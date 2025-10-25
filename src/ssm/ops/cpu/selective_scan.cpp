#include "common.h"

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/vec.h>

#include <array>
#include <cmath>
#include <complex>
#include <tuple>
#include <vector>

namespace ssm {
namespace cpu {

std::tuple<at::Tensor, at::Tensor> selective_scan_cpu(
    const at::Tensor& u, const at::Tensor& delta, const at::Tensor& A,
    const at::Tensor& B, const at::Tensor& C, const c10::optional<at::Tensor>& D,
    const c10::optional<at::Tensor>& z,
    const c10::optional<at::Tensor>& dt_bias, bool softplus,
    bool return_last_state) {
  TORCH_CHECK(u.dim() == 3 && delta.dim() == 3,
              "u and delta must have shape (B, D, L).");
  TORCH_CHECK(u.sizes() == delta.sizes(),
              "delta must match the shape of u.");

  const auto batch = u.size(0);
  const auto dim = u.size(1);
  const auto length = u.size(2);
  const auto state_dim = A.size(-1);

  TORCH_CHECK(A.dim() == 2 && A.size(0) == dim && A.size(1) == state_dim,
              "A must have shape (D, N).");

  auto compute_dtype = get_compute_dtype(u);
  if (A.scalar_type() != at::kFloat && A.scalar_type() != compute_dtype) {
    compute_dtype = promote_compute_dtype(compute_dtype, A.scalar_type());
  }

  const auto device = u.device();

  auto u_compute = to_compute(u, compute_dtype, device).contiguous();
  auto delta_compute = to_compute(delta, compute_dtype, device);
  auto A_compute = to_compute(A, compute_dtype, device).contiguous();

  if (dt_bias.has_value()) {
    const auto& tensor = dt_bias.value();
    TORCH_CHECK(tensor.dim() == 1 && tensor.size(0) == dim,
                "dt_bias must have shape (D,).");
    delta_compute =
        delta_compute +
        to_compute(tensor, compute_dtype, device).view({1, dim, 1});
  }

  if (softplus) {
    delta_compute = at::softplus(delta_compute);
  }

  delta_compute = delta_compute.contiguous();

  auto B_info = make_scan_param("B", B, batch, dim, state_dim, length,
                                compute_dtype, device);
  auto C_info = make_scan_param("C", C, batch, dim, state_dim, length,
                                compute_dtype, device);

  c10::optional<at::Tensor> D_compute = c10::nullopt;
  if (D.has_value()) {
    const auto& tensor = D.value();
    TORCH_CHECK(tensor.dim() == 1 && tensor.size(0) == dim,
                "D must have shape (D,).");
    D_compute = to_compute(tensor, compute_dtype, device).contiguous();
  }

  c10::optional<at::Tensor> z_gate = c10::nullopt;
  if (z.has_value()) {
    const auto& tensor = z.value();
    TORCH_CHECK(tensor.dim() == 3 && tensor.sizes() == u.sizes(),
                "z must have shape (B, D, L).");
    z_gate = at::silu(to_compute(tensor, compute_dtype, device)).contiguous();
  }

  auto state =
      at::zeros({batch, dim, state_dim}, u.options().dtype(compute_dtype))
          .contiguous();
  auto output =
      at::empty({batch, dim, length}, u.options().dtype(compute_dtype));

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      compute_dtype, "selective_scan_cpu", [&]() {
        const auto* u_ptr = u_compute.data_ptr<scalar_t>();
        const auto* delta_ptr = delta_compute.data_ptr<scalar_t>();
        const auto* A_ptr = A_compute.data_ptr<scalar_t>();
        const auto* D_ptr =
            D_compute.has_value() ? D_compute.value().data_ptr<scalar_t>()
                                  : nullptr;
        const auto* z_ptr =
            z_gate.has_value() ? z_gate.value().data_ptr<scalar_t>() : nullptr;
        auto* state_ptr = state.data_ptr<scalar_t>();
        auto* out_ptr = output.data_ptr<scalar_t>();

        const auto row_stride = length;
        const auto state_stride = state_dim;

        at::parallel_for(0, batch * dim, 0, [&](int64_t start, int64_t end) {
          for (const auto idx : c10::irange(start, end)) {
            const auto b = idx / dim;
            const auto d_idx = idx % dim;
            const auto row_offset = idx * row_stride;
            const auto state_offset = idx * state_stride;

            const auto* A_row = A_ptr + d_idx * state_stride;
            auto* state_row = state_ptr + state_offset;
            auto* out_row = out_ptr + row_offset;
            const auto* u_row = u_ptr + row_offset;
            const auto* delta_row = delta_ptr + row_offset;
            const auto* z_row =
                z_ptr != nullptr ? z_ptr + row_offset : nullptr;

            const auto B_row =
                make_scan_param_row<scalar_t>(B_info, b, d_idx);
            const auto C_row =
                make_scan_param_row<scalar_t>(C_info, b, d_idx);

            const auto d_skip =
                D_ptr != nullptr ? D_ptr[d_idx] : scalar_t(0);

            for (const auto t : c10::irange(length)) {
              const auto delta_val = delta_row[t];
              const auto u_val = u_row[t];
              const auto delta_u = delta_val * u_val;

              const auto* B_ptr = B_row.data(t);
              const auto* C_ptr = C_row.data(t);
              const auto B_stride = B_row.state_stride;
              const auto C_stride = C_row.state_stride;

              scalar_t y_val = scalar_t(0);

              auto scalar_update = [&](int64_t n) {
                const auto decay = std::exp(delta_val * A_row[n]);
                const auto drive = B_ptr[n * B_stride] * delta_u;
                const auto updated = decay * state_row[n] + drive;
                state_row[n] = updated;
                y_val += updated * C_ptr[n * C_stride];
              };

              if constexpr (!c10::is_complex<scalar_t>::value &&
                            at::vec::is_vec_specialized_for<scalar_t>::value) {
                if (B_stride == 1 && C_stride == 1) {
                  using Vec = at::vec::Vectorized<scalar_t>;
                  const auto vec_size = Vec::size();
                  const Vec delta_vec(delta_val);
                  const Vec delta_u_vec(delta_u);
                  Vec y_vec = Vec(scalar_t(0));
                  int64_t n = 0;
                  for (; n + vec_size <= state_dim; n += vec_size) {
                    auto a_vec = Vec::loadu(A_row + n);
                    auto prev_state = Vec::loadu(state_row + n);
                    auto b_vec = Vec::loadu(B_ptr + n);
                    auto c_vec = Vec::loadu(C_ptr + n);
                    auto decay_vec = (a_vec * delta_vec).exp();
                    auto updated = decay_vec * prev_state + b_vec * delta_u_vec;
                    y_vec = y_vec + updated * c_vec;
                    updated.store(state_row + n);
                  }
                  alignas(alignof(scalar_t)) std::array<scalar_t, Vec::size()> buf{};
                  y_vec.store(buf.data());
                  for (const auto value : buf) {
                    y_val += value;
                  }
                  for (; n < state_dim; ++n) {
                    scalar_update(n);
                  }
                } else {
                  for (const auto n : c10::irange(state_dim)) {
                    scalar_update(n);
                  }
                }
              } else {
                for (const auto n : c10::irange(state_dim)) {
                  scalar_update(n);
                }
              }

              if (D_ptr != nullptr) {
                y_val += d_skip * u_val;
              }

              if (z_row != nullptr) {
                y_val *= z_row[t];
              }

              out_row[t] = y_val;
            }
          }
        });
      });

  auto output_cast = output.to(u.scalar_type());
  at::Tensor last_state;
  if (return_last_state) {
    last_state = state.to(u.scalar_type());
  }
  return std::make_tuple(output_cast, last_state);
}

namespace {

template <typename scalar_t>
inline scalar_t conj_if_complex(const scalar_t& value) {
  if constexpr (c10::is_complex<scalar_t>::value) {
    return std::conj(value);
  }
  return value;
}

template <typename scalar_t>
inline scalar_t silu_grad(const scalar_t& x) {
  const scalar_t one = scalar_t(1);
  const auto neg_x = -x;
  const scalar_t sigmoid = one / (one + std::exp(neg_x));
  return sigmoid * (one + x * (one - sigmoid));
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
           at::Tensor, at::Tensor, at::Tensor>
selective_scan_backward_cpu(
    const at::Tensor& grad_output,
    const c10::optional<at::Tensor>& grad_last_state, const at::Tensor& u,
    const at::Tensor& delta, const at::Tensor& A, const at::Tensor& B,
    const at::Tensor& C, const c10::optional<at::Tensor>& D,
    const c10::optional<at::Tensor>& z,
    const c10::optional<at::Tensor>& dt_bias, bool softplus) {
  TORCH_CHECK(u.dim() == 3 && delta.dim() == 3,
              "u and delta must have shape (B, D, L).");
  TORCH_CHECK(grad_output.dim() == 3,
              "grad_output must have shape (B, D, L).");
  TORCH_CHECK(u.sizes() == grad_output.sizes(),
              "grad_output must match the output shape of forward.");
  TORCH_CHECK(u.sizes() == delta.sizes(),
              "delta must match the shape of u.");

  const auto batch = u.size(0);
  const auto dim = u.size(1);
  const auto length = u.size(2);
  const auto state_dim = A.size(-1);

  TORCH_CHECK(A.dim() == 2 && A.size(0) == dim && A.size(1) == state_dim,
              "A must have shape (D, N).");

  auto compute_dtype = get_compute_dtype(u);
  if (A.scalar_type() != at::kFloat && A.scalar_type() != compute_dtype) {
    compute_dtype = promote_compute_dtype(compute_dtype, A.scalar_type());
  }

  const auto device = u.device();

  auto u_compute = to_compute(u, compute_dtype, device).contiguous();
  auto delta_base = to_compute(delta, compute_dtype, device);
  auto grad_output_compute =
      to_compute(grad_output, compute_dtype, device).contiguous();
  auto A_compute = to_compute(A, compute_dtype, device).contiguous();

  if (dt_bias.has_value()) {
    const auto& tensor = dt_bias.value();
    TORCH_CHECK(tensor.dim() == 1 && tensor.size(0) == dim,
                "dt_bias must have shape (D,).");
    delta_base =
        delta_base + to_compute(tensor, compute_dtype, device).view({1, dim, 1});
  }

  at::Tensor delta_derivative;
  at::Tensor delta_final;
  if (softplus) {
    delta_derivative = at::sigmoid(delta_base);
    delta_final = at::softplus(delta_base);
  } else {
    delta_derivative = at::ones_like(delta_base);
    delta_final = delta_base;
  }

  delta_final = delta_final.contiguous();
  delta_derivative = delta_derivative.contiguous();

  auto B_info = make_scan_param("B", B, batch, dim, state_dim, length,
                                compute_dtype, device);
  auto C_info = make_scan_param("C", C, batch, dim, state_dim, length,
                                compute_dtype, device);

  c10::optional<at::Tensor> D_compute = c10::nullopt;
  if (D.has_value()) {
    const auto& tensor = D.value();
    TORCH_CHECK(tensor.dim() == 1 && tensor.size(0) == dim,
                "D must have shape (D,).");
    D_compute = to_compute(tensor, compute_dtype, device).contiguous();
  }

  c10::optional<at::Tensor> z_input = c10::nullopt;
  c10::optional<at::Tensor> z_gate = c10::nullopt;
  if (z.has_value()) {
    const auto& tensor = z.value();
    TORCH_CHECK(tensor.dim() == 3 && tensor.sizes() == u.sizes(),
                "z must have shape (B, D, L).");
    auto z_compute = to_compute(tensor, compute_dtype, device).contiguous();
    z_gate = at::silu(z_compute);
    z_input = std::move(z_compute);
  }

  at::Tensor grad_last_state_compute;
  if (grad_last_state.has_value()) {
    const auto& tensor = grad_last_state.value();
    TORCH_CHECK(tensor.dim() == 3 && tensor.size(0) == batch &&
                    tensor.size(1) == dim && tensor.size(2) == state_dim,
                "grad_last_state must have shape (B, D, N).");
    grad_last_state_compute =
        to_compute(tensor, compute_dtype, device).contiguous();
  }

  at::Tensor grad_u_compute;
  at::Tensor grad_delta_post;
  at::Tensor grad_A_compute;
  at::Tensor grad_B_compute;
  at::Tensor grad_C_compute;
  at::Tensor grad_D_compute_tensor;
  at::Tensor grad_z_compute_tensor;
  at::Tensor grad_dt_bias_compute;

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      compute_dtype, "selective_scan_backward_cpu", [&]() {
        auto grad_u_local = at::zeros_like(u_compute);
        auto grad_delta_local = at::zeros_like(delta_final);
        auto grad_A_local = at::zeros_like(A_compute);

        auto B_grad_info = B_info;
        B_grad_info.tensor = at::zeros_like(B_info.tensor);
        auto C_grad_info = C_info;
        C_grad_info.tensor = at::zeros_like(C_info.tensor);

        at::Tensor grad_D_local;
        scalar_t* grad_D_ptr = nullptr;
        const scalar_t* D_ptr = nullptr;
        if (D_compute.has_value()) {
          grad_D_local = at::zeros_like(D_compute.value());
          grad_D_ptr = grad_D_local.data_ptr<scalar_t>();
          D_ptr = D_compute.value().data_ptr<scalar_t>();
        }

        at::Tensor grad_z_local;
        scalar_t* grad_z_ptr = nullptr;
        const scalar_t* z_input_ptr = nullptr;
        const scalar_t* z_gate_ptr = nullptr;
        int64_t z_stride_batch = 0;
        int64_t z_stride_dim = 0;
        int64_t z_stride_time = 0;
        if (z_input.has_value()) {
          grad_z_local = at::zeros_like(z_input.value());
          grad_z_ptr = grad_z_local.data_ptr<scalar_t>();
          const auto& z_in_tensor = z_input.value();
          const auto& z_gate_tensor = z_gate.value();
          z_input_ptr = z_in_tensor.data_ptr<scalar_t>();
          z_gate_ptr = z_gate_tensor.data_ptr<scalar_t>();
          z_stride_batch = z_in_tensor.stride(0);
          z_stride_dim = z_in_tensor.stride(1);
          z_stride_time = z_in_tensor.stride(2);
        }

        scalar_t* grad_last_ptr_base = nullptr;
        int64_t grad_last_stride_batch = 0;
        int64_t grad_last_stride_dim = 0;
        if (grad_last_state_compute.defined()) {
          grad_last_ptr_base =
              grad_last_state_compute.data_ptr<scalar_t>();
          grad_last_stride_batch = grad_last_state_compute.stride(0);
          grad_last_stride_dim = grad_last_state_compute.stride(1);
        }

        auto state_history =
            at::zeros({batch, dim, length + 1, state_dim},
                      u.options().dtype(compute_dtype));

        auto state_acc = state_history.accessor<scalar_t, 4>();
        auto u_acc = u_compute.accessor<scalar_t, 3>();
        auto delta_acc = delta_final.accessor<scalar_t, 3>();
        auto grad_out_acc = grad_output_compute.accessor<scalar_t, 3>();
        auto grad_u_acc = grad_u_local.accessor<scalar_t, 3>();
        auto grad_delta_acc = grad_delta_local.accessor<scalar_t, 3>();
        auto grad_A_acc = grad_A_local.accessor<scalar_t, 2>();

        const auto* A_ptr = A_compute.data_ptr<scalar_t>();

        for (const auto b : c10::irange(batch)) {
          for (const auto d_idx : c10::irange(dim)) {
            const auto* A_row = A_ptr + d_idx * state_dim;
            auto B_row = make_scan_param_row<scalar_t>(B_info, b, d_idx);
            auto C_row = make_scan_param_row<scalar_t>(C_info, b, d_idx);
            auto B_grad_row = make_scan_param_row<scalar_t>(B_grad_info, b, d_idx);
            auto C_grad_row = make_scan_param_row<scalar_t>(C_grad_info, b, d_idx);

            auto state_slice = state_acc[b][d_idx];

            for (const auto t : c10::irange(length)) {
              const auto delta_val = delta_acc[b][d_idx][t];
              const auto u_val = u_acc[b][d_idx][t];
              const auto delta_u = delta_val * u_val;

              auto* prev_state = &state_slice[t][0];
              auto* next_state = &state_slice[t + 1][0];

              const auto* B_ptr = B_row.data(t);
              const auto B_stride = B_row.state_stride;

              for (const auto n : c10::irange(state_dim)) {
                const auto decay = std::exp(delta_val * A_row[n]);
                const auto drive = B_ptr[n * B_stride] * delta_u;
                next_state[n] = decay * prev_state[n] + drive;
              }
            }

            std::vector<scalar_t> grad_state(state_dim, scalar_t(0));
            if (grad_last_ptr_base != nullptr) {
              const auto* grad_last_ptr =
                  grad_last_ptr_base + b * grad_last_stride_batch +
                  d_idx * grad_last_stride_dim;
              for (const auto n : c10::irange(state_dim)) {
                grad_state[n] = grad_last_ptr[n];
              }
            }

            for (int64_t t = length - 1; t >= 0; --t) {
              const auto delta_val = delta_acc[b][d_idx][t];
              const auto u_val = u_acc[b][d_idx][t];
              const auto grad_out = grad_out_acc[b][d_idx][t];

              const auto* B_ptr = B_row.data(t);
              const auto* C_ptr = C_row.data(t);
              auto* B_grad_ptr = const_cast<scalar_t*>(B_grad_row.data(t));
              auto* C_grad_ptr = const_cast<scalar_t*>(C_grad_row.data(t));
              const auto B_stride = B_row.state_stride;
              const auto C_stride = C_row.state_stride;

              auto* prev_state = &state_slice[t][0];
              auto* curr_state = &state_slice[t + 1][0];

              scalar_t y_pre = scalar_t(0);
              for (const auto n : c10::irange(state_dim)) {
                y_pre += curr_state[n] * C_ptr[n * C_stride];
              }

              scalar_t grad_pre_gate = grad_out;
              scalar_t pre_gate = y_pre;
              scalar_t d_skip = scalar_t(0);
              if (D_ptr != nullptr) {
                d_skip = D_ptr[d_idx];
                pre_gate += d_skip * u_val;
              }

              if (z_input_ptr != nullptr) {
                const auto offset =
                    b * z_stride_batch + d_idx * z_stride_dim + t * z_stride_time;
                const auto z_val = z_gate_ptr[offset];
                grad_pre_gate *= z_val;
                const auto z_input_val = z_input_ptr[offset];
                grad_z_ptr[offset] +=
                    grad_out * pre_gate * silu_grad(z_input_val);
              }

              if (D_ptr != nullptr) {
                grad_u_acc[b][d_idx][t] +=
                    grad_pre_gate * conj_if_complex(d_skip);
                grad_D_ptr[d_idx] += grad_pre_gate * conj_if_complex(u_val);
              }

              const auto grad_y_pre = grad_pre_gate;

              std::vector<scalar_t> grad_state_prev(state_dim, scalar_t(0));
              scalar_t grad_delta_val = scalar_t(0);
              scalar_t grad_delta_u = scalar_t(0);

              for (const auto n : c10::irange(state_dim)) {
                const auto C_val = C_ptr[n * C_stride];
                const auto state_grad =
                    grad_state[n] + grad_y_pre * conj_if_complex(C_val);
                grad_state[n] = state_grad;
                C_grad_ptr[n * C_stride] +=
                    grad_y_pre * conj_if_complex(curr_state[n]);

                const auto decay = std::exp(delta_val * A_row[n]);
                grad_state_prev[n] += state_grad * decay;

                const auto state_prev = prev_state[n];
                const auto grad_decay = state_grad * state_prev;
                const auto decay_val = decay;
                grad_A_acc[d_idx][n] += grad_decay * (delta_val * decay_val);
                grad_delta_val += grad_decay * (A_row[n] * decay_val);

                const auto grad_drive = state_grad;
                const auto B_val = B_ptr[n * B_stride];
                const auto delta_u = delta_val * u_val;
                B_grad_ptr[n * B_stride] +=
                    grad_drive * conj_if_complex(delta_u);
                grad_delta_u += grad_drive * conj_if_complex(B_val);
              }

              grad_delta_val += grad_delta_u * conj_if_complex(u_val);
              grad_delta_acc[b][d_idx][t] += grad_delta_val;
              grad_u_acc[b][d_idx][t] +=
                  grad_delta_u * conj_if_complex(delta_val);

              grad_state = std::move(grad_state_prev);
            }
          }
        }

        grad_u_compute = grad_u_local;
        grad_delta_post = grad_delta_local;
        grad_A_compute = grad_A_local;
        grad_B_compute = B_grad_info.tensor;
        grad_C_compute = C_grad_info.tensor;
        if (D_compute.has_value()) {
          grad_D_compute_tensor = grad_D_local;
        }
        if (z_input.has_value()) {
          grad_z_compute_tensor = grad_z_local;
        }
      });

  auto grad_delta_input = grad_delta_post * delta_derivative;
  if (dt_bias.has_value()) {
    grad_dt_bias_compute = grad_delta_input.sum({0, 2});
  }

  auto grad_u_result = grad_u_compute.to(u.scalar_type());
  auto grad_delta_result = grad_delta_input.to(delta.scalar_type());
  auto grad_A_result = grad_A_compute.to(A.scalar_type());
  auto grad_B_result = grad_B_compute.to(B.scalar_type());
  auto grad_C_result = grad_C_compute.to(C.scalar_type());

  at::Tensor grad_D_result;
  if (D.has_value()) {
    grad_D_result = grad_D_compute_tensor.to(D.value().scalar_type());
  }

  at::Tensor grad_z_result;
  if (z.has_value()) {
    grad_z_result = grad_z_compute_tensor.to(z.value().scalar_type());
  }

  at::Tensor grad_dt_bias_result;
  if (dt_bias.has_value()) {
    grad_dt_bias_result =
        grad_dt_bias_compute.to(dt_bias.value().scalar_type());
  }

  return {grad_u_result,      grad_delta_result, grad_A_result,
          grad_B_result,      grad_C_result,      grad_D_result,
          grad_z_result,      grad_dt_bias_result};
}

}  // namespace cpu
}  // namespace ssm

