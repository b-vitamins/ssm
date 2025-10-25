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

at::Tensor selective_state_step_cpu(
    at::Tensor state, const at::Tensor& x, const at::Tensor& dt,
    const at::Tensor& A, const at::Tensor& B, const at::Tensor& C,
    const c10::optional<at::Tensor>& D, const c10::optional<at::Tensor>& z,
    const c10::optional<at::Tensor>& dt_bias, bool softplus) {
  TORCH_CHECK(state.dim() == 3, "state must have shape (B, D, N).");
  TORCH_CHECK(x.dim() == 2 && dt.dim() == 2,
              "x and dt must have shape (B, D).");

  const auto batch = state.size(0);
  const auto dim = state.size(1);
  const auto state_dim = state.size(2);

  TORCH_CHECK(x.size(0) == batch && x.size(1) == dim,
              "x and dt must match the first two dimensions of state.");
  TORCH_CHECK(dt.size(0) == batch && dt.size(1) == dim,
              "x and dt must match the first two dimensions of state.");
  TORCH_CHECK(A.dim() == 2 && A.size(0) == dim && A.size(1) == state_dim,
              "A must have shape (D, N).");

  auto compute_dtype = get_compute_dtype(state);
  if (A.scalar_type() != at::kFloat && A.scalar_type() != compute_dtype) {
    compute_dtype = promote_compute_dtype(compute_dtype, A.scalar_type());
  }

  const auto device = state.device();

  auto state_compute = to_compute(state, compute_dtype, device).contiguous();
  auto x_compute = to_compute(x, compute_dtype, device).contiguous();
  auto dt_compute = to_compute(dt, compute_dtype, device);
  auto A_compute = to_compute(A, compute_dtype, device).contiguous();

  if (dt_bias.has_value()) {
    const auto& tensor = dt_bias.value();
    TORCH_CHECK(tensor.dim() == 1 && tensor.size(0) == dim,
                "dt_bias must have shape (D,).");
    dt_compute =
        dt_compute + to_compute(tensor, compute_dtype, device).view({1, dim});
  }

  if (softplus) {
    dt_compute = at::softplus(dt_compute);
  }

  dt_compute = dt_compute.contiguous();

  auto maybe_group = [&](const std::string& name, const at::Tensor& param) {
    if (param.dim() == 3 && param.size(0) == batch && param.size(1) != dim) {
      auto groups = param.size(1);
      TORCH_CHECK(dim % groups == 0,
                  name, " group dimension must divide D.");
      TORCH_CHECK(param.size(2) == state_dim,
                  name, " must have matching state dimension.");
      return param.repeat_interleave(dim / groups, 1);
    }
    return param;
  };

  auto B_prepared = maybe_group("B", B);
  auto C_prepared = maybe_group("C", C);

  auto B_info = make_scan_param("B", B_prepared, batch, dim, state_dim, 1,
                                compute_dtype, device);
  auto C_info = make_scan_param("C", C_prepared, batch, dim, state_dim, 1,
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
    TORCH_CHECK(tensor.dim() == 2 && tensor.sizes() == x.sizes(),
                "z must have shape (B, D).");
    z_gate = at::silu(to_compute(tensor, compute_dtype, device)).contiguous();
  }

  auto output =
      at::empty({batch, dim}, x.options().dtype(compute_dtype)).contiguous();

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      compute_dtype, "selective_state_step_cpu", [&]() {
        const auto* x_ptr = x_compute.data_ptr<scalar_t>();
        const auto* dt_ptr = dt_compute.data_ptr<scalar_t>();
        const auto* A_ptr = A_compute.data_ptr<scalar_t>();
        const auto* D_ptr =
            D_compute.has_value() ? D_compute.value().data_ptr<scalar_t>()
                                  : nullptr;
        const auto* z_ptr =
            z_gate.has_value() ? z_gate.value().data_ptr<scalar_t>() : nullptr;
        auto* state_ptr = state_compute.data_ptr<scalar_t>();
        auto* out_ptr = output.data_ptr<scalar_t>();

        const auto row_stride = dim;
        const auto state_stride = state_dim;

        at::parallel_for(0, batch * dim, 0, [&](int64_t start, int64_t end) {
          for (const auto idx : c10::irange(start, end)) {
            const auto b = idx / dim;
            const auto d_idx = idx % dim;
            const auto offset = b * row_stride + d_idx;
            const auto state_offset = idx * state_stride;

            const auto dt_val = dt_ptr[offset];
            const auto x_val = x_ptr[offset];

            const auto* A_row = A_ptr + d_idx * state_stride;
            auto* state_row = state_ptr + state_offset;

            const auto B_row =
                make_scan_param_row<scalar_t>(B_info, b, d_idx);
            const auto C_row =
                make_scan_param_row<scalar_t>(C_info, b, d_idx);
            const auto* B_ptr = B_row.data(0);
            const auto* C_ptr = C_row.data(0);
            const auto B_stride = B_row.state_stride;
            const auto C_stride = C_row.state_stride;

            scalar_t y_val = scalar_t(0);

            const bool vectorizable = B_stride == 1 && C_stride == 1;

            if (vectorizable) {
              using Vec = at::vec::Vectorized<scalar_t>;
              constexpr auto kVecSize = Vec::size();
              const auto limit = (state_dim / kVecSize) * kVecSize;

              const Vec dt_vec(dt_val);
              const Vec x_vec(x_val);

              std::array<scalar_t, kVecSize> buffer{};

              int64_t n = 0;
              for (; n < limit; n += kVecSize) {
                const auto A_vec = Vec::loadu(A_row + n);
                const auto state_vec = Vec::loadu(state_row + n);
                const auto B_vec = Vec::loadu(B_ptr + n);
                const auto C_vec = Vec::loadu(C_ptr + n);

                const auto decay_vec = (A_vec * dt_vec).exp();
                const auto drive_vec = B_vec * x_vec;
                const auto updated_vec =
                    decay_vec * state_vec + dt_vec * drive_vec;

                updated_vec.store(state_row + n);

                const auto y_vec = updated_vec * C_vec;
                y_vec.store(buffer.data());
                for (const auto& value : buffer) {
                  y_val += value;
                }
              }

              for (; n < state_dim; ++n) {
                const auto decay = std::exp(dt_val * A_row[n]);
                const auto drive = B_ptr[n] * x_val;
                const auto updated = decay * state_row[n] + dt_val * drive;
                state_row[n] = updated;
                y_val += updated * C_ptr[n];
              }
            } else {
              for (const auto n : c10::irange(state_dim)) {
                const auto decay = std::exp(dt_val * A_row[n]);
                const auto drive = B_ptr[n * B_stride] * x_val;
                const auto updated = decay * state_row[n] + dt_val * drive;
                state_row[n] = updated;
                y_val += updated * C_ptr[n * C_stride];
              }
            }

            if (D_ptr != nullptr) {
              y_val += D_ptr[d_idx] * x_val;
            }

            if (z_ptr != nullptr) {
              y_val *= z_ptr[offset];
            }

            out_ptr[offset] = y_val;
          }
        });
      });

  state.copy_(state_compute.to(state.scalar_type()));
  return output.to(x.scalar_type());
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
           at::Tensor, c10::optional<at::Tensor>, c10::optional<at::Tensor>,
           c10::optional<at::Tensor>>
selective_state_step_backward_cpu(
    const at::Tensor& grad_output,
    const c10::optional<at::Tensor>& grad_state, const at::Tensor& state,
    const at::Tensor& x, const at::Tensor& dt, const at::Tensor& A,
    const at::Tensor& B, const at::Tensor& C,
    const c10::optional<at::Tensor>& D, const c10::optional<at::Tensor>& z,
    const c10::optional<at::Tensor>& dt_bias, bool softplus) {
  TORCH_CHECK(state.dim() == 3, "state must have shape (B, D, N).");
  TORCH_CHECK(x.dim() == 2 && dt.dim() == 2,
              "x and dt must have shape (B, D).");
  TORCH_CHECK(grad_output.dim() == 2,
              "grad_output must have shape (B, D).");

  const auto batch = state.size(0);
  const auto dim = state.size(1);
  const auto state_dim = state.size(2);

  TORCH_CHECK(x.size(0) == batch && x.size(1) == dim,
              "x and dt must match the first two dimensions of state.");
  TORCH_CHECK(dt.size(0) == batch && dt.size(1) == dim,
              "x and dt must match the first two dimensions of state.");
  TORCH_CHECK(grad_output.size(0) == batch && grad_output.size(1) == dim,
              "grad_output must match the output shape of forward.");

  TORCH_CHECK(A.dim() == 2 && A.size(0) == dim && A.size(1) == state_dim,
              "A must have shape (D, N).");

  if (grad_state.has_value()) {
    const auto& tensor = grad_state.value();
    TORCH_CHECK(tensor.dim() == 3 && tensor.size(0) == batch &&
                    tensor.size(1) == dim && tensor.size(2) == state_dim,
                "grad_state must have shape (B, D, N).");
  }

  auto compute_dtype = get_compute_dtype(state);
  if (A.scalar_type() != at::kFloat && A.scalar_type() != compute_dtype) {
    compute_dtype = promote_compute_dtype(compute_dtype, A.scalar_type());
  }

  const auto device = state.device();

  auto state_compute = to_compute(state, compute_dtype, device).contiguous();
  auto x_compute = to_compute(x, compute_dtype, device).contiguous();
  auto dt_base = to_compute(dt, compute_dtype, device);
  auto grad_output_compute =
      to_compute(grad_output, compute_dtype, device).contiguous();
  auto A_compute = to_compute(A, compute_dtype, device).contiguous();

  at::Tensor grad_state_compute;
  if (grad_state.has_value()) {
    grad_state_compute =
        to_compute(grad_state.value(), compute_dtype, device).contiguous();
  } else {
    grad_state_compute = at::zeros_like(state_compute);
  }

  if (dt_bias.has_value()) {
    const auto& tensor = dt_bias.value();
    TORCH_CHECK(tensor.dim() == 1 && tensor.size(0) == dim,
                "dt_bias must have shape (D,).");
    dt_base =
        dt_base + to_compute(tensor, compute_dtype, device).view({1, dim});
  }

  at::Tensor dt_derivative;
  at::Tensor dt_final;
  if (softplus) {
    dt_derivative = at::sigmoid(dt_base);
    dt_final = at::softplus(dt_base);
  } else {
    dt_derivative = at::ones_like(dt_base);
    dt_final = dt_base;
  }

  dt_derivative = dt_derivative.contiguous();
  dt_final = dt_final.contiguous();

  auto maybe_group = [&](const std::string& name, const at::Tensor& param) {
    if (param.dim() == 3 && param.size(0) == batch && param.size(1) != dim) {
      auto groups = param.size(1);
      TORCH_CHECK(dim % groups == 0, name, " group dimension must divide D.");
      TORCH_CHECK(param.size(2) == state_dim,
                  name, " must have matching state dimension.");
      return param.repeat_interleave(dim / groups, 1);
    }
    return param;
  };

  auto B_prepared = maybe_group("B", B);
  auto C_prepared = maybe_group("C", C);

  auto B_info = make_scan_param("B", B_prepared, batch, dim, state_dim, 1,
                                compute_dtype, device);
  auto C_info = make_scan_param("C", C_prepared, batch, dim, state_dim, 1,
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
    TORCH_CHECK(tensor.dim() == 2 && tensor.size(0) == batch &&
                    tensor.size(1) == dim,
                "z must have shape (B, D).");
    auto z_compute = to_compute(tensor, compute_dtype, device).contiguous();
    z_gate = at::silu(z_compute).contiguous();
    z_input = std::move(z_compute);
  }

  at::Tensor grad_state_input_compute;
  at::Tensor grad_x_compute;
  at::Tensor grad_dt_post;
  at::Tensor grad_A_compute;
  at::Tensor grad_B_compute;
  at::Tensor grad_C_compute;
  at::Tensor grad_D_compute_tensor;
  at::Tensor grad_z_compute_tensor;

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      compute_dtype, "selective_state_step_backward_cpu", [&]() {
        auto grad_state_local = at::zeros_like(state_compute);
        auto grad_x_local = at::zeros_like(x_compute);
        auto grad_dt_local = at::zeros_like(dt_final);
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
        if (z_input.has_value()) {
          grad_z_local = at::zeros_like(z_input.value());
          grad_z_ptr = grad_z_local.data_ptr<scalar_t>();
          const auto& z_in_tensor = z_input.value();
          const auto& z_gate_tensor = z_gate.value();
          z_input_ptr = z_in_tensor.data_ptr<scalar_t>();
          z_gate_ptr = z_gate_tensor.data_ptr<scalar_t>();
          z_stride_batch = z_in_tensor.stride(0);
          z_stride_dim = z_in_tensor.stride(1);
        }

        const auto* state_ptr = state_compute.data_ptr<scalar_t>();
        const auto* x_ptr = x_compute.data_ptr<scalar_t>();
        const auto* dt_ptr = dt_final.data_ptr<scalar_t>();
        const auto* grad_out_ptr =
            grad_output_compute.data_ptr<scalar_t>();
        const auto* grad_state_ptr =
            grad_state_compute.data_ptr<scalar_t>();
        const auto* A_ptr = A_compute.data_ptr<scalar_t>();

        auto* grad_state_ptr_out = grad_state_local.data_ptr<scalar_t>();
        auto* grad_x_ptr = grad_x_local.data_ptr<scalar_t>();
        auto* grad_dt_ptr = grad_dt_local.data_ptr<scalar_t>();
        auto* grad_A_ptr = grad_A_local.data_ptr<scalar_t>();

        for (const auto b : c10::irange(batch)) {
          for (const auto d_idx : c10::irange(dim)) {
            const auto offset = b * dim + d_idx;
            const auto state_offset = offset * state_dim;

            const auto x_val = x_ptr[offset];
            const auto dt_val = dt_ptr[offset];
            const auto grad_out_val = grad_out_ptr[offset];

            const auto* state_row = state_ptr + state_offset;
            const auto* grad_state_row = grad_state_ptr + state_offset;
            auto* grad_state_prev_row = grad_state_ptr_out + state_offset;

            const auto* A_row = A_ptr + d_idx * state_dim;
            auto* grad_A_row = grad_A_ptr + d_idx * state_dim;

            auto B_row = make_scan_param_row<scalar_t>(B_info, b, d_idx);
            auto C_row = make_scan_param_row<scalar_t>(C_info, b, d_idx);
            auto B_grad_row =
                make_scan_param_row<scalar_t>(B_grad_info, b, d_idx);
            auto C_grad_row =
                make_scan_param_row<scalar_t>(C_grad_info, b, d_idx);

            const auto* B_ptr = B_row.data(0);
            const auto* C_ptr = C_row.data(0);
            auto* B_grad_ptr = const_cast<scalar_t*>(B_grad_row.data(0));
            auto* C_grad_ptr = const_cast<scalar_t*>(C_grad_row.data(0));
            const auto B_stride = B_row.state_stride;
            const auto C_stride = C_row.state_stride;

            std::vector<scalar_t> decay_vals(state_dim);
            std::vector<scalar_t> updated_vals(state_dim);

            scalar_t y_pre = scalar_t(0);
            for (const auto n : c10::irange(state_dim)) {
              const auto decay = std::exp(dt_val * A_row[n]);
              const auto drive = B_ptr[n * B_stride] * x_val;
              const auto updated = decay * state_row[n] + dt_val * drive;
              decay_vals[n] = decay;
              updated_vals[n] = updated;
              y_pre += updated * C_ptr[n * C_stride];
            }

            scalar_t pre_gate = y_pre;
            if (D_ptr != nullptr) {
              const auto d_val = D_ptr[d_idx];
              pre_gate += d_val * x_val;
            }

            scalar_t grad_pre_gate = grad_out_val;
            if (z_gate_ptr != nullptr) {
              const auto z_offset =
                  b * z_stride_batch + d_idx * z_stride_dim;
              const auto z_val = z_gate_ptr[z_offset];
              grad_pre_gate *= z_val;
              if (grad_z_ptr != nullptr) {
                const auto z_input_val = z_input_ptr[z_offset];
                grad_z_ptr[z_offset] +=
                    grad_out_val * pre_gate * silu_grad(z_input_val);
              }
            }

            if (D_ptr != nullptr) {
              const auto d_val = D_ptr[d_idx];
              grad_x_ptr[offset] += grad_pre_gate * conj_if_complex(d_val);
              grad_D_ptr[d_idx] += grad_pre_gate * conj_if_complex(x_val);
            }

            scalar_t grad_dt_val = scalar_t(0);
            scalar_t grad_dt_x = scalar_t(0);
            const auto dt_x = dt_val * x_val;

            for (const auto n : c10::irange(state_dim)) {
              const auto C_val = C_ptr[n * C_stride];
              const auto B_val = B_ptr[n * B_stride];
              const auto decay = decay_vals[n];
              const auto updated = updated_vals[n];
              const auto state_prev = state_row[n];
              const auto state_grad_next = grad_state_row[n];

              const auto total_grad =
                  state_grad_next + grad_pre_gate * conj_if_complex(C_val);
              grad_state_prev_row[n] = total_grad * decay;

              C_grad_ptr[n * C_stride] +=
                  grad_pre_gate * conj_if_complex(updated);

              const auto grad_decay = total_grad * state_prev;
              grad_A_row[n] += grad_decay * (dt_val * decay);
              grad_dt_val += grad_decay * (A_row[n] * decay);

              grad_dt_x += total_grad * conj_if_complex(B_val);
              B_grad_ptr[n * B_stride] +=
                  total_grad * conj_if_complex(dt_x);
            }

            grad_dt_val += grad_dt_x * conj_if_complex(x_val);
            grad_x_ptr[offset] += grad_dt_x * conj_if_complex(dt_val);

            grad_dt_ptr[offset] = grad_dt_val;
          }
        }

        grad_state_input_compute = grad_state_local;
        grad_x_compute = grad_x_local;
        grad_dt_post = grad_dt_local;
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

  auto grad_dt_input = grad_dt_post * dt_derivative;

  c10::optional<at::Tensor> grad_dt_bias_compute;
  if (dt_bias.has_value()) {
    grad_dt_bias_compute = grad_dt_input.sum(0);
  }

  auto grad_state_result = grad_state_input_compute.to(state.scalar_type());
  auto grad_x_result = grad_x_compute.to(x.scalar_type());
  auto grad_dt_result = grad_dt_input.to(dt.scalar_type());
  auto grad_A_result = grad_A_compute.to(A.scalar_type());
  auto grad_B_result = grad_B_compute.to(B.scalar_type());
  auto grad_C_result = grad_C_compute.to(C.scalar_type());

  c10::optional<at::Tensor> grad_D_result = c10::nullopt;
  if (D.has_value()) {
    grad_D_result = grad_D_compute_tensor.to(D.value().scalar_type());
  }

  c10::optional<at::Tensor> grad_z_result = c10::nullopt;
  if (z.has_value()) {
    grad_z_result = grad_z_compute_tensor.to(z.value().scalar_type());
  }

  c10::optional<at::Tensor> grad_dt_bias_result = c10::nullopt;
  if (dt_bias.has_value()) {
    grad_dt_bias_result =
        grad_dt_bias_compute.value().to(dt_bias.value().scalar_type());
  }

  return {grad_state_result,  grad_x_result,     grad_dt_result,
          grad_A_result,      grad_B_result,     grad_C_result,
          grad_D_result,      grad_z_result,     grad_dt_bias_result};
}

}  // namespace cpu
}  // namespace ssm

