#include "common.h"

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/vec.h>

#include <array>
#include <cmath>
#include <complex>
#include <tuple>

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

}  // namespace cpu
}  // namespace ssm

