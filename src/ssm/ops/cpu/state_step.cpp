#include "common.h"

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/vec.h>

#include <array>
#include <cmath>
#include <complex>

namespace ssm {
namespace cpu {

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

}  // namespace cpu
}  // namespace ssm

