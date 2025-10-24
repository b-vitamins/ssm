#include "common.h"

#include <ATen/ATen.h>

namespace ssm {
namespace cpu {

namespace {

at::Tensor normalize_scan_param_single(const std::string& name,
                                       const at::Tensor& param, int64_t batch,
                                       int64_t dim, int64_t state_dim,
                                       at::ScalarType dtype,
                                       const at::Device& device) {
  if (param.dim() == 2) {
    TORCH_CHECK(param.size(0) == dim && param.size(1) == state_dim,
                name, " must have shape (D, N).");
    return to_compute(param, dtype, device)
        .view({1, dim, state_dim});
  }
  if (param.dim() == 3) {
    TORCH_CHECK(param.size(0) == batch, name,
                " must have shape (B, D, N) when 3-D.");
    if (param.size(1) != dim || param.size(2) != state_dim) {
      TORCH_CHECK(false, name, " must have shape (B, D, N) when 3-D.");
    }
    return to_compute(param, dtype, device);
  }
  TORCH_CHECK(false, "Unsupported rank for ", name, ".");
}

at::Tensor maybe_expand_grouped(const std::string& name,
                                const at::Tensor& param, int64_t dim,
                                int64_t state_dim) {
  if (param.dim() == 3 && param.size(1) != dim) {
    auto groups = param.size(1);
    TORCH_CHECK(dim % groups == 0, name, " group dimension must divide D.");
    TORCH_CHECK(param.size(2) == state_dim, name,
                " must have matching state dimension.");
    return param.repeat_interleave(dim / groups, 1);
  }
  return param;
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

  auto state_compute = to_compute(state, compute_dtype, device);
  auto x_compute = to_compute(x, compute_dtype, device);
  auto dt_compute = to_compute(dt, compute_dtype, device);
  auto A_compute = to_compute(A, compute_dtype, device);

  auto B_prepared = maybe_expand_grouped("B", B, dim, state_dim);
  auto C_prepared = maybe_expand_grouped("C", C, dim, state_dim);

  auto B_expanded = normalize_scan_param_single("B", B_prepared, batch, dim,
                                                state_dim, compute_dtype,
                                                device);
  auto C_expanded = normalize_scan_param_single("C", C_prepared, batch, dim,
                                                state_dim, compute_dtype,
                                                device);

  c10::optional<at::Tensor> D_compute = c10::nullopt;
  if (D.has_value()) {
    const auto& tensor = D.value();
    TORCH_CHECK(tensor.dim() == 1 && tensor.size(0) == dim,
                "D must have shape (D,).");
    D_compute = to_compute(tensor, compute_dtype, device);
  }

  c10::optional<at::Tensor> z_compute = c10::nullopt;
  if (z.has_value()) {
    const auto& tensor = z.value();
    TORCH_CHECK(tensor.dim() == 2 && tensor.sizes() == x.sizes(),
                "z must have shape (B, D).");
    z_compute = to_compute(tensor, compute_dtype, device);
  }

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

  auto decay = (dt_compute.unsqueeze(-1) * A_compute.unsqueeze(0)).exp();
  auto drive = B_expanded * x_compute.unsqueeze(-1);
  auto new_state = decay * state_compute + dt_compute.unsqueeze(-1) * drive;

  auto output = (new_state * C_expanded).sum(-1);

  if (D_compute.has_value()) {
    output = output +
             D_compute.value().view({1, dim}) * x_compute;
  }

  if (z_compute.has_value()) {
    output = output * at::silu(z_compute.value());
  }

  state.copy_(new_state.to(state.scalar_type()));
  return output.to(x.scalar_type());
}

}  // namespace cpu
}  // namespace ssm

