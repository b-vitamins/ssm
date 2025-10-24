#include "common.h"

#include <ATen/ATen.h>

#include <tuple>
#include <vector>

namespace ssm {
namespace cpu {

namespace {

at::Tensor normalize_scan_param(const std::string& name, const at::Tensor& param,
                                int64_t batch, int64_t dim, int64_t state_dim,
                                int64_t length, at::ScalarType dtype,
                                const at::Device& device) {
  if (param.dim() == 2) {
    if (param.size(0) != dim || param.size(1) != state_dim) {
      TORCH_CHECK(false, name, " must have shape (D, N).");
    }
    return to_compute(param, dtype, device)
        .view({1, dim, state_dim, 1})
        .expand({batch, dim, state_dim, length});
  }
  if (param.dim() == 3) {
    if (param.size(0) != batch || param.size(1) != dim ||
        param.size(2) != state_dim) {
      TORCH_CHECK(false, name,
                  " must have shape (B, D, N) when 3-D.");
    }
    return to_compute(param, dtype, device)
        .unsqueeze(-1)
        .expand({batch, dim, state_dim, length});
  }
  if (param.dim() == 4) {
    TORCH_CHECK(param.size(0) == batch && param.size(3) == length,
                name, " must have shape (B, G, N, L) when 4-D.");
    auto groups = param.size(1);
    TORCH_CHECK(dim % groups == 0, "Group dimension must divide D.");
    auto repeated = to_compute(param, dtype, device)
                        .repeat_interleave(dim / groups, 1);
    TORCH_CHECK(repeated.size(2) == state_dim, name,
                " has mismatched state dimension.");
    return repeated;
  }
  TORCH_CHECK(false, "Unsupported rank for ", name, ".");
}

}  // namespace

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

  auto u_compute = to_compute(u, compute_dtype, device);
  auto delta_compute = to_compute(delta, compute_dtype, device);
  auto A_compute = to_compute(A, compute_dtype, device);

  auto B_full = normalize_scan_param("B", B, batch, dim, state_dim, length,
                                     compute_dtype, device);
  auto C_full = normalize_scan_param("C", C, batch, dim, state_dim, length,
                                     compute_dtype, device);

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
    TORCH_CHECK(tensor.dim() == 3 && tensor.sizes() == u.sizes(),
                "z must have shape (B, D, L).");
    z_compute = to_compute(tensor, compute_dtype, device);
  }

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

  auto state = at::zeros({batch, dim, state_dim},
                         u.options().dtype(compute_dtype));
  std::vector<at::Tensor> outputs;
  outputs.reserve(length);

  const auto A_expanded = A_compute.unsqueeze(0);  // (1, D, N)

  for (int64_t t = 0; t < length; ++t) {
    auto delta_t = delta_compute.select(-1, t);        // (B, D)
    auto decay = (delta_t.unsqueeze(-1) * A_expanded).exp();
    auto B_t = B_full.select(-1, t);                   // (B, D, N)
    auto C_t = C_full.select(-1, t);                   // (B, D, N)
    auto drive = B_t * u_compute.select(-1, t).unsqueeze(-1);
    state = decay * state + delta_t.unsqueeze(-1) * drive;
    auto y_t = (state * C_t).sum(-1);

    if (D_compute.has_value()) {
      y_t = y_t +
            D_compute.value().view({1, dim}) * u_compute.select(-1, t);
    }

    if (z_compute.has_value()) {
      y_t = y_t * at::silu(z_compute.value().select(-1, t));
    }

    outputs.push_back(y_t);
  }

  auto output = at::stack(outputs, -1).to(u.scalar_type());
  auto last_state = state.to(u.scalar_type());

  if (!return_last_state) {
    return std::make_tuple(output, last_state);
  }
  return std::make_tuple(output, last_state);
}

}  // namespace cpu
}  // namespace ssm

