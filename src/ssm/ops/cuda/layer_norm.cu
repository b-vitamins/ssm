#include "common.h"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>

namespace ssm {
namespace cuda {

at::Tensor fused_layer_norm_cuda(
    const at::Tensor& x, const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    const c10::optional<at::Tensor>& residual, bool is_rms, double eps,
    bool prenorm, bool residual_in_fp32) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor.");
  TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor.");
  if (bias.has_value()) {
    TORCH_CHECK(bias.value().is_cuda(), "bias must be a CUDA tensor.");
  }
  if (residual.has_value()) {
    TORCH_CHECK(residual.value().is_cuda(), "residual must be a CUDA tensor.");
  }
  c10::cuda::CUDAGuard guard(x.device());

  TORCH_CHECK(x.dim() == 3, "x must have shape (B, L, D).");
  TORCH_CHECK(weight.dim() == 1 && weight.size(0) == x.size(-1),
              "weight must match the last dimension of x.");
  if (bias.has_value()) {
    const auto& bias_tensor = bias.value();
    TORCH_CHECK(bias_tensor.dim() == 1 && bias_tensor.size(0) == x.size(-1),
                "bias must match the last dimension of x.");
  }
  if (residual.has_value()) {
    const auto& residual_tensor = residual.value();
    TORCH_CHECK(residual_tensor.sizes() == x.sizes(),
                "residual must match the shape of x.");
  }

  auto compute_dtype = get_compute_dtype(x);
  auto base_dtype =
      residual_in_fp32 && residual.has_value() ? at::kFloat : compute_dtype;
  const auto device = x.device();

  auto x_compute = to_compute(x, base_dtype, device);
  at::Tensor residual_compute;
  if (residual.has_value()) {
    residual_compute = to_compute(residual.value(), base_dtype, device);
  }

  at::Tensor norm_input;
  if (residual.has_value() && prenorm) {
    norm_input = x_compute + residual_compute;
  } else {
    norm_input = x_compute;
  }

  at::Tensor normed;
  if (is_rms) {
    auto mean_square = norm_input.pow(2).mean(-1, true);
    normed = norm_input * at::rsqrt(mean_square + eps);
  } else {
    auto mean = norm_input.mean(-1, true);
    auto var = (norm_input - mean).pow(2).mean(-1, true);
    normed = (norm_input - mean) * at::rsqrt(var + eps);
  }

  auto weight_compute = to_compute(weight, normed.scalar_type(), device);
  at::Tensor output = normed * weight_compute.view({1, 1, -1});

  if (bias.has_value()) {
    auto bias_compute = to_compute(bias.value(), normed.scalar_type(), device);
    output = output + bias_compute.view({1, 1, -1});
  }

  if (!prenorm && residual.has_value()) {
    output = output + residual_compute.to(output.scalar_type());
  }

  return output.to(x.scalar_type());
}

}  // namespace cuda
}  // namespace ssm

