#include "common.h"

#include <ATen/ATen.h>

#include <algorithm>
#include <cctype>
#include <string>

namespace ssm {
namespace cpu {

at::Tensor dw_causal_conv_cpu(const at::Tensor& x, const at::Tensor& weight,
                              const c10::optional<at::Tensor>& bias,
                              const std::string& activation) {
  TORCH_CHECK(x.dim() == 3, "x must have shape (B, C, L) or (B, L, C).");

  bool channels_first = false;
  int64_t batch = x.size(0);
  int64_t channels = 0;
  int64_t length = 0;

  if (x.size(1) == weight.size(0)) {
    channels_first = true;
    batch = x.size(0);
    channels = x.size(1);
    length = x.size(2);
  } else if (x.size(2) == weight.size(0)) {
    channels_first = false;
    batch = x.size(0);
    length = x.size(1);
    channels = x.size(2);
  } else {
    TORCH_CHECK(false, "weight channel dimension must match x.");
  }

  int64_t kernel_size = 0;
  at::Tensor weight_conv;
  if (weight.dim() == 2) {
    TORCH_CHECK(weight.size(0) == channels,
                "weight has incompatible shape.");
    kernel_size = weight.size(1);
    weight_conv = weight.unsqueeze(1);
  } else if (weight.dim() == 3) {
    TORCH_CHECK(weight.size(0) == channels && weight.size(1) == 1,
                "Expected depthwise weights with shape (C, 1, K).");
    kernel_size = weight.size(2);
    weight_conv = weight;
  } else {
    TORCH_CHECK(false, "weight must have 2 or 3 dimensions.");
  }

  if (bias.has_value()) {
    const auto& bias_tensor = bias.value();
    TORCH_CHECK(bias_tensor.dim() == 1 && bias_tensor.size(0) == channels,
                "bias must have shape (C,).");
  }

  auto compute_dtype = get_compute_dtype(x);
  const auto device = x.device();

  at::Tensor x_conv = channels_first ? x : x.permute({0, 2, 1});
  x_conv = to_compute(x_conv, compute_dtype, device);

  auto weight_conv_cast = to_compute(weight_conv, compute_dtype, device);
  c10::optional<at::Tensor> bias_conv = c10::nullopt;
  if (bias.has_value()) {
    bias_conv = to_compute(bias.value(), compute_dtype, device);
  }

  int64_t padding = kernel_size - 1;
  auto x_padded = at::constant_pad_nd(x_conv, {padding, 0});
  auto out = at::conv1d(x_padded, weight_conv_cast, bias_conv,
                        at::IntArrayRef({1}), at::IntArrayRef({0}),
                        at::IntArrayRef({1}), channels);

  auto act = activation;
  std::transform(act.begin(), act.end(), act.begin(), ::tolower);

  if (act == "silu") {
    out = at::silu(out);
  } else if (act == "relu") {
    out = at::relu(out);
  } else if (act == "identity") {
    // no-op
  } else {
    TORCH_CHECK(false, "Unsupported activation '" + activation + "'.");
  }

  if (!channels_first) {
    out = out.permute({0, 2, 1});
  }

  return out.to(x.scalar_type());
}

}  // namespace cpu
}  // namespace ssm

