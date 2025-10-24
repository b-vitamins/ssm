#include "common.h"

#include <ATen/ATen.h>
#include <ATen/ops/constant_pad_nd.h>
#include <ATen/ops/conv1d.h>
#include <ATen/ops/relu.h>
#include <ATen/ops/silu.h>

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
  int64_t channels = 0;

  if (x.size(1) == weight.size(0)) {
    channels_first = true;
    channels = x.size(1);
  } else if (x.size(2) == weight.size(0)) {
    channels_first = false;
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
  x_conv = to_compute(x_conv, compute_dtype, device).contiguous();

  auto weight_cast = to_compute(weight_conv, compute_dtype, device).contiguous();
  c10::optional<at::Tensor> bias_cast = c10::nullopt;
  if (bias.has_value()) {
    bias_cast = to_compute(bias.value(), compute_dtype, device).contiguous();
  }

  auto act = activation;
  std::transform(act.begin(), act.end(), act.begin(), ::tolower);
  TORCH_CHECK(act == "silu" || act == "relu" || act == "identity",
              "Unsupported activation '" + activation + "'.");

  const auto pad = std::max<int64_t>(kernel_size - 1, 0);
  at::Tensor x_padded = pad > 0 ? at::constant_pad_nd(x_conv, {pad, 0})
                                : x_conv;

  auto conv_out = at::conv1d(x_padded, weight_cast, bias_cast, {1}, {0}, {1},
                             channels);

  if (act == "silu") {
    if (conv_out.is_complex()) {
      conv_out = conv_out / ((-conv_out).exp() + 1);
    } else {
      conv_out = at::silu(conv_out);
    }
  } else if (act == "relu") {
    if (!conv_out.is_complex()) {
      conv_out = at::relu(conv_out);
    }
  }

  if (!channels_first) {
    conv_out = conv_out.permute({0, 2, 1}).contiguous();
  }

  return conv_out.to(x.scalar_type());
}

}  // namespace cpu
}  // namespace ssm

