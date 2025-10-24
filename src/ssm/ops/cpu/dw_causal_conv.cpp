#include "common.h"

#include <ATen/ATen.h>
#include <ATen/Parallel.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <complex>
#include <string>
#include <type_traits>

#include <c10/util/TypeTraits.h>

namespace ssm {
namespace cpu {

namespace {

template <typename scalar_t>
scalar_t apply_activation_value(const scalar_t& value,
                                const std::string& activation) {
  if (activation == "silu") {
    const scalar_t one = scalar_t(1);
    return value / (one + std::exp(-value));
  }
  if (activation == "relu") {
    if constexpr (c10::is_complex<scalar_t>::value) {
      return value;
    } else {
      const scalar_t zero = scalar_t(0);
      return value > zero ? value : zero;
    }
  }
  return value;
}

}  // namespace

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
    channels = x.size(1);
    length = x.size(2);
  } else if (x.size(2) == weight.size(0)) {
    channels_first = false;
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

  auto out = at::empty_like(x_conv);

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      compute_dtype, "dw_causal_conv_cpu", [&]() {
        const auto* x_ptr = x_conv.data_ptr<scalar_t>();
        const auto* w_ptr = weight_cast.data_ptr<scalar_t>();
        const auto* b_ptr =
            bias_cast.has_value() ? bias_cast.value().data_ptr<scalar_t>()
                                  : nullptr;
        auto* out_ptr = out.data_ptr<scalar_t>();

        const auto row_stride = channels * length;
        const auto channel_stride = length;

        at::parallel_for(0, batch * channels, 0, [&](int64_t start, int64_t end) {
          for (const auto idx : c10::irange(start, end)) {
            const auto b = idx / channels;
            const auto c = idx % channels;
            const auto input_offset = b * row_stride + c * channel_stride;
            const auto kernel_offset = c * kernel_size;

            const auto* input_row = x_ptr + input_offset;
            auto* output_row = out_ptr + input_offset;

            for (const auto t : c10::irange(length)) {
              scalar_t acc = b_ptr != nullptr ? b_ptr[c] : scalar_t(0);
              const auto start_k = kernel_size > (t + 1)
                                       ? kernel_size - 1 - static_cast<int64_t>(t)
                                       : int64_t(0);
              for (auto k = start_k; k < kernel_size; ++k) {
                const auto input_idx = t - (kernel_size - 1 - k);
                acc += input_row[input_idx] * w_ptr[kernel_offset + k];
              }
              output_row[t] = apply_activation_value(acc, act);
            }
          }
        });
      });

  if (!channels_first) {
    out = out.permute({0, 2, 1}).contiguous();
  }

  return out.to(x.scalar_type());
}

}  // namespace cpu
}  // namespace ssm

