#include "common.h"

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/Parallel.h>

#include <c10/util/TypeTraits.h>

#include <cmath>

namespace ssm {
namespace cpu {

at::Tensor fused_layer_norm_cpu(
    const at::Tensor& x, const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    const c10::optional<at::Tensor>& residual, bool is_rms, double eps,
    bool prenorm, bool residual_in_fp32) {
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

  auto x_compute = to_compute(x, base_dtype, device).contiguous();
  at::Tensor residual_compute;
  if (residual.has_value()) {
    residual_compute = to_compute(residual.value(), base_dtype, device).contiguous();
  }

  auto weight_compute =
      to_compute(weight, compute_dtype, device).contiguous();
  c10::optional<at::Tensor> bias_compute = c10::nullopt;
  if (bias.has_value()) {
    bias_compute = to_compute(bias.value(), compute_dtype, device).contiguous();
  }

  auto output =
      at::empty({x.size(0), x.size(1), x.size(2)},
                x.options().dtype(compute_dtype));

  const auto batch = x.size(0);
  const auto length = x.size(1);
  const auto dim = x.size(2);

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      output.scalar_type(), "fused_layer_norm_cpu", [&]() {
        using acc_t = at::acc_type<scalar_t, true>;

        const auto* x_ptr = x_compute.data_ptr<scalar_t>();
        const auto* residual_ptr = residual.has_value()
                                       ? residual_compute.data_ptr<scalar_t>()
                                       : nullptr;
        const auto* weight_ptr = weight_compute.data_ptr<scalar_t>();
        const auto* bias_ptr = bias_compute.has_value()
                                   ? bias_compute.value().data_ptr<scalar_t>()
                                   : nullptr;
        auto* out_ptr = output.data_ptr<scalar_t>();

        const auto row_stride = dim;

        at::parallel_for(0, batch * length, 0,
                         [&](int64_t start, int64_t end) {
                           for (const auto idx : c10::irange(start, end)) {
                             const auto b = idx / length;
                             const auto t = idx % length;
                             const auto offset = b * length * row_stride +
                                                 t * row_stride;

                             const auto* x_row = x_ptr + offset;
                             const auto* residual_row =
                                 residual_ptr != nullptr
                                     ? residual_ptr + offset
                                     : nullptr;
                             auto* out_row = out_ptr + offset;

                             acc_t sum = static_cast<acc_t>(0);
                             acc_t sumsq = static_cast<acc_t>(0);

                             for (const auto d : c10::irange(dim)) {
                               auto value = x_row[d];
                               if (prenorm && residual_row != nullptr) {
                                 value = value + residual_row[d];
                               }
                               const acc_t val_acc = static_cast<acc_t>(value);
                               sum += val_acc;
                               sumsq += val_acc * val_acc;
                             }

                             acc_t mean = static_cast<acc_t>(0);
                             acc_t inv_var = static_cast<acc_t>(0);
                             if (is_rms) {
                               const acc_t mean_square = sumsq / dim;
                               inv_var =
                                   acc_t(1) / std::sqrt(mean_square + acc_t(eps));
                             } else {
                               mean = sum / dim;
                               const acc_t mean_square = sumsq / dim;
                               acc_t variance = mean_square - mean * mean;
                               if constexpr (!c10::is_complex<scalar_t>::value) {
                                 variance = std::max(acc_t(0), variance);
                               }
                               inv_var =
                                   acc_t(1) / std::sqrt(variance + acc_t(eps));
                             }

                             for (const auto d : c10::irange(dim)) {
                               auto value = x_row[d];
                               if (prenorm && residual_row != nullptr) {
                                 value = value + residual_row[d];
                               }
                               auto normed = is_rms
                                                 ? value * static_cast<scalar_t>(inv_var)
                                                 : (value - static_cast<scalar_t>(mean)) *
                                                       static_cast<scalar_t>(inv_var);
                               normed *= weight_ptr[d];
                               if (bias_ptr != nullptr) {
                                 normed += bias_ptr[d];
                               }
                               out_row[d] = normed;
                             }

                             if (!prenorm && residual_row != nullptr) {
                               for (const auto d : c10::irange(dim)) {
                                 out_row[d] += residual_row[d];
                               }
                             }
                           }
                         });
      });

  return output.to(x.scalar_type());
}

}  // namespace cpu
}  // namespace ssm

