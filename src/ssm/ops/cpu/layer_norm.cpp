#include "common.h"

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/vec.h>

#include <c10/util/TypeTraits.h>

#include <algorithm>
#include <cmath>
#include <tuple>
#include <vector>

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
        const bool has_residual = residual_ptr != nullptr;
        const bool add_pre = prenorm && has_residual;
        const bool add_post = !prenorm && has_residual;
        const bool has_bias = bias_ptr != nullptr;

        at::parallel_for(0, batch * length, 0,
                         [&](int64_t start, int64_t end) {
                           for (const auto idx : c10::irange(start, end)) {
                             const auto b = idx / length;
                             const auto t = idx % length;
                             const auto offset = b * length * row_stride +
                                                 t * row_stride;

                             const auto* x_row = x_ptr + offset;
                             const auto* residual_row =
                                 has_residual ? residual_ptr + offset : nullptr;
                             auto* out_row = out_ptr + offset;

                             acc_t sum = static_cast<acc_t>(0);
                             acc_t sumsq = static_cast<acc_t>(0);

                             if constexpr (!c10::is_complex<scalar_t>::value) {
                               using Vec = at::vec::Vectorized<scalar_t>;
                               constexpr auto kWidth = Vec::size();
                               Vec vsum = Vec(scalar_t(0));
                               Vec vsumsq = Vec(scalar_t(0));
                               int64_t d = 0;
                               for (; d + kWidth <= dim; d += kWidth) {
                                 auto vx = Vec::loadu(x_row + d);
                                 if (add_pre) {
                                   vx = vx + Vec::loadu(residual_row + d);
                                 }
                                 vsum = vsum + vx;
                                 vsumsq = vsumsq + vx * vx;
                               }
                               if (d != 0) {
                                 alignas(64) scalar_t buffer[kWidth];
                                 vsum.store(buffer);
                                 for (const auto lane : c10::irange(kWidth)) {
                                   sum += static_cast<acc_t>(buffer[lane]);
                                 }
                                 vsumsq.store(buffer);
                                 for (const auto lane : c10::irange(kWidth)) {
                                   sumsq += static_cast<acc_t>(buffer[lane]);
                                 }
                               }
                               for (; d < dim; ++d) {
                                 auto value = x_row[d];
                                 if (add_pre) {
                                   value = value + residual_row[d];
                                 }
                                 const auto val_acc = static_cast<acc_t>(value);
                                 sum += val_acc;
                                 sumsq += val_acc * val_acc;
                               }
                             } else {
                               for (const auto d : c10::irange(dim)) {
                                 auto value = x_row[d];
                                 if (add_pre) {
                                   value = value + residual_row[d];
                                 }
                                 const auto val_acc = static_cast<acc_t>(value);
                                 sum += val_acc;
                                 sumsq += val_acc * val_acc;
                               }
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

                             if constexpr (!c10::is_complex<scalar_t>::value) {
                               using Vec = at::vec::Vectorized<scalar_t>;
                               constexpr auto kWidth = Vec::size();
                               int64_t d = 0;
                               const auto mean_scalar = static_cast<scalar_t>(mean);
                               const auto inv_scalar =
                                   static_cast<scalar_t>(inv_var);
                               const Vec vec_mean(mean_scalar);
                               const Vec vec_inv(inv_scalar);
                               for (; d + kWidth <= dim; d += kWidth) {
                                 auto vx = Vec::loadu(x_row + d);
                                 if (add_pre) {
                                   vx = vx + Vec::loadu(residual_row + d);
                                 }
                                 if (!is_rms) {
                                   vx = vx - vec_mean;
                                 }
                                 vx = vx * vec_inv;
                                 vx = vx * Vec::loadu(weight_ptr + d);
                                 if (has_bias) {
                                   vx = vx + Vec::loadu(bias_ptr + d);
                                 }
                                 if (add_post) {
                                   vx = vx + Vec::loadu(residual_row + d);
                                 }
                                 vx.store(out_row + d);
                               }
                               for (; d < dim; ++d) {
                                 auto value = x_row[d];
                                 if (add_pre) {
                                   value = value + residual_row[d];
                                 }
                                 auto normed = is_rms
                                                   ? value * inv_scalar
                                                   : (value - mean_scalar) * inv_scalar;
                                 normed *= weight_ptr[d];
                                 if (has_bias) {
                                   normed += bias_ptr[d];
                                 }
                                 if (add_post) {
                                   normed += residual_row[d];
                                 }
                                 out_row[d] = normed;
                               }
                             } else {
                               for (const auto d : c10::irange(dim)) {
                                 auto value = x_row[d];
                                 if (add_pre) {
                                   value = value + residual_row[d];
                                 }
                                 auto normed = is_rms
                                                   ? value * static_cast<scalar_t>(inv_var)
                                                   : (value - static_cast<scalar_t>(mean)) *
                                                         static_cast<scalar_t>(inv_var);
                                 normed *= weight_ptr[d];
                                 if (has_bias) {
                                   normed += bias_ptr[d];
                                 }
                                 if (add_post) {
                                   normed += residual_row[d];
                                 }
                                 out_row[d] = normed;
                               }
                             }
                           }
                         });
      });

  return output.to(x.scalar_type());
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
fused_layer_norm_backward_cpu(
    const at::Tensor& grad_output, const at::Tensor& x,
    const at::Tensor& weight, const c10::optional<at::Tensor>& bias,
    const c10::optional<at::Tensor>& residual, bool is_rms, double eps,
    bool prenorm, bool residual_in_fp32) {
  TORCH_CHECK(x.dim() == 3, "x must have shape (B, L, D).");
  TORCH_CHECK(weight.dim() == 1 && weight.size(0) == x.size(-1),
              "weight must match the last dimension of x.");
  TORCH_CHECK(grad_output.dim() == 3,
              "grad_output must have shape (B, L, D).");
  TORCH_CHECK(grad_output.sizes() == x.sizes(),
              "grad_output must match the output shape of forward.");
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

  const auto dim = x.size(2);

  auto compute_dtype = get_compute_dtype(x);
  auto base_dtype =
      residual_in_fp32 && residual.has_value() ? at::kFloat : compute_dtype;
  const auto device = x.device();

  auto x_compute = to_compute(x, base_dtype, device).contiguous();
  at::Tensor residual_compute;
  if (residual.has_value()) {
    residual_compute =
        to_compute(residual.value(), base_dtype, device).contiguous();
  }

  auto weight_compute =
      to_compute(weight, compute_dtype, device).contiguous();
  auto grad_output_compute =
      to_compute(grad_output, compute_dtype, device).contiguous();

  const bool has_residual = residual.has_value();
  const bool has_bias = bias.has_value();
  const bool add_pre = prenorm && has_residual;
  at::Tensor norm_input = add_pre ? x_compute + residual_compute : x_compute;

  at::Tensor inv_std;
  at::Tensor normalized;
  if (is_rms) {
    auto mean_square = norm_input.pow(2).mean(-1, true);
    inv_std = (mean_square + eps).rsqrt();
    normalized = norm_input * inv_std;
  } else {
    auto mean = norm_input.mean(-1, true);
    auto centered = norm_input - mean;
    auto var = centered.pow(2).mean(-1, true);
    inv_std = (var + eps).rsqrt();
    normalized = centered * inv_std;
  }

  auto norm_input_grad = norm_input.to(compute_dtype);
  auto inv_std_grad = inv_std.to(compute_dtype);
  auto normalized_grad = normalized.to(compute_dtype);
  auto norm_input_grad_conj = at::conj(norm_input_grad);
  auto normalized_grad_conj = at::conj(normalized_grad);

  auto weight_view = weight_compute.view({1, 1, dim});
  auto grad_pre_out = grad_output_compute;

  std::vector<int64_t> reduce_dims = {0, 1};
  auto grad_weight_compute =
      (grad_pre_out * normalized_grad_conj).sum(reduce_dims);

  at::Tensor grad_bias_compute;
  if (has_bias) {
    grad_bias_compute = grad_pre_out.sum(reduce_dims);
  }

  auto grad_norm = grad_pre_out * weight_view;

  at::Tensor grad_norm_input;
  if (is_rms) {
    auto cross = (grad_norm * norm_input_grad_conj).sum(-1, true);
    auto factor = (cross / dim) * inv_std_grad * inv_std_grad * inv_std_grad;
    grad_norm_input = grad_norm * inv_std_grad - norm_input_grad * factor;
  } else {
    auto grad_norm_sum = grad_norm.sum(-1, true);
    auto grad_norm_dot = (grad_norm * normalized_grad_conj).sum(-1, true);
    auto scaled = grad_norm * dim - grad_norm_sum - normalized_grad * grad_norm_dot;
    grad_norm_input = scaled * (inv_std_grad / dim);
  }

  at::Tensor grad_x_result = grad_norm_input.to(x.scalar_type());

  at::Tensor grad_residual_result;
  if (has_residual) {
    at::Tensor grad_residual_base;
    if (prenorm) {
      grad_residual_base = grad_norm_input;
    } else {
      grad_residual_base = grad_pre_out;
    }
    grad_residual_result = grad_residual_base.to(residual.value().scalar_type());
  }

  auto grad_weight_result =
      grad_weight_compute.to(weight.scalar_type());

  at::Tensor grad_bias_result;
  if (has_bias) {
    grad_bias_result = grad_bias_compute.to(bias.value().scalar_type());
  }

  return std::make_tuple(grad_x_result, grad_weight_result, grad_bias_result,
                         grad_residual_result);
}

}  // namespace cpu
}  // namespace ssm

