#include "common.h"

#include <ATen/ATen.h>
#include <ATen/OpMathType.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cmath>
#include <limits>

namespace ssm {
namespace cuda {

namespace {

template <typename scalar_t>
__device__ inline scalar_t clamp_min_zero(scalar_t value) {
  return value;
}

template <>
__device__ inline float clamp_min_zero<float>(float value) {
  return value < 0.f ? 0.f : value;
}

template <>
__device__ inline double clamp_min_zero<double>(double value) {
  return value < 0.0 ? 0.0 : value;
}

template <typename scalar_t, bool kHasResidual, bool kHasBias, bool kIsRms,
          bool kPrenorm>
__global__ void fused_layer_norm_kernel(
    const scalar_t* __restrict__ x, const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias, const scalar_t* __restrict__ residual,
    scalar_t* __restrict__ output, int hidden_size, double eps) {
  using acc_t = at::opmath_type<scalar_t>;
  extern __shared__ unsigned char shared_bytes[];
  acc_t* shared_sum = reinterpret_cast<acc_t*>(shared_bytes);
  acc_t* shared_sumsq = shared_sum + blockDim.x;

  const int row = blockIdx.x;
  const int thread = threadIdx.x;

  const scalar_t* x_row = x + row * hidden_size;
  const scalar_t* weight_row = weight;
  const scalar_t* bias_row = kHasBias ? bias : nullptr;
  const scalar_t* residual_row =
      kHasResidual ? residual + row * hidden_size : nullptr;
  scalar_t* out_row = output + row * hidden_size;

  acc_t thread_sum = acc_t(0);
  acc_t thread_sumsq = acc_t(0);

  for (int col = thread; col < hidden_size; col += blockDim.x) {
    acc_t value = static_cast<acc_t>(x_row[col]);
    if constexpr (kPrenorm && kHasResidual) {
      value += static_cast<acc_t>(residual_row[col]);
    }
    thread_sum += value;
    thread_sumsq += value * value;
  }

  shared_sum[thread] = thread_sum;
  shared_sumsq[thread] = thread_sumsq;
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (thread < offset) {
      shared_sum[thread] += shared_sum[thread + offset];
      shared_sumsq[thread] += shared_sumsq[thread + offset];
    }
    __syncthreads();
  }

  acc_t mean = acc_t(0);
  acc_t inv_var = acc_t(0);
  if (thread == 0) {
    const acc_t inv_hidden = acc_t(1) / static_cast<acc_t>(hidden_size);
    if constexpr (kIsRms) {
      const acc_t mean_square = shared_sumsq[0] * inv_hidden;
      inv_var = acc_t(1) /
                static_cast<acc_t>(::sqrt(static_cast<double>(mean_square + eps)));
    } else {
      mean = shared_sum[0] * inv_hidden;
      const acc_t mean_square = shared_sumsq[0] * inv_hidden;
      acc_t variance = mean_square - mean * mean;
      variance = clamp_min_zero<acc_t>(variance);
      inv_var = acc_t(1) /
                static_cast<acc_t>(::sqrt(static_cast<double>(variance + eps)));
    }
    shared_sum[0] = mean;
    shared_sumsq[0] = inv_var;
  }
  __syncthreads();

  mean = shared_sum[0];
  inv_var = shared_sumsq[0];

  for (int col = thread; col < hidden_size; col += blockDim.x) {
    acc_t value = static_cast<acc_t>(x_row[col]);
    acc_t residual_value = acc_t(0);
    if constexpr (kHasResidual) {
      residual_value = static_cast<acc_t>(residual_row[col]);
    }
    acc_t norm_input = value;
    if constexpr (kPrenorm && kHasResidual) {
      norm_input += residual_value;
    }
    acc_t normed = kIsRms ? norm_input * inv_var
                          : (norm_input - mean) * inv_var;
    normed *= static_cast<acc_t>(weight_row[col]);
    if constexpr (kHasBias) {
      normed += static_cast<acc_t>(bias_row[col]);
    }
    if constexpr (!kPrenorm && kHasResidual) {
      normed += residual_value;
    }
    out_row[col] = static_cast<scalar_t>(normed);
  }
}

inline int determine_num_threads(int hidden_size) {
  if (hidden_size >= 1024) {
    return 1024;
  }
  if (hidden_size >= 512) {
    return 512;
  }
  if (hidden_size >= 256) {
    return 256;
  }
  if (hidden_size >= 128) {
    return 128;
  }
  if (hidden_size >= 64) {
    return 64;
  }
  return 32;
}

template <typename scalar_t, bool kHasResidual, bool kHasBias>
void launch_norm_kernel(const at::Tensor& x, const at::Tensor& weight,
                        const at::Tensor& bias, const at::Tensor& residual,
                        at::Tensor& output, bool is_rms, bool prenorm,
                        double eps) {
  const auto hidden = static_cast<int>(x.size(-1));
  const auto rows64 = x.size(0) * x.size(1);
  TORCH_CHECK(rows64 <= std::numeric_limits<int>::max(),
              "fused_layer_norm_cuda: batch * length is too large for CUDA launch.");
  const auto rows = static_cast<int>(rows64);
  const int threads = determine_num_threads(hidden);
  const dim3 blocks(rows);
  const size_t shared_bytes =
      sizeof(at::opmath_type<scalar_t>) * threads * 2;
  auto stream = at::cuda::getCurrentCUDAStream();

  const scalar_t* x_ptr = x.data_ptr<scalar_t>();
  const scalar_t* weight_ptr = weight.data_ptr<scalar_t>();
  const scalar_t* bias_ptr = nullptr;
  if constexpr (kHasBias) {
    bias_ptr = bias.data_ptr<scalar_t>();
  }
  const scalar_t* residual_ptr = nullptr;
  if constexpr (kHasResidual) {
    residual_ptr = residual.data_ptr<scalar_t>();
  }
  scalar_t* out_ptr = output.data_ptr<scalar_t>();

  if (is_rms) {
    if (prenorm) {
      fused_layer_norm_kernel<scalar_t, kHasResidual, kHasBias, true,
                              true><<<blocks, threads, shared_bytes, stream>>>(
          x_ptr, weight_ptr, bias_ptr, residual_ptr, out_ptr, hidden, eps);
    } else {
      fused_layer_norm_kernel<scalar_t, kHasResidual, kHasBias, true,
                              false><<<blocks, threads, shared_bytes, stream>>>(
          x_ptr, weight_ptr, bias_ptr, residual_ptr, out_ptr, hidden, eps);
    }
  } else {
    if (prenorm) {
      fused_layer_norm_kernel<scalar_t, kHasResidual, kHasBias, false,
                              true><<<blocks, threads, shared_bytes, stream>>>(
          x_ptr, weight_ptr, bias_ptr, residual_ptr, out_ptr, hidden, eps);
    } else {
      fused_layer_norm_kernel<scalar_t, kHasResidual, kHasBias, false,
                              false><<<blocks, threads, shared_bytes, stream>>>(
          x_ptr, weight_ptr, bias_ptr, residual_ptr, out_ptr, hidden, eps);
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

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

  TORCH_CHECK(!x.is_complex(), "fused_layer_norm_cuda does not support complex dtypes.");

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
      to_compute(weight, base_dtype, device).contiguous();
  at::Tensor bias_compute;
  if (bias.has_value()) {
    bias_compute = to_compute(bias.value(), base_dtype, device).contiguous();
  }

  auto output = at::empty_like(x_compute);

  AT_DISPATCH_FLOATING_TYPES(
      base_dtype, "fused_layer_norm_cuda", [&]() {
        if (residual_compute.defined()) {
          if (bias_compute.defined()) {
            launch_norm_kernel<scalar_t, true, true>(
                x_compute, weight_compute, bias_compute, residual_compute,
                output, is_rms, prenorm, eps);
          } else {
            launch_norm_kernel<scalar_t, true, false>(
                x_compute, weight_compute, bias_compute, residual_compute,
                output, is_rms, prenorm, eps);
          }
        } else {
          if (bias_compute.defined()) {
            launch_norm_kernel<scalar_t, false, true>(
                x_compute, weight_compute, bias_compute, residual_compute,
                output, is_rms, prenorm, eps);
          } else {
            launch_norm_kernel<scalar_t, false, false>(
                x_compute, weight_compute, bias_compute, residual_compute,
                output, is_rms, prenorm, eps);
          }
        }
      });

  return output.to(x.scalar_type());
}

}  // namespace cuda
}  // namespace ssm

