#include "common.h"

#include <ATen/ATen.h>
#include <ATen/OpMathType.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/TypeTraits.h>

namespace ssm {
namespace cuda {

namespace {

template <typename T>
__inline__ __device__ T warp_reduce_sum(T value) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

template <typename T>
__inline__ __device__ void block_allreduce(
    T thread_sum, T thread_sq_sum, T* shared_sum, T* shared_sq_sum,
    T& total_sum, T& total_sq_sum) {
  const int lane = threadIdx.x % warpSize;
  const int warp = threadIdx.x / warpSize;
  const int num_warps = blockDim.x / warpSize;

  thread_sum = warp_reduce_sum(thread_sum);
  thread_sq_sum = warp_reduce_sum(thread_sq_sum);

  if (lane == 0) {
    shared_sum[warp] = thread_sum;
    shared_sq_sum[warp] = thread_sq_sum;
  }
  __syncthreads();

  total_sum = static_cast<T>(0);
  total_sq_sum = static_cast<T>(0);
  if (warp == 0) {
    total_sum = lane < num_warps ? shared_sum[lane] : static_cast<T>(0);
    total_sq_sum =
        lane < num_warps ? shared_sq_sum[lane] : static_cast<T>(0);

    total_sum = warp_reduce_sum(total_sum);
    total_sq_sum = warp_reduce_sum(total_sq_sum);

    if (lane == 0) {
      shared_sum[0] = total_sum;
      shared_sq_sum[0] = total_sq_sum;
    }
  }
  __syncthreads();

  total_sum = shared_sum[0];
  total_sq_sum = shared_sq_sum[0];
}

template <typename T>
__inline__ __device__ T block_reduce_sum(T value, T* shared) {
  const int lane = threadIdx.x % warpSize;
  const int warp = threadIdx.x / warpSize;
  const int num_warps = blockDim.x / warpSize;

  value = warp_reduce_sum(value);

  if (lane == 0) {
    shared[warp] = value;
  }
  __syncthreads();

  T total = static_cast<T>(0);
  if (warp == 0) {
    total = lane < num_warps ? shared[lane] : static_cast<T>(0);
    total = warp_reduce_sum(total);
    if (lane == 0) {
      shared[0] = total;
    }
  }
  __syncthreads();

  return shared[0];
}

template <typename scalar_t, typename residual_t, typename opmath_t>
__global__ void fused_layer_norm_kernel(
    const scalar_t* __restrict__ x, const residual_t* __restrict__ residual,
    const opmath_t* __restrict__ weight, const opmath_t* __restrict__ bias,
    scalar_t* __restrict__ output, int64_t rows, int64_t hidden, opmath_t eps,
    bool has_residual, bool prenorm, bool is_rms, bool has_bias) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  const int lane_stride = blockDim.x;
  const int64_t offset = static_cast<int64_t>(row) * hidden;
  const scalar_t* __restrict__ x_row = x + offset;
  const residual_t* __restrict__ residual_row =
      has_residual ? residual + offset : nullptr;
  scalar_t* __restrict__ out_row = output + offset;

  opmath_t thread_sum = static_cast<opmath_t>(0);
  opmath_t thread_sq_sum = static_cast<opmath_t>(0);

  for (int64_t col = threadIdx.x; col < hidden; col += lane_stride) {
    opmath_t x_val = static_cast<opmath_t>(x_row[col]);
    opmath_t res_val = has_residual
                           ? static_cast<opmath_t>(residual_row[col])
                           : static_cast<opmath_t>(0);
    opmath_t norm_input = prenorm && has_residual ? x_val + res_val : x_val;
    thread_sum += norm_input;
    thread_sq_sum += norm_input * norm_input;
  }

  extern __shared__ __align__(sizeof(opmath_t)) unsigned char shared_raw[];
  auto* shared = reinterpret_cast<opmath_t*>(shared_raw);
  const int num_warps = blockDim.x / warpSize;
  opmath_t* shared_sum = shared;
  opmath_t* shared_sq_sum = shared + num_warps;

  opmath_t total_sum;
  opmath_t total_sq_sum;
  block_allreduce(thread_sum, thread_sq_sum, shared_sum, shared_sq_sum,
                  total_sum, total_sq_sum);

  opmath_t mean = static_cast<opmath_t>(0);
  opmath_t inv_var = static_cast<opmath_t>(0);
  const opmath_t dim_val = static_cast<opmath_t>(hidden);

  if (is_rms) {
    const opmath_t mean_square = total_sq_sum / dim_val;
    inv_var = static_cast<opmath_t>(1) / sqrt(mean_square + eps);
  } else {
    mean = total_sum / dim_val;
    opmath_t mean_square = total_sq_sum / dim_val;
    opmath_t variance = mean_square - mean * mean;
    variance = variance < static_cast<opmath_t>(0)
                   ? static_cast<opmath_t>(0)
                   : variance;
    inv_var = static_cast<opmath_t>(1) / sqrt(variance + eps);
  }

  for (int64_t col = threadIdx.x; col < hidden; col += lane_stride) {
    opmath_t x_val = static_cast<opmath_t>(x_row[col]);
    opmath_t res_val = has_residual
                           ? static_cast<opmath_t>(residual_row[col])
                           : static_cast<opmath_t>(0);
    opmath_t norm_input = prenorm && has_residual ? x_val + res_val : x_val;
    opmath_t normed = is_rms ? norm_input * inv_var
                             : (norm_input - mean) * inv_var;
    opmath_t scaled = normed * weight[col];
    if (has_bias) {
      scaled += bias[col];
    }
    if (!prenorm && has_residual) {
      scaled += res_val;
    }
    out_row[col] = static_cast<scalar_t>(scaled);
  }
}

inline int64_t next_pow2(int64_t value) {
  int64_t power = 1;
  while (power < value && power < 1024) {
    power <<= 1;
  }
  if (power < 32) {
    power = 32;
  }
  return power;
}

template <typename scalar_t, typename residual_t, typename opmath_t>
void launch_layer_norm_kernel(
    const scalar_t* x_ptr, const residual_t* residual_ptr,
    const opmath_t* weight_ptr, const opmath_t* bias_ptr, scalar_t* out_ptr,
    int64_t rows, int64_t hidden, opmath_t eps, bool has_residual,
    bool prenorm, bool is_rms, bool has_bias, cudaStream_t stream) {
  const int threads = static_cast<int>(next_pow2(hidden));
  const int64_t blocks = rows;
  const size_t shared_bytes =
      sizeof(opmath_t) * static_cast<size_t>(threads / warpSize) * 2;
  fused_layer_norm_kernel<scalar_t, residual_t, opmath_t>
      <<<blocks, threads, shared_bytes, stream>>>(
          x_ptr, residual_ptr, weight_ptr, bias_ptr, out_ptr, rows, hidden,
          eps, has_residual, prenorm, is_rms, has_bias);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t, typename residual_t, typename opmath_t>
__global__ void fused_layer_norm_backward_kernel(
    const scalar_t* __restrict__ x,
    const residual_t* __restrict__ residual,
    const scalar_t* __restrict__ grad_output,
    const opmath_t* __restrict__ weight,
    opmath_t* __restrict__ grad_weight,
    opmath_t* __restrict__ grad_bias,
    residual_t* __restrict__ grad_residual,
    scalar_t* __restrict__ grad_x, int64_t rows, int64_t hidden, opmath_t eps,
    bool has_residual, bool prenorm, bool is_rms, bool has_bias) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  const int lane_stride = blockDim.x;
  const int64_t offset = static_cast<int64_t>(row) * hidden;
  const scalar_t* __restrict__ x_row = x + offset;
  const residual_t* __restrict__ residual_row =
      has_residual ? residual + offset : nullptr;
  const scalar_t* __restrict__ grad_row = grad_output + offset;
  scalar_t* __restrict__ grad_x_row = grad_x + offset;
  residual_t* __restrict__ grad_res_row =
      has_residual && grad_residual != nullptr ? grad_residual + offset
                                               : nullptr;

  opmath_t thread_sum = static_cast<opmath_t>(0);
  opmath_t thread_sq_sum = static_cast<opmath_t>(0);

  for (int64_t col = threadIdx.x; col < hidden; col += lane_stride) {
    const opmath_t x_val = static_cast<opmath_t>(x_row[col]);
    const opmath_t res_val = has_residual
                                 ? static_cast<opmath_t>(residual_row[col])
                                 : static_cast<opmath_t>(0);
    const opmath_t norm_input =
        (prenorm && has_residual) ? x_val + res_val : x_val;
    thread_sum += norm_input;
    thread_sq_sum += norm_input * norm_input;
  }

  extern __shared__ __align__(sizeof(opmath_t)) unsigned char shared_raw[];
  auto* shared = reinterpret_cast<opmath_t*>(shared_raw);
  const int num_warps = blockDim.x / warpSize;
  opmath_t* shared_sum = shared;
  opmath_t* shared_sq_sum = shared + num_warps;

  const opmath_t total_sum = block_reduce_sum(thread_sum, shared_sum);
  const opmath_t total_sq_sum = block_reduce_sum(thread_sq_sum, shared_sq_sum);

  const opmath_t dim_val = static_cast<opmath_t>(hidden);
  opmath_t mean = static_cast<opmath_t>(0);
  opmath_t inv_var = static_cast<opmath_t>(0);

  if (is_rms) {
    const opmath_t mean_square = total_sq_sum / dim_val;
    inv_var = static_cast<opmath_t>(1) / sqrt(mean_square + eps);
  } else {
    mean = total_sum / dim_val;
    opmath_t mean_square = total_sq_sum / dim_val;
    opmath_t variance = mean_square - mean * mean;
    variance = variance < static_cast<opmath_t>(0) ? static_cast<opmath_t>(0)
                                                   : variance;
    inv_var = static_cast<opmath_t>(1) / sqrt(variance + eps);
  }

  opmath_t* shared_grad_norm_sum = shared + 2 * num_warps;
  opmath_t* shared_grad_norm_dot = shared + 3 * num_warps;
  opmath_t* shared_cross = shared + 4 * num_warps;
  opmath_t* shared_bias = shared + 5 * num_warps;

  opmath_t thread_grad_norm_sum = static_cast<opmath_t>(0);
  opmath_t thread_grad_norm_dot = static_cast<opmath_t>(0);
  opmath_t thread_cross = static_cast<opmath_t>(0);
  opmath_t thread_grad_bias = static_cast<opmath_t>(0);

  for (int64_t col = threadIdx.x; col < hidden; col += lane_stride) {
    const opmath_t x_val = static_cast<opmath_t>(x_row[col]);
    const opmath_t res_val = has_residual
                                 ? static_cast<opmath_t>(residual_row[col])
                                 : static_cast<opmath_t>(0);
    const opmath_t norm_input =
        (prenorm && has_residual) ? x_val + res_val : x_val;
    const opmath_t normed = is_rms ? norm_input * inv_var
                                   : (norm_input - mean) * inv_var;
    const opmath_t grad_pre = static_cast<opmath_t>(grad_row[col]);
    const opmath_t grad_norm = grad_pre * weight[col];

    if (!is_rms) {
      thread_grad_norm_sum += grad_norm;
    }
    thread_grad_norm_dot += grad_norm * normed;
    if (is_rms) {
      thread_cross += grad_norm * norm_input;
    }
    if (has_bias) {
      thread_grad_bias += grad_pre;
    }
  }

  const opmath_t grad_norm_sum = block_reduce_sum(
      is_rms ? static_cast<opmath_t>(0) : thread_grad_norm_sum,
      shared_grad_norm_sum);
  const opmath_t grad_norm_dot =
      block_reduce_sum(thread_grad_norm_dot, shared_grad_norm_dot);
  const opmath_t cross_total = block_reduce_sum(
      is_rms ? thread_cross : static_cast<opmath_t>(0), shared_cross);
  block_reduce_sum(has_bias ? thread_grad_bias : static_cast<opmath_t>(0),
                   shared_bias);

  for (int64_t col = threadIdx.x; col < hidden; col += lane_stride) {
    const opmath_t x_val = static_cast<opmath_t>(x_row[col]);
    const opmath_t res_val = has_residual
                                 ? static_cast<opmath_t>(residual_row[col])
                                 : static_cast<opmath_t>(0);
    const opmath_t norm_input =
        (prenorm && has_residual) ? x_val + res_val : x_val;
    const opmath_t normed = is_rms ? norm_input * inv_var
                                   : (norm_input - mean) * inv_var;
    const opmath_t grad_pre = static_cast<opmath_t>(grad_row[col]);
    const opmath_t grad_norm = grad_pre * weight[col];

    opmath_t grad_norm_input;
    if (is_rms) {
      const opmath_t cross_mean = cross_total / dim_val;
      const opmath_t factor =
          cross_mean * inv_var * inv_var * inv_var;  // inv_std^3
      grad_norm_input = grad_norm * inv_var - norm_input * factor;
    } else {
      const opmath_t numerator =
          grad_norm * dim_val - grad_norm_sum - normed * grad_norm_dot;
      grad_norm_input = numerator * (inv_var / dim_val);
    }

    grad_x_row[col] = static_cast<scalar_t>(grad_norm_input);

    if (has_residual && grad_res_row != nullptr) {
      if (prenorm) {
        grad_res_row[col] = static_cast<residual_t>(grad_norm_input);
      } else {
        grad_res_row[col] = static_cast<residual_t>(grad_pre);
      }
    }

    const opmath_t weight_grad = grad_pre * normed;
    atomicAdd(&grad_weight[col], weight_grad);
    if (has_bias && grad_bias != nullptr) {
      atomicAdd(&grad_bias[col], grad_pre);
    }
  }
}

template <typename scalar_t, typename residual_t, typename opmath_t>
void launch_layer_norm_backward_kernel(
    const scalar_t* x_ptr, const residual_t* residual_ptr,
    const scalar_t* grad_out_ptr, const opmath_t* weight_ptr,
    opmath_t* grad_weight_ptr, opmath_t* grad_bias_ptr,
    residual_t* grad_residual_ptr, scalar_t* grad_x_ptr, int64_t rows,
    int64_t hidden, opmath_t eps, bool has_residual, bool prenorm, bool is_rms,
    bool has_bias, cudaStream_t stream) {
  if (rows == 0 || hidden == 0) {
    return;
  }

  const int threads = static_cast<int>(next_pow2(hidden));
  const int64_t blocks = rows;
  const int num_warps = threads / warpSize;
  const size_t shared_bytes =
      sizeof(opmath_t) * static_cast<size_t>(num_warps) * 6;

  fused_layer_norm_backward_kernel<scalar_t, residual_t, opmath_t>
      <<<blocks, threads, shared_bytes, stream>>>(
          x_ptr, residual_ptr, grad_out_ptr, weight_ptr, grad_weight_ptr,
          grad_bias_ptr, grad_residual_ptr, grad_x_ptr, rows, hidden,
          eps, has_residual, prenorm, is_rms, has_bias);
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

  auto x_contig = x.contiguous();
  auto residual_contig = residual.has_value()
                             ? residual.value().contiguous()
                             : at::Tensor();
  auto weight_contig = weight.contiguous();
  c10::optional<at::Tensor> bias_contig = c10::nullopt;
  if (bias.has_value()) {
    bias_contig = bias.value().contiguous();
  }

  const auto rows = x_contig.size(0) * x_contig.size(1);
  const auto hidden = x_contig.size(2);

  auto output = at::empty_like(x_contig);

  if (rows == 0 || hidden == 0) {
    return output;
  }

  const bool has_residual = residual.has_value();
  const bool has_bias = bias.has_value();

  auto stream = c10::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, x_contig.scalar_type(),
      "fused_layer_norm_cuda", [&] {
        using scalar_t_ = scalar_t;
        using opmath_t = at::opmath_type<scalar_t_>;
        constexpr auto opmath_scalar = c10::CppTypeToScalarType<opmath_t>::value;

        at::Tensor weight_compute = weight_contig;
        if (weight_compute.scalar_type() != opmath_scalar) {
          weight_compute = weight_compute.to(
              weight_compute.options().dtype(opmath_scalar));
        }
        if (!weight_compute.is_contiguous()) {
          weight_compute = weight_compute.contiguous();
        }

        at::Tensor bias_compute;
        const opmath_t* bias_ptr = nullptr;
        if (has_bias) {
          bias_compute = bias_contig.value();
          if (bias_compute.scalar_type() != opmath_scalar) {
            bias_compute = bias_compute.to(
                bias_compute.options().dtype(opmath_scalar));
          }
          if (!bias_compute.is_contiguous()) {
            bias_compute = bias_compute.contiguous();
          }
          bias_ptr = bias_compute.data_ptr<opmath_t>();
        }

        const opmath_t* weight_ptr = weight_compute.data_ptr<opmath_t>();

        if (!has_residual) {
          launch_layer_norm_kernel<scalar_t_, scalar_t_, opmath_t>(
              x_contig.data_ptr<scalar_t_>(), nullptr, weight_ptr, bias_ptr,
              output.data_ptr<scalar_t_>(), rows, hidden, static_cast<opmath_t>(eps),
              false, prenorm, is_rms, has_bias, stream);
          return;
        }

        at::Tensor residual_compute = residual_contig;
        if (residual_in_fp32) {
          if (residual_compute.scalar_type() != at::kFloat) {
            residual_compute = residual_compute.to(
                residual_compute.options().dtype(at::kFloat));
          }
          if (!residual_compute.is_contiguous()) {
            residual_compute = residual_compute.contiguous();
          }
          launch_layer_norm_kernel<scalar_t_, float, opmath_t>(
              x_contig.data_ptr<scalar_t_>(),
              residual_compute.data_ptr<float>(), weight_ptr, bias_ptr,
              output.data_ptr<scalar_t_>(), rows, hidden, static_cast<opmath_t>(eps),
              true, prenorm, is_rms, has_bias, stream);
        } else {
          if (residual_compute.scalar_type() != x_contig.scalar_type()) {
            residual_compute =
                residual_compute.to(x_contig.scalar_type());
          }
          if (!residual_compute.is_contiguous()) {
            residual_compute = residual_compute.contiguous();
          }
          launch_layer_norm_kernel<scalar_t_, scalar_t_, opmath_t>(
              x_contig.data_ptr<scalar_t_>(),
              residual_compute.data_ptr<scalar_t_>(), weight_ptr, bias_ptr,
              output.data_ptr<scalar_t_>(), rows, hidden, static_cast<opmath_t>(eps),
              true, prenorm, is_rms, has_bias, stream);
        }
      });

  return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
fused_layer_norm_backward_cuda(
    const at::Tensor& grad_output, const at::Tensor& x,
    const at::Tensor& weight, const c10::optional<at::Tensor>& bias,
    const c10::optional<at::Tensor>& residual, bool is_rms, double eps,
    bool prenorm, bool residual_in_fp32) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor.");
  TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor.");
  TORCH_CHECK(grad_output.is_cuda(), "grad_output must be a CUDA tensor.");
  if (bias.has_value()) {
    TORCH_CHECK(bias->is_cuda(), "bias must be a CUDA tensor.");
  }
  if (residual.has_value()) {
    TORCH_CHECK(residual->is_cuda(), "residual must be a CUDA tensor.");
  }

  TORCH_CHECK(x.dim() == 3, "x must have shape (B, L, D).");
  TORCH_CHECK(weight.dim() == 1 && weight.size(0) == x.size(-1),
              "weight must match the last dimension of x.");
  TORCH_CHECK(grad_output.sizes() == x.sizes(),
              "grad_output must match the shape of x.");
  if (bias.has_value()) {
    TORCH_CHECK(bias->dim() == 1 && bias->size(0) == x.size(-1),
                "bias must match the last dimension of x.");
  }
  if (residual.has_value()) {
    TORCH_CHECK(residual->sizes() == x.sizes(),
                "residual must match the shape of x.");
  }

  c10::cuda::CUDAGuard guard(x.device());

  auto x_contig = x.contiguous();
  auto grad_out_contig = grad_output.contiguous();
  auto weight_contig = weight.contiguous();
  c10::optional<at::Tensor> bias_contig =
      bias.has_value() ? c10::optional<at::Tensor>(bias->contiguous())
                       : c10::nullopt;
  c10::optional<at::Tensor> residual_contig =
      residual.has_value() ? c10::optional<at::Tensor>(residual->contiguous())
                           : c10::nullopt;

  const auto rows = x_contig.size(0) * x_contig.size(1);
  const auto hidden = x_contig.size(2);

  const bool has_bias = bias.has_value();
  const bool has_residual = residual.has_value();

  if (rows == 0 || hidden == 0) {
    auto grad_x_zero = at::zeros_like(x_contig);
    auto grad_weight_zero = at::zeros_like(weight_contig);
    at::Tensor grad_bias_zero;
    if (has_bias) {
      grad_bias_zero = at::zeros_like(bias_contig.value());
    }
    at::Tensor grad_residual_zero;
    if (has_residual) {
      grad_residual_zero = at::zeros_like(residual_contig.value());
    }
    return std::make_tuple(grad_x_zero.to(x.scalar_type()),
                           grad_weight_zero.to(weight.scalar_type()),
                           has_bias ? grad_bias_zero.to(bias->scalar_type())
                                    : at::Tensor(),
                           has_residual
                               ? grad_residual_zero.to(residual->scalar_type())
                               : at::Tensor());
  }

  auto grad_x = at::empty_like(x_contig);
  at::Tensor grad_weight_out;
  at::Tensor grad_bias_out;
  at::Tensor grad_residual_out;

  auto stream = c10::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, x_contig.scalar_type(),
      "fused_layer_norm_backward_cuda", [&] {
        using scalar_t_ = scalar_t;
        using opmath_t = at::opmath_type<scalar_t_>;
        constexpr auto opmath_scalar =
            c10::CppTypeToScalarType<opmath_t>::value;

        at::Tensor weight_compute = weight_contig;
        if (weight_compute.scalar_type() != opmath_scalar) {
          weight_compute = weight_compute.to(
              weight_compute.options().dtype(opmath_scalar));
        }
        if (!weight_compute.is_contiguous()) {
          weight_compute = weight_compute.contiguous();
        }

        at::Tensor grad_weight_local = at::zeros_like(weight_compute);

        opmath_t* grad_bias_ptr = nullptr;
        at::Tensor grad_bias_local;
        if (has_bias) {
          at::Tensor bias_compute = bias_contig.value();
          if (bias_compute.scalar_type() != opmath_scalar) {
            bias_compute = bias_compute.to(
                bias_compute.options().dtype(opmath_scalar));
          }
          if (!bias_compute.is_contiguous()) {
            bias_compute = bias_compute.contiguous();
          }
          grad_bias_local = at::zeros_like(bias_compute);
          grad_bias_ptr = grad_bias_local.data_ptr<opmath_t>();
        }

        scalar_t_* grad_x_ptr = grad_x.data_ptr<scalar_t_>();
        const scalar_t_* x_ptr = x_contig.data_ptr<scalar_t_>();
        const scalar_t_* grad_out_ptr = grad_out_contig.data_ptr<scalar_t_>();
        const opmath_t* weight_ptr = weight_compute.data_ptr<opmath_t>();
        opmath_t* grad_weight_ptr = grad_weight_local.data_ptr<opmath_t>();

        if (!has_residual) {
          launch_layer_norm_backward_kernel<scalar_t_, scalar_t_, opmath_t>(
              x_ptr, nullptr, grad_out_ptr, weight_ptr, grad_weight_ptr,
              grad_bias_ptr, nullptr, grad_x_ptr, rows, hidden,
              static_cast<opmath_t>(eps), false, prenorm, is_rms, has_bias,
              stream);
        } else if (residual_in_fp32) {
          at::Tensor residual_compute = residual_contig.value();
          if (residual_compute.scalar_type() != at::kFloat) {
            residual_compute = residual_compute.to(
                residual_compute.options().dtype(at::kFloat));
          }
          if (!residual_compute.is_contiguous()) {
            residual_compute = residual_compute.contiguous();
          }
          auto grad_res_local =
              at::empty_like(residual_compute, at::MemoryFormat::Contiguous);
          launch_layer_norm_backward_kernel<scalar_t_, float, opmath_t>(
              x_ptr, residual_compute.data_ptr<float>(), grad_out_ptr,
              weight_ptr, grad_weight_ptr, grad_bias_ptr,
              grad_res_local.data_ptr<float>(), grad_x_ptr, rows, hidden,
              static_cast<opmath_t>(eps), true, prenorm, is_rms, has_bias,
              stream);
          grad_residual_out = grad_res_local;
        } else {
          at::Tensor residual_compute = residual_contig.value();
          if (residual_compute.scalar_type() != x_contig.scalar_type()) {
            residual_compute = residual_compute.to(
                residual_compute.options().dtype(x_contig.scalar_type()));
          }
          if (!residual_compute.is_contiguous()) {
            residual_compute = residual_compute.contiguous();
          }
          auto grad_res_local =
              at::empty_like(residual_compute, at::MemoryFormat::Contiguous);
          launch_layer_norm_backward_kernel<scalar_t_, scalar_t_, opmath_t>(
              x_ptr, residual_compute.data_ptr<scalar_t_>(), grad_out_ptr,
              weight_ptr, grad_weight_ptr, grad_bias_ptr,
              grad_res_local.data_ptr<scalar_t_>(), grad_x_ptr, rows, hidden,
              static_cast<opmath_t>(eps), true, prenorm, is_rms, has_bias,
              stream);
          grad_residual_out = grad_res_local;
        }

        grad_weight_out = grad_weight_local;
        if (has_bias) {
          grad_bias_out = grad_bias_local;
        }
      });

  at::Tensor grad_weight_tensor = grad_weight_out.to(weight.scalar_type());
  at::Tensor grad_bias_tensor;
  if (has_bias) {
    grad_bias_tensor = grad_bias_out.to(bias->scalar_type());
  }

  at::Tensor grad_residual_tensor;
  if (has_residual) {
    grad_residual_tensor = grad_residual_out.to(residual->scalar_type());
  }

  return std::make_tuple(grad_x, grad_weight_tensor, grad_bias_tensor,
                         grad_residual_tensor);
}

}  // namespace cuda
}  // namespace ssm

