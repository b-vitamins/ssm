#include "common.h"

#include <ATen/ATen.h>
#include <ATen/ops/constant_pad_nd.h>
#include <ATen/ops/conv1d.h>
#include <ATen/ops/silu.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <tuple>
#include <string>

#ifndef USE_ROCM
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#else
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif

namespace ssm {
namespace cuda {

namespace {

struct ConvParams {
  using index_t = uint32_t;

  int batch;
  int channels;
  int seqlen;
  int width;
  bool silu_activation;

  index_t x_batch_stride;
  index_t x_c_stride;
  index_t x_l_stride;

  index_t weight_c_stride;
  index_t weight_width_stride;

  index_t out_batch_stride;
  index_t out_c_stride;
  index_t out_l_stride;

  void* x_ptr;
  void* weight_ptr;
  void* bias_ptr;
  void* out_ptr;
};

inline constexpr size_t custom_max(std::initializer_list<size_t> values) {
#ifndef USE_ROCM
  return std::max(values);
#else
  return *std::max_element(values.begin(), values.end());
#endif
}

template <int Bytes>
struct BytesToType {};

template <>
struct BytesToType<16> {
  using Type = uint4;
};

template <>
struct BytesToType<8> {
  using Type = uint64_t;
};

template <>
struct BytesToType<4> {
  using Type = uint32_t;
};

template <>
struct BytesToType<2> {
  using Type = uint16_t;
};

template <>
struct BytesToType<1> {
  using Type = uint8_t;
};

template <int kNThreads_, int kWidth_, bool kIsVecLoad_, typename input_t_,
          typename weight_t_>
struct DwConvKernelTraits {
  using input_t = input_t_;
  using weight_t = weight_t_;
  static constexpr int kNThreads = kNThreads_;
  static constexpr int kWidth = kWidth_;
  static constexpr int kNBytes = sizeof(input_t);
  static_assert(kNBytes == 2 || kNBytes == 4);
  static constexpr int kNElts = kNBytes == 4 ? 4 : 8;
  static_assert(kWidth <= kNElts);
  static constexpr bool kIsVecLoad = kIsVecLoad_;
  using vec_t = typename BytesToType<kNBytes * kNElts>::Type;
  using BlockLoadT = cub::BlockLoad<input_t, kNThreads, kNElts,
                                    cub::BLOCK_LOAD_WARP_TRANSPOSE>;
  using BlockLoadVecT =
      cub::BlockLoad<vec_t, kNThreads, 1, cub::BLOCK_LOAD_DIRECT>;
  using BlockStoreT = cub::BlockStore<input_t, kNThreads, kNElts,
                                      cub::BLOCK_STORE_WARP_TRANSPOSE>;
  using BlockStoreVecT =
      cub::BlockStore<vec_t, kNThreads, 1, cub::BLOCK_STORE_DIRECT>;
  static constexpr int kSmemIOSize =
      kIsVecLoad
          ? 0
          : custom_max({sizeof(typename BlockLoadT::TempStorage),
                        sizeof(typename BlockStoreT::TempStorage)});
  static constexpr int kSmemExchangeSize = kNThreads * kNBytes * kNElts;
  static constexpr int kSmemSize = kSmemIOSize + kSmemExchangeSize;
};

template <typename Ktraits>
__global__ __launch_bounds__(Ktraits::kNThreads)
void dw_causal_conv_channels_first_kernel(ConvParams params) {
  constexpr int kWidth = Ktraits::kWidth;
  constexpr int kNThreads = Ktraits::kNThreads;
  constexpr int kNElts = Ktraits::kNElts;
  static constexpr bool kIsVecLoad = Ktraits::kIsVecLoad;
  using input_t = typename Ktraits::input_t;
  using vec_t = typename Ktraits::vec_t;
  using weight_t = typename Ktraits::weight_t;

  extern __shared__ char smem_[];
  auto& smem_load =
      reinterpret_cast<typename Ktraits::BlockLoadT::TempStorage&>(smem_);
  auto& smem_load_vec =
      reinterpret_cast<typename Ktraits::BlockLoadVecT::TempStorage&>(smem_);
  auto& smem_store =
      reinterpret_cast<typename Ktraits::BlockStoreT::TempStorage&>(smem_);
  auto& smem_store_vec =
      reinterpret_cast<typename Ktraits::BlockStoreVecT::TempStorage&>(smem_);
  vec_t* smem_exchange =
      reinterpret_cast<vec_t*>(smem_ + Ktraits::kSmemIOSize);

  const int tidx = threadIdx.x;
  const int batch_id = blockIdx.x;
  const int channel_id = blockIdx.y;
  auto* x = reinterpret_cast<input_t*>(params.x_ptr) +
            batch_id * params.x_batch_stride +
            channel_id * params.x_c_stride;
  auto* weight = reinterpret_cast<weight_t*>(params.weight_ptr) +
                 channel_id * params.weight_c_stride;
  auto* out = reinterpret_cast<input_t*>(params.out_ptr) +
              batch_id * params.out_batch_stride +
              channel_id * params.out_c_stride;
  float bias_val = params.bias_ptr == nullptr
                       ? 0.f
                       : static_cast<float>(reinterpret_cast<weight_t*>(
                             params.bias_ptr)[channel_id]);

  if (tidx == 0) {
    input_t zeros[kNElts] = {0};
    smem_exchange[kNThreads - 1] =
        reinterpret_cast<vec_t*>(zeros)[0];
  }

  float weight_vals[kWidth];
#pragma unroll
  for (int i = 0; i < kWidth; ++i) {
    weight_vals[i] = static_cast<float>(
        weight[i * params.weight_width_stride]);
  }

  constexpr int kChunkSize = kNThreads * kNElts;
  const int n_chunks = (params.seqlen + kChunkSize - 1) / kChunkSize;
  for (int chunk = 0; chunk < n_chunks; ++chunk) {
    input_t x_vals_load[2 * kNElts] = {0};
    if constexpr (kIsVecLoad) {
      typename Ktraits::BlockLoadVecT(smem_load_vec)
          .Load(reinterpret_cast<vec_t*>(x),
                *reinterpret_cast<vec_t (*)[1]>(&x_vals_load[kNElts]),
                (params.seqlen - chunk * kChunkSize) / kNElts);
    } else {
      __syncthreads();
      typename Ktraits::BlockLoadT(smem_load)
          .Load(x,
                *reinterpret_cast<input_t (*)[kNElts]>(
                    &x_vals_load[kNElts]),
                params.seqlen - chunk * kChunkSize);
    }
    x += kChunkSize;
    __syncthreads();
    if (tidx < kNThreads - 1) {
      smem_exchange[tidx] =
          reinterpret_cast<vec_t*>(x_vals_load)[1];
    }
    __syncthreads();
    reinterpret_cast<vec_t*>(x_vals_load)[0] =
        smem_exchange[tidx > 0 ? tidx - 1 : kNThreads - 1];
    __syncthreads();
    if (tidx == kNThreads - 1) {
      smem_exchange[tidx] =
          reinterpret_cast<vec_t*>(x_vals_load)[1];
    }

    float x_vals[2 * kNElts];
#pragma unroll
    for (int i = 0; i < 2 * kNElts; ++i) {
      x_vals[i] = static_cast<float>(x_vals_load[i]);
    }

    float out_vals[kNElts];
#pragma unroll
    for (int i = 0; i < kNElts; ++i) {
      out_vals[i] = bias_val;
#pragma unroll
      for (int w = 0; w < kWidth; ++w) {
        const int x_idx = kNElts + i - (kWidth - w - 1);
        if (x_idx >= 0) {
          out_vals[i] += weight_vals[w] * x_vals[x_idx];
        }
      }
      if (params.silu_activation) {
        out_vals[i] = out_vals[i] / (1.0f + expf(-out_vals[i]));
      }
    }

    input_t out_vals_store[kNElts];
#pragma unroll
    for (int i = 0; i < kNElts; ++i) {
      out_vals_store[i] = static_cast<input_t>(out_vals[i]);
    }

    if constexpr (kIsVecLoad) {
      typename Ktraits::BlockStoreVecT(smem_store_vec)
          .Store(reinterpret_cast<vec_t*>(out),
                 reinterpret_cast<vec_t(&)[1]>(out_vals_store),
                 (params.seqlen - chunk * kChunkSize) / kNElts);
    } else {
      typename Ktraits::BlockStoreT(smem_store)
          .Store(out, out_vals_store,
                 params.seqlen - chunk * kChunkSize);
    }
    out += kChunkSize;
  }
}

template <int kNThreads, int kWidth, typename input_t, typename weight_t>
void launch_channels_first(ConvParams& params, cudaStream_t stream) {
  constexpr int kNElts = sizeof(input_t) == 4 ? 4 : 8;
  if (params.seqlen % kNElts == 0) {
    using Ktraits =
        DwConvKernelTraits<kNThreads, kWidth, true, input_t, weight_t>;
    constexpr int kSmemSize = Ktraits::kSmemSize;
    dim3 grid(params.batch, params.channels);
    auto* kernel = &dw_causal_conv_channels_first_kernel<Ktraits>;
    if (kSmemSize >= 48 * 1024) {
#ifndef USE_ROCM
      C10_CUDA_CHECK(cudaFuncSetAttribute(
          kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
#else
      C10_CUDA_CHECK(cudaFuncSetAttribute(
          reinterpret_cast<void*>(kernel),
          cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
#endif
    }
    kernel<<<grid, Ktraits::kNThreads, kSmemSize, stream>>>(params);
  } else {
    using Ktraits =
        DwConvKernelTraits<kNThreads, kWidth, false, input_t, weight_t>;
    constexpr int kSmemSize = Ktraits::kSmemSize;
    dim3 grid(params.batch, params.channels);
    auto* kernel = &dw_causal_conv_channels_first_kernel<Ktraits>;
    if (kSmemSize >= 48 * 1024) {
#ifndef USE_ROCM
      C10_CUDA_CHECK(cudaFuncSetAttribute(
          kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
#else
      C10_CUDA_CHECK(cudaFuncSetAttribute(
          reinterpret_cast<void*>(kernel),
          cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
#endif
    }
    kernel<<<grid, Ktraits::kNThreads, kSmemSize, stream>>>(params);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int kNThreads_, int kWidth_, int kChunkSizeL_, bool kIsVecLoad_,
          typename input_t_, typename weight_t_>
struct DwConvChannelLastKernelTraits {
  using input_t = input_t_;
  using weight_t = weight_t_;
  static constexpr int kNThreads = kNThreads_;
  static constexpr int kWidth = kWidth_;
  static constexpr int kChunkSizeL = kChunkSizeL_;
  static constexpr int kNBytes = sizeof(input_t);
  static_assert(kNBytes == 2 || kNBytes == 4);
  static constexpr int kNElts = kNBytes == 4 ? 4 : 8;
  static constexpr int kNEltsPerRow = 128 / kNBytes;
  static constexpr int kNThreadsPerRow = kNEltsPerRow / kNElts;
  static_assert(kNThreadsPerRow * kNBytes * kNElts == 128);
  static constexpr int kNColsPerWarp = 32 / kNThreadsPerRow;
  static constexpr int kNWarps = kNThreads / 32;
  static constexpr int kNColsPerLoad = kNColsPerWarp * kNWarps;
  static constexpr int kNLoads = kChunkSizeL / kNColsPerLoad;
  static constexpr bool kIsVecLoad = kIsVecLoad_;
  using vec_t = typename BytesToType<kNBytes * kNElts>::Type;
};

template <typename Ktraits>
__global__ __launch_bounds__(Ktraits::kNThreads)
void dw_causal_conv_channels_last_kernel(ConvParams params) {
  constexpr int kWidth = Ktraits::kWidth;
  constexpr int kNThreads = Ktraits::kNThreads;
  constexpr int kNElts = Ktraits::kNElts;
  constexpr int kNThreadsPerRow = Ktraits::kNThreadsPerRow;
  constexpr int kChunkSizeL = Ktraits::kChunkSizeL;
  constexpr int kChunkSizeC = Ktraits::kNEltsPerRow;
  using input_t = typename Ktraits::input_t;
  using vec_t = typename Ktraits::vec_t;
  using weight_t = typename Ktraits::weight_t;

  __shared__ input_t x_smem[kWidth - 1 + kChunkSizeL][kChunkSizeC + kNElts];

  const int batch_id = blockIdx.x;
  const int chunk_l_id = blockIdx.y;
  const int chunk_c_id = blockIdx.z;
  const int tid = threadIdx.x;
  const int l_idx = tid / kNThreadsPerRow;
  const int c_idx = tid % kNThreadsPerRow;

  auto* x = reinterpret_cast<input_t*>(params.x_ptr) +
            batch_id * params.x_batch_stride +
            (chunk_l_id * kChunkSizeL + l_idx) * params.x_l_stride +
            chunk_c_id * kChunkSizeC + c_idx * kNElts;
  auto* weight = reinterpret_cast<weight_t*>(params.weight_ptr) +
                 chunk_c_id * kChunkSizeC * params.weight_c_stride;
  auto* out = reinterpret_cast<input_t*>(params.out_ptr) +
              batch_id * params.out_batch_stride +
              (chunk_l_id * kChunkSizeL + l_idx) * params.out_l_stride +
              chunk_c_id * kChunkSizeC + c_idx * kNElts;

#pragma unroll
  for (int l = 0; l < Ktraits::kNLoads; ++l) {
    input_t x_vals_load[kNElts] = {0};
    const int seq_offset = chunk_l_id * kChunkSizeL + l * Ktraits::kNColsPerLoad +
                           l_idx;
    const int ch_offset = chunk_c_id * kChunkSizeC + c_idx * kNElts;
    if (seq_offset < params.seqlen && ch_offset < params.channels) {
      reinterpret_cast<vec_t*>(x_vals_load)[0] =
          *reinterpret_cast<vec_t*>(
              x + l * Ktraits::kNColsPerLoad * params.x_l_stride);
    }
    reinterpret_cast<vec_t*>(x_smem[kWidth - 1 +
                                    l * Ktraits::kNColsPerLoad + l_idx])[c_idx] =
        reinterpret_cast<vec_t*>(x_vals_load)[0];
  }

  if (l_idx < kWidth - 1) {
    input_t x_vals_load[kNElts] = {0};
    const int seq_offset = chunk_l_id * kChunkSizeL + l_idx - (kWidth - 1);
    const int ch_offset = chunk_c_id * kChunkSizeC + c_idx * kNElts;
    if (seq_offset >= 0 && seq_offset < params.seqlen &&
        ch_offset < params.channels) {
      reinterpret_cast<vec_t*>(x_vals_load)[0] =
          *reinterpret_cast<vec_t*>(x - (kWidth - 1) * params.x_l_stride);
    }
    reinterpret_cast<vec_t*>(x_smem[l_idx])[c_idx] =
        reinterpret_cast<vec_t*>(x_vals_load)[0];
  }

  __syncthreads();

  constexpr int kLPerThread =
      (kChunkSizeL * kChunkSizeC) / kNThreads >= kChunkSizeL
          ? kChunkSizeL
          : (kChunkSizeL * kChunkSizeC) / kNThreads;
  static_assert(kLPerThread * kNThreads == kChunkSizeL * kChunkSizeC);
  constexpr int kNRows = kChunkSizeL / kLPerThread;
  const int row_idx = tid / kNRows;
  const int col_idx = tid % kNRows;

  const int channel_index = chunk_c_id * kChunkSizeC + row_idx;
  float bias_val = 0.f;
  if (params.bias_ptr != nullptr && channel_index < params.channels) {
    bias_val = static_cast<float>(reinterpret_cast<weight_t*>(params.bias_ptr)
                                      [channel_index]);
  }

  float weight_vals[kWidth] = {0.f};
  if (channel_index < params.channels) {
#pragma unroll
    for (int w = 0; w < kWidth; ++w) {
      weight_vals[w] = static_cast<float>(
          weight[row_idx * params.weight_c_stride +
                 w * params.weight_width_stride]);
    }
  }

  float x_vals[kWidth - 1 + kLPerThread];
#pragma unroll
  for (int i = 0; i < kWidth - 1 + kLPerThread; ++i) {
    x_vals[i] = static_cast<float>(
        x_smem[col_idx * kLPerThread + i][row_idx]);
  }

  float out_vals[kLPerThread];
#pragma unroll
  for (int i = 0; i < kLPerThread; ++i) {
    out_vals[i] = bias_val;
    if (channel_index >= params.channels) {
      out_vals[i] = 0.f;
      continue;
    }
#pragma unroll
    for (int w = 0; w < kWidth; ++w) {
      out_vals[i] += weight_vals[w] * x_vals[i + w];
    }
    if (params.silu_activation) {
      out_vals[i] = out_vals[i] / (1.0f + expf(-out_vals[i]));
    }
  }

  __syncthreads();
#pragma unroll
  for (int i = 0; i < kLPerThread; ++i) {
    x_smem[col_idx * kLPerThread + i][row_idx] =
        static_cast<input_t>(out_vals[i]);
  }
  __syncthreads();

#pragma unroll
  for (int l = 0; l < Ktraits::kNLoads; ++l) {
    input_t out_vals_store[kNElts];
    reinterpret_cast<vec_t*>(out_vals_store)[0] =
        reinterpret_cast<vec_t*>(x_smem[l * Ktraits::kNColsPerLoad + l_idx])[c_idx];
    const int seq_offset = chunk_l_id * kChunkSizeL +
                           l * Ktraits::kNColsPerLoad + l_idx;
    const int ch_offset = chunk_c_id * kChunkSizeC + c_idx * kNElts;
    if (seq_offset < params.seqlen && ch_offset < params.channels) {
      *reinterpret_cast<vec_t*>(out +
                                l * Ktraits::kNColsPerLoad * params.out_l_stride) =
          reinterpret_cast<vec_t*>(out_vals_store)[0];
    }
  }
}

template <int kNThreads, int kWidth, typename input_t, typename weight_t>
void launch_channels_last(ConvParams& params, cudaStream_t stream) {
  using Ktraits = DwConvChannelLastKernelTraits<kNThreads, kWidth, 64, true,
                                                input_t, weight_t>;
  constexpr int kChunkSizeL = Ktraits::kChunkSizeL;
  constexpr int kChunkSizeC = Ktraits::kNEltsPerRow;
  const int n_chunks_L = (params.seqlen + kChunkSizeL - 1) / kChunkSizeL;
  const int n_chunks_C = (params.channels + kChunkSizeC - 1) / kChunkSizeC;
  dim3 grid(params.batch, n_chunks_L, n_chunks_C);
  auto* kernel = &dw_causal_conv_channels_last_kernel<Ktraits>;
  kernel<<<grid, Ktraits::kNThreads, 0, stream>>>(params);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename input_t>
at::Tensor dw_causal_conv_cuda_impl(const at::Tensor& x,
                                    const at::Tensor& weight,
                                    const c10::optional<at::Tensor>& bias,
                                    bool channels_first,
                                    bool silu_activation) {
  auto stream = c10::cuda::getCurrentCUDAStream();
  auto weight_3d =
      weight.dim() == 2 ? weight.unsqueeze(1) : weight;
  auto x_contig = x.contiguous();
  auto weight_contig = weight_3d.contiguous();
  auto out = at::empty_like(x_contig);

  ConvParams params;
  params.batch = static_cast<int>(x_contig.size(0));
  params.channels = static_cast<int>(channels_first ? x_contig.size(1)
                                                    : x_contig.size(2));
  params.seqlen = static_cast<int>(channels_first ? x_contig.size(2)
                                                  : x_contig.size(1));
  params.width = static_cast<int>(weight_contig.size(2));
  params.silu_activation = silu_activation;

  params.x_batch_stride = static_cast<ConvParams::index_t>(x_contig.stride(0));
  params.x_c_stride = static_cast<ConvParams::index_t>(
      channels_first ? x_contig.stride(1) : x_contig.stride(2));
  params.x_l_stride = static_cast<ConvParams::index_t>(
      channels_first ? x_contig.stride(2) : x_contig.stride(1));

  params.weight_c_stride =
      static_cast<ConvParams::index_t>(weight_contig.stride(0));
  params.weight_width_stride =
      static_cast<ConvParams::index_t>(weight_contig.stride(2));

  params.out_batch_stride = static_cast<ConvParams::index_t>(out.stride(0));
  params.out_c_stride = static_cast<ConvParams::index_t>(
      channels_first ? out.stride(1) : out.stride(2));
  params.out_l_stride = static_cast<ConvParams::index_t>(
      channels_first ? out.stride(2) : out.stride(1));

  params.x_ptr = x_contig.data_ptr();
  params.weight_ptr = const_cast<void*>(weight_contig.data_ptr());
  params.bias_ptr = bias.has_value() ? const_cast<void*>(bias->data_ptr())
                                     : nullptr;
  params.out_ptr = out.data_ptr();

  if (channels_first) {
    if (params.width == 2) {
      launch_channels_first<128, 2, input_t, input_t>(params, stream);
    } else if (params.width == 3) {
      launch_channels_first<128, 3, input_t, input_t>(params, stream);
    } else if (params.width == 4) {
      launch_channels_first<128, 4, input_t, input_t>(params, stream);
    } else {
      TORCH_CHECK(false,
                  "Unsupported kernel size for CUDA depthwise causal conv.");
    }
  } else {
    if (params.width == 2) {
      launch_channels_last<128, 2, input_t, input_t>(params, stream);
    } else if (params.width == 3) {
      launch_channels_last<128, 3, input_t, input_t>(params, stream);
    } else if (params.width == 4) {
      launch_channels_last<128, 4, input_t, input_t>(params, stream);
    } else {
      TORCH_CHECK(false,
                  "Unsupported kernel size for CUDA depthwise causal conv.");
    }
  }

  return out;
}

template <typename input_t>
__global__ void silu_backward_kernel(const input_t* preact, input_t* grad,
                                     int64_t total) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }
  float grad_val = static_cast<float>(grad[index]);
  float pre_val = static_cast<float>(preact[index]);
  float sigmoid = 1.0f / (1.0f + expf(-pre_val));
  float derivative = sigmoid * (1.0f + pre_val * (1.0f - sigmoid));
  grad[index] = static_cast<input_t>(grad_val * derivative);
}

template <typename input_t>
__global__ void dw_causal_conv_backward_channels_first_kernel(
    const input_t* grad_act, const input_t* x, const input_t* weight,
    input_t* grad_x, float* grad_weight, float* grad_bias, int batch,
    int channels, int seqlen, int width, bool accumulate_bias) {
  constexpr int kMaxWidth = 4;
  int batch_id = blockIdx.x;
  int channel_id = blockIdx.y;
  int thread_id = threadIdx.x;
  int stride = blockDim.x;

  if (batch_id >= batch || channel_id >= channels) {
    return;
  }

  const int64_t base =
      (static_cast<int64_t>(batch_id) * channels + channel_id) * seqlen;
  const input_t* grad_ptr = grad_act + base;
  const input_t* x_ptr = x + base;
  input_t* grad_x_ptr = grad_x + base;
  const input_t* weight_ptr = weight + static_cast<int64_t>(channel_id) * width;

  float local_weight[kMaxWidth] = {0.f, 0.f, 0.f, 0.f};
  float local_bias = 0.f;

  for (int idx = thread_id; idx < seqlen; idx += stride) {
    float grad_val = static_cast<float>(grad_ptr[idx]);
    if (accumulate_bias) {
      local_bias += grad_val;
    }

    float grad_x_val = 0.f;
    for (int k = 0; k < width; ++k) {
      int grad_offset = idx + k;
      if (grad_offset < seqlen) {
        grad_x_val += static_cast<float>(grad_ptr[grad_offset]) *
                      static_cast<float>(weight_ptr[k]);
      }
      int x_offset = idx - k;
      if (x_offset >= 0) {
        local_weight[k] +=
            grad_val * static_cast<float>(x_ptr[x_offset]);
      }
    }
    grad_x_ptr[idx] = static_cast<input_t>(grad_x_val);
  }

  __syncthreads();

  for (int k = 0; k < width; ++k) {
    if (local_weight[k] != 0.f) {
      atomicAdd(&grad_weight[channel_id * width + k], local_weight[k]);
    }
  }

  if (accumulate_bias && local_bias != 0.f) {
    atomicAdd(&grad_bias[channel_id], local_bias);
  }
}

template <typename input_t>
std::tuple<at::Tensor, at::Tensor, c10::optional<at::Tensor>>
dw_causal_conv_backward_cuda_impl(const at::Tensor& grad_output_cf,
                                  const at::Tensor& x_cf,
                                  const at::Tensor& weight_cf,
                                  const c10::optional<at::Tensor>& bias,
                                  bool silu_activation) {
  TORCH_CHECK(grad_output_cf.is_contiguous(),
              "grad_output must be contiguous");
  TORCH_CHECK(x_cf.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(weight_cf.is_contiguous(), "weight must be contiguous");

  const auto batch = static_cast<int>(x_cf.size(0));
  const auto channels = static_cast<int>(x_cf.size(1));
  const auto seqlen = static_cast<int>(x_cf.size(2));
  const auto width = static_cast<int>(weight_cf.size(2));

  auto stream = c10::cuda::getCurrentCUDAStream();

  at::Tensor grad_act = grad_output_cf.clone();
  if (silu_activation) {
    auto preact = dw_causal_conv_cuda_impl<input_t>(
        x_cf, weight_cf, bias, /*channels_first=*/true,
        /*silu_activation=*/false);
    const int64_t total = grad_act.numel();
    const int threads = 256;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    silu_backward_kernel<input_t><<<blocks, threads, 0, stream>>>(
        preact.data_ptr<input_t>(), grad_act.data_ptr<input_t>(), total);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  auto grad_x = at::empty_like(x_cf);
  auto weight_view = weight_cf.view({channels, width});
  auto grad_weight_accum = at::zeros(
      {channels, width}, grad_act.options().dtype(at::kFloat));
  float* grad_bias_ptr = nullptr;
  at::Tensor grad_bias_accum;
  const bool has_bias = bias.has_value();
  if (has_bias) {
    grad_bias_accum =
        at::zeros({channels}, grad_act.options().dtype(at::kFloat));
    grad_bias_ptr = grad_bias_accum.data_ptr<float>();
  }

  dim3 grid(batch, channels, 1);
  const int threads = 128;
  dw_causal_conv_backward_channels_first_kernel<input_t>
      <<<grid, threads, 0, stream>>>(grad_act.data_ptr<input_t>(),
                                     x_cf.data_ptr<input_t>(),
                                     weight_view.data_ptr<input_t>(),
                                     grad_x.data_ptr<input_t>(),
                                     grad_weight_accum.data_ptr<float>(),
                                     grad_bias_ptr, batch, channels, seqlen,
                                     width, has_bias);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  auto grad_weight = grad_weight_accum.to(weight_cf.scalar_type())
                         .view_as(weight_cf);
  c10::optional<at::Tensor> grad_bias_opt = c10::nullopt;
  if (has_bias) {
    grad_bias_opt = grad_bias_accum.to(weight_cf.scalar_type());
  }

  return std::make_tuple(grad_x, grad_weight, grad_bias_opt);
}

at::Tensor dw_causal_conv_fallback(const at::Tensor& x,
                                   const at::Tensor& weight,
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

  const auto pad = std::max<int64_t>(kernel_size - 1, 0);
  at::Tensor x_padded = pad > 0 ? at::constant_pad_nd(x_conv, {pad, 0})
                                : x_conv;

  auto conv_out = at::conv1d(x_padded, weight_cast, bias_cast, {1}, {0}, {1},
                             channels);

  auto act = activation;
  std::transform(act.begin(), act.end(), act.begin(), ::tolower);
  if (act == "silu" || act == "swish") {
    conv_out = at::silu(conv_out);
  } else if (act == "relu") {
    conv_out = at::relu(conv_out);
  } else if (act == "identity" || act == "none") {
    // no-op
  } else {
    TORCH_CHECK(false, "Unsupported activation '" + activation + "'.");
  }

  if (!channels_first) {
    conv_out = conv_out.permute({0, 2, 1}).contiguous();
  }

  return conv_out.to(x.scalar_type());
}

bool is_supported_activation(const std::string& activation) {
  std::string lowered = activation;
  std::transform(lowered.begin(), lowered.end(), lowered.begin(), ::tolower);
  return lowered == "identity" || lowered == "none" || lowered == "silu" ||
         lowered == "swish";
}

bool use_silu(const std::string& activation) {
  std::string lowered = activation;
  std::transform(lowered.begin(), lowered.end(), lowered.begin(), ::tolower);
  return lowered == "silu" || lowered == "swish";
}

}  // namespace

at::Tensor dw_causal_conv_cuda(const at::Tensor& x, const at::Tensor& weight,
                              const c10::optional<at::Tensor>& bias,
                              const std::string& activation) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor.");
  TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor.");
  if (bias.has_value()) {
    TORCH_CHECK(bias.value().is_cuda(), "bias must be a CUDA tensor.");
  }
  TORCH_CHECK(x.dim() == 3, "x must have shape (B, C, L) or (B, L, C).");

  if (!is_supported_activation(activation)) {
    return dw_causal_conv_fallback(x, weight, bias, activation);
  }

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

  at::Tensor weight_3d;
  if (weight.dim() == 2) {
    TORCH_CHECK(weight.size(0) == channels,
                "weight has incompatible shape.");
    weight_3d = weight.unsqueeze(1);
  } else if (weight.dim() == 3) {
    TORCH_CHECK(weight.size(0) == channels && weight.size(1) == 1,
                "Expected depthwise weights with shape (C, 1, K).");
    weight_3d = weight;
  } else {
    TORCH_CHECK(false, "weight must have 2 or 3 dimensions.");
  }

  if (bias.has_value()) {
    const auto& bias_tensor = bias.value();
    TORCH_CHECK(bias_tensor.dim() == 1 && bias_tensor.size(0) == channels,
                "bias must have shape (C,).");
  }

  const auto dtype = x.scalar_type();
  if (!(dtype == at::kFloat || dtype == at::kHalf || dtype == at::kBFloat16)) {
    return dw_causal_conv_fallback(x, weight, bias, activation);
  }
  if (weight_3d.scalar_type() != dtype) {
    return dw_causal_conv_fallback(x, weight, bias, activation);
  }
  if (bias.has_value() && bias->scalar_type() != dtype) {
    return dw_causal_conv_fallback(x, weight, bias, activation);
  }

  const int64_t kernel_size = weight_3d.size(2);
  if (kernel_size < 2 || kernel_size > 4) {
    return dw_causal_conv_fallback(x, weight, bias, activation);
  }

  if (!x.is_contiguous() || !weight_3d.is_contiguous() ||
      (bias.has_value() && !bias->is_contiguous())) {
    return dw_causal_conv_fallback(x, weight, bias, activation);
  }

  c10::cuda::CUDAGuard guard(x.device());

  at::Tensor result;
  bool silu_activation = use_silu(activation);
  if (dtype == at::kFloat) {
    result = dw_causal_conv_cuda_impl<float>(x, weight_3d, bias,
                                             channels_first,
                                             silu_activation);
  } else if (dtype == at::kHalf) {
    result = dw_causal_conv_cuda_impl<at::Half>(x, weight_3d, bias,
                                                channels_first,
                                                silu_activation);
  } else {
    result = dw_causal_conv_cuda_impl<at::BFloat16>(x, weight_3d, bias,
                                                    channels_first,
                                                    silu_activation);
  }

  return result;
}

std::tuple<at::Tensor, at::Tensor, c10::optional<at::Tensor>>
dw_causal_conv_backward_cuda(const at::Tensor& grad_output,
                             const at::Tensor& x,
                             const at::Tensor& weight,
                             const c10::optional<at::Tensor>& bias,
                             const std::string& activation) {
  TORCH_CHECK(grad_output.is_cuda(),
              "grad_output must be a CUDA tensor.");
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor.");
  TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor.");
  if (bias.has_value()) {
    TORCH_CHECK(bias->is_cuda(), "bias must be a CUDA tensor.");
  }

  TORCH_CHECK(x.dim() == 3, "x must have shape (B, C, L) or (B, L, C).");
  TORCH_CHECK(grad_output.sizes() == x.sizes(),
              "grad_output must have the same shape as x.");

  if (!is_supported_activation(activation)) {
    TORCH_CHECK(false,
                "Unsupported activation '" + activation +
                    "' for CUDA depthwise causal conv backward.");
  }

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

  auto weight_3d = weight.dim() == 2 ? weight.unsqueeze(1) : weight;
  TORCH_CHECK(weight_3d.size(0) == channels && weight_3d.size(1) == 1,
              "Expected depthwise weights with shape (C, 1, K).");

  if (bias.has_value()) {
    const auto& bias_tensor = bias.value();
    TORCH_CHECK(bias_tensor.dim() == 1 && bias_tensor.size(0) == channels,
                "bias must have shape (C,).");
  }

  const auto dtype = x.scalar_type();
  TORCH_CHECK(dtype == at::kFloat || dtype == at::kHalf ||
                  dtype == at::kBFloat16,
              "Unsupported dtype for CUDA depthwise causal conv backward.");
  TORCH_CHECK(weight_3d.scalar_type() == dtype,
              "weight must match x dtype for CUDA depthwise causal conv backward.");
  if (bias.has_value()) {
    TORCH_CHECK(bias->scalar_type() == dtype,
                "bias must match x dtype for CUDA depthwise causal conv backward.");
  }

  const int64_t kernel_size = weight_3d.size(2);
  TORCH_CHECK(2 <= kernel_size && kernel_size <= 4,
              "Unsupported kernel size for CUDA depthwise causal conv backward.");

  auto grad_out_contig = grad_output.contiguous();
  auto x_cf = channels_first ? x.contiguous() : x.permute({0, 2, 1}).contiguous();
  auto grad_cf = channels_first ? grad_out_contig
                                : grad_out_contig.permute({0, 2, 1}).contiguous();
  auto weight_cf = weight_3d.contiguous();
  c10::optional<at::Tensor> bias_cf = c10::nullopt;
  if (bias.has_value()) {
    bias_cf = bias->contiguous();
  }

  c10::cuda::CUDAGuard guard(x.device());
  bool silu_activation = use_silu(activation);

  std::tuple<at::Tensor, at::Tensor, c10::optional<at::Tensor>> grads;
  if (dtype == at::kFloat) {
    grads = dw_causal_conv_backward_cuda_impl<float>(grad_cf, x_cf, weight_cf,
                                                     bias_cf, silu_activation);
  } else if (dtype == at::kHalf) {
    grads = dw_causal_conv_backward_cuda_impl<at::Half>(
        grad_cf, x_cf, weight_cf, bias_cf, silu_activation);
  } else {
    grads = dw_causal_conv_backward_cuda_impl<at::BFloat16>(
        grad_cf, x_cf, weight_cf, bias_cf, silu_activation);
  }

  at::Tensor grad_x_cf;
  at::Tensor grad_weight_cf;
  c10::optional<at::Tensor> grad_bias_cf;
  std::tie(grad_x_cf, grad_weight_cf, grad_bias_cf) = grads;

  at::Tensor grad_x = channels_first
                          ? grad_x_cf
                          : grad_x_cf.permute({0, 2, 1}).contiguous();

  at::Tensor grad_weight = weight.dim() == 2 ? grad_weight_cf.squeeze(1)
                                             : grad_weight_cf;

  c10::optional<at::Tensor> grad_bias = c10::nullopt;
  if (bias.has_value()) {
    TORCH_CHECK(grad_bias_cf.has_value(),
                "Bias gradients were not produced by CUDA backward kernel.");
    grad_bias = grad_bias_cf->view(bias->sizes());
  }

  return std::make_tuple(grad_x, grad_weight, grad_bias);
}

}  // namespace cuda
}  // namespace ssm
