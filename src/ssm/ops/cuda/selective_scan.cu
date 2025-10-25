#include "common.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_store.cuh>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <tuple>
#include <type_traits>

#ifndef USE_ROCM
#include <cuda_bf16.h>
#else
#include <hip/hip_bf16.h>
#endif
#include <cuda_fp16.h>
#include <c10/util/complex.h>

namespace ssm {
namespace cuda {
namespace {

// The CUDA kernels below are adapted from the reference implementation in the
// state-spaces/mamba project (Tri Dao, 2023). The code has been trimmed to the
// forward pass and integrated with the ssm dispatch/broadcast conventions.

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------

enum class ScanParamKind { kShared, kPerBatch, kGroupedTime };

struct PreparedScanParam {
  ScanParamKind kind{ScanParamKind::kShared};
  at::Tensor tensor;
  int64_t stride_batch{0};
  int64_t stride_dim{0};
  int64_t stride_group{0};
  int64_t stride_state{0};
  int64_t stride_time{0};
  int64_t groups{1};
  int64_t dim_per_group{1};
  int64_t time_size{1};
};

PreparedScanParam prepare_scan_param(const std::string& name,
                                     const at::Tensor& param, int64_t batch,
                                     int64_t dim, int64_t state_dim,
                                     int64_t length, at::ScalarType dtype,
                                     const at::Device& device,
                                     bool expect_variable_dtype) {
  PreparedScanParam info;
  if (param.dim() == 2) {
    TORCH_CHECK(param.size(0) == dim && param.size(1) == state_dim,
                name, " must have shape (D, N).");
    info.tensor = to_compute(param, dtype, device).contiguous();
    info.kind = ScanParamKind::kShared;
    info.stride_dim = info.tensor.stride(0);
    info.stride_state = info.tensor.stride(1);
    return info;
  }
  if (param.dim() == 3) {
    TORCH_CHECK(param.size(0) == batch && param.size(2) == state_dim,
                name, " must have shape (B, D, N) when 3-D.");
    auto target_dtype = expect_variable_dtype ? dtype : param.scalar_type();
    auto tensor = param.to(device, target_dtype).contiguous();
    TORCH_CHECK(tensor.size(1) == dim,
                name,
                " second dimension must match number of channels when 3-D.");
    info.groups = tensor.size(1);
    info.dim_per_group = dim / info.groups;
    if (expect_variable_dtype) {
      info.kind = ScanParamKind::kGroupedTime;
      info.time_size = length;
      info.tensor =
          tensor.view({batch, info.groups, state_dim, 1})
              .expand({batch, info.groups, state_dim, length})
              .contiguous();
      info.stride_batch = info.tensor.stride(0);
      info.stride_group = info.tensor.stride(1);
      info.stride_state = info.tensor.stride(2);
      info.stride_time = info.tensor.stride(3);
    } else {
      info.tensor = tensor;
      info.kind = ScanParamKind::kPerBatch;
      info.stride_batch = info.tensor.stride(0);
      info.stride_dim = info.tensor.stride(1);
      info.stride_state = info.tensor.stride(2);
    }
    return info;
  }
  if (param.dim() == 4) {
    TORCH_CHECK(param.size(0) == batch && param.size(2) == state_dim,
                name, " must have shape (B, G, N, L) when 4-D.");
    info.tensor =
        param.to(device, expect_variable_dtype ? dtype : param.scalar_type());
    info.tensor = info.tensor.contiguous();
    info.kind = ScanParamKind::kGroupedTime;
    info.groups = info.tensor.size(1);
    TORCH_CHECK(dim % info.groups == 0,
                "Group dimension must divide number of channels.");
    info.dim_per_group = dim / info.groups;
    info.time_size = info.tensor.size(3);
    TORCH_CHECK(info.time_size == length || info.time_size == 1,
                name, " last dimension must match sequence length or 1.");
    if (info.time_size == 1 && length != 1) {
      info.tensor =
          info.tensor.expand({batch, info.groups, state_dim, length}).contiguous();
      info.time_size = length;
    }
    info.stride_batch = info.tensor.stride(0);
    info.stride_group = info.tensor.stride(1);
    info.stride_state = info.tensor.stride(2);
    info.stride_time = info.tensor.stride(3);
    return info;
  }
  TORCH_CHECK(false, "Unsupported rank for ", name, ".");
}

// ---------------------------------------------------------------------------
// Kernel parameter struct (forward only)
// ---------------------------------------------------------------------------

struct SSMParamsBase {
  using index_t = uint32_t;

  int batch{0}, dim{0}, seqlen{0}, dstate{0}, n_groups{1}, n_chunks{1};
  int dim_ngroups_ratio{1};
  bool is_variable_B{false};
  bool is_variable_C{false};
  bool delta_softplus{false};

  index_t A_d_stride{0};
  index_t A_dstate_stride{0};
  index_t B_batch_stride{0};
  index_t B_d_stride{0};
  index_t B_dstate_stride{0};
  index_t B_group_stride{0};
  index_t C_batch_stride{0};
  index_t C_d_stride{0};
  index_t C_dstate_stride{0};
  index_t C_group_stride{0};
  index_t u_batch_stride{0};
  index_t u_d_stride{0};
  index_t delta_batch_stride{0};
  index_t delta_d_stride{0};
  index_t z_batch_stride{0};
  index_t z_d_stride{0};
  index_t out_batch_stride{0};
  index_t out_d_stride{0};
  index_t out_z_batch_stride{0};
  index_t out_z_d_stride{0};

  void* __restrict__ A_ptr{nullptr};
  void* __restrict__ B_ptr{nullptr};
  void* __restrict__ C_ptr{nullptr};
  void* __restrict__ D_ptr{nullptr};
  void* __restrict__ u_ptr{nullptr};
  void* __restrict__ delta_ptr{nullptr};
  void* __restrict__ delta_bias_ptr{nullptr};
  void* __restrict__ out_ptr{nullptr};
  void* __restrict__ x_ptr{nullptr};
  void* __restrict__ z_ptr{nullptr};
  void* __restrict__ out_z_ptr{nullptr};
};

// ---------------------------------------------------------------------------
// Kernel implementation helpers (adapted from upstream)
// ---------------------------------------------------------------------------

#ifndef USE_ROCM
constexpr size_t custom_max(std::initializer_list<size_t> ilist) {
  return std::max(ilist);
}

template <typename T>
constexpr T constexpr_min(T a, T b) {
  return std::min(a, b);
}
#else
constexpr size_t custom_max(std::initializer_list<size_t> ilist) {
  return *std::max_element(ilist.begin(), ilist.end());
}

template <typename T>
constexpr T constexpr_min(T a, T b) {
  return a < b ? a : b;
}
#endif

#define MAX_DSTATE 256

using complex_t = c10::complex<float>;

inline __device__ float2 operator+(const float2& a, const float2& b) {
  return {a.x + b.x, a.y + b.y};
}

inline __device__ float4 operator+(const float4& a, const float4& b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
}

// Static switch helper.
#define BOOL_SWITCH(COND, CONST_NAME, ...)                                           \
  [&] {                                                                             \
    if (COND) {                                                                     \
      constexpr bool CONST_NAME = true;                                             \
      return __VA_ARGS__();                                                         \
    } else {                                                                        \
      constexpr bool CONST_NAME = false;                                            \
      return __VA_ARGS__();                                                         \
    }                                                                               \
  }()

// Bytes to vector type mapping.
template <int BYTES>
struct BytesToType {};

template <>
struct BytesToType<16> {
  using Type = uint4;
  static_assert(sizeof(Type) == 16);
};

template <>
struct BytesToType<8> {
  using Type = uint64_t;
  static_assert(sizeof(Type) == 8);
};

template <>
struct BytesToType<4> {
  using Type = uint32_t;
  static_assert(sizeof(Type) == 4);
};

template <>
struct BytesToType<2> {
  using Type = uint16_t;
  static_assert(sizeof(Type) == 2);
};

template <>
struct BytesToType<1> {
  using Type = uint8_t;
  static_assert(sizeof(Type) == 1);
};

template <typename scalar_t, int N>
struct Converter {
  static inline __device__ void to_float(const scalar_t (&src)[N],
                                         float (&dst)[N]) {
    #pragma unroll
    for (int i = 0; i < N; ++i) {
      dst[i] = src[i];
    }
  }
};

template <int N>
struct Converter<at::Half, N> {
  static inline __device__ void to_float(const at::Half (&src)[N],
                                         float (&dst)[N]) {
    static_assert(N % 2 == 0);
    auto& src2 = reinterpret_cast<const half2 (&)[N / 2]>(src);
    auto& dst2 = reinterpret_cast<float2 (&)[N / 2]>(dst);
    #pragma unroll
    for (int i = 0; i < N / 2; ++i) {
      dst2[i] = __half22float2(src2[i]);
    }
  }
};

#if __CUDA_ARCH__ >= 800
template <int N>
struct Converter<at::BFloat16, N> {
  static inline __device__ void to_float(const at::BFloat16 (&src)[N],
                                         float (&dst)[N]) {
    static_assert(N % 2 == 0);
    auto& src2 = reinterpret_cast<const nv_bfloat162 (&)[N / 2]>(src);
    auto& dst2 = reinterpret_cast<float2 (&)[N / 2]>(dst);
    #pragma unroll
    for (int i = 0; i < N / 2; ++i) {
      dst2[i] = __bfloat1622float2(src2[i]);
    }
  }
};
#endif

__device__ __forceinline__ complex_t cexp2f(complex_t z) {
  float t = exp2f(z.real_);
  float c, s;
  sincosf(z.imag_, &s, &c);
  return complex_t(c * t, s * t);
}

template <typename scalar_t>
struct SSMScanOp;

template <>
struct SSMScanOp<float> {
  __device__ __forceinline__ float2 operator()(const float2& ab0,
                                               const float2& ab1) const {
    return make_float2(ab1.x * ab0.x, ab1.x * ab0.y + ab1.y);
  }
};

template <>
struct SSMScanOp<complex_t> {
  __device__ __forceinline__ float4 operator()(const float4& ab0,
                                               const float4& ab1) const {
    complex_t a0 = complex_t(ab0.x, ab0.y);
    complex_t b0 = complex_t(ab0.z, ab0.w);
    complex_t a1 = complex_t(ab1.x, ab1.y);
    complex_t b1 = complex_t(ab1.z, ab1.w);
    complex_t out_a = a1 * a0;
    complex_t out_b = a1 * b0 + b1;
    return make_float4(out_a.real_, out_a.imag_, out_b.real_, out_b.imag_);
  }
};

template <typename scalar_t>
struct SSMScanPrefixCallbackOp {
  using scan_t = std::conditional_t<std::is_same_v<scalar_t, float>, float2, float4>;
  scan_t running_prefix;
  __device__ explicit SSMScanPrefixCallbackOp(scan_t running_prefix_)
      : running_prefix(running_prefix_) {}
  __device__ scan_t operator()(scan_t block_aggregate) {
    scan_t old_prefix = running_prefix;
    running_prefix = SSMScanOp<scalar_t>()(running_prefix, block_aggregate);
    return old_prefix;
  }
};

template <typename Ktraits>
inline __device__ void load_input(typename Ktraits::input_t* u,
                                  typename Ktraits::input_t (&u_vals)[Ktraits::kNItems],
                                  typename Ktraits::BlockLoadT::TempStorage& smem_load,
                                  int seqlen) {
  if constexpr (Ktraits::kIsEvenLen) {
    auto& smem_load_vec = reinterpret_cast<typename Ktraits::BlockLoadVecT::TempStorage&>(smem_load);
    using vec_t = typename Ktraits::vec_t;
    typename Ktraits::BlockLoadVecT(smem_load_vec)
        .Load(reinterpret_cast<vec_t*>(u),
              reinterpret_cast<typename Ktraits::vec_t(&)[Ktraits::kNLoads]>(u_vals));
  } else {
    typename Ktraits::BlockLoadT(smem_load).Load(u, u_vals, seqlen, 0.f);
  }
}

template <typename Ktraits>
inline __device__ void load_weight(
    typename Ktraits::input_t* Bvar,
    typename Ktraits::weight_t (&B_vals)[Ktraits::kNItems],
    typename Ktraits::BlockLoadWeightT::TempStorage& smem_load_weight,
    int seqlen) {
  constexpr int kNItems = Ktraits::kNItems;
  if constexpr (!Ktraits::kIsComplex) {
    typename Ktraits::input_t B_vals_load[kNItems];
    if constexpr (Ktraits::kIsEvenLen) {
      auto& smem_load_weight_vec =
          reinterpret_cast<typename Ktraits::BlockLoadWeightVecT::TempStorage&>(smem_load_weight);
      using vec_t = typename Ktraits::vec_t;
      typename Ktraits::BlockLoadWeightVecT(smem_load_weight_vec)
          .Load(reinterpret_cast<vec_t*>(Bvar),
                reinterpret_cast<typename Ktraits::vec_t(&)[Ktraits::kNLoads]>(B_vals_load));
    } else {
      typename Ktraits::BlockLoadWeightT(smem_load_weight)
          .Load(Bvar, B_vals_load, seqlen, 0.f);
    }
    Converter<typename Ktraits::input_t, kNItems>::to_float(B_vals_load, B_vals);
  } else {
    typename Ktraits::input_t B_vals_load[kNItems * 2];
    if constexpr (Ktraits::kIsEvenLen) {
      auto& smem_load_weight_vec =
          reinterpret_cast<typename Ktraits::BlockLoadWeightVecT::TempStorage&>(smem_load_weight);
      using vec_t = typename Ktraits::vec_t;
      typename Ktraits::BlockLoadWeightVecT(smem_load_weight_vec)
          .Load(reinterpret_cast<vec_t*>(Bvar),
                reinterpret_cast<typename Ktraits::vec_t(&)[Ktraits::kNLoads * 2]>(B_vals_load));
    } else {
      typename Ktraits::BlockLoadWeightT(smem_load_weight)
          .Load(Bvar, B_vals_load, seqlen, 0.f);
    }
    #pragma unroll
    for (int i = 0; i < kNItems; ++i) {
      B_vals[i] = complex_t(B_vals_load[i * 2], B_vals_load[i * 2 + 1]);
    }
  }
}

template <typename Ktraits>
inline __device__ void store_output(
    typename Ktraits::input_t* out,
    const float (&out_vals)[Ktraits::kNItems],
    typename Ktraits::BlockStoreT::TempStorage& smem_store,
    int seqlen) {
  typename Ktraits::input_t write_vals[Ktraits::kNItems];
  #pragma unroll
  for (int i = 0; i < Ktraits::kNItems; ++i) {
    write_vals[i] = out_vals[i];
  }
  if constexpr (Ktraits::kIsEvenLen) {
    auto& smem_store_vec =
        reinterpret_cast<typename Ktraits::BlockStoreVecT::TempStorage&>(smem_store);
    using vec_t = typename Ktraits::vec_t;
    typename Ktraits::BlockStoreVecT(smem_store_vec)
        .Store(reinterpret_cast<vec_t*>(out),
               reinterpret_cast<typename Ktraits::vec_t(&)[Ktraits::kNLoads]>(write_vals));
  } else {
    typename Ktraits::BlockStoreT(smem_store).Store(out, write_vals, seqlen);
  }
}

template <int kNThreads_, int kNItems_, int kNRows_, bool kIsEvenLen_,
          bool kIsVariableB_, bool kIsVariableC_, bool kHasZ_, typename input_t_,
          typename weight_t_>
struct SelectiveScanKernelTraits {
  static_assert(kNItems_ % 4 == 0);
  using input_t = input_t_;
  using weight_t = weight_t_;
  static constexpr int kNThreads = kNThreads_;
  static constexpr int kMinBlocks = kNThreads < 128 ? 5 : 3;
  static constexpr int kNItems = kNItems_;
  static constexpr int kNRows = kNRows_;
  static constexpr int kNBytes = sizeof(input_t);
  static_assert(kNBytes == 2 || kNBytes == 4);
  static constexpr int kNElts = kNBytes == 4 ? 4 : constexpr_min(8, kNItems);
  static_assert(kNItems % kNElts == 0);
  static constexpr int kNLoads = kNItems / kNElts;
  static constexpr bool kIsComplex = std::is_same_v<weight_t, complex_t>;
  static constexpr bool kIsEvenLen = kIsEvenLen_;
  static constexpr bool kIsVariableB = kIsVariableB_;
  static constexpr bool kIsVariableC = kIsVariableC_;
  static constexpr bool kHasZ = kHasZ_;
  static constexpr bool kDirectIO = kIsEvenLen && kNLoads == 1;
  using vec_t = typename BytesToType<kNBytes * kNElts>::Type;
  using scan_t = std::conditional_t<!kIsComplex, float2, float4>;
  using BlockLoadT = cub::BlockLoad<input_t, kNThreads, kNItems,
                                    cub::BLOCK_LOAD_WARP_TRANSPOSE>;
  using BlockLoadVecT = cub::BlockLoad<vec_t, kNThreads, kNLoads,
                                       !kDirectIO ? cub::BLOCK_LOAD_WARP_TRANSPOSE
                                                  : cub::BLOCK_LOAD_DIRECT>;
  using BlockLoadWeightT = cub::BlockLoad<input_t, kNThreads,
                                          !kIsComplex ? kNItems : kNItems * 2,
                                          cub::BLOCK_LOAD_WARP_TRANSPOSE>;
  using BlockLoadWeightVecT = cub::BlockLoad<vec_t, kNThreads,
                                             !kIsComplex ? kNLoads : kNLoads * 2,
                                             !kDirectIO ? cub::BLOCK_LOAD_WARP_TRANSPOSE
                                                        : cub::BLOCK_LOAD_DIRECT>;
  using BlockStoreT = cub::BlockStore<input_t, kNThreads, kNItems,
                                      cub::BLOCK_STORE_WARP_TRANSPOSE>;
  using BlockStoreVecT = cub::BlockStore<vec_t, kNThreads, kNLoads,
                                         !kDirectIO ? cub::BLOCK_STORE_WARP_TRANSPOSE
                                                    : cub::BLOCK_STORE_DIRECT>;
  using BlockScanT = cub::BlockScan<scan_t, kNThreads,
                                    cub::BLOCK_SCAN_WARP_SCANS>;
  static constexpr int kSmemIOSize =
      custom_max({sizeof(typename BlockLoadT::TempStorage),
                  sizeof(typename BlockLoadVecT::TempStorage),
                  (int(kIsVariableB) + int(kIsVariableC)) *
                      sizeof(typename BlockLoadWeightT::TempStorage),
                  (int(kIsVariableB) + int(kIsVariableC)) *
                      sizeof(typename BlockLoadWeightVecT::TempStorage),
                  sizeof(typename BlockStoreT::TempStorage),
                  sizeof(typename BlockStoreVecT::TempStorage)});
  static constexpr int kSmemSize =
      kSmemIOSize + sizeof(typename BlockScanT::TempStorage);
};

template <typename Ktraits>
__global__ __launch_bounds__(Ktraits::kNThreads, Ktraits::kMinBlocks)
void selective_scan_fwd_kernel(SSMParamsBase params) {
  constexpr bool kIsComplex = Ktraits::kIsComplex;
  constexpr bool kIsVariableB = Ktraits::kIsVariableB;
  constexpr bool kIsVariableC = Ktraits::kIsVariableC;
  constexpr bool kHasZ = Ktraits::kHasZ;
  constexpr int kNThreads = Ktraits::kNThreads;
  constexpr int kNItems = Ktraits::kNItems;
  constexpr int kNRows = Ktraits::kNRows;
  constexpr bool kDirectIO = Ktraits::kDirectIO;
  using input_t = typename Ktraits::input_t;
  using weight_t = typename Ktraits::weight_t;
  using scan_t = typename Ktraits::scan_t;

  extern __shared__ char smem_[];
  auto& smem_load = reinterpret_cast<typename Ktraits::BlockLoadT::TempStorage&>(smem_);
  auto& smem_load_weight = reinterpret_cast<typename Ktraits::BlockLoadWeightT::TempStorage&>(smem_);
  auto& smem_load_weight1 =
      *reinterpret_cast<typename Ktraits::BlockLoadWeightT::TempStorage*>(
          smem_ + sizeof(typename Ktraits::BlockLoadWeightT::TempStorage));
  auto& smem_store = reinterpret_cast<typename Ktraits::BlockStoreT::TempStorage&>(smem_);
  auto& smem_scan = *reinterpret_cast<typename Ktraits::BlockScanT::TempStorage*>(
      smem_ + Ktraits::kSmemIOSize);
  scan_t* smem_running_prefix =
      reinterpret_cast<scan_t*>(smem_ + Ktraits::kSmemSize);

  const int batch_id = blockIdx.x;
  const int dim_id = blockIdx.y;
  const int group_id = dim_id / (params.dim_ngroups_ratio);
  input_t* u = reinterpret_cast<input_t*>(params.u_ptr) +
               batch_id * params.u_batch_stride +
               dim_id * kNRows * params.u_d_stride;
  input_t* delta = reinterpret_cast<input_t*>(params.delta_ptr) +
                   batch_id * params.delta_batch_stride +
                   dim_id * kNRows * params.delta_d_stride;
  weight_t* A = reinterpret_cast<weight_t*>(params.A_ptr) +
                dim_id * kNRows * params.A_d_stride;
  weight_t* B = reinterpret_cast<weight_t*>(params.B_ptr) +
                dim_id * kNRows * params.B_d_stride;
  input_t* Bvar = reinterpret_cast<input_t*>(params.B_ptr) +
                  batch_id * params.B_batch_stride +
                  group_id * params.B_group_stride;
  weight_t* C = reinterpret_cast<weight_t*>(params.C_ptr) +
                dim_id * kNRows * params.C_d_stride;
  input_t* Cvar = reinterpret_cast<input_t*>(params.C_ptr) +
                  batch_id * params.C_batch_stride +
                  group_id * params.C_group_stride;
  scan_t* x = reinterpret_cast<scan_t*>(params.x_ptr) +
              (batch_id * params.dim + dim_id * kNRows) * params.n_chunks *
                  params.dstate;

  float D_val[kNRows] = {0};
  if (params.D_ptr != nullptr) {
    #pragma unroll
    for (int r = 0; r < kNRows; ++r) {
      D_val[r] = reinterpret_cast<float*>(params.D_ptr)[dim_id * kNRows + r];
    }
  }
  float delta_bias[kNRows] = {0};
  if (params.delta_bias_ptr != nullptr) {
    #pragma unroll
    for (int r = 0; r < kNRows; ++r) {
      delta_bias[r] =
          reinterpret_cast<float*>(params.delta_bias_ptr)[dim_id * kNRows + r];
    }
  }

  constexpr int kChunkSize = kNThreads * kNItems;
  for (int chunk = 0; chunk < params.n_chunks; ++chunk) {
    input_t u_vals[kNRows][kNItems], delta_vals_load[kNRows][kNItems];
    __syncthreads();
    #pragma unroll
    for (int r = 0; r < kNRows; ++r) {
      if constexpr (!kDirectIO) {
        if (r > 0) {
          __syncthreads();
        }
      }
      load_input<Ktraits>(u + r * params.u_d_stride, u_vals[r], smem_load,
                          params.seqlen - chunk * kChunkSize);
      if constexpr (!kDirectIO) {
        __syncthreads();
      }
      load_input<Ktraits>(delta + r * params.delta_d_stride, delta_vals_load[r],
                          smem_load, params.seqlen - chunk * kChunkSize);
    }
    u += kChunkSize;
    delta += kChunkSize;

    float delta_vals[kNRows][kNItems], delta_u_vals[kNRows][kNItems],
        out_vals[kNRows][kNItems];
    #pragma unroll
    for (int r = 0; r < kNRows; ++r) {
      #pragma unroll
      for (int i = 0; i < kNItems; ++i) {
        float u_val = float(u_vals[r][i]);
        delta_vals[r][i] = float(delta_vals_load[r][i]) + delta_bias[r];
        if (params.delta_softplus) {
          delta_vals[r][i] = delta_vals[r][i] <= 20.f
                                 ? log1pf(expf(delta_vals[r][i]))
                                 : delta_vals[r][i];
        }
        delta_u_vals[r][i] = delta_vals[r][i] * u_val;
        out_vals[r][i] = D_val[r] * u_val;
      }
    }

    __syncthreads();
    for (int state_idx = 0; state_idx < params.dstate; ++state_idx) {
      weight_t A_val[kNRows];
      #pragma unroll
      for (int r = 0; r < kNRows; ++r) {
        A_val[r] = A[state_idx * params.A_dstate_stride +
                     r * params.A_d_stride];
        constexpr float kLog2e = M_LOG2E;
        if constexpr (!kIsComplex) {
          A_val[r] *= kLog2e;
        } else {
          A_val[r].real_ *= kLog2e;
        }
      }
      weight_t BC_val[kNRows];
      weight_t B_vals[kNItems], C_vals[kNItems];
      if constexpr (kIsVariableB) {
        load_weight<Ktraits>(Bvar + state_idx * params.B_dstate_stride, B_vals,
                             smem_load_weight,
                             (params.seqlen - chunk * kChunkSize) *
                                 (!kIsComplex ? 1 : 2));
        if constexpr (!kIsVariableC) {
          #pragma unroll
          for (int r = 0; r < kNRows; ++r) {
            BC_val[r] =
                C[state_idx * params.C_dstate_stride + r * params.C_d_stride];
          }
        }
      }
      if constexpr (kIsVariableC) {
        auto& smem_load_weight_C =
            !kIsVariableB ? smem_load_weight : smem_load_weight1;
        load_weight<Ktraits>(Cvar + state_idx * params.C_dstate_stride, C_vals,
                             smem_load_weight_C,
                             (params.seqlen - chunk * kChunkSize) *
                                 (!kIsComplex ? 1 : 2));
        if constexpr (!kIsVariableB) {
          #pragma unroll
          for (int r = 0; r < kNRows; ++r) {
            BC_val[r] =
                B[state_idx * params.B_dstate_stride + r * params.B_d_stride];
          }
        }
      }
      if constexpr (!kIsVariableB && !kIsVariableC) {
        #pragma unroll
        for (int r = 0; r < kNRows; ++r) {
          BC_val[r] =
              B[state_idx * params.B_dstate_stride + r * params.B_d_stride] *
              C[state_idx * params.C_dstate_stride + r * params.C_d_stride];
        }
      }

      #pragma unroll
      for (int r = 0; r < kNRows; ++r) {
        if (r > 0) {
          __syncthreads();
        }
        typename Ktraits::scan_t thread_data[kNItems];
        #pragma unroll
        for (int i = 0; i < kNItems; ++i) {
          if constexpr (!kIsComplex) {
            thread_data[i] = make_float2(
                exp2f(delta_vals[r][i] * A_val[r]),
                !kIsVariableB ? delta_u_vals[r][i]
                              : B_vals[i] * delta_u_vals[r][i]);
            if constexpr (!Ktraits::kIsEvenLen) {
              if (threadIdx.x * kNItems + i >=
                  params.seqlen - chunk * kChunkSize) {
                thread_data[i] = make_float2(1.f, 0.f);
              }
            }
          } else {
            complex_t delta_a_exp = cexp2f(delta_vals[r][i] * A_val[r]);
            weight_t B_delta_u_val = !kIsVariableB
                                         ? delta_u_vals[r][i]
                                         : B_vals[i] * delta_u_vals[r][i];
            thread_data[i] = make_float4(delta_a_exp.real_, delta_a_exp.imag_,
                                         B_delta_u_val.real_,
                                         B_delta_u_val.imag_);
            if constexpr (!Ktraits::kIsEvenLen) {
              if (threadIdx.x * kNItems + i >=
                  params.seqlen - chunk * kChunkSize) {
                thread_data[i] = make_float4(1.f, 0.f, 0.f, 0.f);
              }
            }
          }
        }
        typename Ktraits::scan_t running_prefix;
        if constexpr (!kIsComplex) {
          running_prefix = chunk > 0 && threadIdx.x % 32 == 0
                               ? smem_running_prefix[state_idx + r * MAX_DSTATE]
                               : make_float2(1.f, 0.f);
        } else {
          running_prefix = chunk > 0 && threadIdx.x % 32 == 0
                               ? smem_running_prefix[state_idx + r * MAX_DSTATE]
                               : make_float4(1.f, 0.f, 0.f, 0.f);
        }
        SSMScanPrefixCallbackOp<weight_t> prefix_op(running_prefix);
        typename Ktraits::BlockScanT(smem_scan)
            .InclusiveScan(thread_data, thread_data, SSMScanOp<weight_t>(),
                           prefix_op);
        if (threadIdx.x == 0) {
          smem_running_prefix[state_idx] = prefix_op.running_prefix;
          x[(r * params.n_chunks + chunk) * params.dstate + state_idx] =
              prefix_op.running_prefix;
        }
        #pragma unroll
        for (int i = 0; i < kNItems; ++i) {
          const weight_t C_val = !kIsVariableC
                                     ? BC_val[r]
                                     : (!kIsVariableB ? BC_val[r] * C_vals[i]
                                                      : C_vals[i]);
          if constexpr (!kIsComplex) {
            out_vals[r][i] += thread_data[i].y * C_val;
          } else {
            out_vals[r][i] +=
                (complex_t(thread_data[i].z, thread_data[i].w) * C_val).real_ *
                2;
          }
        }
      }
    }

    input_t* out = reinterpret_cast<input_t*>(params.out_ptr) +
                   batch_id * params.out_batch_stride +
                   dim_id * kNRows * params.out_d_stride + chunk * kChunkSize;
    __syncthreads();
    #pragma unroll
    for (int r = 0; r < kNRows; ++r) {
      if constexpr (!kDirectIO) {
        if (r > 0) {
          __syncthreads();
        }
      }
      store_output<Ktraits>(out + r * params.out_d_stride, out_vals[r],
                            smem_store, params.seqlen - chunk * kChunkSize);
    }

    if constexpr (kHasZ) {
      input_t* z = reinterpret_cast<input_t*>(params.z_ptr) +
                   batch_id * params.z_batch_stride +
                   dim_id * kNRows * params.z_d_stride + chunk * kChunkSize;
      input_t* out_z = reinterpret_cast<input_t*>(params.out_z_ptr) +
                       batch_id * params.out_z_batch_stride +
                       dim_id * kNRows * params.out_z_d_stride +
                       chunk * kChunkSize;
      #pragma unroll
      for (int r = 0; r < kNRows; ++r) {
        input_t z_vals[kNItems];
        __syncthreads();
        load_input<Ktraits>(z + r * params.z_d_stride, z_vals, smem_load,
                            params.seqlen - chunk * kChunkSize);
        #pragma unroll
        for (int i = 0; i < kNItems; ++i) {
          float z_val = z_vals[i];
          out_vals[r][i] *= z_val / (1 + expf(-z_val));
        }
        __syncthreads();
        store_output<Ktraits>(out_z + r * params.out_z_d_stride, out_vals[r],
                              smem_store, params.seqlen - chunk * kChunkSize);
      }
    }

    Bvar += kChunkSize * (!kIsComplex ? 1 : 2);
    Cvar += kChunkSize * (!kIsComplex ? 1 : 2);
  }
}

template <int kNThreads, int kNItems, typename input_t, typename weight_t>
void selective_scan_fwd_launch(SSMParamsBase& params, cudaStream_t stream) {
  constexpr int kNRows = 1;
  BOOL_SWITCH(params.seqlen % (kNThreads * kNItems) == 0, kIsEvenLen, [&] {
    BOOL_SWITCH(params.is_variable_B, kIsVariableB, [&] {
      BOOL_SWITCH(params.is_variable_C, kIsVariableC, [&] {
        BOOL_SWITCH(params.z_ptr != nullptr, kHasZ, [&] {
          using Ktraits = SelectiveScanKernelTraits<kNThreads, kNItems, kNRows,
                                                   kIsEvenLen, kIsVariableB,
                                                   kIsVariableC, kHasZ, input_t,
                                                   weight_t>;
          constexpr int kSmemSize =
              Ktraits::kSmemSize + kNRows * MAX_DSTATE * sizeof(typename Ktraits::scan_t);
          dim3 grid(params.batch, params.dim / kNRows);
          auto kernel = &selective_scan_fwd_kernel<Ktraits>;
          if (kSmemSize >= 48 * 1024) {
#ifndef USE_ROCM
            C10_CUDA_CHECK(cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                kSmemSize));
#else
            C10_CUDA_CHECK(cudaFuncSetAttribute(
                (void*)kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                kSmemSize));
#endif
          }
          kernel<<<grid, Ktraits::kNThreads, kSmemSize, stream>>>(params);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
      });
    });
  });
}

template <typename input_t, typename weight_t>
void selective_scan_fwd_cuda(SSMParamsBase& params, cudaStream_t stream) {
#ifndef USE_ROCM
  if (params.seqlen <= 128) {
    selective_scan_fwd_launch<32, 4, input_t, weight_t>(params, stream);
  } else if (params.seqlen <= 256) {
    selective_scan_fwd_launch<32, 8, input_t, weight_t>(params, stream);
  } else if (params.seqlen <= 512) {
    selective_scan_fwd_launch<32, 16, input_t, weight_t>(params, stream);
  } else if (params.seqlen <= 1024) {
    selective_scan_fwd_launch<64, 16, input_t, weight_t>(params, stream);
  } else {
    selective_scan_fwd_launch<128, 16, input_t, weight_t>(params, stream);
  }
#else
  if (params.seqlen <= 256) {
    selective_scan_fwd_launch<64, 4, input_t, weight_t>(params, stream);
  } else if (params.seqlen <= 512) {
    selective_scan_fwd_launch<64, 8, input_t, weight_t>(params, stream);
  } else if (params.seqlen <= 1024) {
    selective_scan_fwd_launch<64, 16, input_t, weight_t>(params, stream);
  } else {
    selective_scan_fwd_launch<128, 16, input_t, weight_t>(params, stream);
  }
#endif
}

// Compute the final recurrent state by combining chunk-level prefix scans.
__global__ void accumulate_last_state_kernel(const float2* chunk_prefix,
                                             float* last_state, int batch,
                                             int dim, int n_chunks,
                                             int dstate) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch * dim * dstate;
  if (idx >= total) {
    return;
  }
  int state_idx = idx % dstate;
  int dim_idx = (idx / dstate) % dim;
  int batch_idx = idx / (dstate * dim);
  float state_val = 0.f;
  for (int chunk = 0; chunk < n_chunks; ++chunk) {
    const auto prefix = chunk_prefix[(((batch_idx * dim + dim_idx) * n_chunks +
                                       chunk) * dstate) + state_idx];
    state_val = prefix.x * state_val + prefix.y;
  }
  last_state[idx] = state_val;
}

inline void set_params(SSMParamsBase& params, int batch, int dim, int seqlen,
                       int dstate, int n_groups, int n_chunks,
                       bool is_variable_B, bool is_variable_C, bool softplus,
                       const at::Tensor& u, const at::Tensor& delta,
                       const at::Tensor& A, const at::Tensor& B,
                       const at::Tensor& C, const at::Tensor& out,
                       const at::Tensor& x,
                       const c10::optional<at::Tensor>& z,
                       const c10::optional<at::Tensor>& D,
                       const c10::optional<at::Tensor>& dt_bias) {
  std::memset(&params, 0, sizeof(params));
  params.batch = batch;
  params.dim = dim;
  params.seqlen = seqlen;
  params.dstate = dstate;
  params.n_groups = n_groups;
  params.n_chunks = n_chunks;
  params.dim_ngroups_ratio = dim / n_groups;
  params.is_variable_B = is_variable_B;
  params.is_variable_C = is_variable_C;
  params.delta_softplus = softplus;
  params.A_ptr = const_cast<void*>(A.data_ptr());
  params.B_ptr = const_cast<void*>(B.data_ptr());
  params.C_ptr = const_cast<void*>(C.data_ptr());
  params.u_ptr = const_cast<void*>(u.data_ptr());
  params.delta_ptr = const_cast<void*>(delta.data_ptr());
  params.out_ptr = const_cast<void*>(out.data_ptr());
  params.x_ptr = const_cast<void*>(x.data_ptr());
  if (D.has_value()) {
    params.D_ptr = const_cast<void*>(D.value().data_ptr());
  }
  if (dt_bias.has_value()) {
    params.delta_bias_ptr = const_cast<void*>(dt_bias.value().data_ptr());
  }
  if (z.has_value()) {
    params.z_ptr = const_cast<void*>(z.value().data_ptr());
    params.out_z_ptr = params.out_ptr;
  }
  params.A_d_stride = A.stride(0);
  params.A_dstate_stride = A.stride(1);
  if (!is_variable_B) {
    params.B_d_stride = B.stride(0);
  } else {
    params.B_batch_stride = B.stride(0);
    params.B_group_stride = B.stride(1);
  }
  params.B_dstate_stride = !is_variable_B ? B.stride(1) : B.stride(2);
  if (!is_variable_C) {
    params.C_d_stride = C.stride(0);
  } else {
    params.C_batch_stride = C.stride(0);
    params.C_group_stride = C.stride(1);
  }
  params.C_dstate_stride = !is_variable_C ? C.stride(1) : C.stride(2);
  params.u_batch_stride = u.stride(0);
  params.u_d_stride = u.stride(1);
  params.delta_batch_stride = delta.stride(0);
  params.delta_d_stride = delta.stride(1);
  if (z.has_value()) {
    params.z_batch_stride = z.value().stride(0);
    params.z_d_stride = z.value().stride(1);
    params.out_z_batch_stride = out.stride(0);
    params.out_z_d_stride = out.stride(1);
  }
  params.out_batch_stride = out.stride(0);
  params.out_d_stride = out.stride(1);
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> selective_scan_cuda(
    const at::Tensor& u, const at::Tensor& delta, const at::Tensor& A,
    const at::Tensor& B, const at::Tensor& C, const c10::optional<at::Tensor>& D,
    const c10::optional<at::Tensor>& z,
    const c10::optional<at::Tensor>& dt_bias, bool softplus,
    bool return_last_state) {
  TORCH_CHECK(u.is_cuda(), "u must be a CUDA tensor.");
  TORCH_CHECK(delta.is_cuda(), "delta must be a CUDA tensor.");
  TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor.");
  TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor.");
  TORCH_CHECK(C.is_cuda(), "C must be a CUDA tensor.");

  TORCH_CHECK(u.dim() == 3 && delta.dim() == 3,
              "u and delta must have shape (B, D, L).");
  TORCH_CHECK(u.sizes() == delta.sizes(),
              "delta must match the shape of u.");

  const auto batch = u.size(0);
  const auto dim = u.size(1);
  const auto length = u.size(2);
  TORCH_CHECK(length > 0, "Sequence length must be positive.");
  const auto state_dim = A.size(-1);

  TORCH_CHECK(A.dim() == 2 && A.size(0) == dim && A.size(1) == state_dim,
              "A must have shape (D, N).");
  TORCH_CHECK(state_dim <= MAX_DSTATE,
              "selective_scan only supports state dimension <= 256.");

  const auto device = u.device();
  auto input_dtype = u.scalar_type();
  TORCH_CHECK(input_dtype == at::kFloat || input_dtype == at::kHalf ||
              input_dtype == at::kBFloat16,
              "selective_scan CUDA input must be float, half, or bfloat16.");

  auto weight_dtype = A.scalar_type();
  TORCH_CHECK(weight_dtype == at::kFloat,
              "selective_scan CUDA weights must be float.");

  auto u_prepared = u.contiguous();
  auto delta_prepared = delta.contiguous();
  auto A_prepared = to_compute(A, weight_dtype, device).contiguous();

  c10::optional<at::Tensor> D_prepared = c10::nullopt;
  if (D.has_value()) {
    TORCH_CHECK(D.value().dim() == 1 && D.value().size(0) == dim,
                "D must have shape (D,).");
    TORCH_CHECK(D.value().scalar_type() == at::kFloat,
                "D must be float.");
    D_prepared = D.value().to(device, at::kFloat).contiguous();
  }

  c10::optional<at::Tensor> dt_bias_prepared = c10::nullopt;
  if (dt_bias.has_value()) {
    TORCH_CHECK(dt_bias.value().dim() == 1 && dt_bias.value().size(0) == dim,
                "dt_bias must have shape (D,).");
    TORCH_CHECK(dt_bias.value().scalar_type() == at::kFloat,
                "dt_bias must be float.");
    dt_bias_prepared = dt_bias.value().to(device, at::kFloat).contiguous();
  }

  c10::optional<at::Tensor> z_prepared = c10::nullopt;
  if (z.has_value()) {
    TORCH_CHECK(z.value().dim() == 3 && z.value().sizes() == u.sizes(),
                "z must have shape (B, D, L).");
    TORCH_CHECK(z.value().scalar_type() == input_dtype,
                "z must match input dtype.");
    z_prepared = z.value().contiguous();
  }

  bool is_variable_B = B.dim() >= 3;
  bool is_variable_C = C.dim() >= 3;

  auto B_info = prepare_scan_param("B", B, batch, dim, state_dim, length,
                                   weight_dtype, device, is_variable_B);
  auto C_info = prepare_scan_param("C", C, batch, dim, state_dim, length,
                                   weight_dtype, device, is_variable_C);

  // When parameters vary over time we store them in the input dtype as required
  // by the fused kernel; constant parameters stay in fp32.
  if (is_variable_B) {
    B_info.tensor = B_info.tensor.to(device, input_dtype).contiguous();
  }
  if (is_variable_C) {
    C_info.tensor = C_info.tensor.to(device, input_dtype).contiguous();
  }

  auto out = at::empty_like(delta_prepared);
  const int chunk_limit = 2048;
  const int n_chunks = (length + chunk_limit - 1) / chunk_limit;
  auto x = at::empty({batch, dim, n_chunks, state_dim, 2},
                     A_prepared.options());

  SSMParamsBase params;
  set_params(params, batch, dim, length, state_dim,
             is_variable_B ? B_info.groups : 1, n_chunks, is_variable_B,
             is_variable_C, softplus, u_prepared, delta_prepared, A_prepared,
             B_info.tensor, C_info.tensor, out, x, z_prepared, D_prepared,
             dt_bias_prepared);

  c10::cuda::CUDAGuard device_guard(device);
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  if (input_dtype == at::kFloat) {
    selective_scan_fwd_cuda<float, float>(params, stream);
  } else if (input_dtype == at::kHalf) {
    selective_scan_fwd_cuda<at::Half, float>(params, stream);
  } else {
    selective_scan_fwd_cuda<at::BFloat16, float>(params, stream);
  }

  at::Tensor last_state;
  if (return_last_state) {
    last_state = at::empty({batch, dim, state_dim},
                           A_prepared.options().dtype(at::kFloat));
    const int threads = 256;
    const int blocks = (batch * dim * state_dim + threads - 1) / threads;
    accumulate_last_state_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const float2*>(x.data_ptr()),
        last_state.data_ptr<float>(), batch, dim, n_chunks, state_dim);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    last_state = last_state.to(u.scalar_type());
  } else {
    last_state = at::empty({0}, out.options());
  }

  auto output = out.to(u.scalar_type());
  return std::make_tuple(output, last_state);
}

}  // namespace cuda
}  // namespace ssm
