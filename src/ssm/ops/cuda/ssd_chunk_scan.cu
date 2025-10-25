#include "common.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <cmath>

namespace ssm {
namespace cuda {

namespace {

template <typename scalar_t>
struct Tensor4Accessor {
  const scalar_t* ptr;
  int64_t s0;
  int64_t s1;
  int64_t s2;
  int64_t s3;
};

at::Tensor normalize_chunk_param(const std::string& name,
                                 const at::Tensor& param, int64_t batch,
                                 int64_t seqlen, int64_t heads, int64_t proj,
                                 at::ScalarType dtype, const at::Device& device) {
  if (param.dim() == 2) {
    TORCH_CHECK(param.size(0) == heads && param.size(1) == proj,
                name, " must have shape (H, P).");
    return to_compute(param, dtype, device)
        .view({1, 1, heads, proj})
        .expand({batch, seqlen, heads, proj});
  }
  if (param.dim() == 3) {
    TORCH_CHECK(param.size(0) == batch && param.size(1) == heads &&
                    param.size(2) == proj,
                name, " must be (B, H, P) when 3-D.");
    return to_compute(param, dtype, device)
        .unsqueeze(1)
        .expand({batch, seqlen, heads, proj});
  }
  if (param.dim() == 4) {
    TORCH_CHECK(param.size(0) == batch && param.size(1) == seqlen &&
                    param.size(2) == heads && param.size(3) == proj,
                name, " must have shape (B, L, H, P) when 4-D.");
    return to_compute(param, dtype, device).contiguous();
  }
  TORCH_CHECK(false, "Unsupported rank for ", name, ".");
}

template <typename scalar_t, typename compute_t>
__global__ void ssd_chunk_scan_kernel(
    const scalar_t* __restrict__ x_ptr,
    const scalar_t* __restrict__ dt_ptr,
    const compute_t* __restrict__ A_ptr,
    Tensor4Accessor<scalar_t> B_accessor,
    Tensor4Accessor<scalar_t> C_accessor,
    const scalar_t* __restrict__ D_ptr,
    Tensor4Accessor<scalar_t> Z_accessor,
    compute_t* __restrict__ state_ptr,
    scalar_t* __restrict__ out_ptr,
    const int32_t* __restrict__ lengths_ptr, int64_t batch, int64_t seqlen,
    int64_t heads, int64_t proj, int64_t chunk_size, int64_t chunk_idx,
    int64_t z_proj) {
  const int b = blockIdx.x;
  const int h = blockIdx.y;
  if (b >= batch || h >= heads) {
    return;
  }

  const int32_t seqlen_valid = lengths_ptr[b];
  const int64_t chunk_start = chunk_idx * chunk_size;
  if (chunk_start >= seqlen_valid) {
    return;
  }
  const int64_t chunk_len =
      ::min<int64_t>(chunk_size, seqlen_valid - chunk_start);

  extern __shared__ compute_t smem[];
  compute_t* A_cache = smem;
  compute_t* D_cache = D_ptr != nullptr ? smem + proj : nullptr;

  for (int64_t i = threadIdx.x; i < proj; i += blockDim.x) {
    A_cache[i] = A_ptr[h * proj + i];
    if (D_cache != nullptr) {
      D_cache[i] = static_cast<compute_t>(D_ptr[h * proj + i]);
    }
  }
  __syncthreads();

  const int64_t head_proj_offset = (b * heads + h) * proj;
  compute_t* state_head = state_ptr + head_proj_offset;

  const int64_t chunk_time_offset = (b * seqlen + chunk_start) * heads + h;
  const int64_t chunk_elem_offset =
      ((b * seqlen + chunk_start) * heads + h) * proj;
  const int64_t time_stride = heads * proj;

  const scalar_t* X_chunk = x_ptr + chunk_elem_offset;
  const bool has_z = Z_accessor.ptr != nullptr;
  const scalar_t* dt_chunk = dt_ptr + chunk_time_offset;
  scalar_t* out_chunk = out_ptr + chunk_elem_offset;

  for (int64_t i = threadIdx.x; i < proj; i += blockDim.x) {
    compute_t state = state_head[i];
    for (int64_t t = 0; t < chunk_len; ++t) {
      const int64_t time_offset = t * time_stride;
      const compute_t dt_val = static_cast<compute_t>(dt_chunk[t * heads]);
      const compute_t decay = ::exp(dt_val * A_cache[i]);
      const compute_t x_val = static_cast<compute_t>(X_chunk[time_offset + i]);
      const int64_t time_index = chunk_start + t;
      const int64_t B_index = b * B_accessor.s0 +
                              time_index * B_accessor.s1 +
                              h * B_accessor.s2 + i * B_accessor.s3;
      const compute_t drive = static_cast<compute_t>(B_accessor.ptr[B_index]);
      const compute_t delta_u = dt_val * drive * x_val;
      state = decay * state + delta_u;
      const int64_t C_index = b * C_accessor.s0 +
                              time_index * C_accessor.s1 +
                              h * C_accessor.s2 + i * C_accessor.s3;
      compute_t out_val =
          state * static_cast<compute_t>(C_accessor.ptr[C_index]);
      if (D_cache != nullptr) {
        out_val += D_cache[i] * x_val;
      }
      if (has_z) {
        compute_t gate_val;
        if (z_proj == 1) {
          const int64_t Z_index = b * Z_accessor.s0 +
                                  time_index * Z_accessor.s1 +
                                  h * Z_accessor.s2;
          const compute_t raw_gate =
              static_cast<compute_t>(Z_accessor.ptr[Z_index]);
          gate_val = raw_gate /
                     (compute_t(1) + ::exp(-raw_gate));
        } else {
          const int64_t Z_index = b * Z_accessor.s0 +
                                  time_index * Z_accessor.s1 +
                                  h * Z_accessor.s2 +
                                  i * Z_accessor.s3;
          const compute_t raw_gate =
              static_cast<compute_t>(Z_accessor.ptr[Z_index]);
          gate_val = raw_gate /
                     (compute_t(1) + ::exp(-raw_gate));
        }
        out_val *= gate_val;
      }
      out_chunk[time_offset + i] = static_cast<scalar_t>(out_val);
    }
    state_head[i] = state;
  }
}

}  // namespace

at::Tensor ssd_chunk_scan_cuda(
    const at::Tensor& X, const at::Tensor& dt, const at::Tensor& A,
    const at::Tensor& B, const at::Tensor& C, int64_t chunk_size,
    const c10::optional<at::Tensor>& D, const c10::optional<at::Tensor>& z,
    const c10::optional<at::Tensor>& seq_lens,
    const c10::optional<at::Tensor>& cu_seqlens,
    const c10::optional<at::Tensor>& initial_states) {
  TORCH_CHECK(X.is_cuda(), "X must be a CUDA tensor.");
  TORCH_CHECK(dt.is_cuda(), "dt must be a CUDA tensor.");
  TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor.");
  TORCH_CHECK(B.device().is_cuda(), "B must be a CUDA tensor.");
  TORCH_CHECK(C.device().is_cuda(), "C must be a CUDA tensor.");
  c10::cuda::CUDAGuard guard(X.device());

  TORCH_CHECK(X.dim() == 4, "X must have shape (B, L, H, P).");
  TORCH_CHECK(dt.dim() == 3, "dt must have shape (B, L, H).");

  const auto batch = X.size(0);
  const auto seqlen = X.size(1);
  const auto heads = X.size(2);
  const auto proj = X.size(3);

  TORCH_CHECK(dt.size(0) == batch && dt.size(1) == seqlen && dt.size(2) == heads,
              "dt must align with (B, L, H).");

  auto compute_dtype = get_compute_dtype(X);
  if (A.scalar_type() != at::kFloat && A.scalar_type() != compute_dtype) {
    compute_dtype = promote_compute_dtype(compute_dtype, A.scalar_type());
  }

  const auto device = X.device();

  auto X_compute = to_compute(X, compute_dtype, device).contiguous();
  auto dt_compute = to_compute(dt, compute_dtype, device).contiguous();

  at::Tensor A_compute;
  if (A.dim() == 1) {
    TORCH_CHECK(A.size(0) == heads, "A must have shape (H,) when 1-D.");
    A_compute = to_compute(A, compute_dtype, device)
                    .unsqueeze(-1)
                    .expand({heads, proj})
                    .contiguous();
  } else if (A.dim() == 2) {
    TORCH_CHECK(A.size(0) == heads && A.size(1) == proj,
                "A must have shape (H, P) when 2-D.");
    A_compute = to_compute(A, compute_dtype, device).contiguous();
  } else {
    TORCH_CHECK(false, "A must be 1-D or 2-D.");
  }

  auto B_full = normalize_chunk_param("B", B, batch, seqlen, heads, proj,
                                      compute_dtype, device);
  auto C_full = normalize_chunk_param("C", C, batch, seqlen, heads, proj,
                                      compute_dtype, device);

  c10::optional<at::Tensor> D_compute = c10::nullopt;
  if (D.has_value()) {
    const auto& tensor = D.value();
    if (tensor.dim() == 1) {
      TORCH_CHECK(tensor.size(0) == heads, "D must have shape (H,).");
      D_compute = to_compute(tensor, compute_dtype, device)
                      .view({heads, 1})
                      .expand({heads, proj})
                      .contiguous();
    } else if (tensor.dim() == 2) {
      TORCH_CHECK(tensor.size(0) == heads && tensor.size(1) == proj,
                  "D must have shape (H, P) when 2-D.");
      D_compute = to_compute(tensor, compute_dtype, device).contiguous();
    } else {
      TORCH_CHECK(false, "Unsupported rank for D.");
    }
  }

  c10::optional<at::Tensor> z_compute = c10::nullopt;
  if (z.has_value()) {
    const auto& tensor = z.value();
    if (tensor.dim() == 3) {
      TORCH_CHECK(tensor.size(0) == batch && tensor.size(1) == seqlen &&
                      tensor.size(2) == heads,
                  "z must have shape (B, L, H) when 3-D.");
      z_compute = to_compute(tensor, compute_dtype, device)
                       .unsqueeze(-1)
                       .contiguous();
    } else if (tensor.dim() == 4) {
      TORCH_CHECK(tensor.size(0) == batch && tensor.size(1) == seqlen &&
                      tensor.size(2) == heads && tensor.size(3) == proj,
                  "z must have shape (B, L, H, P) when 4-D.");
      z_compute = to_compute(tensor, compute_dtype, device).contiguous();
    } else {
      TORCH_CHECK(false, "Unsupported rank for z.");
    }
  }

  at::Tensor initial_state;
  if (initial_states.has_value()) {
    const auto& tensor = initial_states.value();
    TORCH_CHECK(tensor.dim() == 3 && tensor.size(0) == batch &&
                    tensor.size(1) == heads && tensor.size(2) == proj,
                "initial_states must have shape (B, H, P).");
    initial_state = to_compute(tensor, compute_dtype, device).contiguous();
  } else {
    initial_state = at::zeros({batch, heads, proj},
                              X.options().dtype(compute_dtype));
  }

  at::Tensor lengths;
  if (cu_seqlens.has_value()) {
    const auto& tensor = cu_seqlens.value();
    TORCH_CHECK(tensor.dim() == 1 && tensor.numel() == batch + 1,
                "cu_seqlens must have length B + 1.");
    auto cu_long = tensor.to(at::kLong);
    auto slice1 = cu_long.slice(0, 1, cu_long.size(0));
    auto slice0 = cu_long.slice(0, 0, cu_long.size(0) - 1);
    lengths = slice1 - slice0;
  } else if (seq_lens.has_value()) {
    const auto& tensor = seq_lens.value();
    TORCH_CHECK(tensor.dim() == 1 && tensor.numel() == batch,
                "seq_lens must have length B.");
    lengths = tensor.to(at::kLong);
  } else {
    lengths = at::full({batch}, seqlen,
                       at::TensorOptions().dtype(at::kLong).device(device));
  }

  TORCH_CHECK(chunk_size > 0, "chunk_size must be positive");

  auto lengths_host = lengths.to(at::kLong).cpu();
  auto max_len = lengths_host.max().item<int64_t>();
  TORCH_CHECK(max_len <= seqlen,
              "Sequence length exceeds input range for some batch item.");
  const int64_t num_chunks = (max_len + chunk_size - 1) / chunk_size;

  auto lengths_device = lengths_host.to(device, /*non_blocking=*/true)
                            .to(at::kInt)
                            .contiguous();

  auto outputs = at::zeros({batch, seqlen, heads, proj},
                           X.options().dtype(compute_dtype));
  auto states = initial_state.clone().contiguous();

  auto round_up_32 = [](int64_t value) {
    return ((value + 31) / 32) * 32;
  };
  const int threads = static_cast<int>(
      std::min<int64_t>(256, std::max<int64_t>(32, round_up_32(proj))));
  dim3 grid(batch, heads, 1);

  auto stream = at::cuda::getCurrentCUDAStream();

  at::Tensor D_full;
  if (D_compute.has_value()) {
    D_full = D_compute.value().contiguous();
  }

  at::Tensor z_full;
  int64_t z_proj = 0;
  if (z_compute.has_value()) {
    z_full = z_compute.value();
    z_proj = z_full.size(-1);
    TORCH_CHECK(z_proj == 1 || z_proj == proj,
                "z must have last dimension 1 or match proj");
  }

  AT_DISPATCH_FLOATING_TYPES(X_compute.scalar_type(), "ssd_chunk_scan_cuda",
                             [&] {
                               using scalar_t_ = scalar_t;
                               const auto* x_ptr =
                                   X_compute.data_ptr<scalar_t_>();
                               const auto* dt_ptr =
                                   dt_compute.data_ptr<scalar_t_>();
                               const auto* A_ptr =
                                   A_compute.data_ptr<scalar_t_>();
                               Tensor4Accessor<scalar_t_> B_accessor{
                                   B_full.data_ptr<scalar_t_>(),
                                   B_full.stride(0),
                                   B_full.stride(1),
                                   B_full.stride(2),
                                   B_full.stride(3)};
                               Tensor4Accessor<scalar_t_> C_accessor{
                                   C_full.data_ptr<scalar_t_>(),
                                   C_full.stride(0),
                                   C_full.stride(1),
                                   C_full.stride(2),
                                   C_full.stride(3)};
                               const auto* D_ptr = D_full.defined()
                                                       ? D_full
                                                             .data_ptr<scalar_t_>()
                                                       : nullptr;
                               Tensor4Accessor<scalar_t_> Z_accessor{};
                               if (z_full.defined()) {
                                 Z_accessor = Tensor4Accessor<scalar_t_>{
                                     z_full.data_ptr<scalar_t_>(),
                                     z_full.stride(0),
                                     z_full.stride(1),
                                     z_full.stride(2),
                                     z_full.stride(3)};
                               } else {
                                 Z_accessor = Tensor4Accessor<scalar_t_>{
                                     nullptr, 0, 0, 0, 0};
                               }
                               auto* state_ptr =
                                   states.data_ptr<scalar_t_>();
                               auto* out_ptr =
                                   outputs.data_ptr<scalar_t_>();
                               const auto* lengths_ptr =
                                   lengths_device.data_ptr<int32_t>();
                               const size_t smem = proj * sizeof(scalar_t_) *
                                                   (D_ptr != nullptr ? 2 : 1);

                               for (int64_t chunk_idx = 0;
                                    chunk_idx < num_chunks; ++chunk_idx) {
                                 ssd_chunk_scan_kernel<scalar_t_, scalar_t_>
                                     <<<grid, threads, smem, stream>>>(
                                         x_ptr, dt_ptr, A_ptr, B_accessor,
                                         C_accessor, D_ptr, Z_accessor,
                                         state_ptr, out_ptr,
                                         lengths_ptr, batch, seqlen, heads,
                                         proj, chunk_size, chunk_idx, z_proj);
                                 C10_CUDA_KERNEL_LAUNCH_CHECK();
                               }
                             });

  return outputs.to(X.scalar_type());
}

}  // namespace cuda
}  // namespace ssm

