#include "common.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <cmath>

namespace ssm {
namespace cuda {

namespace {

at::Tensor normalize_scan_param_single(const std::string& name,
                                       const at::Tensor& param, int64_t batch,
                                       int64_t dim, int64_t state_dim,
                                       at::ScalarType dtype,
                                       const at::Device& device) {
  if (param.dim() == 2) {
    TORCH_CHECK(param.size(0) == dim && param.size(1) == state_dim,
                name, " must have shape (D, N).");
    return to_compute(param, dtype, device).view({1, dim, state_dim});
  }
  if (param.dim() == 3) {
    TORCH_CHECK(param.size(0) == batch && param.size(1) == dim &&
                    param.size(2) == state_dim,
                name, " must have shape (B, D, N) when 3-D.");
    return to_compute(param, dtype, device);
  }
  TORCH_CHECK(false, "Unsupported rank for ", name, ".");
}

at::Tensor maybe_expand_grouped(const std::string& name,
                                const at::Tensor& param, int64_t dim,
                                int64_t state_dim) {
  if (param.dim() == 3 && param.size(1) != dim) {
    const auto groups = param.size(1);
    TORCH_CHECK(dim % groups == 0, name, " group dimension must divide D.");
    TORCH_CHECK(param.size(2) == state_dim, name,
                " must have matching state dimension.");
    return param.repeat_interleave(dim / groups, 1);
  }
  return param;
}

template <typename scalar_t>
__device__ inline scalar_t sigmoid(scalar_t value) {
  return scalar_t(1) / (scalar_t(1) + exp(-value));
}

template <typename scalar_t>
__device__ inline scalar_t silu(scalar_t value) {
  const auto s = sigmoid(value);
  return value * s;
}

template <typename scalar_t>
__global__ void selective_state_step_kernel(
    scalar_t* __restrict__ state, const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ dt, const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B, const scalar_t* __restrict__ C,
    const scalar_t* __restrict__ D, const scalar_t* __restrict__ z,
    scalar_t* __restrict__ output, int64_t state_batch_stride,
    int64_t state_dim_stride, int64_t state_state_stride,
    int64_t x_batch_stride, int64_t x_dim_stride, int64_t dt_batch_stride,
    int64_t dt_dim_stride, int64_t A_dim_stride, int64_t A_state_stride,
    int64_t B_batch_stride, int64_t B_dim_stride, int64_t B_state_stride,
    int64_t C_batch_stride, int64_t C_dim_stride, int64_t C_state_stride,
    int64_t out_batch_stride, int64_t out_dim_stride, int64_t D_stride,
    int64_t z_batch_stride, int64_t z_dim_stride, int64_t batch,
    int64_t dim, int64_t state_dim, int64_t B_batch, int64_t C_batch,
    bool has_D, bool has_z) {
  const auto b = static_cast<int64_t>(blockIdx.x);
  const auto d = static_cast<int64_t>(blockIdx.y);

  const auto state_base = state + b * state_batch_stride + d * state_dim_stride;
  const auto A_row = A + d * A_dim_stride;
  const auto B_row =
      B + (B_batch == 1 ? 0 : b * B_batch_stride) + d * B_dim_stride;
  const auto C_row =
      C + (C_batch == 1 ? 0 : b * C_batch_stride) + d * C_dim_stride;
  const auto x_val =
      x[b * x_batch_stride + d * x_dim_stride];
  const auto dt_val =
      dt[b * dt_batch_stride + d * dt_dim_stride];

  scalar_t partial = scalar_t(0);

  for (int64_t idx = threadIdx.x; idx < state_dim; idx += blockDim.x) {
    const auto state_offset = idx * state_state_stride;
    const auto coeff_offset = idx * A_state_stride;
    const auto proj_offset = idx * B_state_stride;
    const auto out_offset = idx * C_state_stride;

    const auto prev_state = state_base[state_offset];
    const auto a_val = A_row[coeff_offset];
    const auto b_val = B_row[proj_offset];
    const auto c_val = C_row[out_offset];

    const auto decay = exp(dt_val * a_val);
    const auto drive = b_val * x_val;
    const auto updated = decay * prev_state + dt_val * drive;
    state_base[state_offset] = updated;
    partial += updated * c_val;
  }

  extern __shared__ __align__(sizeof(scalar_t)) unsigned char smem_raw[];
  auto* smem = reinterpret_cast<scalar_t*>(smem_raw);
  smem[threadIdx.x] = partial;
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      smem[threadIdx.x] += smem[threadIdx.x + offset];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    scalar_t y_val = smem[0];
    if (has_D) {
      y_val += D[d * D_stride] * x_val;
    }
    if (has_z) {
      const auto z_val = z[b * z_batch_stride + d * z_dim_stride];
      y_val *= silu(z_val);
    }
    output[b * out_batch_stride + d * out_dim_stride] = y_val;
  }
}

inline int compute_block_size(int64_t state_dim) {
  int threads = 256;
  while (threads > 32 && threads / 2 >= state_dim) {
    threads /= 2;
  }
  return std::max(32, threads);
}

}  // namespace

at::Tensor selective_state_step_cuda(
    at::Tensor state, const at::Tensor& x, const at::Tensor& dt,
    const at::Tensor& A, const at::Tensor& B, const at::Tensor& C,
    const c10::optional<at::Tensor>& D, const c10::optional<at::Tensor>& z,
    const c10::optional<at::Tensor>& dt_bias, bool softplus) {
  TORCH_CHECK(state.is_cuda(), "state must be a CUDA tensor.");
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor.");
  TORCH_CHECK(dt.is_cuda(), "dt must be a CUDA tensor.");
  TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor.");
  TORCH_CHECK(B.device().is_cuda(), "B must be a CUDA tensor.");
  TORCH_CHECK(C.device().is_cuda(), "C must be a CUDA tensor.");
  c10::cuda::CUDAGuard guard(state.device());

  TORCH_CHECK(state.dim() == 3, "state must have shape (B, D, N).");
  TORCH_CHECK(x.dim() == 2 && dt.dim() == 2,
              "x and dt must have shape (B, D).");

  const auto batch = state.size(0);
  const auto dim = state.size(1);
  const auto state_dim = state.size(2);

  TORCH_CHECK(x.size(0) == batch && x.size(1) == dim,
              "x and dt must match the first two dimensions of state.");
  TORCH_CHECK(dt.size(0) == batch && dt.size(1) == dim,
              "x and dt must match the first two dimensions of state.");
  TORCH_CHECK(A.dim() == 2 && A.size(0) == dim && A.size(1) == state_dim,
              "A must have shape (D, N).");

  auto compute_dtype = get_compute_dtype(state);
  if (A.scalar_type() != at::kFloat && A.scalar_type() != compute_dtype) {
    compute_dtype = promote_compute_dtype(compute_dtype, A.scalar_type());
  }

  const auto device = state.device();

  auto state_compute = to_compute(state, compute_dtype, device).contiguous();
  auto x_compute = to_compute(x, compute_dtype, device).contiguous();
  auto dt_compute = to_compute(dt, compute_dtype, device);
  auto A_compute = to_compute(A, compute_dtype, device).contiguous();

  if (dt_bias.has_value()) {
    const auto& tensor = dt_bias.value();
    TORCH_CHECK(tensor.dim() == 1 && tensor.size(0) == dim,
                "dt_bias must have shape (D,).");
    dt_compute =
        dt_compute + to_compute(tensor, compute_dtype, device).view({1, dim});
  }

  if (softplus) {
    dt_compute = at::softplus(dt_compute);
  }

  dt_compute = dt_compute.contiguous();

  auto B_prepared = maybe_expand_grouped("B", B, dim, state_dim);
  auto C_prepared = maybe_expand_grouped("C", C, dim, state_dim);

  auto B_expanded =
      normalize_scan_param_single("B", B_prepared, batch, dim, state_dim,
                                  compute_dtype, device)
          .contiguous();
  auto C_expanded =
      normalize_scan_param_single("C", C_prepared, batch, dim, state_dim,
                                  compute_dtype, device)
          .contiguous();

  c10::optional<at::Tensor> D_compute = c10::nullopt;
  if (D.has_value()) {
    const auto& tensor = D.value();
    TORCH_CHECK(tensor.dim() == 1 && tensor.size(0) == dim,
                "D must have shape (D,).");
    D_compute = to_compute(tensor, compute_dtype, device).contiguous();
  }

  c10::optional<at::Tensor> z_compute = c10::nullopt;
  if (z.has_value()) {
    const auto& tensor = z.value();
    TORCH_CHECK(tensor.dim() == 2 && tensor.sizes() == x.sizes(),
                "z must have shape (B, D).");
    z_compute = to_compute(tensor, compute_dtype, device).contiguous();
  }

  auto output =
      at::empty({batch, dim}, x.options().dtype(compute_dtype)).contiguous();

  const auto stream = at::cuda::getCurrentCUDAStream();
  const auto threads = compute_block_size(state_dim);
  const dim3 grid(batch, dim);

  AT_DISPATCH_FLOATING_TYPES(
      compute_dtype, "selective_state_step_cuda", [&]() {
        const auto shared_size =
            static_cast<size_t>(threads) * sizeof(scalar_t);
        selective_state_step_kernel<scalar_t><<<grid, threads, shared_size,
                                                stream.stream()>>>(
            state_compute.data_ptr<scalar_t>(),
            x_compute.data_ptr<scalar_t>(),
            dt_compute.data_ptr<scalar_t>(),
            A_compute.data_ptr<scalar_t>(),
            B_expanded.data_ptr<scalar_t>(),
            C_expanded.data_ptr<scalar_t>(),
            D_compute.has_value() ? D_compute.value().data_ptr<scalar_t>()
                                  : nullptr,
            z_compute.has_value() ? z_compute.value().data_ptr<scalar_t>()
                                  : nullptr,
            output.data_ptr<scalar_t>(), state_compute.stride(0),
            state_compute.stride(1), state_compute.stride(2),
            x_compute.stride(0), x_compute.stride(1), dt_compute.stride(0),
            dt_compute.stride(1), A_compute.stride(0), A_compute.stride(1),
            B_expanded.stride(0), B_expanded.stride(1),
            B_expanded.stride(2), C_expanded.stride(0),
            C_expanded.stride(1), C_expanded.stride(2), output.stride(0),
            output.stride(1),
            D_compute.has_value() ? D_compute.value().stride(0) : int64_t{0},
            z_compute.has_value() ? z_compute.value().stride(0) : int64_t{0},
            z_compute.has_value() ? z_compute.value().stride(1) : int64_t{0},
            batch, dim, state_dim, B_expanded.size(0), C_expanded.size(0),
            D_compute.has_value(), z_compute.has_value());
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();

  state.copy_(state_compute.to(state.scalar_type()));
  return output.to(x.scalar_type());
}

}  // namespace cuda
}  // namespace ssm

