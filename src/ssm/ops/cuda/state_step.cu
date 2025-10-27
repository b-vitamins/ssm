#include "common.h"

#include <ATen/ATen.h>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <cmath>
#include <tuple>

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
__device__ inline scalar_t silu_grad(scalar_t value) {
  const auto s = sigmoid(value);
  return s * (scalar_t(1) + value * (scalar_t(1) - s));
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

template <typename scalar_t>
__global__ void selective_state_step_backward_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ grad_state_next,
    const scalar_t* __restrict__ state,
    const scalar_t* __restrict__ x, const scalar_t* __restrict__ dt,
    const scalar_t* __restrict__ A, const scalar_t* __restrict__ B,
    const scalar_t* __restrict__ C, const scalar_t* __restrict__ D,
    const scalar_t* __restrict__ z, scalar_t* __restrict__ grad_state_prev,
    scalar_t* __restrict__ grad_x, scalar_t* __restrict__ grad_dt_post,
    scalar_t* __restrict__ grad_A, scalar_t* __restrict__ grad_B,
    scalar_t* __restrict__ grad_C, scalar_t* __restrict__ grad_D,
    scalar_t* __restrict__ grad_z, int64_t state_dim,
    int64_t B_batch, int64_t C_batch, bool has_D, bool has_z,
    bool has_grad_state) {
  const auto b = static_cast<int64_t>(blockIdx.x);
  const auto d = static_cast<int64_t>(blockIdx.y);

  const auto row_offset = b * gridDim.y + d;
  const auto state_offset = row_offset * state_dim;

  const auto x_val = x[row_offset];
  const auto dt_val = dt[row_offset];
  const auto grad_out_val = grad_output[row_offset];

  const auto* state_row = state + state_offset;
  const auto* A_row = A + d * state_dim;

  const auto batch_stride = gridDim.y * state_dim;
  const auto* B_row =
      B + (B_batch == 1 ? 0 : b * batch_stride) + d * state_dim;
  const auto* C_row =
      C + (C_batch == 1 ? 0 : b * batch_stride) + d * state_dim;

  const auto* grad_state_row =
      has_grad_state ? grad_state_next + state_offset : nullptr;
  auto* grad_state_prev_row = grad_state_prev + state_offset;

  auto* grad_B_row =
      grad_B + (B_batch == 1 ? 0 : b * batch_stride) + d * state_dim;
  auto* grad_C_row =
      grad_C + (C_batch == 1 ? 0 : b * batch_stride) + d * state_dim;

  scalar_t partial_pre = scalar_t(0);
  for (int64_t n = threadIdx.x; n < state_dim; n += blockDim.x) {
    const auto decay = exp(dt_val * A_row[n]);
    const auto drive = B_row[n] * x_val;
    const auto updated = decay * state_row[n] + dt_val * drive;
    partial_pre += updated * C_row[n];
  }

  extern __shared__ __align__(sizeof(scalar_t)) unsigned char smem_raw[];
  auto* smem = reinterpret_cast<scalar_t*>(smem_raw);
  smem[threadIdx.x] = partial_pre;
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      smem[threadIdx.x] += smem[threadIdx.x + offset];
    }
    __syncthreads();
  }

  scalar_t pre_gate = smem[0];
  scalar_t grad_pre_gate = grad_out_val;
  scalar_t grad_x_extra = scalar_t(0);

  if (has_D) {
    const auto d_val = D[d];
    pre_gate += d_val * x_val;
  }

  if (has_z) {
    const auto z_val = z[row_offset];
    const auto gate = silu(z_val);
    grad_pre_gate *= gate;
    if (threadIdx.x == 0 && grad_z != nullptr) {
      grad_z[row_offset] =
          grad_out_val * pre_gate * silu_grad(z_val);
    }
  } else if (threadIdx.x == 0 && grad_z != nullptr) {
    grad_z[row_offset] = scalar_t(0);
  }

  if (has_D) {
    const auto d_val = D[d];
    grad_x_extra += grad_pre_gate * d_val;
    if (threadIdx.x == 0 && grad_D != nullptr) {
      at::cuda::atomic::atomicAdd(grad_D + d, grad_pre_gate * x_val);
    }
  }

  auto* grad_A_row = grad_A + d * state_dim;

  scalar_t grad_dt_val_local = scalar_t(0);
  scalar_t grad_dt_x_local = scalar_t(0);

  for (int64_t n = threadIdx.x; n < state_dim; n += blockDim.x) {
    const auto A_val = A_row[n];
    const auto B_val = B_row[n];
    const auto C_val = C_row[n];
    const auto decay = exp(dt_val * A_val);
    const auto drive = B_val * x_val;
    const auto updated = decay * state_row[n] + dt_val * drive;
    const auto grad_next =
        has_grad_state ? grad_state_row[n] : scalar_t(0);
    const auto total_grad = grad_next + grad_pre_gate * C_val;

    grad_state_prev_row[n] = total_grad * decay;

    const auto grad_decay = total_grad * state_row[n];
    grad_dt_val_local += grad_decay * (A_val * decay);
    grad_dt_x_local += total_grad * B_val;

    at::cuda::atomic::atomicAdd(grad_A_row + n, grad_decay * (dt_val * decay));

    const auto grad_updated = grad_pre_gate * updated;
    if (C_batch == 1) {
      at::cuda::atomic::atomicAdd(grad_C_row + n, grad_updated);
    } else {
      grad_C_row[n] += grad_updated;
    }

    const auto grad_B_val = total_grad * (dt_val * x_val);
    if (B_batch == 1) {
      at::cuda::atomic::atomicAdd(grad_B_row + n, grad_B_val);
    } else {
      grad_B_row[n] += grad_B_val;
    }
  }

  smem[threadIdx.x] = grad_dt_val_local;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      smem[threadIdx.x] += smem[threadIdx.x + offset];
    }
    __syncthreads();
  }
  scalar_t grad_dt_val_total = smem[0];

  smem[threadIdx.x] = grad_dt_x_local;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      smem[threadIdx.x] += smem[threadIdx.x + offset];
    }
    __syncthreads();
  }
  scalar_t grad_dt_x_total = smem[0];

  if (threadIdx.x == 0) {
    grad_dt_val_total += grad_dt_x_total * x_val;
    const auto grad_x_total = grad_x_extra + grad_dt_x_total * dt_val;
    grad_dt_post[row_offset] = grad_dt_val_total;
    grad_x[row_offset] = grad_x_total;
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

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
           c10::optional<at::Tensor>, c10::optional<at::Tensor>,
           c10::optional<at::Tensor>>
selective_state_step_backward_cuda(
    const at::Tensor& grad_output,
    const c10::optional<at::Tensor>& grad_state, const at::Tensor& state,
    const at::Tensor& x, const at::Tensor& dt, const at::Tensor& A,
    const at::Tensor& B, const at::Tensor& C,
    const c10::optional<at::Tensor>& D, const c10::optional<at::Tensor>& z,
    const c10::optional<at::Tensor>& dt_bias, bool softplus) {
  TORCH_CHECK(state.dim() == 3, "state must have shape (B, D, N).");
  TORCH_CHECK(x.dim() == 2 && dt.dim() == 2,
              "x and dt must have shape (B, D).");
  TORCH_CHECK(grad_output.dim() == 2,
              "grad_output must have shape (B, D).");

  const auto batch = state.size(0);
  const auto dim = state.size(1);
  const auto state_dim = state.size(2);

  TORCH_CHECK(x.size(0) == batch && x.size(1) == dim,
              "x must match the first two dimensions of state.");
  TORCH_CHECK(dt.size(0) == batch && dt.size(1) == dim,
              "dt must match the first two dimensions of state.");
  TORCH_CHECK(grad_output.size(0) == batch && grad_output.size(1) == dim,
              "grad_output must match the output shape.");

  TORCH_CHECK(A.dim() == 2 && A.size(0) == dim && A.size(1) == state_dim,
              "A must have shape (D, N).");

  if (grad_state.has_value()) {
    const auto& tensor = grad_state.value();
    TORCH_CHECK(tensor.dim() == 3 && tensor.size(0) == batch &&
                    tensor.size(1) == dim && tensor.size(2) == state_dim,
                "grad_state must have shape (B, D, N).");
  }

  TORCH_CHECK(state.is_cuda(), "state must be CUDA.");
  TORCH_CHECK(x.is_cuda(), "x must be CUDA.");
  TORCH_CHECK(dt.is_cuda(), "dt must be CUDA.");
  TORCH_CHECK(A.is_cuda(), "A must be CUDA.");
  TORCH_CHECK(B.is_cuda(), "B must be CUDA.");
  TORCH_CHECK(C.is_cuda(), "C must be CUDA.");
  TORCH_CHECK(grad_output.is_cuda(), "grad_output must be CUDA.");
  if (grad_state.has_value()) {
    TORCH_CHECK(grad_state.value().is_cuda(),
                "grad_state must be CUDA when provided.");
  }

  c10::cuda::CUDAGuard guard(state.device());

  auto compute_dtype = get_compute_dtype(state);
  if (A.scalar_type() != at::kFloat && A.scalar_type() != compute_dtype) {
    compute_dtype = promote_compute_dtype(compute_dtype, A.scalar_type());
  }

  const auto device = state.device();

  auto state_compute = to_compute(state, compute_dtype, device).contiguous();
  auto x_compute = to_compute(x, compute_dtype, device).contiguous();
  auto dt_base = to_compute(dt, compute_dtype, device);
  auto grad_output_compute =
      to_compute(grad_output, compute_dtype, device).contiguous();
  auto A_compute = to_compute(A, compute_dtype, device).contiguous();

  at::Tensor grad_state_next_compute;
  bool has_grad_state = false;
  if (grad_state.has_value()) {
    grad_state_next_compute =
        to_compute(grad_state.value(), compute_dtype, device).contiguous();
    has_grad_state = true;
  }

  if (dt_bias.has_value()) {
    const auto& tensor = dt_bias.value();
    TORCH_CHECK(tensor.dim() == 1 && tensor.size(0) == dim,
                "dt_bias must have shape (D,).");
    dt_base =
        dt_base + to_compute(tensor, compute_dtype, device).view({1, dim});
  }

  at::Tensor dt_derivative;
  at::Tensor dt_final;
  if (softplus) {
    dt_derivative = at::sigmoid(dt_base);
    dt_final = at::softplus(dt_base);
  } else {
    dt_derivative = at::ones_like(dt_base);
    dt_final = dt_base;
  }

  dt_derivative = dt_derivative.contiguous();
  dt_final = dt_final.contiguous();

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
    TORCH_CHECK(tensor.dim() == 2 && tensor.size(0) == batch &&
                    tensor.size(1) == dim,
                "z must have shape (B, D).");
    z_compute = to_compute(tensor, compute_dtype, device).contiguous();
  }

  auto grad_state_prev = at::empty_like(state_compute);
  auto grad_x = at::zeros_like(x_compute);
  auto grad_dt_post = at::zeros_like(dt_final);
  auto grad_A = at::zeros_like(A_compute);
  auto grad_B = at::zeros_like(B_expanded);
  auto grad_C = at::zeros_like(C_expanded);

  c10::optional<at::Tensor> grad_D_storage = c10::nullopt;
  if (D_compute.has_value()) {
    grad_D_storage = at::zeros_like(D_compute.value());
  }

  c10::optional<at::Tensor> grad_z_storage = c10::nullopt;
  if (z_compute.has_value()) {
    grad_z_storage = at::empty_like(z_compute.value());
  }

  const auto stream = at::cuda::getCurrentCUDAStream();
  const auto threads = compute_block_size(state_dim);
  const dim3 grid(batch, dim);

  AT_DISPATCH_FLOATING_TYPES(
      compute_dtype, "selective_state_step_backward_cuda", [&]() {
        const auto shared = static_cast<size_t>(threads) * sizeof(scalar_t);
        selective_state_step_backward_kernel<scalar_t><<<grid, threads, shared,
                                                        stream.stream()>>>(
            grad_output_compute.data_ptr<scalar_t>(),
            has_grad_state ? grad_state_next_compute.data_ptr<scalar_t>()
                           : nullptr,
            state_compute.data_ptr<scalar_t>(),
            x_compute.data_ptr<scalar_t>(),
            dt_final.data_ptr<scalar_t>(),
            A_compute.data_ptr<scalar_t>(),
            B_expanded.data_ptr<scalar_t>(),
            C_expanded.data_ptr<scalar_t>(),
            D_compute.has_value()
                ? D_compute.value().data_ptr<scalar_t>()
                : nullptr,
            z_compute.has_value()
                ? z_compute.value().data_ptr<scalar_t>()
                : nullptr,
            grad_state_prev.data_ptr<scalar_t>(),
            grad_x.data_ptr<scalar_t>(),
            grad_dt_post.data_ptr<scalar_t>(),
            grad_A.data_ptr<scalar_t>(),
            grad_B.data_ptr<scalar_t>(),
            grad_C.data_ptr<scalar_t>(),
            grad_D_storage.has_value()
                ? grad_D_storage.value().data_ptr<scalar_t>()
                : nullptr,
            grad_z_storage.has_value()
                ? grad_z_storage.value().data_ptr<scalar_t>()
                : nullptr,
            state_dim, B_expanded.size(0), C_expanded.size(0),
            D_compute.has_value(), z_compute.has_value(), has_grad_state);
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();

  auto grad_dt_input = grad_dt_post * dt_derivative;

  c10::optional<at::Tensor> grad_dt_bias_compute = c10::nullopt;
  if (dt_bias.has_value()) {
    grad_dt_bias_compute = grad_dt_input.sum(0);
  }

  auto collapse_grouped_grad = [&](const at::Tensor& grad,
                                   const at::Tensor& original) {
    if (original.dim() == 3 && original.size(1) != dim) {
      const auto groups = original.size(1);
      TORCH_INTERNAL_ASSERT(dim % groups == 0);
      const auto per_group = dim / groups;
      auto reshaped = grad.view({grad.size(0), dim, state_dim})
                          .view({grad.size(0), groups, per_group, state_dim})
                          .sum(2);
      return reshaped.contiguous();
    }
    return grad;
  };

  auto grad_B_collapsed = collapse_grouped_grad(grad_B, B);
  auto grad_C_collapsed = collapse_grouped_grad(grad_C, C);

  if (grad_B_collapsed.size(0) == 1 && B.dim() == 3 && B.size(0) > 1) {
    grad_B_collapsed = grad_B_collapsed.expand({B.size(0), grad_B_collapsed.size(1),
                                                grad_B_collapsed.size(2)})
                          .clone();
  }
  if (grad_C_collapsed.size(0) == 1 && C.dim() == 3 && C.size(0) > 1) {
    grad_C_collapsed = grad_C_collapsed.expand({C.size(0), grad_C_collapsed.size(1),
                                                grad_C_collapsed.size(2)})
                          .clone();
  }

  at::Tensor grad_B_result;
  if (B.dim() == 2) {
    grad_B_result = grad_B_collapsed.squeeze(0).contiguous();
  } else {
    grad_B_result = grad_B_collapsed.view(B.sizes()).contiguous();
  }

  at::Tensor grad_C_result;
  if (C.dim() == 2) {
    grad_C_result = grad_C_collapsed.squeeze(0).contiguous();
  } else {
    grad_C_result = grad_C_collapsed.view(C.sizes()).contiguous();
  }

  auto grad_state_result = grad_state_prev.to(state.scalar_type());
  auto grad_x_result = grad_x.to(x.scalar_type());
  auto grad_dt_result = grad_dt_input.to(dt.scalar_type());
  auto grad_A_result = grad_A.to(A.scalar_type());
  grad_B_result = grad_B_result.to(B.scalar_type());
  grad_C_result = grad_C_result.to(C.scalar_type());

  c10::optional<at::Tensor> grad_D_result = c10::nullopt;
  if (D.has_value()) {
    grad_D_result = grad_D_storage.value().to(D.value().scalar_type());
  }

  c10::optional<at::Tensor> grad_z_result = c10::nullopt;
  if (z.has_value()) {
    grad_z_result = grad_z_storage.value().to(z.value().scalar_type());
  }

  c10::optional<at::Tensor> grad_dt_bias_result = c10::nullopt;
  if (dt_bias.has_value()) {
    grad_dt_bias_result = grad_dt_bias_compute.value().to(
        dt_bias.value().scalar_type());
  }

  return {grad_state_result, grad_x_result, grad_dt_result, grad_A_result,
          grad_B_result, grad_C_result, grad_D_result, grad_z_result,
          grad_dt_bias_result};
}

}  // namespace cuda
}  // namespace ssm

