#include "common.h"

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/vec.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <tuple>
#include <vector>

namespace ssm {
namespace cpu {

namespace {

template <typename scalar_t>
inline scalar_t conj_if_complex(const scalar_t& value) {
  if constexpr (c10::is_complex<scalar_t>::value) {
    return std::conj(value);
  }
  return value;
}

template <typename scalar_t>
inline scalar_t silu_grad(const scalar_t& x) {
  const scalar_t one = scalar_t(1);
  const auto neg_x = -x;
  const scalar_t sigmoid = one / (one + std::exp(neg_x));
  return sigmoid * (one + x * (one - sigmoid));
}

}  // namespace

at::Tensor ssd_chunk_scan_cpu(
    const at::Tensor& X, const at::Tensor& dt, const at::Tensor& A,
    const at::Tensor& B, const at::Tensor& C, int64_t chunk_size,
    const c10::optional<at::Tensor>& D, const c10::optional<at::Tensor>& z,
    const c10::optional<at::Tensor>& seq_lens,
    const c10::optional<at::Tensor>& cu_seqlens,
    const c10::optional<at::Tensor>& initial_states) {
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

  TORCH_CHECK(chunk_size > 0, "chunk_size must be positive.");

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

  auto B_info = make_chunk_param("B", B, batch, seqlen, heads, proj,
                                 compute_dtype, device);
  auto C_info = make_chunk_param("C", C, batch, seqlen, heads, proj,
                                 compute_dtype, device);

  enum class SkipKind { kNone, kVector, kMatrix };
  SkipKind skip_kind = SkipKind::kNone;
  at::Tensor D_tensor;
  if (D.has_value()) {
    const auto& tensor = D.value();
    if (tensor.dim() == 1) {
      TORCH_CHECK(tensor.size(0) == heads, "D must have shape (H,).");
      D_tensor = to_compute(tensor, compute_dtype, device).contiguous();
      skip_kind = SkipKind::kVector;
    } else if (tensor.dim() == 2) {
      TORCH_CHECK(tensor.size(0) == heads && tensor.size(1) == proj,
                  "D must have shape (H, P) when 2-D.");
      D_tensor = to_compute(tensor, compute_dtype, device).contiguous();
      skip_kind = SkipKind::kMatrix;
    } else {
      TORCH_CHECK(false, "Unsupported rank for D.");
    }
  }

  enum class GateKind { kNone, kScalar, kVector };
  GateKind gate_kind = GateKind::kNone;
  at::Tensor z_tensor;
  if (z.has_value()) {
    const auto& tensor = z.value();
    if (tensor.dim() == 3) {
      TORCH_CHECK(tensor.size(0) == batch && tensor.size(1) == seqlen &&
                      tensor.size(2) == heads,
                  "z must have shape (B, L, H) when 3-D.");
      z_tensor =
          at::silu(to_compute(tensor, compute_dtype, device)).contiguous();
      gate_kind = GateKind::kScalar;
    } else if (tensor.dim() == 4) {
      TORCH_CHECK(tensor.size(0) == batch && tensor.size(1) == seqlen &&
                      tensor.size(2) == heads && tensor.size(3) == proj,
                  "z must have shape (B, L, H, P) when 4-D.");
      z_tensor =
          at::silu(to_compute(tensor, compute_dtype, device)).contiguous();
      gate_kind = GateKind::kVector;
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
  auto state_buffer = initial_state.clone();

  at::Tensor lengths;
  if (cu_seqlens.has_value()) {
    const auto& tensor = cu_seqlens.value();
    TORCH_CHECK(tensor.dim() == 1 && tensor.numel() == batch + 1,
                "cu_seqlens must have length B + 1.");
    auto cu_long = tensor.to(at::kLong).contiguous();
    lengths = cu_long.slice(0, 1, cu_long.size(0)) -
              cu_long.slice(0, 0, cu_long.size(0) - 1);
  } else if (seq_lens.has_value()) {
    const auto& tensor = seq_lens.value();
    TORCH_CHECK(tensor.dim() == 1 && tensor.numel() == batch,
                "seq_lens must have length B.");
    lengths = tensor.to(at::kLong).contiguous();
  } else {
    lengths = at::full({batch}, seqlen,
                       at::TensorOptions().dtype(at::kLong).device(device));
  }

  auto outputs = at::zeros({batch, seqlen, heads, proj},
                           X.options().dtype(compute_dtype));

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      compute_dtype, "ssd_chunk_scan_cpu", [&]() {
        const auto* X_ptr = X_compute.data_ptr<scalar_t>();
        const auto* dt_ptr = dt_compute.data_ptr<scalar_t>();
        const auto* A_ptr = A_compute.data_ptr<scalar_t>();
        const auto* D_ptr = skip_kind != SkipKind::kNone
                                ? D_tensor.data_ptr<scalar_t>()
                                : nullptr;
        const auto* z_ptr = gate_kind != GateKind::kNone
                                ? z_tensor.data_ptr<scalar_t>()
                                : nullptr;
        auto* state_ptr = state_buffer.data_ptr<scalar_t>();
        auto* out_ptr = outputs.data_ptr<scalar_t>();
        const auto* lengths_ptr = lengths.data_ptr<int64_t>();

        const auto proj_stride = proj;
        const auto head_stride = proj;
        const auto token_stride = heads * proj;
        const auto batch_stride = seqlen * token_stride;
        const auto dt_batch_stride = seqlen * heads;
        const auto dt_time_stride = heads;
        const auto dt_head_stride = 1;

        const auto B_time_stride =
            B_info.kind == ChunkParamKind::kTimeVarying ? B_info.stride_length : 0;
        const auto C_time_stride =
            C_info.kind == ChunkParamKind::kTimeVarying ? C_info.stride_length : 0;
        const bool B_has_time =
            B_info.kind == ChunkParamKind::kTimeVarying && B_info.length_size != 1;
        const bool C_has_time =
            C_info.kind == ChunkParamKind::kTimeVarying && C_info.length_size != 1;

        const bool skip_has_matrix = skip_kind == SkipKind::kMatrix;
        const bool skip_has_vector = skip_kind == SkipKind::kVector;

        const bool gate_has_scalar = gate_kind == GateKind::kScalar;
        const bool gate_has_vector = gate_kind == GateKind::kVector;

        at::parallel_for(0, batch * heads, 0, [&](int64_t start, int64_t end) {
          for (const auto idx : c10::irange(start, end)) {
            const auto b = idx / heads;
            const auto h = idx % heads;
            const auto valid = std::min<int64_t>(lengths_ptr[b], seqlen);
            if (valid <= 0) {
              continue;
            }

            auto* state_head = state_ptr + idx * head_stride;
            const auto* A_row = A_ptr + h * proj_stride;

            const auto token_base = b * batch_stride + h * proj_stride;
            const auto dt_base = b * dt_batch_stride + h * dt_head_stride;

            const auto* D_row = skip_has_matrix ? D_ptr + h * proj_stride : nullptr;
            const auto D_scalar = skip_has_vector ? D_ptr[h] : scalar_t(0);

            const auto* z_scalar_base =
                gate_has_scalar ? z_ptr + dt_base : nullptr;
            const auto* z_vector_base =
                gate_has_vector ? z_ptr + token_base : nullptr;

            for (int64_t start_t = 0; start_t < valid; start_t += chunk_size) {
              const auto chunk_len =
                  std::min<int64_t>(chunk_size, valid - start_t);

              const auto* dt_chunk = dt_ptr + dt_base + start_t * dt_time_stride;
              const auto* gate_scalar_chunk =
                  z_scalar_base != nullptr
                      ? z_scalar_base + start_t * dt_time_stride
                      : nullptr;
              const auto* gate_vector_chunk =
                  z_vector_base != nullptr
                      ? z_vector_base + start_t * token_stride
                      : nullptr;

              const auto token_offset = token_base + start_t * token_stride;
              const auto* X_chunk = X_ptr + token_offset;
              auto* out_chunk = out_ptr + token_offset;

              const auto B_slice =
                  chunk_param_slice<scalar_t>(B_info, b, start_t, h);
              const auto C_slice =
                  chunk_param_slice<scalar_t>(C_info, b, start_t, h);
              const auto* B_chunk = B_slice.ptr;
              const auto* C_chunk = C_slice.ptr;
              const auto B_proj_stride = B_slice.stride;
              const auto C_proj_stride = C_slice.stride;

              for (const auto step : c10::irange<int64_t>(chunk_len)) {
                const auto dt_val = dt_chunk[step * dt_time_stride];
                const auto gate_scalar =
                    gate_scalar_chunk != nullptr
                        ? gate_scalar_chunk[step * dt_time_stride]
                        : scalar_t(1);

                const auto* X_token = X_chunk + step * token_stride;
                auto* out_token = out_chunk + step * token_stride;

                const auto* gate_vec =
                    gate_vector_chunk != nullptr
                        ? gate_vector_chunk + step * token_stride
                        : nullptr;

                const auto* B_head =
                    B_chunk + (B_has_time ? step * B_time_stride : 0);
                const auto* C_head =
                    C_chunk + (C_has_time ? step * C_time_stride : 0);

                auto scalar_update = [&](int64_t p) {
                  const auto decay = std::exp(dt_val * A_row[p]);
                  const auto drive = B_head[p * B_proj_stride] * X_token[p];
                  const auto updated = decay * state_head[p] + dt_val * drive;
                  state_head[p] = updated;

                  scalar_t out_val = updated * C_head[p * C_proj_stride];
                  if (skip_has_vector) {
                    out_val += D_scalar * X_token[p];
                  } else if (skip_has_matrix) {
                    out_val += D_row[p] * X_token[p];
                  }
                  if (gate_vec != nullptr) {
                    out_val *= gate_vec[p];
                  }
                  out_token[p] = out_val * gate_scalar;
                };

                if constexpr (!c10::is_complex<scalar_t>::value &&
                              at::vec::is_vec_specialized_for<scalar_t>::value) {
                  if (B_proj_stride == 1 && C_proj_stride == 1) {
                    using Vec = at::vec::Vectorized<scalar_t>;
                    const auto vec_size = Vec::size();
                    const Vec dt_vec(dt_val);
                    const Vec gate_scalar_vec(gate_scalar);
                    const Vec D_scalar_vec(D_scalar);
                    int64_t p = 0;
                    for (; p + vec_size <= proj; p += vec_size) {
                      auto a_vec = Vec::loadu(A_row + p);
                      auto prev_state = Vec::loadu(state_head + p);
                      auto b_vec = Vec::loadu(B_head + p);
                      auto x_vec = Vec::loadu(X_token + p);
                      auto c_vec = Vec::loadu(C_head + p);
                      auto decay_vec = (a_vec * dt_vec).exp();
                      auto updated = decay_vec * prev_state +
                                     (b_vec * x_vec) * dt_vec;
                      updated.store(state_head + p);
                      auto out_vec = updated * c_vec;
                      if (skip_has_vector) {
                        out_vec = out_vec + D_scalar_vec * x_vec;
                      } else if (skip_has_matrix) {
                        auto d_vec = Vec::loadu(D_row + p);
                        out_vec = out_vec + d_vec * x_vec;
                      }
                      if (gate_vec != nullptr) {
                        auto gate_vec_v = Vec::loadu(gate_vec + p);
                        out_vec = out_vec * gate_vec_v;
                      }
                      out_vec = out_vec * gate_scalar_vec;
                      out_vec.store(out_token + p);
                    }
                    for (; p < proj; ++p) {
                      scalar_update(p);
                    }
                  } else {
                    for (const auto p : c10::irange(proj)) {
                      scalar_update(p);
                    }
                  }
                } else {
                  for (const auto p : c10::irange(proj)) {
                    scalar_update(p);
                  }
                }
              }
            }
          }
        });
      });

  return outputs.to(X.scalar_type());
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
           c10::optional<at::Tensor>, c10::optional<at::Tensor>,
           c10::optional<at::Tensor>>
ssd_chunk_scan_backward_cpu(
    const at::Tensor& grad_output, const at::Tensor& X, const at::Tensor& dt,
    const at::Tensor& A, const at::Tensor& B, const at::Tensor& C,
    int64_t /*chunk_size*/, const c10::optional<at::Tensor>& D,
    const c10::optional<at::Tensor>& z,
    const c10::optional<at::Tensor>& seq_lens,
    const c10::optional<at::Tensor>& cu_seqlens,
    const c10::optional<at::Tensor>& initial_states) {
  TORCH_CHECK(X.dim() == 4, "X must have shape (B, L, H, P).");
  TORCH_CHECK(dt.dim() == 3, "dt must have shape (B, L, H).");
  TORCH_CHECK(grad_output.dim() == 4,
              "grad_output must have shape (B, L, H, P).");
  TORCH_CHECK(grad_output.sizes() == X.sizes(),
              "grad_output must match the forward output shape.");

  const auto batch = X.size(0);
  const auto seqlen = X.size(1);
  const auto heads = X.size(2);
  const auto proj = X.size(3);

  TORCH_CHECK(dt.size(0) == batch && dt.size(1) == seqlen &&
                  dt.size(2) == heads,
              "dt must align with (B, L, H).");

  TORCH_CHECK(A.dim() == 1 || A.dim() == 2,
              "A must be 1-D or 2-D matching head/state dims.");

  auto compute_dtype = get_compute_dtype(X);
  if (A.scalar_type() != at::kFloat && A.scalar_type() != compute_dtype) {
    compute_dtype = promote_compute_dtype(compute_dtype, A.scalar_type());
  }

  const auto device = X.device();

  auto X_compute = to_compute(X, compute_dtype, device).contiguous();
  auto dt_compute = to_compute(dt, compute_dtype, device).contiguous();
  auto grad_output_compute =
      to_compute(grad_output, compute_dtype, device).contiguous();

  at::Tensor A_compute;
  bool A_was_vector = false;
  if (A.dim() == 1) {
    TORCH_CHECK(A.size(0) == heads, "A must have shape (H,) when 1-D.");
    A_compute = to_compute(A, compute_dtype, device)
                    .unsqueeze(-1)
                    .expand({heads, proj})
                    .contiguous();
    A_was_vector = true;
  } else {
    TORCH_CHECK(A.size(0) == heads && A.size(1) == proj,
                "A must have shape (H, P) when 2-D.");
    A_compute = to_compute(A, compute_dtype, device).contiguous();
  }

  auto B_info = make_chunk_param("B", B, batch, seqlen, heads, proj,
                                 compute_dtype, device);
  auto C_info = make_chunk_param("C", C, batch, seqlen, heads, proj,
                                 compute_dtype, device);

  enum class SkipKind { kNone, kVector, kMatrix };
  SkipKind skip_kind = SkipKind::kNone;
  at::Tensor D_tensor;
  if (D.has_value()) {
    const auto& tensor = D.value();
    if (tensor.dim() == 1) {
      TORCH_CHECK(tensor.size(0) == heads, "D must have shape (H,).");
      D_tensor = to_compute(tensor, compute_dtype, device).contiguous();
      skip_kind = SkipKind::kVector;
    } else if (tensor.dim() == 2) {
      TORCH_CHECK(tensor.size(0) == heads && tensor.size(1) == proj,
                  "D must have shape (H, P) when 2-D.");
      D_tensor = to_compute(tensor, compute_dtype, device).contiguous();
      skip_kind = SkipKind::kMatrix;
    } else {
      TORCH_CHECK(false, "Unsupported rank for D.");
    }
  }

  enum class GateKind { kNone, kScalar, kVector };
  GateKind gate_kind = GateKind::kNone;
  at::Tensor z_input_tensor;
  at::Tensor z_gate_tensor;
  if (z.has_value()) {
    const auto& tensor = z.value();
    if (tensor.dim() == 3) {
      TORCH_CHECK(tensor.size(0) == batch && tensor.size(1) == seqlen &&
                      tensor.size(2) == heads,
                  "z must have shape (B, L, H) when 3-D.");
      z_input_tensor = to_compute(tensor, compute_dtype, device).contiguous();
      z_gate_tensor = at::silu(z_input_tensor).contiguous();
      gate_kind = GateKind::kScalar;
    } else if (tensor.dim() == 4) {
      TORCH_CHECK(tensor.size(0) == batch && tensor.size(1) == seqlen &&
                      tensor.size(2) == heads && tensor.size(3) == proj,
                  "z must have shape (B, L, H, P) when 4-D.");
      z_input_tensor = to_compute(tensor, compute_dtype, device).contiguous();
      z_gate_tensor = at::silu(z_input_tensor).contiguous();
      gate_kind = GateKind::kVector;
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
    auto cu_long = tensor.to(at::kLong).contiguous();
    lengths = cu_long.slice(0, 1, cu_long.size(0)) -
              cu_long.slice(0, 0, cu_long.size(0) - 1);
  } else if (seq_lens.has_value()) {
    const auto& tensor = seq_lens.value();
    TORCH_CHECK(tensor.dim() == 1 && tensor.numel() == batch,
                "seq_lens must have length B.");
    lengths = tensor.to(at::kLong).contiguous();
  } else {
    lengths = at::full({batch}, seqlen,
                       at::TensorOptions().dtype(at::kLong).device(device));
  }

  auto grad_X_compute = at::zeros_like(X_compute);
  auto grad_dt_compute = at::zeros_like(dt_compute);
  auto grad_A_compute = at::zeros_like(A_compute);
  auto B_grad_info = B_info;
  B_grad_info.tensor = at::zeros_like(B_info.tensor);
  auto C_grad_info = C_info;
  C_grad_info.tensor = at::zeros_like(C_info.tensor);

  at::Tensor grad_D_tensor;
  if (skip_kind != SkipKind::kNone) {
    grad_D_tensor = at::zeros_like(D_tensor);
  }

  at::Tensor grad_z_tensor;
  if (gate_kind != GateKind::kNone) {
    grad_z_tensor = at::zeros_like(z_input_tensor);
  }

  auto grad_initial_state = at::zeros_like(initial_state);

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      compute_dtype, "ssd_chunk_scan_backward_cpu", [&]() {
        const auto* X_ptr = X_compute.data_ptr<scalar_t>();
        const auto* dt_ptr = dt_compute.data_ptr<scalar_t>();
        const auto* grad_out_ptr =
            grad_output_compute.data_ptr<scalar_t>();
        const auto* A_ptr = A_compute.data_ptr<scalar_t>();
        auto* grad_A_ptr = grad_A_compute.data_ptr<scalar_t>();
        const auto* B_ptr = B_info.tensor.data_ptr<scalar_t>();
        const auto* C_ptr = C_info.tensor.data_ptr<scalar_t>();
        auto* B_grad_ptr = B_grad_info.tensor.data_ptr<scalar_t>();
        auto* C_grad_ptr = C_grad_info.tensor.data_ptr<scalar_t>();
        const auto* D_ptr = skip_kind != SkipKind::kNone
                                ? D_tensor.data_ptr<scalar_t>()
                                : nullptr;
        auto* grad_D_ptr = skip_kind != SkipKind::kNone
                               ? grad_D_tensor.data_ptr<scalar_t>()
                               : nullptr;
        const auto* z_input_ptr = gate_kind != GateKind::kNone
                                      ? z_input_tensor.data_ptr<scalar_t>()
                                      : nullptr;
        const auto* z_gate_ptr = gate_kind != GateKind::kNone
                                     ? z_gate_tensor.data_ptr<scalar_t>()
                                     : nullptr;
        auto* grad_z_ptr = gate_kind != GateKind::kNone
                               ? grad_z_tensor.data_ptr<scalar_t>()
                               : nullptr;
        const auto* lengths_ptr = lengths.data_ptr<int64_t>();
        const auto* init_ptr = initial_state.data_ptr<scalar_t>();
        auto* grad_init_ptr = grad_initial_state.data_ptr<scalar_t>();
        auto* grad_X_ptr = grad_X_compute.data_ptr<scalar_t>();
        auto* grad_dt_ptr = grad_dt_compute.data_ptr<scalar_t>();

        const auto proj_stride = proj;
        const auto head_stride = proj;
        const auto token_stride = heads * proj;
        const auto batch_stride = seqlen * token_stride;
        const auto dt_batch_stride = seqlen * heads;
        const auto dt_time_stride = heads;
        const auto dt_head_stride = 1;

        const auto B_time_stride =
            B_info.kind == ChunkParamKind::kTimeVarying ? B_info.stride_length
                                                        : 0;
        const auto C_time_stride =
            C_info.kind == ChunkParamKind::kTimeVarying ? C_info.stride_length
                                                        : 0;
        const bool B_has_time =
            B_info.kind == ChunkParamKind::kTimeVarying &&
            B_info.length_size != 1;
        const bool C_has_time =
            C_info.kind == ChunkParamKind::kTimeVarying &&
            C_info.length_size != 1;

        const bool skip_has_matrix = skip_kind == SkipKind::kMatrix;
        const bool skip_has_vector = skip_kind == SkipKind::kVector;

        int64_t z_scalar_batch_stride = 0;
        int64_t z_scalar_time_stride = 0;
        int64_t z_scalar_head_stride = 0;
        int64_t z_vector_batch_stride = 0;
        int64_t z_vector_time_stride = 0;
        int64_t z_vector_head_stride = 0;
        int64_t z_vector_proj_stride = 0;

        if (gate_kind == GateKind::kScalar) {
          z_scalar_batch_stride = z_input_tensor.stride(0);
          z_scalar_time_stride = z_input_tensor.stride(1);
          z_scalar_head_stride = z_input_tensor.stride(2);
        } else if (gate_kind == GateKind::kVector) {
          z_vector_batch_stride = z_input_tensor.stride(0);
          z_vector_time_stride = z_input_tensor.stride(1);
          z_vector_head_stride = z_input_tensor.stride(2);
          z_vector_proj_stride = z_input_tensor.stride(3);
        }

        at::parallel_for(0, heads, 0, [&](int64_t start, int64_t end) {
          for (const auto h : c10::irange(start, end)) {
            for (const auto b : c10::irange<int64_t>(batch)) {
              const auto valid =
                  std::min<int64_t>(lengths_ptr[b], seqlen);
              if (valid <= 0) {
                continue;
              }

              const auto token_base = b * batch_stride + h * proj_stride;
              const auto dt_base = b * dt_batch_stride + h * dt_head_stride;
              const auto grad_token_base = token_base;
              const auto grad_dt_base = dt_base;
              const auto init_offset =
                  b * heads * head_stride + h * head_stride;

              std::vector<scalar_t> state_hist((valid + 1) * proj,
                                               scalar_t(0));
              const auto* init_state_head = init_ptr + init_offset;
              std::copy(init_state_head, init_state_head + proj,
                        state_hist.data());

              for (const auto t : c10::irange<int64_t>(valid)) {
                const auto dt_offset = dt_base + t * dt_time_stride;
                const auto dt_val = dt_ptr[dt_offset];
                const auto* prev_state = state_hist.data() + t * proj;
                auto* curr_state = state_hist.data() + (t + 1) * proj;

                const auto B_slice =
                    chunk_param_slice<scalar_t>(B_info, b, t, h);
                const auto* B_head = B_slice.ptr;
                const auto B_proj_stride = B_slice.stride;

                for (const auto p : c10::irange<int64_t>(proj)) {
                  const auto decay =
                      std::exp(dt_val * A_ptr[h * proj_stride + p]);
                  const auto drive = B_head[p * B_proj_stride] *
                                     X_ptr[token_base + t * token_stride + p];
                  curr_state[p] = decay * prev_state[p] + dt_val * drive;
                }
              }

              std::vector<scalar_t> grad_state_curr(proj, scalar_t(0));
              std::vector<scalar_t> grad_state_prev(proj, scalar_t(0));

              for (int64_t t = valid - 1; t >= 0; --t) {
                std::fill(grad_state_prev.begin(), grad_state_prev.end(),
                          scalar_t(0));

                const auto dt_offset = grad_dt_base + t * dt_time_stride;
                const auto dt_val = dt_ptr[dt_offset];
                scalar_t grad_dt_val = scalar_t(0);

                const auto token_offset = grad_token_base + t * token_stride;
                const auto* X_token = X_ptr + token_offset;
                auto* grad_X_token = grad_X_ptr + token_offset;
                const auto* grad_out_token = grad_out_ptr + token_offset;

                const auto B_slice =
                    chunk_param_slice<scalar_t>(B_info, b, t, h);
                const auto* B_head = B_slice.ptr;
                const auto B_proj_stride = B_slice.stride;
                auto B_grad_slice =
                    chunk_param_slice<scalar_t>(B_grad_info, b, t, h);
                auto* B_grad_head = const_cast<scalar_t*>(B_grad_slice.ptr);
                const auto B_grad_stride = B_grad_slice.stride;

                const auto C_slice =
                    chunk_param_slice<scalar_t>(C_info, b, t, h);
                const auto* C_head = C_slice.ptr;
                const auto C_proj_stride = C_slice.stride;
                auto C_grad_slice =
                    chunk_param_slice<scalar_t>(C_grad_info, b, t, h);
                auto* C_grad_head = const_cast<scalar_t*>(C_grad_slice.ptr);
                const auto C_grad_stride = C_grad_slice.stride;

                const auto* D_row = skip_has_matrix
                                        ? D_ptr + h * proj_stride
                                        : nullptr;
                auto* grad_D_row = skip_has_matrix
                                       ? grad_D_ptr + h * proj_stride
                                       : nullptr;
                const auto D_scalar =
                    skip_has_vector ? D_ptr[h] : scalar_t(0);
                auto* grad_D_scalar = skip_has_vector ? grad_D_ptr + h : nullptr;

                scalar_t gate_scalar = scalar_t(1);
                scalar_t gate_scalar_input = scalar_t(0);
                scalar_t* grad_gate_scalar_ptr = nullptr;
                if (gate_kind == GateKind::kScalar) {
                  const auto offset = b * z_scalar_batch_stride +
                                      t * z_scalar_time_stride +
                                      h * z_scalar_head_stride;
                  gate_scalar = z_gate_ptr[offset];
                  gate_scalar_input = z_input_ptr[offset];
                  grad_gate_scalar_ptr = grad_z_ptr + offset;
                }

                const scalar_t* gate_vector = nullptr;
                const scalar_t* gate_vector_input = nullptr;
                scalar_t* grad_gate_vector_ptr = nullptr;
                if (gate_kind == GateKind::kVector) {
                  const auto offset = b * z_vector_batch_stride +
                                      t * z_vector_time_stride +
                                      h * z_vector_head_stride;
                  gate_vector = z_gate_ptr + offset;
                  gate_vector_input = z_input_ptr + offset;
                  grad_gate_vector_ptr = grad_z_ptr + offset;
                }

                const auto* prev_state = state_hist.data() + t * proj;
                const auto* curr_state = state_hist.data() + (t + 1) * proj;

                scalar_t grad_gate_scalar = scalar_t(0);

                for (const auto p : c10::irange<int64_t>(proj)) {
                  const auto A_val = A_ptr[h * proj_stride + p];
                  const auto C_val = C_head[p * C_proj_stride];
                  const auto updated = curr_state[p];
                  scalar_t base = updated * C_val;
                  if (skip_has_vector) {
                    base += D_scalar * X_token[p];
                  } else if (skip_has_matrix) {
                    base += D_row[p] * X_token[p];
                  }

                  scalar_t gate_vec_val = scalar_t(1);
                  scalar_t gate_vec_input_val = scalar_t(0);
                  scalar_t* grad_gate_vec_slot = nullptr;
                  if (gate_vector != nullptr) {
                    gate_vec_val = gate_vector[p * z_vector_proj_stride];
                    gate_vec_input_val =
                        gate_vector_input[p * z_vector_proj_stride];
                    grad_gate_vec_slot =
                        grad_gate_vector_ptr + p * z_vector_proj_stride;
                  }

                  const auto pre_scalar = gate_vector != nullptr
                                              ? base * gate_vec_val
                                              : base;
                  const auto grad_out_val = grad_out_token[p];
                  grad_gate_scalar +=
                      grad_out_val * conj_if_complex(pre_scalar);
                  const auto grad_pre_scalar =
                      grad_out_val * conj_if_complex(gate_scalar);

                  scalar_t grad_base = grad_pre_scalar;
                  if (gate_vector != nullptr) {
                    const auto grad_gate_vec_val =
                        grad_pre_scalar * conj_if_complex(base);
                    grad_base = grad_pre_scalar * conj_if_complex(gate_vec_val);
                    *grad_gate_vec_slot +=
                        grad_gate_vec_val * silu_grad(gate_vec_input_val);
                  }

                  scalar_t state_grad = grad_state_curr[p] +
                                        grad_base * conj_if_complex(C_val);
                  C_grad_head[p * C_grad_stride] +=
                      grad_base * conj_if_complex(updated);

                  if (skip_has_vector) {
                    grad_X_token[p] += grad_base * conj_if_complex(D_scalar);
                    *grad_D_scalar += grad_base * conj_if_complex(X_token[p]);
                  } else if (skip_has_matrix) {
                    grad_X_token[p] += grad_base * conj_if_complex(D_row[p]);
                    grad_D_row[p] += grad_base * conj_if_complex(X_token[p]);
                  }

                  const auto decay = std::exp(dt_val * A_val);
                  const auto state_prev = prev_state[p];
                  grad_state_prev[p] += state_grad * decay;

                  const auto grad_decay =
                      state_grad * conj_if_complex(state_prev);
                  grad_A_ptr[h * proj_stride + p] +=
                      grad_decay * (dt_val * decay);
                  grad_dt_val += grad_decay * (A_val * decay);

                  const auto B_val = B_head[p * B_proj_stride];
                  const auto drive = B_val * X_token[p];
                  B_grad_head[p * B_grad_stride] +=
                      state_grad * conj_if_complex(dt_val * X_token[p]);
                  grad_X_token[p] +=
                      state_grad * conj_if_complex(dt_val * B_val);
                  grad_dt_val += state_grad * conj_if_complex(drive);

                  grad_state_curr[p] = state_grad;
                }

                grad_dt_ptr[dt_offset] += grad_dt_val;
                if (grad_gate_scalar_ptr != nullptr) {
                  *grad_gate_scalar_ptr +=
                      grad_gate_scalar * silu_grad(gate_scalar_input);
                }

                grad_state_curr.swap(grad_state_prev);
              }

              auto* grad_init_head = grad_init_ptr + init_offset;
              for (const auto p : c10::irange<int64_t>(proj)) {
                grad_init_head[p] = grad_state_curr[p];
              }
            }
          }
        });
      });

  auto grad_X = grad_X_compute.to(X.scalar_type());
  auto grad_dt = grad_dt_compute.to(dt.scalar_type());
  at::Tensor grad_A;
  if (A_was_vector) {
    grad_A = grad_A_compute.sum(-1).to(A.scalar_type());
  } else {
    grad_A = grad_A_compute.to(A.scalar_type());
  }

  auto grad_B = B_grad_info.tensor.to(B.scalar_type());
  auto grad_C = C_grad_info.tensor.to(C.scalar_type());

  c10::optional<at::Tensor> grad_D = c10::nullopt;
  if (skip_kind != SkipKind::kNone && D.has_value()) {
    if (D.value().dim() == 1) {
      grad_D = grad_D_tensor.to(D.value().scalar_type());
    } else {
      grad_D = grad_D_tensor.to(D.value().scalar_type());
    }
  }

  c10::optional<at::Tensor> grad_z = c10::nullopt;
  if (gate_kind != GateKind::kNone && z.has_value()) {
    grad_z = grad_z_tensor.to(z.value().scalar_type());
  }

  c10::optional<at::Tensor> grad_initial = c10::nullopt;
  if (initial_states.has_value()) {
    grad_initial =
        grad_initial_state.to(initial_states.value().scalar_type());
  }

  return std::make_tuple(grad_X, grad_dt, grad_A, grad_B, grad_C, grad_D,
                         grad_z, grad_initial);
}

}  // namespace cpu
}  // namespace ssm

