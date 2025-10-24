#include "common.h"

#include <ATen/ATen.h>
#include <ATen/Parallel.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <tuple>

namespace ssm {
namespace cpu {

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

        at::parallel_for(0, batch * heads, 0, [&](int64_t start, int64_t end) {
          for (const auto idx : c10::irange(start, end)) {
            const auto b = idx / heads;
            const auto h = idx % heads;
            const auto valid = std::min<int64_t>(lengths_ptr[b], seqlen);

            auto* state_head = state_ptr + idx * head_stride;
            const auto* A_row = A_ptr + h * proj_stride;

            for (int64_t start_t = 0; start_t < valid; start_t += chunk_size) {
              const auto end_t = std::min<int64_t>(valid, start_t + chunk_size);
              for (int64_t t = start_t; t < end_t; ++t) {
                const auto token_index = b * batch_stride + t * token_stride +
                                         h * proj_stride;
                const auto dt_offset = b * seqlen * heads + t * heads + h;
                const auto dt_val = dt_ptr[dt_offset];
                const auto* X_token = X_ptr + token_index;
                auto* out_token = out_ptr + token_index;

                const auto B_slice =
                    chunk_param_slice<scalar_t>(B_info, b, t, h);
                const auto C_slice =
                    chunk_param_slice<scalar_t>(C_info, b, t, h);
                const auto* B_head = B_slice.ptr;
                const auto* C_head = C_slice.ptr;
                const auto B_stride = B_slice.stride;
                const auto C_stride = C_slice.stride;

                scalar_t gate_scalar = scalar_t(1);
                const scalar_t* gate_vec = nullptr;
                if (gate_kind == GateKind::kScalar) {
                  gate_scalar = z_ptr[dt_offset];
                } else if (gate_kind == GateKind::kVector) {
                  gate_vec = z_ptr + token_index;
                }

                const scalar_t* D_row = nullptr;
                scalar_t D_scalar = scalar_t(0);
                if (skip_kind == SkipKind::kVector) {
                  D_scalar = D_ptr[h];
                } else if (skip_kind == SkipKind::kMatrix) {
                  D_row = D_ptr + h * proj_stride;
                }

                for (const auto p : c10::irange(proj)) {
                  const auto decay = std::exp(dt_val * A_row[p]);
                  const auto drive = B_head[p * B_stride] * X_token[p];
                  const auto updated = decay * state_head[p] + dt_val * drive;
                  state_head[p] = updated;

                  scalar_t out_val = updated * C_head[p * C_stride];
                  if (skip_kind == SkipKind::kVector) {
                    out_val += D_scalar * X_token[p];
                  } else if (skip_kind == SkipKind::kMatrix) {
                    out_val += D_row[p] * X_token[p];
                  }
                  if (gate_vec != nullptr) {
                    out_val *= gate_vec[p];
                  }
                  out_token[p] = out_val;
                }

                if (gate_kind == GateKind::kScalar) {
                  for (const auto p : c10::irange(proj)) {
                    out_token[p] *= gate_scalar;
                  }
                }
              }
            }
          }
        });
      });

  return outputs.to(X.scalar_type());
}

}  // namespace cpu
}  // namespace ssm

