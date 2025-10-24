#include "common.h"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>

namespace ssm {
namespace cuda {

namespace {

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
    return to_compute(param, dtype, device);
  }
  TORCH_CHECK(false, "Unsupported rank for ", name, ".");
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

  auto X_compute = to_compute(X, compute_dtype, device);
  auto dt_compute = to_compute(dt, compute_dtype, device);

  at::Tensor A_compute;
  if (A.dim() == 1) {
    TORCH_CHECK(A.size(0) == heads, "A must have shape (H,) when 1-D.");
    A_compute = to_compute(A, compute_dtype, device)
                    .unsqueeze(-1)
                    .expand({heads, proj});
  } else if (A.dim() == 2) {
    TORCH_CHECK(A.size(0) == heads && A.size(1) == proj,
                "A must have shape (H, P) when 2-D.");
    A_compute = to_compute(A, compute_dtype, device);
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
      D_compute = to_compute(tensor, compute_dtype, device).view({heads, 1});
    } else if (tensor.dim() == 2) {
      TORCH_CHECK(tensor.size(0) == heads && tensor.size(1) == proj,
                  "D must have shape (H, P) when 2-D.");
      D_compute = to_compute(tensor, compute_dtype, device);
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
      z_compute = to_compute(tensor, compute_dtype, device).unsqueeze(-1);
    } else if (tensor.dim() == 4) {
      TORCH_CHECK(tensor.size(0) == batch && tensor.size(1) == seqlen &&
                      tensor.size(2) == heads && tensor.size(3) == proj,
                  "z must have shape (B, L, H, P) when 4-D.");
      z_compute = to_compute(tensor, compute_dtype, device);
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
    initial_state = to_compute(tensor, compute_dtype, device);
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

  auto outputs = at::zeros({batch, seqlen, heads, proj},
                           X.options().dtype(compute_dtype));

  for (int64_t b = 0; b < batch; ++b) {
    auto valid = lengths[b].item<int64_t>();
    TORCH_CHECK(valid <= seqlen, "Sequence length exceeds input range.");
    auto state_b = initial_state[b];
    for (int64_t start = 0; start < valid; start += chunk_size) {
      auto end = std::min<int64_t>(valid, start + chunk_size);
      for (int64_t t = start; t < end; ++t) {
        auto dt_bt = dt_compute[b][t];
        auto decay = (dt_bt.unsqueeze(-1) * A_compute).exp();
        auto drive = B_full[b][t] * X_compute[b][t];
        state_b = decay * state_b + dt_bt.unsqueeze(-1) * drive;
        auto y_bt = state_b * C_full[b][t];

        if (D_compute.has_value()) {
          auto skip = D_compute.value();
          if (skip.dim() == 2 && skip.size(1) != proj) {
            skip = skip.expand({heads, proj});
          }
          y_bt = y_bt + skip * X_compute[b][t];
        }

        if (z_compute.has_value()) {
          auto gate = z_compute.value()[b][t];
          if (!(gate.sizes() == y_bt.sizes())) {
            gate = gate.expand_as(y_bt);
          }
          y_bt = y_bt * at::silu(gate);
        }

        outputs[b][t].copy_(y_bt);
      }
    }
    if (valid < seqlen) {
      outputs[b].narrow(0, valid, seqlen - valid).zero_();
    }
  }

  return outputs.to(X.scalar_type());
}

}  // namespace cuda
}  // namespace ssm

