#pragma once

#include <ATen/ATen.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>

#include <string>
#include <tuple>

namespace ssm {
namespace cpu {

enum class ScanParamKind { kShared, kPerBatch, kGroupedTime };  // NOLINT

struct ScanParamInfo {
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

enum class ChunkParamKind { kShared, kPerBatch, kTimeVarying };  // NOLINT

struct ChunkParamInfo {
  ChunkParamKind kind{ChunkParamKind::kShared};
  at::Tensor tensor;
  int64_t stride_batch{0};
  int64_t stride_length{0};
  int64_t stride_head{0};
  int64_t stride_proj{0};
  int64_t length_size{1};
};

inline at::ScalarType get_compute_dtype(const at::Tensor& tensor) {
  auto dtype = tensor.scalar_type();
  if (dtype == at::kHalf || dtype == at::kBFloat16) {
    return at::kFloat;
  }
  if (tensor.is_complex()) {
    if (dtype == at::kComplexHalf) {
      return at::kComplexFloat;
    }
    return dtype;
  }
  return dtype;
}

inline at::Tensor to_compute(const at::Tensor& tensor, at::ScalarType dtype,
                             const at::Device& device) {
  return tensor.to(tensor.options().dtype(dtype).device(device));
}

inline at::ScalarType promote_compute_dtype(at::ScalarType lhs,
                                            at::ScalarType rhs) {
  return at::promote_types(lhs, rhs);
}

inline ScanParamInfo make_scan_param(const std::string& name,
                                     const at::Tensor& param, int64_t batch,
                                     int64_t dim, int64_t state_dim,
                                     int64_t length, at::ScalarType dtype,
                                     const at::Device& device) {
  ScanParamInfo info;
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
    TORCH_CHECK(param.size(0) == batch && param.size(1) == dim &&
                    param.size(2) == state_dim,
                name, " must have shape (B, D, N) when 3-D.");
    info.tensor = to_compute(param, dtype, device).contiguous();
    info.kind = ScanParamKind::kPerBatch;
    info.stride_batch = info.tensor.stride(0);
    info.stride_dim = info.tensor.stride(1);
    info.stride_state = info.tensor.stride(2);
    return info;
  }
  if (param.dim() == 4) {
    TORCH_CHECK(param.size(0) == batch && param.size(3) == length,
                name, " must have shape (B, G, N, L) when 4-D.");
    info.tensor = to_compute(param, dtype, device).contiguous();
    info.kind = ScanParamKind::kGroupedTime;
    info.groups = info.tensor.size(1);
    TORCH_CHECK(dim % info.groups == 0,
                "Group dimension must divide number of channels.");
    info.dim_per_group = dim / info.groups;
    TORCH_CHECK(info.tensor.size(2) == state_dim,
                name, " has mismatched state dimension.");
    info.time_size = info.tensor.size(3);
    TORCH_CHECK(info.time_size == length || info.time_size == 1,
                name, " last dimension must match sequence length or 1.");
    info.stride_batch = info.tensor.stride(0);
    info.stride_group = info.tensor.stride(1);
    info.stride_state = info.tensor.stride(2);
    info.stride_time = info.tensor.stride(3);
    return info;
  }
  TORCH_CHECK(false, "Unsupported rank for ", name, ".");
}

inline ChunkParamInfo make_chunk_param(const std::string& name,
                                       const at::Tensor& param,
                                       int64_t batch, int64_t seqlen,
                                       int64_t heads, int64_t proj,
                                       at::ScalarType dtype,
                                       const at::Device& device) {
  ChunkParamInfo info;
  if (param.dim() == 2) {
    TORCH_CHECK(param.size(0) == heads && param.size(1) == proj,
                name, " must have shape (H, P).");
    info.tensor = to_compute(param, dtype, device).contiguous();
    info.kind = ChunkParamKind::kShared;
    info.stride_head = info.tensor.stride(0);
    info.stride_proj = info.tensor.stride(1);
    return info;
  }
  if (param.dim() == 3) {
    TORCH_CHECK(param.size(0) == batch && param.size(1) == heads &&
                    param.size(2) == proj,
                name, " must have shape (B, H, P) when 3-D.");
    info.tensor = to_compute(param, dtype, device).contiguous();
    info.kind = ChunkParamKind::kPerBatch;
    info.stride_batch = info.tensor.stride(0);
    info.stride_head = info.tensor.stride(1);
    info.stride_proj = info.tensor.stride(2);
    return info;
  }
  if (param.dim() == 4) {
    TORCH_CHECK(param.size(0) == batch && param.size(1) == seqlen &&
                    param.size(2) == heads && param.size(3) == proj,
                name, " must have shape (B, L, H, P) when 4-D.");
    info.tensor = to_compute(param, dtype, device).contiguous();
    info.kind = ChunkParamKind::kTimeVarying;
    info.length_size = info.tensor.size(1);
    TORCH_CHECK(info.length_size == seqlen || info.length_size == 1,
                name, " length dimension must match sequence length or 1.");
    info.stride_batch = info.tensor.stride(0);
    info.stride_length = info.tensor.stride(1);
    info.stride_head = info.tensor.stride(2);
    info.stride_proj = info.tensor.stride(3);
    return info;
  }
  TORCH_CHECK(false, "Unsupported rank for ", name, ".");
}

template <typename scalar_t>
struct ParamSlice {
  const scalar_t* ptr;
  int64_t stride;
};

template <typename scalar_t>
struct ScanParamRowView {
  const scalar_t* base{nullptr};
  int64_t state_stride{0};
  int64_t time_stride{0};
  int64_t time_size{1};
  bool has_time{false};

  inline const scalar_t* data(int64_t time_index) const {
    if (!has_time) {
      return base;
    }
    const auto resolved = time_size == 1 ? 0 : time_index;
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(resolved < time_size);
    return base + resolved * time_stride;
  }
};

template <typename scalar_t>
inline ParamSlice<scalar_t> scan_param_slice(const ScanParamInfo& info,
                                             int64_t batch, int64_t dim,
                                             int64_t t) {
  const auto* base = info.tensor.data_ptr<scalar_t>();
  switch (info.kind) {
    case ScanParamKind::kShared: {
      auto offset = dim * info.stride_dim;
      return {base + offset, info.stride_state};
    }
    case ScanParamKind::kPerBatch: {
      auto offset = batch * info.stride_batch + dim * info.stride_dim;
      return {base + offset, info.stride_state};
    }
    case ScanParamKind::kGroupedTime: {
      auto group = dim / info.dim_per_group;
      auto time_index = info.time_size == 1 ? 0 : t;
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(time_index < info.time_size);
      auto offset = batch * info.stride_batch + group * info.stride_group +
                    time_index * info.stride_time;
      return {base + offset, info.stride_state};
    }
  }
  TORCH_INTERNAL_ASSERT(false, "Unknown scan parameter kind");
}

template <typename scalar_t>
inline ScanParamRowView<scalar_t> make_scan_param_row(
    const ScanParamInfo& info, int64_t batch, int64_t dim) {
  const auto* base = info.tensor.data_ptr<scalar_t>();
  switch (info.kind) {
    case ScanParamKind::kShared: {
      auto offset = dim * info.stride_dim;
      return {base + offset, info.stride_state, 0, 1, false};
    }
    case ScanParamKind::kPerBatch: {
      auto offset = batch * info.stride_batch + dim * info.stride_dim;
      return {base + offset, info.stride_state, 0, 1, false};
    }
    case ScanParamKind::kGroupedTime: {
      auto group = dim / info.dim_per_group;
      auto offset = batch * info.stride_batch + group * info.stride_group;
      return {base + offset, info.stride_state, info.stride_time,
              info.time_size, true};
    }
  }
  TORCH_INTERNAL_ASSERT(false, "Unknown scan parameter kind");
}

template <typename scalar_t>
inline ParamSlice<scalar_t> chunk_param_slice(const ChunkParamInfo& info,
                                              int64_t batch, int64_t length,
                                              int64_t head) {
  const auto* base = info.tensor.data_ptr<scalar_t>();
  switch (info.kind) {
    case ChunkParamKind::kShared: {
      auto offset = head * info.stride_head;
      return {base + offset, info.stride_proj};
    }
    case ChunkParamKind::kPerBatch: {
      auto offset = batch * info.stride_batch + head * info.stride_head;
      return {base + offset, info.stride_proj};
    }
    case ChunkParamKind::kTimeVarying: {
      auto time_index = info.length_size == 1 ? 0 : length;
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(time_index < info.length_size);
      auto offset = batch * info.stride_batch +
                    time_index * info.stride_length + head * info.stride_head;
      return {base + offset, info.stride_proj};
    }
  }
  TORCH_INTERNAL_ASSERT(false, "Unknown chunk parameter kind");
}

}  // namespace cpu
}  // namespace ssm

