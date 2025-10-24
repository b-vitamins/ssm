#pragma once

#include <ATen/ATen.h>

namespace ssm {
namespace cpu {

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

}  // namespace cpu
}  // namespace ssm

