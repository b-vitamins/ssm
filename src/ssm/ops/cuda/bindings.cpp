// CUDA backend pybind11 bindings (stub)
// This file documents intended registration points for CUDA fused ops.

#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "ssm CUDA fused ops (stub)";
  // m.def("selective_scan", &selective_scan_cuda, "Selective scan (CUDA)");
  // m.def("selective_state_step", &selective_state_step_cuda, "Selective state step (CUDA)");
  // m.def("ssd_chunk_scan", &ssd_chunk_scan_cuda, "SSD chunk scan (CUDA)");
  // m.def("dw_causal_conv", &dw_causal_conv_cuda, "Depthwise causal conv (CUDA)");
  // m.def("fused_layer_norm", &fused_layer_norm_cuda, "Fused LayerNorm/RMSNorm (CUDA)");
}
