// CPU backend pybind11 bindings (stub)
// This file documents the intended registration points for CPU fused ops.
// Implementations are added per ROADMAP; do not export any real kernels yet.

#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "ssm CPU fused ops (stub)";
  // m.def("selective_scan", &selective_scan_cpu, "Selective scan (CPU)");
  // m.def("selective_state_step", &selective_state_step_cpu, "Selective state step (CPU)");
  // m.def("ssd_chunk_scan", &ssd_chunk_scan_cpu, "SSD chunk scan (CPU)");
  // m.def("dw_causal_conv", &dw_causal_conv_cpu, "Depthwise causal conv (CPU)");
  // m.def("fused_layer_norm", &fused_layer_norm_cpu, "Fused LayerNorm/RMSNorm (CPU)");
}
