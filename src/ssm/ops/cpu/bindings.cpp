#include <torch/extension.h>

#include <string>
#include <tuple>

namespace py = pybind11;

namespace ssm {
namespace cpu {

std::tuple<at::Tensor, at::Tensor> selective_scan_cpu(
    const at::Tensor& u, const at::Tensor& delta, const at::Tensor& A,
    const at::Tensor& B, const at::Tensor& C, const c10::optional<at::Tensor>& D,
    const c10::optional<at::Tensor>& z,
    const c10::optional<at::Tensor>& dt_bias, bool softplus,
    bool return_last_state);

at::Tensor selective_state_step_cpu(
    at::Tensor state, const at::Tensor& x, const at::Tensor& dt,
    const at::Tensor& A, const at::Tensor& B, const at::Tensor& C,
    const c10::optional<at::Tensor>& D, const c10::optional<at::Tensor>& z,
    const c10::optional<at::Tensor>& dt_bias, bool softplus);

at::Tensor ssd_chunk_scan_cpu(
    const at::Tensor& X, const at::Tensor& dt, const at::Tensor& A,
    const at::Tensor& B, const at::Tensor& C, int64_t chunk_size,
    const c10::optional<at::Tensor>& D, const c10::optional<at::Tensor>& z,
    const c10::optional<at::Tensor>& seq_lens,
    const c10::optional<at::Tensor>& cu_seqlens,
    const c10::optional<at::Tensor>& initial_states);

at::Tensor dw_causal_conv_cpu(const at::Tensor& x, const at::Tensor& weight,
                              const c10::optional<at::Tensor>& bias,
                              const std::string& activation);

at::Tensor fused_layer_norm_cpu(
    const at::Tensor& x, const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    const c10::optional<at::Tensor>& residual, bool is_rms, double eps,
    bool prenorm, bool residual_in_fp32);

}  // namespace cpu
}  // namespace ssm

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "ssm CPU fused ops";
  m.def("selective_scan", &ssm::cpu::selective_scan_cpu,
        "Selective scan (CPU)", py::arg("u"), py::arg("delta"), py::arg("A"),
        py::arg("B"), py::arg("C"), py::arg("D") = c10::nullopt,
        py::arg("z") = c10::nullopt, py::arg("dt_bias") = c10::nullopt,
        py::arg("softplus") = false, py::arg("return_last_state") = false);
  m.def("selective_state_step", &ssm::cpu::selective_state_step_cpu,
        "Selective state step (CPU)", py::arg("state"), py::arg("x"),
        py::arg("dt"), py::arg("A"), py::arg("B"), py::arg("C"),
        py::arg("D") = c10::nullopt, py::arg("z") = c10::nullopt,
        py::arg("dt_bias") = c10::nullopt, py::arg("softplus") = true);
  m.def("ssd_chunk_scan", &ssm::cpu::ssd_chunk_scan_cpu,
        "SSD chunk scan (CPU)", py::arg("X"), py::arg("dt"), py::arg("A"),
        py::arg("B"), py::arg("C"), py::arg("chunk_size"),
        py::arg("D") = c10::nullopt, py::arg("z") = c10::nullopt,
        py::arg("seq_lens") = c10::nullopt,
        py::arg("cu_seqlens") = c10::nullopt,
        py::arg("initial_states") = c10::nullopt);
  m.def("dw_causal_conv", &ssm::cpu::dw_causal_conv_cpu,
        "Depthwise causal conv (CPU)", py::arg("x"), py::arg("weight"),
        py::arg("bias") = c10::nullopt, py::arg("activation") = "silu");
  m.def("fused_layer_norm", &ssm::cpu::fused_layer_norm_cpu,
        "Fused LayerNorm/RMSNorm (CPU)", py::arg("x"), py::arg("weight"),
        py::arg("bias") = c10::nullopt, py::arg("residual") = c10::nullopt,
        py::arg("is_rms") = false, py::arg("eps") = 1e-5,
        py::arg("prenorm") = true, py::arg("residual_in_fp32") = true);
}

