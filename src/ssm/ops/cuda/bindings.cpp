#include <torch/extension.h>

#include <string>
#include <tuple>

namespace py = pybind11;

namespace ssm {
namespace cuda {

std::tuple<at::Tensor, at::Tensor, at::Tensor> selective_scan_cuda(
    const at::Tensor& u, const at::Tensor& delta, const at::Tensor& A,
    const at::Tensor& B, const at::Tensor& C, const c10::optional<at::Tensor>& D,
    const c10::optional<at::Tensor>& z,
    const c10::optional<at::Tensor>& dt_bias, bool softplus,
    bool return_last_state);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
           c10::optional<at::Tensor>, c10::optional<at::Tensor>,
           c10::optional<at::Tensor>>
selective_scan_backward_cuda(
    const at::Tensor& grad_output,
    const c10::optional<at::Tensor>& grad_last_state, const at::Tensor& u,
    const at::Tensor& delta, const at::Tensor& A, const at::Tensor& B,
    const at::Tensor& C, const c10::optional<at::Tensor>& D,
    const c10::optional<at::Tensor>& z,
    const c10::optional<at::Tensor>& dt_bias, bool softplus,
    const at::Tensor& forward_output, const at::Tensor& cache);

at::Tensor selective_state_step_cuda(
    at::Tensor state, const at::Tensor& x, const at::Tensor& dt,
    const at::Tensor& A, const at::Tensor& B, const at::Tensor& C,
    const c10::optional<at::Tensor>& D, const c10::optional<at::Tensor>& z,
    const c10::optional<at::Tensor>& dt_bias, bool softplus);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
           c10::optional<at::Tensor>, c10::optional<at::Tensor>,
           c10::optional<at::Tensor>>
selective_state_step_backward_cuda(
    const at::Tensor& grad_output,
    const c10::optional<at::Tensor>& grad_state, const at::Tensor& state,
    const at::Tensor& x, const at::Tensor& dt, const at::Tensor& A,
    const at::Tensor& B, const at::Tensor& C,
    const c10::optional<at::Tensor>& D, const c10::optional<at::Tensor>& z,
    const c10::optional<at::Tensor>& dt_bias, bool softplus);

at::Tensor ssd_chunk_scan_cuda(
    const at::Tensor& X, const at::Tensor& dt, const at::Tensor& A,
    const at::Tensor& B, const at::Tensor& C, int64_t chunk_size,
    const c10::optional<at::Tensor>& D, const c10::optional<at::Tensor>& z,
    const c10::optional<at::Tensor>& seq_lens,
    const c10::optional<at::Tensor>& cu_seqlens,
    const c10::optional<at::Tensor>& initial_states);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
           c10::optional<at::Tensor>, c10::optional<at::Tensor>,
           c10::optional<at::Tensor>>
ssd_chunk_scan_backward_cuda(
    const at::Tensor& grad_output, const at::Tensor& X, const at::Tensor& dt,
    const at::Tensor& A, const at::Tensor& B, const at::Tensor& C,
    int64_t chunk_size, const c10::optional<at::Tensor>& D,
    const c10::optional<at::Tensor>& z,
    const c10::optional<at::Tensor>& seq_lens,
    const c10::optional<at::Tensor>& cu_seqlens,
    const c10::optional<at::Tensor>& initial_states);

at::Tensor dw_causal_conv_cuda(const at::Tensor& x, const at::Tensor& weight,
                              const c10::optional<at::Tensor>& bias,
                              const std::string& activation);

std::tuple<at::Tensor, at::Tensor, c10::optional<at::Tensor>>
dw_causal_conv_backward_cuda(const at::Tensor& grad_output,
                             const at::Tensor& x,
                             const at::Tensor& weight,
                             const c10::optional<at::Tensor>& bias,
                             const std::string& activation);

at::Tensor fused_layer_norm_cuda(
    const at::Tensor& x, const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    const c10::optional<at::Tensor>& residual, bool is_rms, double eps,
    bool prenorm, bool residual_in_fp32);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
fused_layer_norm_backward_cuda(
    const at::Tensor& grad_output, const at::Tensor& x,
    const at::Tensor& weight, const c10::optional<at::Tensor>& bias,
    const c10::optional<at::Tensor>& residual, bool is_rms, double eps,
    bool prenorm, bool residual_in_fp32);

}  // namespace cuda
}  // namespace ssm

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "ssm CUDA fused ops";
  m.def("selective_scan", &ssm::cuda::selective_scan_cuda,
        "Selective scan (CUDA)", py::arg("u"), py::arg("delta"), py::arg("A"),
        py::arg("B"), py::arg("C"), py::arg("D") = c10::nullopt,
        py::arg("z") = c10::nullopt, py::arg("dt_bias") = c10::nullopt,
        py::arg("softplus") = false, py::arg("return_last_state") = false);
  m.def("selective_scan_backward", &ssm::cuda::selective_scan_backward_cuda,
        "Selective scan backward (CUDA)", py::arg("grad_output"),
        py::arg("grad_last_state") = c10::nullopt, py::arg("u"),
        py::arg("delta"), py::arg("A"), py::arg("B"), py::arg("C"),
        py::arg("D") = c10::nullopt, py::arg("z") = c10::nullopt,
        py::arg("dt_bias") = c10::nullopt, py::arg("softplus") = false,
        py::arg("forward_output"), py::arg("cache"));
  m.def("selective_state_step", &ssm::cuda::selective_state_step_cuda,
        "Selective state step (CUDA)", py::arg("state"), py::arg("x"),
        py::arg("dt"), py::arg("A"), py::arg("B"), py::arg("C"),
        py::arg("D") = c10::nullopt, py::arg("z") = c10::nullopt,
        py::arg("dt_bias") = c10::nullopt, py::arg("softplus") = true);
  m.def("selective_state_step_backward",
        &ssm::cuda::selective_state_step_backward_cuda,
        "Selective state step backward (CUDA)", py::arg("grad_output"),
        py::arg("grad_state") = c10::nullopt, py::arg("state"), py::arg("x"),
        py::arg("dt"), py::arg("A"), py::arg("B"), py::arg("C"),
        py::arg("D") = c10::nullopt, py::arg("z") = c10::nullopt,
        py::arg("dt_bias") = c10::nullopt, py::arg("softplus") = true);
  m.def("ssd_chunk_scan", &ssm::cuda::ssd_chunk_scan_cuda,
        "SSD chunk scan (CUDA)", py::arg("X"), py::arg("dt"), py::arg("A"),
        py::arg("B"), py::arg("C"), py::arg("chunk_size"),
        py::arg("D") = c10::nullopt, py::arg("z") = c10::nullopt,
        py::arg("seq_lens") = c10::nullopt,
        py::arg("cu_seqlens") = c10::nullopt,
        py::arg("initial_states") = c10::nullopt);
  m.def("ssd_chunk_scan_backward", &ssm::cuda::ssd_chunk_scan_backward_cuda,
        "SSD chunk scan backward (CUDA)", py::arg("grad_output"),
        py::arg("X"), py::arg("dt"), py::arg("A"), py::arg("B"),
        py::arg("C"), py::arg("chunk_size"), py::arg("D") = c10::nullopt,
        py::arg("z") = c10::nullopt, py::arg("seq_lens") = c10::nullopt,
        py::arg("cu_seqlens") = c10::nullopt,
        py::arg("initial_states") = c10::nullopt);
  m.def("dw_causal_conv", &ssm::cuda::dw_causal_conv_cuda,
        "Depthwise causal conv (CUDA)", py::arg("x"), py::arg("weight"),
        py::arg("bias") = c10::nullopt, py::arg("activation") = "silu");
  m.def("dw_causal_conv_backward", &ssm::cuda::dw_causal_conv_backward_cuda,
        "Depthwise causal conv backward (CUDA)", py::arg("grad_output"),
        py::arg("x"), py::arg("weight"), py::arg("bias") = c10::nullopt,
        py::arg("activation") = "silu");
  m.def("fused_layer_norm", &ssm::cuda::fused_layer_norm_cuda,
        "Fused LayerNorm/RMSNorm (CUDA)", py::arg("x"), py::arg("weight"),
        py::arg("bias") = c10::nullopt, py::arg("residual") = c10::nullopt,
        py::arg("is_rms") = false, py::arg("eps") = 1e-5,
        py::arg("prenorm") = true, py::arg("residual_in_fp32") = true);
  m.def("fused_layer_norm_backward",
        &ssm::cuda::fused_layer_norm_backward_cuda,
        "Fused LayerNorm/RMSNorm backward (CUDA)", py::arg("grad_output"),
        py::arg("x"), py::arg("weight"), py::arg("bias") = c10::nullopt,
        py::arg("residual") = c10::nullopt, py::arg("is_rms") = false,
        py::arg("eps") = 1e-5, py::arg("prenorm") = true,
        py::arg("residual_in_fp32") = true);
}

TORCH_LIBRARY_IMPL(ssm, CUDA, m) {
  m.impl("selective_scan", ssm::cuda::selective_scan_cuda);
  m.impl("selective_scan_backward", ssm::cuda::selective_scan_backward_cuda);
  m.impl("selective_state_step", ssm::cuda::selective_state_step_cuda);
  m.impl("selective_state_step_backward",
         ssm::cuda::selective_state_step_backward_cuda);
  m.impl("ssd_chunk_scan_backward", ssm::cuda::ssd_chunk_scan_backward_cuda);
  m.impl("dw_causal_conv_backward", ssm::cuda::dw_causal_conv_backward_cuda);
  m.impl("fused_layer_norm_backward",
         ssm::cuda::fused_layer_norm_backward_cuda);
}

