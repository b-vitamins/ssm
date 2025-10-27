from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

import pytest
import torch

import ssm.ops as ops


_REFERENCE_PATH = Path(__file__).resolve().parents[1] / "mamba_reference_cases.json"
_REFERENCE_AVAILABLE = _REFERENCE_PATH.exists()
if _REFERENCE_AVAILABLE:
    with _REFERENCE_PATH.open("r", encoding="utf-8") as handle:
        _REFERENCE_CASES: Dict[str, Any] = json.load(handle)
else:  # pragma: no cover - exercised in environments without the JSON
    _REFERENCE_CASES = {}
pytestmark = pytest.mark.skipif(
    not _REFERENCE_AVAILABLE,
    reason=(
        "Mamba CUDA reference JSON not found. Generate it with "
        "scripts/refresh_mamba_goldens.py --mamba-repo <path> --device cuda:0 "
        "--output tests/mamba_reference_cases.json"
    ),
)


def _dtype_from_string(name: str) -> torch.dtype:
    try:
        return getattr(torch, name)
    except AttributeError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported dtype in golden tensor: {name}") from exc


def _tensor_from_record(
    record: Dict[str, Any] | None,
    *,
    device: torch.device | None = None,
    requires_grad: bool = False,
) -> torch.Tensor | None:
    if record is None:
        return None
    dtype = _dtype_from_string(record["dtype"])
    if "data_real" in record and "data_imag" in record:
        real = torch.tensor(record["data_real"], dtype=torch.float32)
        imag = torch.tensor(record["data_imag"], dtype=torch.float32)
        tensor = torch.complex(real, imag).to(dtype)
    else:
        tensor = torch.tensor(record["data"], dtype=dtype)
    if device is not None:
        tensor = tensor.to(device)
    if requires_grad:
        tensor.requires_grad_(True)
    return tensor


def _prepare_inputs(
    records: Mapping[str, Dict[str, Any] | None],
    *,
    gradients: Mapping[str, Dict[str, Any]],
    device: torch.device,
) -> Dict[str, torch.Tensor | None]:
    prepared: Dict[str, torch.Tensor | None] = {}
    for name, record in records.items():
        prepared[name] = _tensor_from_record(
            record,
            device=device,
            requires_grad=name in gradients,
        )
    return prepared


def _expected_tensors(records: Mapping[str, Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    expected: Dict[str, torch.Tensor] = {}
    for name, record in records.items():
        tensor = _tensor_from_record(record)
        if tensor is None:
            raise AssertionError(f"Missing tensor data for expected output '{name}'")
        expected[name] = tensor
    return expected


def _expected_gradients(
    records: Mapping[str, Dict[str, Any]],
) -> Dict[str, torch.Tensor]:
    grads: Dict[str, torch.Tensor] = {}
    for name, record in records.items():
        tensor = _tensor_from_record(record)
        if tensor is None:
            raise AssertionError(f"Missing tensor data for expected gradient '{name}'")
        grads[name] = tensor
    return grads


def _require_tensor(
    inputs: Mapping[str, torch.Tensor | None],
    name: str,
    case_name: str,
) -> torch.Tensor:
    tensor = inputs.get(name)
    if tensor is None:
        raise AssertionError(f"Expected tensor for '{name}' in case '{case_name}'")
    return tensor


@pytest.fixture(autouse=True)
def _require_cuda_backend() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    status = ops._cuda_backend_status()
    if not status.available:
        pytest.skip(f"CUDA backend unavailable: {status}")


@pytest.mark.parametrize(
    "case_name, case",
    sorted(_REFERENCE_CASES.get("selective_scan", {}).items()),
)
def test_selective_scan_against_goldens(case_name: str, case: Dict[str, Any]) -> None:
    device = torch.device("cuda")
    gradients = case.get("gradients", {})
    inputs = _prepare_inputs(case["inputs"], gradients=gradients, device=device)

    result = ops.selective_scan(
        _require_tensor(inputs, "u", case_name),
        _require_tensor(inputs, "delta", case_name),
        _require_tensor(inputs, "A", case_name),
        _require_tensor(inputs, "B", case_name),
        _require_tensor(inputs, "C", case_name),
        D=inputs.get("D"),
        z=inputs.get("z"),
        dt_bias=inputs.get("delta_bias"),
        softplus=case.get("options", {}).get(
            "softplus",
            inputs.get("delta_bias") is not None,
        ),
        return_last_state="last_state" in case.get("outputs", {}),
    )
    if isinstance(result, tuple):
        out_cuda, last_state_cuda = result
        expected_last_state = _tensor_from_record(case["outputs"].get("last_state"))
        assert expected_last_state is not None
        torch.testing.assert_close(last_state_cuda.detach().cpu(), expected_last_state)
    else:
        out_cuda = result
    outputs = _expected_tensors(case["outputs"])
    expected_y = outputs.get("y") or outputs.get("out")
    if expected_y is None:
        raise AssertionError(f"Missing expected selective-scan output for {case_name}")
    torch.testing.assert_close(out_cuda.detach().cpu(), expected_y)

    if gradients:
        out_cuda.sum().backward()
        for name, expected in _expected_gradients(gradients).items():
            tensor = _require_tensor(inputs, name, case_name)
            grad_actual = tensor.grad
            if grad_actual is None:
                raise AssertionError(
                    f"Gradient for {name} was not populated in {case_name}"
                )
            torch.testing.assert_close(grad_actual.detach().cpu(), expected)


@pytest.mark.parametrize(
    "case_name, case",
    sorted(_REFERENCE_CASES.get("selective_state_step", {}).items()),
)
def test_selective_state_step_against_goldens(
    case_name: str, case: Dict[str, Any]
) -> None:
    device = torch.device("cuda")
    gradients = case.get("gradients", {})
    inputs = _prepare_inputs(case["inputs"], gradients=gradients, device=device)

    state_tensor = inputs["state"]
    assert isinstance(state_tensor, torch.Tensor)
    state_copy = state_tensor.clone()

    out_cuda = ops.selective_state_step(
        state_tensor,
        _require_tensor(inputs, "x", case_name),
        _require_tensor(inputs, "dt", case_name),
        _require_tensor(inputs, "A", case_name),
        _require_tensor(inputs, "B", case_name),
        _require_tensor(inputs, "C", case_name),
        D=inputs.get("D"),
        z=inputs.get("z"),
        dt_bias=inputs.get("dt_bias"),
        softplus=case.get("options", {}).get(
            "softplus", inputs.get("dt_bias") is not None
        ),
    )
    expected = _expected_tensors(case["outputs"])
    torch.testing.assert_close(out_cuda.detach().cpu(), expected["out"])
    torch.testing.assert_close(state_tensor.detach().cpu(), expected["state"])

    if gradients:
        out_cuda.sum().backward()
        for name, expected in _expected_gradients(gradients).items():
            source = (
                state_tensor
                if name == "state"
                else _require_tensor(inputs, name, case_name)
            )
            grad_actual = source.grad
            if grad_actual is None:
                raise AssertionError(f"Gradient for {name} missing in {case_name}")
            torch.testing.assert_close(grad_actual.detach().cpu(), expected)
        state_tensor.data.copy_(state_copy)  # restore for safety


@pytest.mark.parametrize(
    "case_name, case",
    sorted(_REFERENCE_CASES.get("ssd_chunk_scan", {}).items()),
)
def test_ssd_chunk_scan_against_goldens(case_name: str, case: Dict[str, Any]) -> None:
    device = torch.device("cuda")
    gradients = case.get("gradients", {})
    inputs = _prepare_inputs(case["inputs"], gradients=gradients, device=device)

    seq_meta = case.get("options", {}).get("seq_meta")
    meta_device: Dict[str, torch.Tensor] | None
    if seq_meta is None:
        meta_device = None
    else:
        meta_device = {}
        for key, value in seq_meta.items():
            tensor = torch.tensor(value, device=device, dtype=torch.long)
            meta_device[key] = tensor

    out_cuda = ops.ssd_chunk_scan(
        _require_tensor(inputs, "X", case_name),
        _require_tensor(inputs, "dt", case_name),
        _require_tensor(inputs, "A", case_name),
        _require_tensor(inputs, "B", case_name),
        _require_tensor(inputs, "C", case_name),
        case.get("options", {}).get("chunk_size"),
        D=inputs.get("D"),
        z=inputs.get("z"),
        seq_meta=meta_device,
        initial_states=inputs.get("initial_states"),
    )
    expected = _expected_tensors(case["outputs"])
    torch.testing.assert_close(out_cuda.detach().cpu(), expected["out"])

    if gradients:
        out_cuda.sum().backward()
        for name, expected in _expected_gradients(gradients).items():
            tensor = _require_tensor(inputs, name, case_name)
            grad_actual = tensor.grad
            if grad_actual is None:
                raise AssertionError(f"Gradient for {name} missing in {case_name}")
            torch.testing.assert_close(grad_actual.detach().cpu(), expected)


@pytest.mark.parametrize(
    "case_name, case",
    sorted(_REFERENCE_CASES.get("dw_causal_conv", {}).items()),
)
def test_dw_causal_conv_against_goldens(case_name: str, case: Dict[str, Any]) -> None:
    device = torch.device("cuda")
    gradients = case.get("gradients", {})
    inputs = _prepare_inputs(case["inputs"], gradients=gradients, device=device)

    out_cuda = ops.dw_causal_conv(
        _require_tensor(inputs, "x", case_name),
        _require_tensor(inputs, "weight", case_name),
        bias=inputs.get("bias"),
        activation=case.get("options", {}).get("activation"),
    )
    expected = _expected_tensors(case["outputs"])
    torch.testing.assert_close(out_cuda.detach().cpu(), expected["out"])

    if gradients:
        out_cuda.sum().backward()
        for name, expected in _expected_gradients(gradients).items():
            tensor = _require_tensor(inputs, name, case_name)
            grad_actual = tensor.grad
            if grad_actual is None:
                raise AssertionError(f"Gradient for {name} missing in {case_name}")
            torch.testing.assert_close(grad_actual.detach().cpu(), expected)


@pytest.mark.parametrize(
    "case_name, case",
    sorted(_REFERENCE_CASES.get("fused_layer_norm", {}).items()),
)
def test_fused_layer_norm_against_goldens(case_name: str, case: Dict[str, Any]) -> None:
    device = torch.device("cuda")
    gradients = case.get("gradients", {})
    inputs = _prepare_inputs(case["inputs"], gradients=gradients, device=device)

    out_cuda = ops.fused_layer_norm(
        _require_tensor(inputs, "x", case_name),
        _require_tensor(inputs, "weight", case_name),
        _require_tensor(inputs, "bias", case_name),
        residual=inputs.get("residual"),
        is_rms=case.get("options", {}).get("is_rms", False),
        eps=case.get("options", {}).get("eps", 1e-5),
        prenorm=case.get("options", {}).get("prenorm", True),
        residual_in_fp32=case.get("options", {}).get("residual_in_fp32", False),
    )
    expected = _expected_tensors(case["outputs"])
    torch.testing.assert_close(out_cuda.detach().cpu(), expected["out"])

    if gradients:
        out_cuda.sum().backward()
        for name, expected in _expected_gradients(gradients).items():
            source = _require_tensor(inputs, name, case_name)
            grad_actual = source.grad
            if grad_actual is None:
                raise AssertionError(f"Gradient for {name} missing in {case_name}")
            torch.testing.assert_close(grad_actual.detach().cpu(), expected)
