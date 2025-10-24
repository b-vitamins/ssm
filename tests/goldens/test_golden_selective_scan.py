import json
from pathlib import Path

import pytest
import torch

from ssm.ops import selective_scan


@pytest.mark.golden
@pytest.mark.parametrize(
    "case_name",
    ["selective_scan_case1", "selective_scan_grouped"],
)
def test_selective_scan_golden(run_goldens, case_name):
    if not run_goldens:
        pytest.skip("enable with --run-goldens")
    fp = Path(__file__).parent / "data" / f"{case_name}.json"
    data = json.loads(fp.read_text())
    inputs = data["inputs"]
    outputs = data["outputs"]

    u = torch.tensor(inputs["u"], dtype=torch.float32)
    delta = torch.tensor(inputs["delta"], dtype=torch.float32)
    A = torch.tensor(inputs["A"], dtype=torch.float32)
    B_val = inputs["B"]
    C_val = inputs["C"]
    D_val = inputs["D"]
    z_val = inputs["z"]
    dt_bias_val = inputs["dt_bias"]

    Bm = torch.tensor(B_val, dtype=torch.float32)
    Cm = torch.tensor(C_val, dtype=torch.float32)
    D = None if D_val is None else torch.tensor(D_val, dtype=torch.float32)
    z = None if z_val is None else torch.tensor(z_val, dtype=torch.float32)
    dt_bias = (
        None if dt_bias_val is None else torch.tensor(dt_bias_val, dtype=torch.float32)
    )
    softplus = bool(inputs["softplus"])

    out, last_state = selective_scan(
        u,
        delta,
        A,
        Bm,
        Cm,
        D=D,
        z=z,
        dt_bias=dt_bias,
        softplus=softplus,
        return_last_state=True,
    )

    expected_out = torch.tensor(outputs["out"], dtype=torch.float32)
    expected_state = torch.tensor(outputs["last_state"], dtype=torch.float32)

    assert torch.allclose(out, expected_out)
    assert torch.allclose(last_state, expected_state)
