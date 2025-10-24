import json
from pathlib import Path

import pytest
import torch

from ssm.ops import selective_scan


@pytest.mark.golden
def test_selective_scan_golden_case1(run_goldens):
    if not run_goldens:
        pytest.skip("enable with --run-goldens")
    fp = Path(__file__).parent / "data" / "selective_scan_case1.json"
    data = json.loads(fp.read_text())
    inputs = data["inputs"]
    outputs = data["outputs"]

    u = torch.tensor(inputs["u"], dtype=torch.float32)
    delta = torch.tensor(inputs["delta"], dtype=torch.float32)
    A = torch.tensor(inputs["A"], dtype=torch.float32)
    Bm = torch.tensor(inputs["B"], dtype=torch.float32)
    Cm = torch.tensor(inputs["C"], dtype=torch.float32)
    D = torch.tensor(inputs["D"], dtype=torch.float32)
    z = torch.tensor(inputs["z"], dtype=torch.float32)
    dt_bias = torch.tensor(inputs["dt_bias"], dtype=torch.float32)
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
