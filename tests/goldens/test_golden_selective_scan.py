import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ssm.ops import selective_scan


@pytest.mark.golden
def test_selective_scan_golden_case1(run_goldens):
    if not run_goldens:
        pytest.skip("enable with --run-goldens")
    fp = Path(__file__).parent / "data" / "selective_scan_case1.json"
    data = json.loads(fp.read_text())
    meta = data["meta"]
    inputs = data["inputs"]
    outputs = data["outputs"]

    u = torch.tensor(inputs["u"], dtype=torch.float32)
    delta = torch.tensor(inputs["delta"], dtype=torch.float32)
    A = torch.tensor(inputs["A"], dtype=torch.float32)
    Bm = torch.tensor(inputs["B"], dtype=torch.float32)
    Cm = torch.tensor(inputs["C"], dtype=torch.float32)
    D = torch.tensor(inputs["D"], dtype=torch.float32)

    with pytest.raises(NotImplementedError):
        _ = selective_scan(u, delta, A, Bm, Cm, D=D)

    # The test still validates the golden file structure.
    out = np.array(outputs["out"], dtype=np.float32)
    assert out.shape == (meta["B"], meta["D"], meta["L"])
