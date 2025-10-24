import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ssm.ops import ssd_chunk_scan


@pytest.mark.golden
def test_ssd_chunk_scan_golden_case1(run_goldens):
    if not run_goldens:
        pytest.skip("enable with --run-goldens")
    fp = Path(__file__).parent / "data" / "ssd_chunk_scan_case1.json"
    data = json.loads(fp.read_text())
    meta = data["meta"]
    inputs = data["inputs"]
    outputs = data["outputs"]

    X = torch.tensor(inputs["X"], dtype=torch.float32)
    dt = torch.tensor(inputs["dt"], dtype=torch.float32)
    A = torch.tensor(inputs["A"], dtype=torch.float32)
    Bv = torch.tensor(inputs["B"], dtype=torch.float32)
    Cv = torch.tensor(inputs["C"], dtype=torch.float32)
    chunk = int(inputs["chunk_size"]) 

    with pytest.raises(NotImplementedError):
        _ = ssd_chunk_scan(X, dt, A, Bv, Cv, chunk_size=chunk)

    # The test still validates the golden file structure.
    Y = np.array(outputs["Y"], dtype=np.float32)
    assert Y.shape == (meta["B"], meta["L"], meta["H"], meta["P"]) 
