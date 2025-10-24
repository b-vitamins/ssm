import json
from pathlib import Path

import pytest
import torch

from ssm.ops import ssd_chunk_scan


@pytest.mark.golden
@pytest.mark.parametrize(
    "case_name",
    ["ssd_chunk_scan_case1", "ssd_chunk_scan_varlen"],
)
def test_ssd_chunk_scan_golden(run_goldens, case_name):
    if not run_goldens:
        pytest.skip("enable with --run-goldens")
    fp = Path(__file__).parent / "data" / f"{case_name}.json"
    data = json.loads(fp.read_text())
    inputs = data["inputs"]
    outputs = data["outputs"]

    X = torch.tensor(inputs["X"], dtype=torch.float32)
    dt = torch.tensor(inputs["dt"], dtype=torch.float32)
    A = torch.tensor(inputs["A"], dtype=torch.float32)
    Bv = torch.tensor(inputs["B"], dtype=torch.float32)
    Cv = torch.tensor(inputs["C"], dtype=torch.float32)
    Dv = torch.tensor(inputs["D"], dtype=torch.float32)
    Zv = torch.tensor(inputs["z"], dtype=torch.float32)
    chunk = int(inputs["chunk_size"])
    seq_meta = inputs["seq_meta"]
    seq_meta_tensor = {}
    if seq_meta.get("seq_lens") is not None:
        seq_meta_tensor["seq_lens"] = seq_meta["seq_lens"]
    if seq_meta.get("cu_seqlens") is not None:
        seq_meta_tensor["cu_seqlens"] = torch.tensor(
            seq_meta["cu_seqlens"], dtype=torch.long
        )
    init_state = torch.tensor(inputs["initial_states"], dtype=torch.float32)

    out = ssd_chunk_scan(
        X,
        dt,
        A,
        Bv,
        Cv,
        chunk_size=chunk,
        D=Dv,
        z=Zv,
        seq_meta=seq_meta_tensor,
        initial_states=init_state,
    )

    expected = torch.tensor(outputs["Y"], dtype=torch.float32)
    assert torch.allclose(out, expected)
