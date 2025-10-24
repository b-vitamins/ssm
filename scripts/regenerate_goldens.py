#!/usr/bin/env python3
"""Regenerate golden files for tests.

This script is optional and only used when you have an authoritative implementation
available (e.g., after implementing reference ops or when referencing an external library).

It writes JSON files into tests/goldens/data/ with small test tensors and outputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parents[1] / "tests" / "goldens" / "data"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Placeholder: These should be overwritten by real results when available.
    (out_dir / "selective_scan_case1.json").write_text(json.dumps({
        "meta": {"description": "placeholder", "B": 1, "D": 2, "L": 4, "N": 2, "dtype": "float32", "seed": 1234},
        "inputs": {"u": [[[0,0,0,0],[0,0,0,0]]], "delta": [[[0,0,0,0],[0,0,0,0]]], "A": [[0,0],[0,0]], "B": [[0,0],[0,0]], "C": [[0,0],[0,0]], "D": [0,0]},
        "outputs": {"out": [[[0,0,0,0],[0,0,0,0]]]}
    }, indent=2))

    (out_dir / "ssd_chunk_scan_case1.json").write_text(json.dumps({
        "meta": {"description": "placeholder", "B": 1, "L": 8, "H": 2, "P": 2, "G": 1, "N": 4, "dtype": "float32", "seed": 5678},
        "inputs": {
            "X": [[[[0,0],[0,0]]]*8],
            "dt": [[[0,0]]*8],
            "A": [0,0],
            "B": [[[[0,0,0,0]]]*8],
            "C": [[[[0,0,0,0]]]*8],
            "chunk_size": 4
        },
        "outputs": {"Y": [[[[0,0],[0,0]]]*8]}
    }, indent=2))

    print(f"Goldens written to {out_dir}")


if __name__ == "__main__":
    main()
