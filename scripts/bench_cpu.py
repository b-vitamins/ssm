"""Microbenchmarks comparing CPU fused kernels with the reference path."""

from __future__ import annotations

import argparse
import time
from typing import Callable, Tuple

import torch

import ssm.ops as ops
from ssm.ops.python import reference as reference_ops


def _timeit(fn: Callable[[], None], iters: int = 25, warmup: int = 5) -> float:
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    end = time.perf_counter()
    return (end - start) / iters


def _selective_scan_inputs(batch: int, dim: int, state: int, length: int) -> Tuple:
    torch.manual_seed(0)
    u = torch.randn(batch, dim, length)
    delta = torch.randn(batch, dim, length)
    A = torch.randn(dim, state, dtype=torch.float32)
    B = torch.randn(dim, state)
    C = torch.randn(dim, state)
    return u, delta, A, B, C


def bench_selective_scan(args: argparse.Namespace) -> None:
    u, delta, A, B, C = _selective_scan_inputs(
        args.batch, args.dim, args.state, args.length
    )

    def run_reference() -> None:
        reference_ops.selective_scan(u, delta, A, B, C)

    def run_cpu() -> None:
        ops.selective_scan(u, delta, A, B, C)

    ref = _timeit(run_reference, iters=args.iters)
    cpu = _timeit(run_cpu, iters=args.iters)
    print(f"selective_scan ref={ref*1e3:.3f} ms | cpu={cpu*1e3:.3f} ms")


def bench_dw_causal_conv(args: argparse.Namespace) -> None:
    torch.manual_seed(1)
    x = torch.randn(args.batch, args.dim, args.length)
    weight = torch.randn(args.dim, args.kernel)

    def run_reference() -> None:
        reference_ops.dw_causal_conv(x, weight, activation=args.activation)

    def run_cpu() -> None:
        ops.dw_causal_conv(x, weight, activation=args.activation)

    ref = _timeit(run_reference, iters=args.iters)
    cpu = _timeit(run_cpu, iters=args.iters)
    print(f"dw_causal_conv ref={ref*1e3:.3f} ms | cpu={cpu*1e3:.3f} ms")


def bench_fused_layer_norm(args: argparse.Namespace) -> None:
    torch.manual_seed(2)
    x = torch.randn(args.batch, args.length, args.dim)
    weight = torch.randn(args.dim)

    def run_reference() -> None:
        reference_ops.fused_layer_norm(x, weight, None, prenorm=True)

    def run_cpu() -> None:
        ops.fused_layer_norm(x, weight, None, prenorm=True)

    ref = _timeit(run_reference, iters=args.iters)
    cpu = _timeit(run_cpu, iters=args.iters)
    print(f"fused_layer_norm ref={ref*1e3:.3f} ms | cpu={cpu*1e3:.3f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="op", required=True)

    scan = sub.add_parser("selective_scan", help="Benchmark selective_scan")
    scan.add_argument("--batch", type=int, default=2)
    scan.add_argument("--dim", type=int, default=64)
    scan.add_argument("--state", type=int, default=16)
    scan.add_argument("--length", type=int, default=256)
    scan.add_argument("--iters", type=int, default=50)
    scan.set_defaults(func=bench_selective_scan)

    conv = sub.add_parser("dw_causal_conv", help="Benchmark depthwise causal conv")
    conv.add_argument("--batch", type=int, default=2)
    conv.add_argument("--dim", type=int, default=64)
    conv.add_argument("--length", type=int, default=256)
    conv.add_argument("--kernel", type=int, default=7)
    conv.add_argument("--activation", type=str, default="silu")
    conv.add_argument("--iters", type=int, default=100)
    conv.set_defaults(func=bench_dw_causal_conv)

    ln = sub.add_parser("fused_layer_norm", help="Benchmark fused layer norm")
    ln.add_argument("--batch", type=int, default=2)
    ln.add_argument("--length", type=int, default=256)
    ln.add_argument("--dim", type=int, default=512)
    ln.add_argument("--iters", type=int, default=100)
    ln.set_defaults(func=bench_fused_layer_norm)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

