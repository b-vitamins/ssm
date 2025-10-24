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


def _selective_scan_inputs(
    batch: int,
    dim: int,
    state: int,
    length: int,
    groups: int,
    time_varying: bool,
    with_dt_bias: bool,
    with_z: bool,
) -> Tuple:
    torch.manual_seed(0)
    u = torch.randn(batch, dim, length)
    delta = torch.randn(batch, dim, length)
    A = torch.randn(dim, state, dtype=torch.float32)
    if groups > 1:
        B = torch.randn(batch, groups, state, length if time_varying else 1)
        C = torch.randn(batch, groups, state, length if time_varying else 1)
    else:
        B = torch.randn(dim, state)
        C = torch.randn(dim, state)
    dt_bias = torch.randn(dim) if with_dt_bias else None
    z = torch.randn(batch, dim, length) if with_z else None
    return u, delta, A, B, C, dt_bias, z


def _ssd_chunk_inputs(
    batch: int,
    seqlen: int,
    heads: int,
    proj: int,
    time_varying: bool,
    varlen: bool,
    with_d: bool,
    with_z: bool,
) -> Tuple:
    torch.manual_seed(1)
    X = torch.randn(batch, seqlen, heads, proj)
    dt = torch.rand(batch, seqlen, heads)
    A = torch.randn(heads, proj)
    if time_varying:
        B = torch.randn(batch, seqlen, heads, proj)
        C = torch.randn(batch, heads, proj)
    else:
        B = torch.randn(heads, proj)
        C = torch.randn(heads, proj)
    D = torch.randn(heads, proj) if with_d else None
    if with_z:
        z = (
            torch.randn(batch, seqlen, heads, proj)
            if time_varying
            else torch.randn(batch, seqlen, heads)
        )
    else:
        z = None
    if varlen:
        lengths = list(range(seqlen, seqlen - batch, -1))
        lengths = [max(1, min(seqlen, length_val)) for length_val in lengths]
        cu = torch.tensor([0] + list(torch.cumsum(torch.tensor(lengths), 0).tolist()))
        seq_meta = {"seq_lens": lengths, "cu_seqlens": cu.tolist()}
    else:
        seq_meta = {"seq_lens": [seqlen] * batch}
    init_state = torch.randn(batch, heads, proj)
    return X, dt, A, B, C, D, z, seq_meta, init_state


def bench_selective_scan(args: argparse.Namespace) -> None:
    u, delta, A, B, C, dt_bias, z = _selective_scan_inputs(
        args.batch,
        args.dim,
        args.state,
        args.length,
        args.groups,
        args.time_varying,
        args.dt_bias,
        args.with_z,
    )

    def run_reference() -> None:
        reference_ops.selective_scan(
            u,
            delta,
            A,
            B,
            C,
            dt_bias=dt_bias,
            z=z,
            softplus=args.softplus,
        )

    def run_cpu() -> None:
        ops.selective_scan(
            u,
            delta,
            A,
            B,
            C,
            dt_bias=dt_bias,
            z=z,
            softplus=args.softplus,
        )

    ref = _timeit(run_reference, iters=args.iters)
    cpu = _timeit(run_cpu, iters=args.iters)
    print(
        f"selective_scan ref={ref * 1e3:.3f} ms | cpu={cpu * 1e3:.3f} ms | speedup={ref / cpu:.2f}x"
    )


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
    print(
        f"dw_causal_conv ref={ref * 1e3:.3f} ms | cpu={cpu * 1e3:.3f} ms | speedup={ref / cpu:.2f}x"
    )


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
    print(
        f"fused_layer_norm ref={ref * 1e3:.3f} ms | cpu={cpu * 1e3:.3f} ms | speedup={ref / cpu:.2f}x"
    )


def bench_ssd_chunk_scan(args: argparse.Namespace) -> None:
    X, dt, A, B, C, D, z, seq_meta, init_state = _ssd_chunk_inputs(
        args.batch,
        args.length,
        args.heads,
        args.proj,
        args.time_varying,
        args.varlen,
        args.with_d,
        args.with_z,
    )

    def run_reference() -> None:
        reference_ops.ssd_chunk_scan(
            X,
            dt,
            A,
            B,
            C,
            args.chunk,
            D=D,
            z=z,
            seq_meta=seq_meta,
            initial_states=init_state,
        )

    def run_cpu() -> None:
        ops.ssd_chunk_scan(
            X,
            dt,
            A,
            B,
            C,
            args.chunk,
            D=D,
            z=z,
            seq_meta=seq_meta,
            initial_states=init_state,
        )

    ref = _timeit(run_reference, iters=args.iters)
    cpu = _timeit(run_cpu, iters=args.iters)
    print(
        f"ssd_chunk_scan ref={ref * 1e3:.3f} ms | cpu={cpu * 1e3:.3f} ms | speedup={ref / cpu:.2f}x"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="op", required=True)

    scan = sub.add_parser("selective_scan", help="Benchmark selective_scan")
    scan.add_argument("--batch", type=int, default=2)
    scan.add_argument("--dim", type=int, default=64)
    scan.add_argument("--state", type=int, default=16)
    scan.add_argument("--length", type=int, default=256)
    scan.add_argument("--groups", type=int, default=1)
    scan.add_argument("--time-varying", action="store_true")
    scan.add_argument("--dt-bias", action="store_true")
    scan.add_argument("--with-z", action="store_true")
    scan.add_argument("--softplus", action="store_true")
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

    ssd = sub.add_parser("ssd_chunk_scan", help="Benchmark SSD chunk scan")
    ssd.add_argument("--batch", type=int, default=2)
    ssd.add_argument("--length", type=int, default=256)
    ssd.add_argument("--heads", type=int, default=8)
    ssd.add_argument("--proj", type=int, default=16)
    ssd.add_argument("--chunk", type=int, default=16)
    ssd.add_argument("--time-varying", action="store_true")
    ssd.add_argument("--varlen", action="store_true")
    ssd.add_argument("--with-d", action="store_true")
    ssd.add_argument("--with-z", action="store_true")
    ssd.add_argument("--iters", type=int, default=50)
    ssd.set_defaults(func=bench_ssd_chunk_scan)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
