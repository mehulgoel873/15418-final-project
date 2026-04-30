#!/usr/bin/env python3
"""Sweep bench across sparsities, granularities, token lengths, impls, and seeds."""
import argparse
import csv
import re
import subprocess
import sys
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BENCH = ROOT / "bin" / "bench"

SPARSITIES   = [0.5, 0.9, 0.95, 0.99]
GRANULARITIES = [1, 2, 8, 32]
TOKEN_LENS   = [1 << 10, 1 << 12, 1 << 14, 1 << 15]
IMPLS        = ["sparse", "naive", "sparse2"]
SEEDS        = [0, 1, 2]

D_DEFAULT = 768
ITERS_DEFAULT = 10

RESULT_RE = re.compile(r"^(Transformer\s+\S.*?)\s{2,}([0-9]+\.[0-9]+)\s*$")
KERNEL_RE = re.compile(r"^\[([^\]]+)\]\s+([0-9]+\.[0-9]+)\s*ms\s*$")


def parse_ms(stdout: str):
    for line in stdout.splitlines():
        m = RESULT_RE.match(line.strip())
        if m:
            return m.group(1).strip(), float(m.group(2))
    return None, None


def parse_kernels(stdout: str):
    """Extract sddmm / softmax / spmm timings from bench stdout.

    bench prints '[kernel_label]  X.XXX ms' lines; for each kernel we keep the
    last occurrence (steady-state) since matmul_tiled and friends repeat."""
    last: dict[str, float | None] = {"sddmm_ms": None, "softmax_ms": None, "spmm_ms": None}
    for line in stdout.splitlines():
        m = KERNEL_RE.match(line.strip())
        if not m:
            continue
        head = m.group(1).strip().split()[0].lower()
        ms = float(m.group(2))
        if head.startswith("sddmm"):
            last["sddmm_ms"] = ms
        elif head.startswith("softmax"):
            last["softmax_ms"] = ms
        elif head.startswith("spmm"):
            last["spmm_ms"] = ms
    return last


def run_one(impl, N, d, iters, sparsity, granularity, seed):
    cmd = [
        str(BENCH),
        "--impl", impl,
        "--sparsity", str(sparsity),
        "--granularity", str(granularity),
        "--seed", str(seed),
        str(N), str(d), str(iters),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    label, ms = parse_ms(proc.stdout)
    kernels = parse_kernels(proc.stdout)
    return cmd, proc.returncode, proc.stdout, proc.stderr, label, ms, kernels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=D_DEFAULT)
    ap.add_argument("--iters", type=int, default=ITERS_DEFAULT)
    ap.add_argument("--out", default=str(ROOT / "results.csv"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not BENCH.exists():
        print(f"error: {BENCH} not found; build it first.", file=sys.stderr)
        sys.exit(1)

    combos = list(product(IMPLS, TOKEN_LENS, SPARSITIES, GRANULARITIES, SEEDS))
    print(f"running {len(combos)} configurations -> {args.out}")

    out_path = Path(args.out)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["impl", "N", "d", "iters", "sparsity", "granularity",
                    "seed", "label", "ms_per_iter",
                    "sddmm_ms", "softmax_ms", "spmm_ms", "returncode"])

        for i, (impl, N, sparsity, granularity, seed) in enumerate(combos, 1):
            tag = f"[{i}/{len(combos)}] impl={impl} N={N} s={sparsity} g={granularity} seed={seed}"
            if args.dry_run:
                print("DRY", tag)
                continue
            print(tag, flush=True)
            cmd, rc, stdout, stderr, label, ms, kernels = run_one(
                impl, N, args.d, args.iters, sparsity, granularity, seed)
            sddmm = kernels["sddmm_ms"]
            softmax = kernels["softmax_ms"]
            spmm = kernels["spmm_ms"]
            if rc != 0 or ms is None:
                print(f"  FAILED rc={rc}", file=sys.stderr)
                if stderr.strip():
                    print(stderr, file=sys.stderr)
                w.writerow([impl, N, args.d, args.iters, sparsity, granularity,
                            seed, label or "", "",
                            "" if sddmm is None else sddmm,
                            "" if softmax is None else softmax,
                            "" if spmm is None else spmm, rc])
            else:
                extras = []
                if sddmm is not None:   extras.append(f"sddmm={sddmm:.3f}")
                if softmax is not None: extras.append(f"softmax={softmax:.3f}")
                if spmm is not None:    extras.append(f"spmm={spmm:.3f}")
                print(f"  -> {ms:.3f} ms/iter  " + " ".join(extras))
                w.writerow([impl, N, args.d, args.iters, sparsity, granularity,
                            seed, label, ms,
                            "" if sddmm is None else sddmm,
                            "" if softmax is None else softmax,
                            "" if spmm is None else spmm, rc])
            f.flush()

    print(f"done -> {out_path}")


if __name__ == "__main__":
    main()
