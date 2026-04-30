#!/usr/bin/env python3
"""Plot naive / sparse timing ratio vs sparsity, one plot per N.

For each N in the results CSV, draw 4 lines (one per granularity) showing
naive_metric / sparse_metric (i.e. speedup), averaged across seeds.

Use --metric to choose which timing column to compare:
    ms_per_iter  (default) total transformer ms per iter
    sddmm        Q @ K^T  (sparse: [sddmm], naive: 4th matmul_tiled)
    spmm         attn @ V (sparse: [spmm],  naive: 5th matmul_tiled)
    softmax      attention softmax
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent

METRIC_COLUMNS = {
    "ms_per_iter": "ms_per_iter",
    "sddmm":       "sddmm_ms",
    "spmm":        "spmm_ms",
    "softmax":     "softmax_ms",
}


def load(path: Path, column: str):
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            try:
                r["N"] = int(r["N"])
                r["sparsity"] = float(r["sparsity"])
                r["granularity"] = int(r["granularity"])
                r["seed"] = int(r["seed"])
                v = r.get(column, "")
                r["_metric"] = float(v) if v not in ("", None) else None
            except ValueError:
                continue
            if r["_metric"] is None:
                continue
            rows.append(r)
    return rows


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "main-results.csv"))
    ap.add_argument("--impl", default="sparse",
                    help="sparse impl to compare against naive (sparse | sparse2)")
    ap.add_argument("--metric", default="ms_per_iter",
                    choices=sorted(METRIC_COLUMNS.keys()),
                    help="which timing to compare")
    ap.add_argument("--out", default=None,
                    help="output image path (default: speedup_<impl>_<metric>.png)")
    args = ap.parse_args()

    column = METRIC_COLUMNS[args.metric]
    out = args.out or str(ROOT / f"speedup_{args.impl}_{args.metric}.png")

    rows = load(Path(args.csv), column)

    bucket = defaultdict(list)
    for r in rows:
        bucket[(r["impl"], r["N"], r["sparsity"], r["granularity"])].append(r["_metric"])
    avg = {k: mean(v) for k, v in bucket.items()}

    Ns = sorted({r["N"] for r in rows})
    granularities = sorted({r["granularity"] for r in rows})
    sparsities = sorted({r["sparsity"] for r in rows})

    if len(Ns) != 4:
        print(f"warning: expected 4 N values, found {Ns}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=False)
    for ax, N in zip(axes.flat, Ns):
        for g in granularities:
            xs, ys = [], []
            for s in sparsities:
                sp = avg.get((args.impl, N, s, g))
                nv = avg.get(("naive", N, s, g))
                if sp is None or nv is None or sp == 0:
                    continue
                xs.append(s)
                ys.append(nv / sp)
            if xs:
                ax.plot(xs, ys, marker="o", label=f"granularity={g}")
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title(f"N = {N}")
        ax.set_xlabel("sparsity")
        ax.set_ylabel(f"naive / {args.impl} ({args.metric} ratio)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(f"{args.impl} vs naive — {args.metric} speedup (higher = faster)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
