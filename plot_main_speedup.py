#!/usr/bin/env python3
"""Plot naive / sparse ms-per-iter ratio vs sparsity, one plot per N.

For each N in the results CSV, draw 4 lines (one per granularity) showing
sparse_ms_per_iter / naive_ms_per_iter, averaged across seeds.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


def load(path: Path):
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            try:
                r["N"] = int(r["N"])
                r["sparsity"] = float(r["sparsity"])
                r["granularity"] = int(r["granularity"])
                r["seed"] = int(r["seed"])
                r["ms_per_iter"] = float(r["ms_per_iter"]) if r["ms_per_iter"] else None
            except ValueError:
                continue
            if r["ms_per_iter"] is None:
                continue
            rows.append(r)
    return rows


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "results.csv"))
    ap.add_argument("--impl", default="sparse",
                    help="sparse impl to compare against naive (sparse | sparse2)")
    ap.add_argument("--out", default=str(ROOT / "speedup.png"),
                    help="output image path; one figure with 2x2 subplots")
    args = ap.parse_args()

    rows = load(Path(args.csv))

    # Average ms_per_iter across seeds, keyed by (impl, N, sparsity, granularity).
    bucket = defaultdict(list)
    for r in rows:
        bucket[(r["impl"], r["N"], r["sparsity"], r["granularity"])].append(r["ms_per_iter"])
    avg = {k: mean(v) for k, v in bucket.items()}

    Ns = sorted({r["N"] for r in rows})
    granularities = sorted({r["granularity"] for r in rows})
    sparsities = sorted({r["sparsity"] for r in rows})

    if len(Ns) != 4:
        print(f"warning: expected 4 N values, found {Ns}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=False)
    for ax, N in zip(axes.flat, Ns):
        for g in granularities:
            ys = []
            xs = []
            for s in sparsities:
                sp = avg.get((args.impl, N, s, g))
                nv = avg.get(("naive", N, s, g))
                if sp is None or nv is None or nv == 0:
                    continue
                xs.append(s)
                ys.append(nv / sp)
            if xs:
                ax.plot(xs, ys, marker="o", label=f"granularity={g}")
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title(f"N = {N}")
        ax.set_xlabel("sparsity")
        ax.set_ylabel(f"naive / {args.impl} (ms/iter ratio)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(f"{args.impl} vs naive — ms/iter ratio (higher = faster)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
