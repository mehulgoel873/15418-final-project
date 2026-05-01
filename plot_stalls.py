#!/usr/bin/env python3
"""Plot per-kernel time breakdown from `ncu --csv --page raw --metrics ...`.

The --page raw export is wide format: each metric is its own column. Row 0
is column names, row 1 is units (e.g. "%", "us", "cycle"), then data rows.

Usage:
    python3 plot_stalls.py stalls.csv
"""
import sys
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

CSV = sys.argv[1] if len(sys.argv) > 1 else "stalls.csv"

# Skip the units row.
df = pd.read_csv(CSV, skiprows=[1], thousands=",")

if "Kernel Name" not in df.columns:
    sys.exit("Expected wide-format ncu csv (--page raw). Got columns: "
             + ", ".join(df.columns))

# Coerce all metric columns to numeric.
meta_cols = {"ID", "Process ID", "Process Name", "Host Name", "Kernel Name",
             "Context", "Stream", "Block Size", "Grid Size", "Device", "CC"}
metric_cols = [c for c in df.columns if c not in meta_cols]
for c in metric_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

def short_name(s: str) -> str:
    s = re.sub(r"\s*\([^)]*\)\s*$", "", str(s))
    s = s.replace("void ", "")
    return s[:60]

df["Kernel"] = df["Kernel Name"].map(short_name)

# Keep only the kernels we care about: spmm, sddmm, softmax variants.
KERNEL_PATTERN = re.compile(r"spmm|sddmm|softmax", re.IGNORECASE)
df = df[df["Kernel"].str.contains(KERNEL_PATTERN)]
if df.empty:
    sys.exit("No spmm/sddmm/softmax kernels found in the CSV.")

# Aggregate launches of the same kernel.
DURATION = "gpu__time_duration.sum"
agg = {c: "mean" for c in metric_cols}
if DURATION in metric_cols:
    agg[DURATION] = "sum"   # total wall time across all launches
wide = df.groupby("Kernel").agg(agg)

# `--set full` collects the warp-count breakdown:
#   smsp__average_warps_issue_stalled_<reason>_per_issue_active.ratio
# (number of resident warps per issue cycle waiting for each reason).
# "selected" lives under the same prefix and represents the issuing share.
STALL_PREFIX = "smsp__average_warps_issue_stalled_"

# Match the prefix; allow optional ncu-appended suffix like ".avg" or ".ratio".
def matches_stall(col: str) -> bool:
    base = col.split(".")[0]
    return base.startswith(STALL_PREFIX)

stall_cols = [c for c in wide.columns if matches_stall(c)]
if not stall_cols:
    sys.exit("No smsp__average_warp_latency_issue_stalled_* metrics found.\n"
             "Available columns:\n  " + "\n  ".join(wide.columns))

def stall_label(c: str) -> str:
    base = c.split(".")[0]
    return base[len(STALL_PREFIX):]

components = {}
for c in stall_cols:
    label = stall_label(c)
    if label == "selected":
        label = "selected (issuing)"
    components[label] = wide[c]
comp_df = pd.DataFrame(components).fillna(0.0)

# Sort kernels by total duration if available.
if DURATION in wide.columns:
    order = wide[DURATION].sort_values(ascending=False).index
else:
    order = comp_df.sum(axis=1).sort_values(ascending=False).index
comp_df = comp_df.loc[order]

# Each kernel sums to 100% of its avg warp latency.
totals = comp_df.sum(axis=1).replace(0, np.nan)
pct = comp_df.div(totals, axis=0) * 100.0

# ---- Plot 1: stacked bar of stall reasons (%) per kernel -------------------
fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(pct)), 6))
bottom = np.zeros(len(pct))
col_order = (["selected (issuing)"] if "selected (issuing)" in pct.columns else [])
col_order += [c for c in pct.columns if c not in col_order]
cmap = plt.get_cmap("tab20")
for i, col in enumerate(col_order):
    color = "#3a8a3a" if col == "selected (issuing)" else cmap(i % 20)
    ax.bar(pct.index, pct[col].values, bottom=bottom, label=col, color=color)
    bottom += pct[col].values
ax.set_ylabel("Share of avg warp latency per inst issued (%)")
ax.set_title("Per-kernel stall breakdown (issuing vs. each stall reason)")
ax.set_xticklabels(pct.index, rotation=45, ha="right")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig("stalls_breakdown.png", dpi=150)
print("wrote stalls_breakdown.png")

# ---- Plot 2: total kernel duration (us) -----------------------------------
if DURATION in wide.columns:
    dur_us = wide[DURATION].loc[order]   # already in us per ncu units row
    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(dur_us))))
    ax.barh(dur_us.index[::-1], dur_us.values[::-1], color="steelblue")
    ax.set_xlabel("Total wall time across all launches (us)")
    ax.set_title("Per-kernel wall time (summed across launches)")
    for i, v in enumerate(dur_us.values[::-1]):
        ax.text(v, i, f" {v:,.0f}", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig("stalls_duration.png", dpi=150)
    print("wrote stalls_duration.png")

# ---- Plot 3: compute vs memory throughput (%) -----------------------------
THROUGHPUT_COLS = {
    "compute (FMA)":         "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active",
    "compute+mem (overall)": "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "DRAM":                  "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "L1TEX":                 "l1tex__throughput.avg.pct_of_peak_sustained_active",
}
present = {k: v for k, v in THROUGHPUT_COLS.items() if v in wide.columns}
if present:
    tp = wide[list(present.values())].loc[order].rename(
        columns={v: k for k, v in present.items()})
    fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(tp)), 4))
    x = np.arange(len(tp))
    width = 0.8 / len(tp.columns)
    for i, col in enumerate(tp.columns):
        ax.bar(x + i * width, tp[col].values, width, label=col)
    ax.set_xticks(x + width * (len(tp.columns) - 1) / 2)
    ax.set_xticklabels(tp.index, rotation=45, ha="right")
    ax.set_ylabel("% of peak")
    ax.set_title("Per-kernel throughput (%)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("stalls_throughput.png", dpi=150)
    print("wrote stalls_throughput.png")
