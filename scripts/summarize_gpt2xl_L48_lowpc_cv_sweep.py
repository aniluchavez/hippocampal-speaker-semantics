#!/usr/bin/env python3
"""Summarize GPT-2 XL L48 low-PC fixed-window CV sweep."""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path("/scratch/aniluchavez/ConvoDATAS/SemanticGLM")
PLOTS = ROOT / "plots"
PCS = [5, 10, 20]
RUNS = {
    "Random": (
        "gpt2-xl_ctx200_bestfixed_selfm300_len500_otherp20_len500_"
        "notebookexact_shuf_xcirc_r2only"
    ),
    "Purged 300 ms": (
        "gpt2-xl_ctx200_bestfixed_selfm300_len500_otherp20_len500_"
        "notebookexact_purgedshuf_emb300ms_xcirc_r2only"
    ),
    "Purged 500 ms": (
        "gpt2-xl_ctx200_bestfixed_selfm300_len500_otherp20_len500_"
        "notebookexact_purgedshuf_emb500ms_xcirc_r2only"
    ),
    "Contiguous block": (
        "gpt2-xl_ctx200_bestfixed_selfm300_len500_otherp20_len500_"
        "notebookexact_xcirc_r2only"
    ),
}


rows = []
for cv, run in RUNS.items():
    for pc in PCS:
        files = sorted((ROOT / run / f"pc{pc}").glob("PTY*_L48_sem.pkl"))
        if len(files) != 15:
            raise SystemExit(
                f"{cv}, pc={pc}: expected 15 files, found {len(files)}"
            )
        for path in files:
            with path.open("rb") as handle:
                data = pickle.load(handle)["df"]
            data = data.loc[data["region"].str.lower().eq("hippocampus")]
            for (patient, condition), group in data.groupby(
                ["patient", "condition"]
            ):
                rows.append(
                    {
                        "cv": cv,
                        "pc": pc,
                        "patient": patient,
                        "condition": condition,
                        "n_neurons": len(group),
                        "median_r2": group["r2"].median(),
                        "mean_r2": group["r2"].mean(),
                        "pct_neurons_r2_gt0": 100 * group["r2"].gt(0).mean(),
                        "median_train_r2": group["r2_train"].median(),
                    }
                )

patient = pd.DataFrame(rows)
summary = (
    patient.groupby(["condition", "cv", "pc"], as_index=False)
    .agg(
        n_patients=("patient", "nunique"),
        median_patient_r2=("median_r2", "median"),
        mean_patient_r2=("median_r2", "mean"),
        median_pct_neurons_r2_gt0=("pct_neurons_r2_gt0", "median"),
        median_train_r2=("median_train_r2", "median"),
    )
)

test_rows = []
for (condition, pc), group in patient.groupby(["condition", "pc"]):
    wide = group.pivot(index="patient", columns="cv", values="median_r2")
    for cv in ["Purged 300 ms", "Purged 500 ms"]:
        result = stats.wilcoxon(wide[cv], alternative="greater")
        test_rows.append(
            {
                "condition": condition,
                "pc": pc,
                "comparison": f"{cv} > 0",
                "median_r2": wide[cv].median(),
                "wilcoxon_W": result.statistic,
                "p_one_sided": result.pvalue,
            }
        )
tests = pd.DataFrame(test_rows)

PLOTS.mkdir(parents=True, exist_ok=True)
stem = PLOTS / "all15_gpt2xl_L48_lowpc_fixed_random_purged_block_cv"
patient.to_csv(f"{stem}_patient.csv", index=False)
summary.to_csv(f"{stem}_summary.csv", index=False)
tests.to_csv(f"{stem}_paired_tests.csv", index=False)

colors = {
    "Random": "#222222",
    "Purged 300 ms": "#2b8cbe",
    "Purged 500 ms": "#7b3294",
    "Contiguous block": "#d95f0e",
}
markers = {
    "Random": "o",
    "Purged 300 ms": "s",
    "Purged 500 ms": "^",
    "Contiguous block": "D",
}
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True, constrained_layout=True)
for ax, condition in zip(axes, ["self", "other"]):
    subset = summary.loc[summary["condition"].eq(condition)]
    for cv in RUNS:
        line = subset.loc[subset["cv"].eq(cv)].set_index("pc").loc[PCS]
        ax.plot(
            PCS,
            line["median_patient_r2"],
            marker=markers[cv],
            color=colors[cv],
            linewidth=2,
            markersize=7,
            label=cv,
        )
    ax.axhline(0, color="0.5", linestyle="--", linewidth=1)
    ax.set_xticks(PCS)
    ax.set_xlabel("PCA components")
    ax.set_ylabel("Median patient held-out pseudo-$R^2$")
    ax.set_title(condition.capitalize(), fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
axes[1].legend(frameon=False, fontsize=9)
fig.suptitle(
    "Lower dimensionality does not rescue embargoed speaking performance\n"
    "GPT-2 XL final layer (L48) • fixed windows • 15 patients",
    fontweight="bold",
)
fig.savefig(f"{stem}.png", dpi=220, bbox_inches="tight")
fig.savefig(f"{stem}.pdf", bbox_inches="tight")

print(summary.to_string(index=False))
print()
print(tests.to_string(index=False))
print(f"{stem}.pdf")
