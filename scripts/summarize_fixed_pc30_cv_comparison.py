#!/usr/bin/env python3
"""Summarize matched random, purged-shuffle, and contiguous block CV fits."""

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
RUNS = {
    "Random": (
        "gpt2-large_ctx200_bestfixed_selfm300_len500_otherp20_len500_"
        "notebookexact_shuf_xcirc_r2only/pc30"
    ),
    "Purged 300 ms": (
        "gpt2-large_ctx200_bestfixed_selfm300_len500_otherp20_len500_"
        "notebookexact_purgedshuf_emb300ms_xcirc_r2only/pc30"
    ),
    "Purged 500 ms": (
        "gpt2-large_ctx200_bestfixed_selfm300_len500_otherp20_len500_"
        "notebookexact_purgedshuf_emb500ms_xcirc_r2only/pc30"
    ),
    "Contiguous block": (
        "gpt2-large_ctx200_bestfixed_selfm300_len500_otherp20_len500_"
        "notebookexact_xcirc_r2only/pc30"
    ),
}
ORDER = list(RUNS)


rows = []
for cv, relative in RUNS.items():
    files = sorted((ROOT / relative).glob("PTY*_L36_sem.pkl"))
    if len(files) != 15:
        raise SystemExit(f"{cv}: expected 15 patient files, found {len(files)}")
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
    patient.groupby(["condition", "cv"], as_index=False)
    .agg(
        n_patients=("patient", "nunique"),
        median_patient_r2=("median_r2", "median"),
        mean_patient_r2=("median_r2", "mean"),
        median_pct_neurons_r2_gt0=("pct_neurons_r2_gt0", "median"),
        median_train_r2=("median_train_r2", "median"),
    )
)

test_rows = []
for condition, group in patient.groupby("condition"):
    wide = group.pivot(index="patient", columns="cv", values="median_r2")
    comparisons = [
        ("Purged 300 ms", "Contiguous block"),
        ("Random", "Purged 300 ms"),
        ("Purged 500 ms", "Contiguous block"),
        ("Random", "Purged 500 ms"),
        ("Purged 300 ms", "Purged 500 ms"),
    ]
    for higher, lower in comparisons:
        difference = wide[higher] - wide[lower]
        result = stats.wilcoxon(difference, alternative="greater")
        test_rows.append(
            {
                "condition": condition,
                "comparison": f"{higher} > {lower}",
                "median_paired_difference": difference.median(),
                "wilcoxon_W": result.statistic,
                "p_one_sided": result.pvalue,
            }
        )
    difference = wide["Purged 300 ms"]
    result = stats.wilcoxon(difference, alternative="greater")
    test_rows.append(
        {
            "condition": condition,
            "comparison": "Purged 300 ms > 0",
            "median_paired_difference": difference.median(),
            "wilcoxon_W": result.statistic,
            "p_one_sided": result.pvalue,
        }
    )
    difference = wide["Purged 500 ms"]
    result = stats.wilcoxon(difference, alternative="greater")
    test_rows.append(
        {
            "condition": condition,
            "comparison": "Purged 500 ms > 0",
            "median_paired_difference": difference.median(),
            "wilcoxon_W": result.statistic,
            "p_one_sided": result.pvalue,
        }
    )
tests = pd.DataFrame(test_rows)

PLOTS.mkdir(parents=True, exist_ok=True)
stem = PLOTS / "all15_gpt2large_L36_pc30_fixed_random_purged_block_cv"
patient.to_csv(f"{stem}_patient.csv", index=False)
summary.to_csv(f"{stem}_summary.csv", index=False)
tests.to_csv(f"{stem}_paired_tests.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True, constrained_layout=True)
colors = {"self": "#c83e3e", "other": "#3656b3"}
for ax, condition in zip(axes, ["self", "other"]):
    subset = patient.loc[patient["condition"].eq(condition)]
    wide = subset.pivot(index="patient", columns="cv", values="median_r2")
    x = np.arange(len(ORDER))
    for _, values in wide[ORDER].iterrows():
        ax.plot(x, values, color="0.72", linewidth=0.8, alpha=0.65)
        ax.scatter(x, values, color=colors[condition], s=22, alpha=0.72)
    medians = wide[ORDER].median()
    ax.plot(x, medians, color="black", linewidth=2.5, marker="o", markersize=7)
    ax.axhline(0, color="0.35", linestyle="--", linewidth=1)
    ax.set_xticks(
        x,
        ["Random", "Purged\n300 ms", "Purged\n500 ms", "Contiguous\nblock"],
    )
    ax.set_title(
        f"{condition.capitalize()}\nmedian R²: "
        + " → ".join(f"{value:.4f}" for value in medians),
        fontweight="bold",
    )
    ax.set_ylabel("Patient median held-out pseudo-$R^2$")
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle(
    "Removing local temporal adjacency explains part—but not all—of random-CV performance\n"
    "GPT-2 Large L36 • 30 PCs • fixed windows • 15 patients",
    fontweight="bold",
)
fig.savefig(f"{stem}.png", dpi=220, bbox_inches="tight")
fig.savefig(f"{stem}.pdf", bbox_inches="tight")

print(summary.to_string(index=False))
print()
print(tests.to_string(index=False))
print(f"{stem}.pdf")
