#!/usr/bin/env python3
"""Test whether BERT last-layer (L12) hippocampal encoding differs with vs.
without speaker tags in the input tokenization.

Paired across patients (patient-level median held-out pseudo-R^2), separately
for self/other, using the same fixed-window pc50 fit used elsewhere in the
paper (self: -300 to +200 ms; other: +20 to +520 ms).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


SEMGLM_ROOT = Path("/scratch/aniluchavez/ConvoDATAS/SemanticGLM")
TAG = "bestfixed_selfm300_len500_otherp20_len500"
RUNS = {
    "no_tag": "bert-base_ctx200",
    "spktag": "bert-base-causal_ctx200spktag",
}
LAYER = 12
OUT = Path(
    "/scratch/aniluchavez/ConvoDATAS/SemanticGLM/plots/"
    "bert_spktag_vs_notag_L12_paired_tests.csv"
)


def load_patient_medians(folder_prefix: str) -> pd.DataFrame:
    path = (
        SEMGLM_ROOT
        / f"{folder_prefix}_{TAG}_notebookexact_shuf_xcirc_r2only"
        / "pc50"
        / f"L{LAYER:02d}_all.pkl"
    )
    df = pickle.load(path.open("rb"))
    df = df.loc[df["region"].eq("hippocampus")]
    return (
        df.groupby(["patient", "condition"], as_index=False)["r2"]
        .median()
        .rename(columns={"r2": "patient_median_r2"})
    )


def main() -> int:
    no_tag = load_patient_medians(RUNS["no_tag"])
    spktag = load_patient_medians(RUNS["spktag"])

    rows = []
    for condition in sorted(no_tag["condition"].unique()):
        a = no_tag.loc[no_tag["condition"].eq(condition)].set_index("patient")["patient_median_r2"]
        b = spktag.loc[spktag["condition"].eq(condition)].set_index("patient")["patient_median_r2"]
        common = a.index.intersection(b.index)
        a, b = a.loc[common].to_numpy(), b.loc[common].to_numpy()
        diff = b - a  # spktag - no_tag
        n = len(diff)

        t_stat, p_two = stats.ttest_rel(b, a)
        sd = np.std(diff, ddof=1)
        sem = stats.sem(diff)
        crit = stats.t.ppf(0.975, n - 1)
        try:
            w_stat, p_wilcoxon = stats.wilcoxon(diff, zero_method="wilcox")
        except ValueError:
            w_stat, p_wilcoxon = np.nan, np.nan

        rows.append(
            {
                "condition": condition,
                "n_patients": n,
                "median_no_tag": float(np.median(a)),
                "median_spktag": float(np.median(b)),
                "mean_diff_spktag_minus_notag": float(np.mean(diff)),
                "ci95_low": float(np.mean(diff) - crit * sem),
                "ci95_high": float(np.mean(diff) + crit * sem),
                "t": float(t_stat),
                "df": n - 1,
                "p_two_sided_ttest": float(p_two),
                "cohen_dz": float(np.mean(diff) / sd) if sd > 0 else np.nan,
                "wilcoxon_stat": float(w_stat),
                "p_two_sided_wilcoxon": float(p_wilcoxon),
            }
        )

    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(out.to_string(index=False))
    print(f"\nSaved: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
