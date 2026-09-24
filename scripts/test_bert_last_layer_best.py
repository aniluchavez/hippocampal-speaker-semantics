#!/usr/bin/env python3
"""Test whether BERT's final layer has the strongest encoding.

Uses patient-level median held-out McFadden pseudo-R² from the model/layer
encoding sweep. Tests are paired across patients:

  1. Friedman omnibus across all BERT layers, separately for self/other.
  2. One-sided Wilcoxon signed-rank tests: layer 12 > each earlier layer.
     BH-FDR is applied across the 12 pairwise tests within each condition.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests


PATIENT_CSV = Path(
    "/scratch/aniluchavez/ConvoDATAS/SemanticGLM/plots/"
    "all15_bestfixed_model_layer_performance_patient.csv"
)
OUT_PAIRWISE = Path(
    "/scratch/aniluchavez/ConvoDATAS/SemanticGLM/plots/"
    "bert_L12_vs_other_layers_paired_tests.csv"
)
OUT_OMNIBUS = Path(
    "/scratch/aniluchavez/ConvoDATAS/SemanticGLM/plots/"
    "bert_layer_omnibus_friedman.csv"
)


def main() -> int:
    df = pd.read_csv(PATIENT_CSV)
    bert = df.loc[df["model"].eq("bert-base")].copy()
    if bert.empty:
        raise RuntimeError(f"No bert-base rows found in {PATIENT_CSV}")

    last_layer = int(bert["layer"].max())
    pair_rows: list[dict] = []
    omni_rows: list[dict] = []

    for condition in sorted(bert["condition"].unique()):
        sub = bert.loc[bert["condition"].eq(condition)]
        piv = sub.pivot_table(
            index="patient", columns="layer", values="patient_median_r2"
        )
        layers = sorted(int(x) for x in piv.columns)
        piv = piv[layers].dropna()

        stat, p_omni = friedmanchisquare(*[piv[layer].values for layer in layers])
        layer_medians = piv.median(axis=0)
        best_layer = int(layer_medians.idxmax())

        omni_rows.append(
            {
                "model": "bert-base",
                "condition": condition,
                "n_patients": len(piv),
                "n_layers": len(layers),
                "last_layer": last_layer,
                "median_best_layer": best_layer,
                "median_best_r2": float(layer_medians.loc[best_layer]),
                "last_layer_median_r2": float(layer_medians.loc[last_layer]),
                "friedman_chi2": float(stat),
                "friedman_p": float(p_omni),
            }
        )

        x = piv[last_layer].values
        for layer in layers:
            if layer == last_layer:
                continue
            y = piv[layer].values
            diff = x - y
            try:
                w_greater, p_greater = wilcoxon(
                    diff, alternative="greater", zero_method="wilcox"
                )
            except ValueError:
                w_greater, p_greater = np.nan, np.nan
            try:
                w_two, p_two = wilcoxon(
                    diff, alternative="two-sided", zero_method="wilcox"
                )
            except ValueError:
                w_two, p_two = np.nan, np.nan

            pair_rows.append(
                {
                    "model": "bert-base",
                    "condition": condition,
                    "comparison": f"L{last_layer} > L{layer}",
                    "layer_ref": last_layer,
                    "layer_other": layer,
                    "n_patients": len(piv),
                    "median_L12": float(np.median(x)),
                    "median_other": float(np.median(y)),
                    "median_diff_L12_minus_other": float(np.median(diff)),
                    "mean_diff_L12_minus_other": float(np.mean(diff)),
                    "wilcoxon_stat_greater": float(w_greater),
                    "p_greater": float(p_greater),
                    "wilcoxon_stat_two_sided": float(w_two),
                    "p_two_sided": float(p_two),
                    "n_patients_L12_gt_other": int((diff > 0).sum()),
                    "n_patients_L12_lt_other": int((diff < 0).sum()),
                    "n_ties": int((diff == 0).sum()),
                }
            )

    pair = pd.DataFrame(pair_rows)
    for condition in sorted(pair["condition"].unique()):
        mask = pair["condition"].eq(condition)
        pair.loc[mask, "p_greater_fdr_bh"] = multipletests(
            pair.loc[mask, "p_greater"].values, method="fdr_bh"
        )[1]
        pair.loc[mask, "p_two_sided_fdr_bh"] = multipletests(
            pair.loc[mask, "p_two_sided"].values, method="fdr_bh"
        )[1]
        pair.loc[mask, "survives_fdr_greater_0p05"] = (
            pair.loc[mask, "p_greater_fdr_bh"] < 0.05
        )

    omnibus = pd.DataFrame(omni_rows)
    OUT_PAIRWISE.parent.mkdir(parents=True, exist_ok=True)
    pair.to_csv(OUT_PAIRWISE, index=False)
    omnibus.to_csv(OUT_OMNIBUS, index=False)

    print("Omnibus Friedman:")
    print(omnibus.to_string(index=False))
    print("\nPairwise L12 > other layers:")
    cols = [
        "condition",
        "comparison",
        "median_L12",
        "median_other",
        "median_diff_L12_minus_other",
        "p_greater",
        "p_greater_fdr_bh",
        "survives_fdr_greater_0p05",
        "n_patients_L12_gt_other",
        "n_patients_L12_lt_other",
    ]
    print(pair[cols].to_string(index=False))
    print(f"\nSaved: {OUT_OMNIBUS}")
    print(f"Saved: {OUT_PAIRWISE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
