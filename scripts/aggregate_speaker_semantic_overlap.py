#!/usr/bin/env python3
"""Aggregate joint speaker/semantic results and test overlap enrichment."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata


def fdr_bh(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    ranked = pvalues[order]
    q = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1].clip(0, 1)
    restored = np.empty_like(q)
    restored[order] = q
    return restored


def overlap_null(data, speaker_col, semantic_col, n_perm, rng):
    observed = int((data[speaker_col] & data[semantic_col]).sum())
    groups = list(data.groupby("patient").indices.values())
    expected = sum(
        data.loc[idx, speaker_col].sum()
        * data.loc[idx, semantic_col].sum()
        / len(idx)
        for idx in groups
    )
    null = np.zeros(n_perm, dtype=np.int32)
    speaker = data[speaker_col].to_numpy(bool)
    semantic = data[semantic_col].to_numpy(bool)
    for permutation in range(n_perm):
        shuffled = semantic.copy()
        for idx in groups:
            shuffled[idx] = rng.permutation(shuffled[idx])
        null[permutation] = int((speaker & shuffled).sum())
    p = (1 + np.sum(null >= observed)) / (n_perm + 1)
    return observed, float(expected), null, float(p)


def bootstrap_enrichment(data, speaker_col, semantic_col, n_boot, rng):
    patients = data["patient"].unique()
    values = []
    for _ in range(n_boot):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        observed = 0
        expected = 0.0
        for patient in sampled:
            group = data.loc[data["patient"].eq(patient)]
            observed += int((group[speaker_col] & group[semantic_col]).sum())
            expected += (
                group[speaker_col].sum()
                * group[semantic_col].sum()
                / len(group)
            )
        if expected > 0:
            values.append(observed / expected)
    if not values:
        return np.array([np.nan, np.nan])
    return np.quantile(values, [0.025, 0.975])


def within_patient_rank_correlation(data, n_perm, rng):
    ranked_speaker = np.empty(len(data))
    ranked_semantic = np.empty(len(data))
    groups = list(data.groupby("patient").indices.values())
    for idx in groups:
        ranked_speaker[idx] = rankdata(
            data.loc[idx, "unique_speaker_delta_ll_per_word"]
        )
        ranked_semantic[idx] = rankdata(
            data.loc[idx, "unique_semantic_delta_ll_per_word"]
        )
        ranked_speaker[idx] -= ranked_speaker[idx].mean()
        ranked_semantic[idx] -= ranked_semantic[idx].mean()
    observed = np.corrcoef(ranked_speaker, ranked_semantic)[0, 1]
    null = np.zeros(n_perm)
    for permutation in range(n_perm):
        shuffled = ranked_semantic.copy()
        for idx in groups:
            shuffled[idx] = rng.permutation(shuffled[idx])
        null[permutation] = np.corrcoef(ranked_speaker, shuffled)[0, 1]
    p = (1 + np.sum(null >= observed)) / (n_perm + 1)
    return float(observed), null, float(p)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_dir",
        type=Path,
        default=Path(
            "/scratch/aniluchavez/ConvoDATAS/SpeakerSemanticJoint/"
            "llama-3.1-8b_ctx200_L14_pc50_fixed_self0_other0_len500_"
            "blockcv_yfoldcirc_perm100"
        ),
    )
    parser.add_argument("--n_perm", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--window_label",
        default="Common 0–500 ms window",
        help="Window description shown in the aggregate figure title.",
    )
    args = parser.parse_args()

    files = sorted(args.run_dir.glob("PTY*_joint_results.csv"))
    if len(files) != 15:
        raise SystemExit(f"Expected 15 patient files, found {len(files)}")
    data = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    data["q_speaker_global"] = fdr_bh(data["p_speaker"])
    data["q_semantic_global"] = fdr_bh(data["p_semantic"])
    data["speaker_global_fdr_sig"] = (
        (data["q_speaker_global"] < 0.05)
        & (data["unique_speaker_delta_ll"] > 0)
    )
    data["semantic_global_fdr_sig"] = (
        (data["q_semantic_global"] < 0.05)
        & (data["unique_semantic_delta_ll"] > 0)
    )
    data["overlap_global_fdr"] = (
        data["speaker_global_fdr_sig"] & data["semantic_global_fdr_sig"]
    )

    criteria = [
        ("raw_p05", "speaker_raw_sig", "semantic_raw_sig"),
        ("patient_fdr", "speaker_fdr_sig", "semantic_fdr_sig"),
        (
            "global_fdr",
            "speaker_global_fdr_sig",
            "semantic_global_fdr_sig",
        ),
    ]
    rng = np.random.default_rng(args.seed)
    rows = []
    nulls = {}
    for label, speaker_col, semantic_col in criteria:
        observed, expected, null, p = overlap_null(
            data, speaker_col, semantic_col, args.n_perm, rng
        )
        ci_low, ci_high = bootstrap_enrichment(
            data, speaker_col, semantic_col, 10000, rng
        )
        enrichment = observed / expected if expected > 0 else np.nan
        rows.append(
            {
                "criterion": label,
                "n_neurons": len(data),
                "n_speaker": int(data[speaker_col].sum()),
                "n_semantic": int(data[semantic_col].sum()),
                "observed_overlap": observed,
                "expected_overlap": expected,
                "enrichment": enrichment,
                "enrichment_ci95_low": ci_low,
                "enrichment_ci95_high": ci_high,
                "permutation_p": p,
            }
        )
        nulls[label] = null

    correlation, correlation_null, correlation_p = (
        within_patient_rank_correlation(data, args.n_perm, rng)
    )
    rows.append(
        {
            "criterion": "continuous_within_patient_rank",
            "n_neurons": len(data),
            "n_speaker": np.nan,
            "n_semantic": np.nan,
            "observed_overlap": np.nan,
            "expected_overlap": np.nan,
            "enrichment": correlation,
            "enrichment_ci95_low": np.nan,
            "enrichment_ci95_high": np.nan,
            "permutation_p": correlation_p,
        }
    )
    summary = pd.DataFrame(rows)

    patient_rows = []
    for patient, group in data.groupby("patient"):
        patient_rows.append(
            {
                "patient": patient,
                "n_neurons": len(group),
                "speaker_raw": int(group["speaker_raw_sig"].sum()),
                "semantic_raw": int(group["semantic_raw_sig"].sum()),
                "overlap_raw": int(
                    (group["speaker_raw_sig"] & group["semantic_raw_sig"]).sum()
                ),
                "speaker_global_fdr": int(
                    group["speaker_global_fdr_sig"].sum()
                ),
                "semantic_global_fdr": int(
                    group["semantic_global_fdr_sig"].sum()
                ),
                "overlap_global_fdr": int(
                    group["overlap_global_fdr"].sum()
                ),
            }
        )
    patient_summary = pd.DataFrame(patient_rows)

    data.to_csv(args.run_dir / "all15_neuron_results.csv", index=False)
    summary.to_csv(args.run_dir / "all15_overlap_summary.csv", index=False)
    patient_summary.to_csv(
        args.run_dir / "all15_patient_overlap_counts.csv", index=False
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)
    for ax, criterion, title in zip(
        axes[:2],
        ["raw_p05", "global_fdr"],
        ["Raw p<0.05", "Global FDR q<0.05"],
    ):
        row = summary.loc[summary["criterion"].eq(criterion)].iloc[0]
        null = nulls[criterion]
        ax.hist(null, bins="auto", color="#9bb7d4", edgecolor="white")
        ax.axvline(
            row["observed_overlap"],
            color="#b2182b",
            linewidth=2.5,
            label=f"observed={int(row['observed_overlap'])}",
        )
        ax.axvline(
            row["expected_overlap"],
            color="black",
            linestyle="--",
            label=f"expected={row['expected_overlap']:.1f}",
        )
        ax.set_title(
            f"{title}\nenrichment={row['enrichment']:.2f}, "
            f"p={row['permutation_p']:.3g}",
            fontweight="bold",
        )
        ax.set_xlabel("Overlap under within-patient label shuffle")
        ax.set_ylabel("Permutations")
        ax.legend(frameon=False, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    axes[2].scatter(
        data["unique_speaker_delta_ll_per_word"],
        data["unique_semantic_delta_ll_per_word"],
        s=15,
        alpha=0.35,
        color="#4d4d4d",
    )
    axes[2].axhline(0, color="0.6", linewidth=0.8)
    axes[2].axvline(0, color="0.6", linewidth=0.8)
    axes[2].set_title(
        f"Continuous association\nwithin-patient rank r={correlation:.2f}, "
        f"p={correlation_p:.3g}",
        fontweight="bold",
    )
    axes[2].set_xlabel("Unique speaker ΔLL / word")
    axes[2].set_ylabel("Unique semantic ΔLL / word")
    axes[2].spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Do speaker-identity neurons also encode semantics above chance?\n"
        f"{args.window_label}; individual-speaker and semantic unique effects",
        fontweight="bold",
    )
    fig.savefig(
        args.run_dir / "all15_speaker_semantic_overlap.png",
        dpi=220,
        bbox_inches="tight",
    )
    fig.savefig(
        args.run_dir / "all15_speaker_semantic_overlap.pdf",
        bbox_inches="tight",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
