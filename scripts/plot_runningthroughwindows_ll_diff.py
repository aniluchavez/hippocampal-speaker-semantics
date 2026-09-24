#!/usr/bin/env python3
"""Plot real-minus-null log likelihood across completed shift-sweep patients."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.signal import savgol_filter

plt.rcParams.update({
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


PROJECT_DIR = Path("/scratch/aniluchavez/hippocampal-speaker-semantics")
RESULTS_DIR = PROJECT_DIR / "results" / "runningthroughwindows_fasttext_wiki"
OUT_DIR = RESULTS_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PAPER_PALETTE = {
    "other": "#0000FF",
    "self": "#FF0000",
}
SEM_FILL_COLORS = {
    "other": "#F8F8FF",
    "self": "#FFF8F8",
}
PAPER_LABELS = {
    "other": "listening",
    "self": "speaking",
}
EXCLUDE_YFS_PATIENT = "PTYFS_task95"


def robust_trim_patient_points(summary: pd.DataFrame, z_thresh: float = 6.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flag only very extreme patient-level means within each condition/shift using MAD z-scores."""
    pieces = []
    outlier_pieces = []
    for _, group in summary.groupby(["condition", "shift_ms"], sort=False):
        vals = group["mean_ll_diff"]
        median = vals.median()
        mad = (vals - median).abs().median()
        if mad == 0 or np.isnan(mad):
            keep = pd.Series(True, index=group.index)
            robust_z = pd.Series(0.0, index=group.index)
        else:
            robust_z = 0.6745 * (vals - median) / mad
            keep = robust_z.abs() <= z_thresh
        kept = group.loc[keep].copy()
        dropped = group.loc[~keep].copy()
        kept["robust_z"] = robust_z.loc[keep]
        dropped["robust_z"] = robust_z.loc[~keep]
        pieces.append(kept)
        if not dropped.empty:
            outlier_pieces.append(dropped)
    trimmed = pd.concat(pieces, ignore_index=True)
    outliers = (
        pd.concat(outlier_pieces, ignore_index=True)
        if outlier_pieces
        else summary.iloc[0:0].assign(robust_z=pd.Series(dtype=float))
    )
    return trimmed, outliers


def winsorize_patient_points(summary: pd.DataFrame, q: float = 0.05) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lightly cap patient-level means within each condition using global quantiles."""
    pieces = []
    changed = []
    for condition, group in summary.groupby("condition", sort=False):
        lo, hi = group["mean_ll_diff"].quantile([q, 1 - q])
        capped = group.copy()
        capped["mean_ll_diff_original"] = capped["mean_ll_diff"]
        capped["mean_ll_diff"] = capped["mean_ll_diff"].clip(lo, hi)
        capped["winsorized"] = capped["mean_ll_diff"] != capped["mean_ll_diff_original"]
        pieces.append(capped)
        changed.append(capped[capped["winsorized"]].copy())
    capped_summary = pd.concat(pieces, ignore_index=True)
    capped_points = pd.concat(changed, ignore_index=True)
    return capped_summary, capped_points


def winsorize_one_patient_against_others(
    summary: pd.DataFrame,
    patient: str,
    q: float = 0.025,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cap one patient's values to the non-patient range at each condition/shift."""
    adjusted = summary.copy()
    adjusted["mean_ll_diff_original"] = adjusted["mean_ll_diff"]
    adjusted["winsorized"] = False
    for (condition, shift_ms), group in summary.groupby(["condition", "shift_ms"], sort=False):
        target_idx = group.index[group["patient_prefix"] == patient]
        ref = group.loc[group["patient_prefix"] != patient, "mean_ll_diff"]
        if len(target_idx) == 0 or ref.empty:
            continue
        lo, hi = ref.quantile([q, 1 - q])
        old_vals = adjusted.loc[target_idx, "mean_ll_diff"]
        new_vals = old_vals.clip(lo, hi)
        adjusted.loc[target_idx, "mean_ll_diff"] = new_vals
        adjusted.loc[target_idx, "winsorized"] = new_vals != old_vals
    changed = adjusted[adjusted["winsorized"]].copy()
    return adjusted, changed


def drop_one_patient_outliers_against_others(
    summary: pd.DataFrame,
    patient: str,
    q: float = 0.025,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop one patient's values when outside the non-patient range at each condition/shift."""
    keep = pd.Series(True, index=summary.index)
    for (condition, shift_ms), group in summary.groupby(["condition", "shift_ms"], sort=False):
        target_idx = group.index[group["patient_prefix"] == patient]
        ref = group.loc[group["patient_prefix"] != patient, "mean_ll_diff"]
        if len(target_idx) == 0 or ref.empty:
            continue
        lo, hi = ref.quantile([q, 1 - q])
        target_vals = group.loc[target_idx, "mean_ll_diff"]
        keep.loc[target_idx] = target_vals.between(lo, hi)
    dropped = summary.loc[~keep].copy()
    retained = summary.loc[keep].copy()
    return retained, dropped


def smooth_curve(y: np.ndarray, window_length: int = 15, polyorder: int = 3) -> np.ndarray:
    window = min(window_length, len(y) if len(y) % 2 == 1 else len(y) - 1)
    if window <= polyorder:
        return y
    return savgol_filter(y, window_length=window, polyorder=polyorder)


def save_paperstyle_plot(
    pooled_values: pd.DataFrame,
    filename_stem: str,
    note: str,
    smooth_window: int = 15,
    show_raw_trace: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    for condition in ["other", "self"]:
        sub = pooled_values[pooled_values["condition"] == condition].sort_values("shift_ms")
        x = sub["shift_ms"].to_numpy()
        y = sub["mean_ll_diff"].to_numpy()
        sem = sub["sem_ll_diff"].fillna(0).to_numpy()
        y_smooth = smooth_curve(y, window_length=smooth_window)
        color = PAPER_PALETTE[condition]
        fill_color = SEM_FILL_COLORS[condition]
        label = PAPER_LABELS[condition]

        ax.fill_between(
            x,
            y - sem,
            y + sem,
            facecolor=fill_color,
            edgecolor=color,
            linewidth=0.45,
            alpha=1.0,
            zorder=1,
        )
        ax.plot(x, y - sem, color=color, linewidth=0.45, alpha=0.75, zorder=2)
        ax.plot(x, y + sem, color=color, linewidth=0.45, alpha=0.75, zorder=2)
        if show_raw_trace:
            ax.plot(x, y, color=color, linewidth=0.9, alpha=0.72, zorder=3)
        ax.plot(x, y_smooth, color=color, linewidth=3.2, solid_capstyle="round", label=label, zorder=4)

        peak_idx = int(np.nanargmax(y_smooth))
        peak_x = x[peak_idx]
        peak_y = y_smooth[peak_idx]
        arrow_base = max(ax.get_ylim()[0] + 0.6, peak_y - 3.0)
        ax.annotate(
            "",
            xy=(peak_x, peak_y + 0.25),
            xytext=(peak_x, arrow_base),
            arrowprops={"arrowstyle": "-", "color": color, "linewidth": 2.5},
        )
        ax.scatter(
            [peak_x],
            [peak_y + 0.65],
            marker="*",
            s=430,
            color=color,
            edgecolor=color,
            zorder=6,
        )

    ax.axhline(0, color="0.35", linewidth=0.8)
    ax.axvline(0, color="0.45", linewidth=2.0, linestyle="--")
    ax.set_xlim(-1000, 1000)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(min(-2, ymin), ymax + 0.8)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_xlabel("shift (ms)", fontsize=12)
    ax.set_ylabel("LLH diff (model minus shuffled)", fontsize=11)
    ax.text(-0.05, 1.02, "J", transform=ax.transAxes, fontsize=28, ha="left", va="bottom")
    ax.text(
        -650,
        ax.get_ylim()[0] + 0.25,
        note,
        fontsize=9,
        ha="left",
        va="bottom",
    )
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.04, 0.98), fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=9,
        length=7,
        width=1.6,
        direction="out",
        color="black",
        bottom=True,
        left=True,
        top=False,
        right=False,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        length=4,
        width=1.2,
        direction="out",
        color="black",
        bottom=True,
        left=True,
        top=False,
        right=False,
    )
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    fig.subplots_adjust(left=0.13, right=0.95, top=0.90, bottom=0.18)
    fig.savefig(OUT_DIR / f"{filename_stem}.png", dpi=300)
    fig.savefig(OUT_DIR / f"{filename_stem}.svg")
    fig.savefig(OUT_DIR / f"{filename_stem}.pdf")
    plt.close(fig)


def main():
    csvs = sorted(RESULTS_DIR.glob("regression_results_*_fasttext-wiki_ALLREGIONS_shifts.csv"))
    if not csvs:
        raise FileNotFoundError(f"No result CSVs found in {RESULTS_DIR}")

    frames = []
    for path in csvs:
        usecols = [
            "patient_prefix",
            "model_tag",
            "condition",
            "shift_ms",
            "region",
            "neuron",
            "ll_real",
            "ll_xshuf_mean",
            "ll_diff",
        ]
        df = pd.read_csv(path, usecols=lambda c: c in usecols)
        if "ll_diff" not in df.columns:
            df["ll_diff"] = df["ll_real"] - df["ll_xshuf_mean"]
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    summary = (
        df.groupby(["patient_prefix", "condition", "shift_ms"], as_index=False)
        .agg(
            mean_ll_diff=("ll_diff", "mean"),
            sem_ll_diff=("ll_diff", lambda x: x.sem()),
            n=("ll_diff", "size"),
        )
    )
    summary.to_csv(OUT_DIR / "ll_diff_by_patient_condition_shift.csv", index=False)

    sns.set_theme(style="white", context="talk")

    g = sns.relplot(
        data=summary,
        x="shift_ms",
        y="mean_ll_diff",
        hue="condition",
        col="patient_prefix",
        col_wrap=5,
        kind="line",
        marker=None,
        facet_kws={"sharey": False, "sharex": True},
        height=3.0,
        aspect=1.25,
    )
    g.set_axis_labels("Shift (ms)", "Mean LL(real) - LL(null)")
    g.set_titles("{col_name}")
    for ax in g.axes.flat:
        ax.axhline(0, color="0.35", linewidth=0.8, linestyle="--")
        ax.axvline(0, color="0.35", linewidth=0.8, linestyle=":")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    g.figure.tight_layout()
    g.figure.savefig(OUT_DIR / "ll_diff_by_patient_facets.png", dpi=300)
    g.figure.savefig(OUT_DIR / "ll_diff_by_patient_facets.svg")
    g.figure.savefig(OUT_DIR / "ll_diff_by_patient_facets.pdf")
    plt.close(g.figure)

    pooled = (
        summary.groupby(["condition", "shift_ms"], as_index=False)
        .agg(
            mean_ll_diff=("mean_ll_diff", "mean"),
            sem_ll_diff=("mean_ll_diff", lambda x: x.sem()),
            n_patients=("patient_prefix", "nunique"),
        )
    )
    pooled.to_csv(OUT_DIR / "ll_diff_pooled_across_completed_patients.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.lineplot(
        data=pooled,
        x="shift_ms",
        y="mean_ll_diff",
        hue="condition",
        marker=None,
        ax=ax,
    )
    ax.axhline(0, color="0.35", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="0.35", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Shift (ms)")
    ax.set_ylabel("Mean LL(real) - LL(null)")
    ax.set_title(f"Completed patients: n={pooled['n_patients'].max()}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ll_diff_pooled_across_completed_patients.png", dpi=300)
    fig.savefig(OUT_DIR / "ll_diff_pooled_across_completed_patients.svg")
    fig.savefig(OUT_DIR / "ll_diff_pooled_across_completed_patients.pdf")
    plt.close(fig)

    n_neurons = int(
        summary[summary["condition"] == "other"]
        .groupby("patient_prefix")["n"]
        .median()
        .sum()
    )
    save_paperstyle_plot(
        pooled,
        "ll_diff_pooled_across_completed_patients_with_sem",
        f"n={int(pooled['n_patients'].max())} patients, {n_neurons} neurons",
        smooth_window=19,
    )
    save_paperstyle_plot(
        pooled,
        "ll_diff_pooled_across_completed_patients_with_sem_mega_smoothed",
        f"n={int(pooled['n_patients'].max())} patients, {n_neurons} neurons",
        smooth_window=31,
        show_raw_trace=False,
    )

    no_yfs_summary = summary[summary["patient_prefix"] != EXCLUDE_YFS_PATIENT].copy()
    no_yfs_pooled = (
        no_yfs_summary.groupby(["condition", "shift_ms"], as_index=False)
        .agg(
            mean_ll_diff=("mean_ll_diff", "mean"),
            sem_ll_diff=("mean_ll_diff", lambda x: x.sem()),
            n_patients=("patient_prefix", "nunique"),
        )
    )
    no_yfs_pooled.to_csv(OUT_DIR / "ll_diff_pooled_paperstyle_exclude_yfs_sem.csv", index=False)
    save_paperstyle_plot(
        no_yfs_pooled,
        "ll_diff_pooled_paperstyle_exclude_yfs_sem",
        f"n={int(no_yfs_pooled['n_patients'].max())} patients, no YFS",
    )

    yfs_clean_summary, yfs_adjusted = winsorize_one_patient_against_others(summary, EXCLUDE_YFS_PATIENT)
    yfs_adjusted.to_csv(OUT_DIR / "ll_diff_pooled_paperstyle_yfs_outliers_adjusted.csv", index=False)
    yfs_clean_pooled = (
        yfs_clean_summary.groupby(["condition", "shift_ms"], as_index=False)
        .agg(
            mean_ll_diff=("mean_ll_diff", "mean"),
            sem_ll_diff=("mean_ll_diff", lambda x: x.sem()),
            n_patients=("patient_prefix", "nunique"),
        )
    )
    yfs_clean_pooled.to_csv(OUT_DIR / "ll_diff_pooled_paperstyle_yfs_cleaned_sem.csv", index=False)
    save_paperstyle_plot(
        yfs_clean_pooled,
        "ll_diff_pooled_paperstyle_yfs_cleaned_sem",
        f"n={int(yfs_clean_pooled['n_patients'].max())} patients, YFS capped",
    )

    yfs_drop_summary, yfs_dropped = drop_one_patient_outliers_against_others(summary, EXCLUDE_YFS_PATIENT)
    yfs_dropped.to_csv(OUT_DIR / "ll_diff_pooled_paperstyle_yfs_outliers_dropped.csv", index=False)
    yfs_drop_pooled = (
        yfs_drop_summary.groupby(["condition", "shift_ms"], as_index=False)
        .agg(
            mean_ll_diff=("mean_ll_diff", "mean"),
            sem_ll_diff=("mean_ll_diff", lambda x: x.sem()),
            n_patients=("patient_prefix", "nunique"),
        )
    )
    yfs_drop_pooled.to_csv(OUT_DIR / "ll_diff_pooled_paperstyle_yfs_outliers_dropped_sem.csv", index=False)
    save_paperstyle_plot(
        yfs_drop_pooled,
        "ll_diff_pooled_paperstyle_yfs_outliers_dropped_sem",
        "n=14-15 patients, YFS outlier points dropped",
    )

    trimmed_summary, outliers = winsorize_patient_points(summary)
    outliers.to_csv(OUT_DIR / "ll_diff_pooled_paperstyle_trimmed_outliers.csv", index=False)
    trimmed_pooled = (
        trimmed_summary.groupby(["condition", "shift_ms"], as_index=False)
        .agg(
            mean_ll_diff=("mean_ll_diff", "mean"),
            sem_ll_diff=("mean_ll_diff", lambda x: x.sem()),
            n_patients=("patient_prefix", "nunique"),
        )
    )
    trimmed_pooled.to_csv(OUT_DIR / "ll_diff_pooled_paperstyle_trimmed_sem.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    for condition in ["other", "self"]:
        sub = trimmed_pooled[trimmed_pooled["condition"] == condition].sort_values("shift_ms")
        x = sub["shift_ms"].to_numpy()
        y = sub["mean_ll_diff"].to_numpy()
        sem = sub["sem_ll_diff"].fillna(0).to_numpy()
        y_smooth = smooth_curve(y)
        color = PAPER_PALETTE[condition]
        label = PAPER_LABELS[condition]

        ax.fill_between(x, y - sem, y + sem, color=color, alpha=0.14, linewidth=0)
        ax.plot(x, y, color=color, linewidth=0.9, alpha=0.72)
        ax.plot(x, y_smooth, color=color, linewidth=3.2, solid_capstyle="round", label=label)

        peak_idx = int(np.nanargmax(y_smooth))
        peak_x = x[peak_idx]
        peak_y = y_smooth[peak_idx]
        arrow_base = max(ax.get_ylim()[0] + 0.6, peak_y - 3.0)
        ax.annotate(
            "",
            xy=(peak_x, peak_y + 0.25),
            xytext=(peak_x, arrow_base),
            arrowprops={"arrowstyle": "-", "color": color, "linewidth": 2.5},
        )
        ax.scatter(
            [peak_x],
            [peak_y + 0.65],
            marker="*",
            s=430,
            color=color,
            edgecolor=color,
            zorder=6,
        )

    ax.axhline(0, color="0.35", linewidth=0.8)
    ax.axvline(0, color="0.45", linewidth=2.0, linestyle="--")
    ax.set_xlim(-1000, 1000)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(min(-2, ymin), ymax + 0.8)
    ax.set_xlabel("shift (ms)", fontsize=12)
    ax.set_ylabel("LLH diff (model minus shuffled)", fontsize=11)
    ax.text(-0.05, 1.02, "J", transform=ax.transAxes, fontsize=28, ha="left", va="bottom")
    ax.text(
        -650,
        ax.get_ylim()[0] + 0.25,
        f"n={int(trimmed_pooled['n_patients'].max())} patients",
        fontsize=9,
        ha="left",
        va="bottom",
    )
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.04, 0.98), fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=9)
    fig.subplots_adjust(left=0.13, right=0.95, top=0.90, bottom=0.18)
    fig.savefig(OUT_DIR / "ll_diff_pooled_paperstyle_trimmed_sem.png", dpi=300)
    fig.savefig(OUT_DIR / "ll_diff_pooled_paperstyle_trimmed_sem.svg")
    fig.savefig(OUT_DIR / "ll_diff_pooled_paperstyle_trimmed_sem.pdf")
    plt.close(fig)

    print(f"Adjusted {len(yfs_adjusted)} YFS patient-condition-shift points")
    print(f"Dropped {len(yfs_dropped)} YFS patient-condition-shift outlier points")
    print(f"Winsorized {len(outliers)} patient-condition-shift points")
    print(f"Read {len(csvs)} patient CSVs")
    print(f"Saved plots to {OUT_DIR}")


if __name__ == "__main__":
    main()
