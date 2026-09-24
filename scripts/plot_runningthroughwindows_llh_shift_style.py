#!/usr/bin/env python3
"""Make paper-style LLH-diff shift plots from runningthroughwindows results."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.signal import savgol_filter


PROJECT_DIR = Path("/scratch/aniluchavez/hippocampal-speaker-semantics")
DATA_DIR = PROJECT_DIR / "results" / "runningthroughwindows_fasttext_wiki"
SAVE_DIR = DATA_DIR / "llh_shift_plots_eps"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
EXCLUDE_PATIENTS = set()

CUSTOM_PALETTE = {
    "self": "#C63E73",
    "other": "#5F7D37",
}
SMOOTH_WINDOW_LENGTH = 51
SMOOTH_POLYORDER = 3

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Nimbus Sans", "Arial", "DejaVu Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def patient_from_path(path: Path) -> str:
    name = path.name
    prefix = "regression_results_"
    suffix = "_fasttext-wiki_ALLREGIONS_shifts.csv"
    if name.startswith(prefix) and name.endswith(suffix):
        return name.removeprefix(prefix).removesuffix(suffix)
    return name.replace("regression_results_", "").replace("_ALLREGIONS_shifts.csv", "")


def make_plots(df_all: pd.DataFrame, save_dir: Path, label_suffix: str = "") -> None:
    save_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = sorted(DATA_DIR.glob("regression_results_*_fasttext-wiki_ALLREGIONS_shifts.csv"))
    stem_suffix = f"_{label_suffix}" if label_suffix else ""
    if label_suffix:
        print(f"\nPlot subset: {label_suffix}")

    # Average within patient first so the CI is across patients, not across neurons.
    df_patient_mean = (
        df_all.groupby(["patient_id", "region", "condition", "shift_ms"], as_index=False)["ll_diff"]
        .mean()
        .rename(columns={"ll_diff": "patient_mean_ll_diff"})
    )
    df_patient_mean.to_csv(save_dir / f"patient_mean_llh_diff_by_shift{stem_suffix}.csv", index=False)

    regions = sorted(df_patient_mean["region"].dropna().unique())
    maxima_rows = []
    smooth_maxima_rows = []

    for region in regions:
        df_region = df_patient_mean[df_patient_mean["region"] == region].copy()

        df_mean = (
            df_region.groupby(["condition", "shift_ms"], as_index=False)["patient_mean_ll_diff"]
            .mean()
            .rename(columns={"patient_mean_ll_diff": "ll_diff"})
        )

        max_vals = (
            df_mean.loc[df_mean.groupby("condition")["ll_diff"].idxmax(), ["condition", "shift_ms", "ll_diff"]]
            .sort_values("condition")
            .reset_index(drop=True)
        )
        max_vals["region"] = region
        maxima_rows.append(max_vals)

        print(f"\n=== Region: {region} ===")
        for _, row in max_vals.iterrows():
            print(
                f"{row['condition']}: max mean LLH diff = "
                f"{row['ll_diff']:.3f} at shift {row['shift_ms']} ms"
            )

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(
            data=df_region,
            x="shift_ms",
            y="patient_mean_ll_diff",
            hue="condition",
            palette=CUSTOM_PALETTE,
            errorbar="ci",
            marker="o",
            markersize=4,
            linewidth=2,
            ax=ax,
        )

        ax.axvline(0, color="gray", linestyle="--", label="Word onset")

        for _, row in max_vals.iterrows():
            cond = row["condition"]
            shift = row["shift_ms"]
            val = row["ll_diff"]
            ax.scatter(
                shift,
                val,
                color=CUSTOM_PALETTE.get(cond, "black"),
                s=90,
                edgecolor="black",
                zorder=5,
                label=f"{cond} max",
            )
            ax.text(shift, val, f"{val:.2f}", fontsize=10, ha="left", va="bottom")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)
        ax.set_title(f"Avg LLH Diff +/- 95% CI vs. Shift - Region: {region}", fontsize=16)
        ax.set_xlabel("Shift (ms)", fontsize=14)
        ax.set_ylabel("LLH Diff (model - shuffled)", fontsize=14)
        ax.tick_params(axis="both", labelsize=12)
        ax.legend(title="Condition", frameon=False)
        fig.tight_layout()

        stem = f"{region}_llh_shift_plot{stem_suffix}"
        fig.savefig(save_dir / f"{stem}.eps", format="eps", dpi=300)
        fig.savefig(save_dir / f"{stem}.png", dpi=300)
        fig.savefig(save_dir / f"{stem}.svg", format="svg")
        fig.savefig(save_dir / f"{stem}.pdf")
        plt.close(fig)

        smoothed_patient_vals = []
        for (patient_id, cond), sub in df_region.groupby(["patient_id", "condition"]):
            sub = sub.copy().sort_values("shift_ms")
            n = len(sub)
            window_length = min(SMOOTH_WINDOW_LENGTH, n if n % 2 == 1 else n - 1)
            if window_length <= SMOOTH_POLYORDER:
                sub["ll_diff_smooth"] = sub["patient_mean_ll_diff"]
            else:
                sub["ll_diff_smooth"] = savgol_filter(
                    sub["patient_mean_ll_diff"].to_numpy(),
                    window_length=window_length,
                    polyorder=SMOOTH_POLYORDER,
                )
            smoothed_patient_vals.append(sub)

        df_smooth_patients = pd.concat(smoothed_patient_vals, ignore_index=True)
        df_smooth = (
            df_smooth_patients.groupby(["condition", "shift_ms"], as_index=False)
            .agg(
                ll_diff_smooth=("ll_diff_smooth", "mean"),
                sem=("ll_diff_smooth", "sem"),
                n_patients=("patient_id", "nunique"),
            )
        )
        df_smooth["ci95"] = 1.96 * df_smooth["sem"].fillna(0)
        smooth_max_vals = (
            df_smooth.loc[
                df_smooth.groupby("condition")["ll_diff_smooth"].idxmax(),
                ["condition", "shift_ms", "ll_diff_smooth"],
            ]
            .sort_values("condition")
            .reset_index(drop=True)
        )
        smooth_max_vals["region"] = region
        smooth_maxima_rows.append(smooth_max_vals)

        print(f"\n=== Region: {region} smoothed ===")
        for _, row in smooth_max_vals.iterrows():
            print(
                f"{row['condition']}: max smoothed mean LLH diff = "
                f"{row['ll_diff_smooth']:.3f} at shift {row['shift_ms']} ms"
            )

        fig, ax = plt.subplots(figsize=(10, 5))
        for cond, sub in df_smooth.groupby("condition"):
            sub = sub.sort_values("shift_ms")
            color = CUSTOM_PALETTE.get(cond, "black")
            ax.plot(
                sub["shift_ms"],
                sub["ll_diff_smooth"],
                color=color,
                marker="o",
                markersize=4,
                linewidth=2,
                label=cond,
            )
            ax.fill_between(
                sub["shift_ms"],
                sub["ll_diff_smooth"] - sub["ci95"],
                sub["ll_diff_smooth"] + sub["ci95"],
                color=color,
                alpha=0.20,
                linewidth=0,
            )
        ax.axvline(0, color="gray", linestyle="--", label="Word onset")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)
        ax.set_title(f"Smoothed Avg LLH Diff vs. Shift - Region: {region}", fontsize=16)
        ax.set_xlabel("Shift (ms)", fontsize=14)
        ax.set_ylabel("Smoothed LLH Diff (model - shuffled)", fontsize=14)
        ax.tick_params(axis="both", labelsize=12)
        ax.legend(title="Condition", frameon=False)
        fig.tight_layout()

        smooth_stem = f"{region}_llh_shift_plot_smoothed{stem_suffix}"
        fig.savefig(save_dir / f"{smooth_stem}.eps", format="eps", dpi=300)
        fig.savefig(save_dir / f"{smooth_stem}.png", dpi=300)
        fig.savefig(save_dir / f"{smooth_stem}.svg", format="svg")
        fig.savefig(save_dir / f"{smooth_stem}.pdf")
        plt.close(fig)

        df_smooth.to_csv(save_dir / f"{region}_llh_shift_smoothed_values{stem_suffix}.csv", index=False)
        df_smooth_patients.to_csv(save_dir / f"{region}_llh_shift_smoothed_patient_values{stem_suffix}.csv", index=False)

    maxima = pd.concat(maxima_rows, ignore_index=True) if maxima_rows else pd.DataFrame()
    maxima.to_csv(save_dir / f"max_mean_llh_diff_by_region_condition{stem_suffix}.csv", index=False)
    smooth_maxima = pd.concat(smooth_maxima_rows, ignore_index=True) if smooth_maxima_rows else pd.DataFrame()
    smooth_maxima.to_csv(save_dir / f"max_smoothed_mean_llh_diff_by_region_condition{stem_suffix}.csv", index=False)


def make_zscore_plots(df_all: pd.DataFrame, save_dir: Path, label_suffix: str = "zscore") -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    stem_suffix = f"_{label_suffix}" if label_suffix else ""

    df_patient_mean = (
        df_all.groupby(["patient_id", "region", "condition", "shift_ms"], as_index=False)["ll_diff"]
        .mean()
        .rename(columns={"ll_diff": "patient_mean_ll_diff"})
    )

    groups = df_patient_mean.groupby(["patient_id", "region", "condition"])["patient_mean_ll_diff"]
    curve_mean = groups.transform("mean")
    curve_sd = groups.transform(lambda x: x.std(ddof=0))
    df_z = df_patient_mean.copy()
    df_z["ll_diff_z"] = (df_z["patient_mean_ll_diff"] - curve_mean) / curve_sd.replace(0, pd.NA)
    df_z["ll_diff_z"] = df_z["ll_diff_z"].fillna(0.0)
    df_z.to_csv(save_dir / f"patient_mean_llh_diff_by_shift{stem_suffix}.csv", index=False)

    regions = sorted(df_z["region"].dropna().unique())
    z_maxima_rows = []
    z_smooth_maxima_rows = []

    for region in regions:
        df_region = df_z[df_z["region"] == region].copy()
        df_mean = (
            df_region.groupby(["condition", "shift_ms"], as_index=False)["ll_diff_z"]
            .mean()
            .rename(columns={"ll_diff_z": "mean_ll_diff_z"})
        )

        max_vals = (
            df_mean.loc[
                df_mean.groupby("condition")["mean_ll_diff_z"].idxmax(),
                ["condition", "shift_ms", "mean_ll_diff_z"],
            ]
            .sort_values("condition")
            .reset_index(drop=True)
        )
        max_vals["region"] = region
        z_maxima_rows.append(max_vals)

        print(f"\n=== Region: {region} z-scored ===")
        for _, row in max_vals.iterrows():
            print(
                f"{row['condition']}: max mean z-scored LLH diff = "
                f"{row['mean_ll_diff_z']:.3f} at shift {row['shift_ms']} ms"
            )

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(
            data=df_region,
            x="shift_ms",
            y="ll_diff_z",
            hue="condition",
            palette=CUSTOM_PALETTE,
            errorbar="ci",
            marker="o",
            markersize=4,
            linewidth=2,
            ax=ax,
        )
        ax.axhline(0, color="0.35", linestyle=":", linewidth=0.8)
        ax.axvline(0, color="gray", linestyle="--", label="Word onset")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)
        ax.set_title(f"Patient-normalized LLH Diff +/- 95% CI vs. Shift - Region: {region}", fontsize=16)
        ax.set_xlabel("Shift (ms)", fontsize=14)
        ax.set_ylabel("Within-patient z-scored LLH diff", fontsize=14)
        ax.tick_params(axis="both", labelsize=12)
        ax.legend(title="Condition", frameon=False)
        fig.tight_layout()

        stem = f"{region}_llh_shift_plot{stem_suffix}"
        fig.savefig(save_dir / f"{stem}.eps", format="eps", dpi=300)
        fig.savefig(save_dir / f"{stem}.png", dpi=300)
        fig.savefig(save_dir / f"{stem}.svg", format="svg")
        fig.savefig(save_dir / f"{stem}.pdf")
        plt.close(fig)

        smooth_patient_vals = []
        for (patient_id, cond), sub in df_region.groupby(["patient_id", "condition"]):
            sub = sub.copy().sort_values("shift_ms")
            n = len(sub)
            window_length = min(SMOOTH_WINDOW_LENGTH, n if n % 2 == 1 else n - 1)
            if window_length <= SMOOTH_POLYORDER:
                sub["ll_diff_z_smooth"] = sub["ll_diff_z"]
            else:
                sub["ll_diff_z_smooth"] = savgol_filter(
                    sub["ll_diff_z"].to_numpy(),
                    window_length=window_length,
                    polyorder=SMOOTH_POLYORDER,
                )
            smooth_patient_vals.append(sub)

        df_smooth_patients = pd.concat(smooth_patient_vals, ignore_index=True)
        df_smooth = (
            df_smooth_patients.groupby(["condition", "shift_ms"], as_index=False)
            .agg(
                ll_diff_z_smooth=("ll_diff_z_smooth", "mean"),
                sem=("ll_diff_z_smooth", "sem"),
                n_patients=("patient_id", "nunique"),
            )
        )
        df_smooth["ci95"] = 1.96 * df_smooth["sem"].fillna(0)

        smooth_max_vals = (
            df_smooth.loc[
                df_smooth.groupby("condition")["ll_diff_z_smooth"].idxmax(),
                ["condition", "shift_ms", "ll_diff_z_smooth"],
            ]
            .sort_values("condition")
            .reset_index(drop=True)
        )
        smooth_max_vals["region"] = region
        z_smooth_maxima_rows.append(smooth_max_vals)

        print(f"\n=== Region: {region} z-scored smoothed ===")
        for _, row in smooth_max_vals.iterrows():
            print(
                f"{row['condition']}: max smoothed z-scored LLH diff = "
                f"{row['ll_diff_z_smooth']:.3f} at shift {row['shift_ms']} ms"
            )

        fig, ax = plt.subplots(figsize=(10, 5))
        for cond, sub in df_smooth.groupby("condition"):
            sub = sub.sort_values("shift_ms")
            color = CUSTOM_PALETTE.get(cond, "black")
            ax.plot(
                sub["shift_ms"],
                sub["ll_diff_z_smooth"],
                color=color,
                marker="o",
                markersize=4,
                linewidth=2,
                label=cond,
            )
            ax.fill_between(
                sub["shift_ms"],
                sub["ll_diff_z_smooth"] - sub["ci95"],
                sub["ll_diff_z_smooth"] + sub["ci95"],
                color=color,
                alpha=0.20,
                linewidth=0,
            )
        ax.axhline(0, color="0.35", linestyle=":", linewidth=0.8)
        ax.axvline(0, color="gray", linestyle="--", label="Word onset")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)
        ax.set_title(f"Smoothed Patient-normalized LLH Diff vs. Shift - Region: {region}", fontsize=16)
        ax.set_xlabel("Shift (ms)", fontsize=14)
        ax.set_ylabel("Smoothed within-patient z-scored LLH diff", fontsize=14)
        ax.tick_params(axis="both", labelsize=12)
        ax.legend(title="Condition", frameon=False)
        fig.tight_layout()

        smooth_stem = f"{region}_llh_shift_plot_smoothed{stem_suffix}"
        fig.savefig(save_dir / f"{smooth_stem}.eps", format="eps", dpi=300)
        fig.savefig(save_dir / f"{smooth_stem}.png", dpi=300)
        fig.savefig(save_dir / f"{smooth_stem}.svg", format="svg")
        fig.savefig(save_dir / f"{smooth_stem}.pdf")
        plt.close(fig)

        df_smooth.to_csv(save_dir / f"{region}_llh_shift_smoothed_values{stem_suffix}.csv", index=False)
        df_smooth_patients.to_csv(
            save_dir / f"{region}_llh_shift_smoothed_patient_values{stem_suffix}.csv",
            index=False,
        )

    z_maxima = pd.concat(z_maxima_rows, ignore_index=True) if z_maxima_rows else pd.DataFrame()
    z_maxima.to_csv(save_dir / f"max_mean_llh_diff_by_region_condition{stem_suffix}.csv", index=False)
    z_smooth_maxima = (
        pd.concat(z_smooth_maxima_rows, ignore_index=True) if z_smooth_maxima_rows else pd.DataFrame()
    )
    z_smooth_maxima.to_csv(
        save_dir / f"max_smoothed_mean_llh_diff_by_region_condition{stem_suffix}.csv",
        index=False,
    )


def make_minmax_plots(df_all: pd.DataFrame, save_dir: Path, label_suffix: str = "minmax") -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    stem_suffix = f"_{label_suffix}" if label_suffix else ""

    df_patient_mean = (
        df_all.groupby(["patient_id", "region", "condition", "shift_ms"], as_index=False)["ll_diff"]
        .mean()
        .rename(columns={"ll_diff": "patient_mean_ll_diff"})
    )

    groups = df_patient_mean.groupby(["patient_id", "region", "condition"])["patient_mean_ll_diff"]
    curve_min = groups.transform("min")
    curve_range = groups.transform(lambda x: x.max() - x.min())
    df_norm = df_patient_mean.copy()
    df_norm["ll_diff_minmax"] = (
        (df_norm["patient_mean_ll_diff"] - curve_min) / curve_range.replace(0, pd.NA)
    )
    df_norm["ll_diff_minmax"] = df_norm["ll_diff_minmax"].fillna(0.0)
    df_norm.to_csv(save_dir / f"patient_mean_llh_diff_by_shift{stem_suffix}.csv", index=False)

    regions = sorted(df_norm["region"].dropna().unique())
    maxima_rows = []
    smooth_maxima_rows = []

    for region in regions:
        df_region = df_norm[df_norm["region"] == region].copy()
        df_mean = (
            df_region.groupby(["condition", "shift_ms"], as_index=False)["ll_diff_minmax"]
            .mean()
            .rename(columns={"ll_diff_minmax": "mean_ll_diff_minmax"})
        )

        max_vals = (
            df_mean.loc[
                df_mean.groupby("condition")["mean_ll_diff_minmax"].idxmax(),
                ["condition", "shift_ms", "mean_ll_diff_minmax"],
            ]
            .sort_values("condition")
            .reset_index(drop=True)
        )
        max_vals["region"] = region
        maxima_rows.append(max_vals)

        print(f"\n=== Region: {region} min-max normalized ===")
        for _, row in max_vals.iterrows():
            print(
                f"{row['condition']}: max mean min-max LLH diff = "
                f"{row['mean_ll_diff_minmax']:.3f} at shift {row['shift_ms']} ms"
            )

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(
            data=df_region,
            x="shift_ms",
            y="ll_diff_minmax",
            hue="condition",
            palette=CUSTOM_PALETTE,
            errorbar="ci",
            marker="o",
            markersize=4,
            linewidth=2,
            ax=ax,
        )
        ax.axvline(0, color="gray", linestyle="--", label="Word onset")
        ax.set_ylim(bottom=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)
        ax.set_title(f"Patient-normalized LLH Diff +/- 95% CI vs. Shift - Region: {region}", fontsize=16)
        ax.set_xlabel("Shift (ms)", fontsize=14)
        ax.set_ylabel("Within-patient min-max LLH diff", fontsize=14)
        ax.tick_params(axis="both", labelsize=12)
        ax.legend(title="Condition", frameon=False)
        fig.tight_layout()

        stem = f"{region}_llh_shift_plot{stem_suffix}"
        fig.savefig(save_dir / f"{stem}.eps", format="eps", dpi=300)
        fig.savefig(save_dir / f"{stem}.png", dpi=300)
        fig.savefig(save_dir / f"{stem}.svg", format="svg")
        fig.savefig(save_dir / f"{stem}.pdf")
        plt.close(fig)

        smooth_patient_vals = []
        for (patient_id, cond), sub in df_region.groupby(["patient_id", "condition"]):
            sub = sub.copy().sort_values("shift_ms")
            n = len(sub)
            window_length = min(SMOOTH_WINDOW_LENGTH, n if n % 2 == 1 else n - 1)
            if window_length <= SMOOTH_POLYORDER:
                sub["ll_diff_minmax_smooth"] = sub["ll_diff_minmax"]
            else:
                sub["ll_diff_minmax_smooth"] = savgol_filter(
                    sub["ll_diff_minmax"].to_numpy(),
                    window_length=window_length,
                    polyorder=SMOOTH_POLYORDER,
                )
            sub["ll_diff_minmax_smooth"] = sub["ll_diff_minmax_smooth"].clip(0, 1)
            smooth_patient_vals.append(sub)

        df_smooth_patients = pd.concat(smooth_patient_vals, ignore_index=True)
        df_smooth = (
            df_smooth_patients.groupby(["condition", "shift_ms"], as_index=False)
            .agg(
                ll_diff_minmax_smooth=("ll_diff_minmax_smooth", "mean"),
                sem=("ll_diff_minmax_smooth", "sem"),
                n_patients=("patient_id", "nunique"),
            )
        )
        df_smooth["ci95"] = 1.96 * df_smooth["sem"].fillna(0)

        smooth_max_vals = (
            df_smooth.loc[
                df_smooth.groupby("condition")["ll_diff_minmax_smooth"].idxmax(),
                ["condition", "shift_ms", "ll_diff_minmax_smooth"],
            ]
            .sort_values("condition")
            .reset_index(drop=True)
        )
        smooth_max_vals["region"] = region
        smooth_maxima_rows.append(smooth_max_vals)

        print(f"\n=== Region: {region} min-max normalized smoothed ===")
        for _, row in smooth_max_vals.iterrows():
            print(
                f"{row['condition']}: max smoothed min-max LLH diff = "
                f"{row['ll_diff_minmax_smooth']:.3f} at shift {row['shift_ms']} ms"
            )

        fig, ax = plt.subplots(figsize=(10, 5))
        for cond, sub in df_smooth.groupby("condition"):
            sub = sub.sort_values("shift_ms")
            color = CUSTOM_PALETTE.get(cond, "black")
            ax.plot(
                sub["shift_ms"],
                sub["ll_diff_minmax_smooth"],
                color=color,
                marker="o",
                markersize=4,
                linewidth=2,
                label=cond,
            )
            ax.fill_between(
                sub["shift_ms"],
                (sub["ll_diff_minmax_smooth"] - sub["ci95"]).clip(lower=0),
                (sub["ll_diff_minmax_smooth"] + sub["ci95"]).clip(upper=1),
                color=color,
                alpha=0.20,
                linewidth=0,
            )
        ax.axvline(0, color="gray", linestyle="--", label="Word onset")
        ax.set_ylim(0, 1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)
        ax.set_title(f"Smoothed Patient-normalized LLH Diff vs. Shift - Region: {region}", fontsize=16)
        ax.set_xlabel("Shift (ms)", fontsize=14)
        ax.set_ylabel("Smoothed within-patient min-max LLH diff", fontsize=14)
        ax.tick_params(axis="both", labelsize=12)
        ax.legend(title="Condition", frameon=False)
        fig.tight_layout()

        smooth_stem = f"{region}_llh_shift_plot_smoothed{stem_suffix}"
        fig.savefig(save_dir / f"{smooth_stem}.eps", format="eps", dpi=300)
        fig.savefig(save_dir / f"{smooth_stem}.png", dpi=300)
        fig.savefig(save_dir / f"{smooth_stem}.svg", format="svg")
        fig.savefig(save_dir / f"{smooth_stem}.pdf")
        plt.close(fig)

        df_smooth.to_csv(save_dir / f"{region}_llh_shift_smoothed_values{stem_suffix}.csv", index=False)
        df_smooth_patients.to_csv(
            save_dir / f"{region}_llh_shift_smoothed_patient_values{stem_suffix}.csv",
            index=False,
        )

    maxima = pd.concat(maxima_rows, ignore_index=True) if maxima_rows else pd.DataFrame()
    maxima.to_csv(save_dir / f"max_mean_llh_diff_by_region_condition{stem_suffix}.csv", index=False)
    smooth_maxima = (
        pd.concat(smooth_maxima_rows, ignore_index=True) if smooth_maxima_rows else pd.DataFrame()
    )
    smooth_maxima.to_csv(
        save_dir / f"max_smoothed_mean_llh_diff_by_region_condition{stem_suffix}.csv",
        index=False,
    )


def main() -> None:
    csv_paths = sorted(DATA_DIR.glob("regression_results_*_fasttext-wiki_ALLREGIONS_shifts.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No shift result CSVs found in {DATA_DIR}")

    print("Loaded patients:", [patient_from_path(p) for p in csv_paths])

    df_list = []
    for path in csv_paths:
        df = pd.read_csv(path)
        df["patient_id"] = patient_from_path(path)
        df_list.append(df)

    df_all = pd.concat(df_list, ignore_index=True)
    df_all.to_csv(SAVE_DIR / "all_patients_resultswindowing_fasttext_wiki.csv", index=False)
    make_plots(df_all, SAVE_DIR)
    make_zscore_plots(df_all, SAVE_DIR)
    make_minmax_plots(df_all, SAVE_DIR)

    df_no_yfs = df_all[df_all["patient_id"] != "PTYFS_task95"].copy()
    make_plots(df_no_yfs, SAVE_DIR, label_suffix="exclude_PTYFS")

    print(f"\nSaved plots and summaries to: {SAVE_DIR}")


if __name__ == "__main__":
    main()
