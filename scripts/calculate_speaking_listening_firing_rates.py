from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


SPIKE_ROOT = Path("/scratch/aniluchavez/ConvoDATAS/SpikeWindows")
OUT_DIR = Path("/scratch/aniluchavez/hippocampal-speaker-semantics/results/firing_rate_speaking_listening")

PATIENTS = [
    "PTYEU_task147",
    "PTYEV_task37",
    "PTYEY_task86",
    "PTYEZ_task60",
    "PTYFA_task25",
    "PTYFC_task28",
    "PTYFF_task17",
    "PTYFG_task18",
    "PTYFI_task81",
    "PTYFK_task40",
    "PTYFM_task104",
    "PTYFP_task88",
    "PTYFR_task91",
    "PTYFS_task95",
    "PTYFU_task224",
]


def patient_to_folder_prefix(patient_id: str) -> str:
    return "pt" + patient_id[2:]


def find_window_dir(patient_id: str, window_tag: str) -> Path | None:
    short = patient_to_folder_prefix(patient_id)
    candidates = sorted(SPIKE_ROOT.glob(f"output_{short}_english_only_{window_tag}"))
    if candidates:
        return candidates[0]
    candidates = sorted(SPIKE_ROOT.glob(f"output_{patient_id}_english_only_{window_tag}"))
    return candidates[0] if candidates else None


def region_files_for_speaker(speaker_dir: Path) -> dict[str, Path]:
    return {p.name.replace("_spike_counts.npy", ""): p for p in speaker_dir.glob("*_spike_counts.npy")}


def compute_for_window(window_tag: str, window_length_ms: float, label: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    per_region_rows = []
    per_neuron_rows = []

    for patient_id in PATIENTS:
        window_dir = find_window_dir(patient_id, window_tag)
        if window_dir is None:
            print(f"missing {patient_id}: {window_tag}")
            continue

        speaker_dirs = sorted([p for p in window_dir.glob("Speaker*") if p.is_dir()])
        if not speaker_dirs:
            print(f"no speaker dirs {patient_id}: {window_dir}")
            continue

        regions = sorted(set().union(*(region_files_for_speaker(s).keys() for s in speaker_dirs)))
        for region in regions:
            self_path = window_dir / "Speaker1" / f"{region}_spike_counts.npy"
            if not self_path.exists():
                continue

            self_counts = np.load(self_path)
            other_mats = []
            for speaker_dir in speaker_dirs:
                if speaker_dir.name == "Speaker1":
                    continue
                path = speaker_dir / f"{region}_spike_counts.npy"
                if path.exists():
                    other_mats.append(np.load(path))
            if not other_mats:
                continue

            other_counts = np.vstack(other_mats)
            if self_counts.shape[1] != other_counts.shape[1]:
                print(f"shape mismatch {patient_id} {region}: self {self_counts.shape}, other {other_counts.shape}")
                continue

            seconds = window_length_ms / 1000.0
            self_fr_by_neuron = np.nanmean(self_counts, axis=0) / seconds
            other_fr_by_neuron = np.nanmean(other_counts, axis=0) / seconds

            per_region_rows.append({
                "window_label": label,
                "window_tag": window_tag,
                "patient_id": patient_id,
                "region": region,
                "n_neurons": int(self_counts.shape[1]),
                "n_self_events": int(self_counts.shape[0]),
                "n_other_events": int(other_counts.shape[0]),
                "self_fr_hz": float(np.mean(self_fr_by_neuron)),
                "other_fr_hz": float(np.mean(other_fr_by_neuron)),
                "other_minus_self_fr_hz": float(np.mean(other_fr_by_neuron) - np.mean(self_fr_by_neuron)),
            })

            for neuron_i, (self_fr, other_fr) in enumerate(zip(self_fr_by_neuron, other_fr_by_neuron), start=1):
                per_neuron_rows.append({
                    "window_label": label,
                    "window_tag": window_tag,
                    "patient_id": patient_id,
                    "region": region,
                    "neuron_i": neuron_i,
                    "self_fr_hz": float(self_fr),
                    "other_fr_hz": float(other_fr),
                    "other_minus_self_fr_hz": float(other_fr - self_fr),
                })

    per_region = pd.DataFrame(per_region_rows)
    per_neuron = pd.DataFrame(per_neuron_rows)
    if per_region.empty:
        return per_region, per_neuron, pd.DataFrame()

    pooled_rows = []
    for (window_label, window_tag, patient_id), sub in per_region.groupby(["window_label", "window_tag", "patient_id"], sort=False):
        weights = sub["n_neurons"].to_numpy()
        self_fr = np.average(sub["self_fr_hz"], weights=weights)
        other_fr = np.average(sub["other_fr_hz"], weights=weights)
        pooled_rows.append({
            "window_label": window_label,
            "window_tag": window_tag,
            "patient_id": patient_id,
            "region": "all_regions",
            "n_neurons": int(weights.sum()),
            "n_self_events": int(sub["n_self_events"].max()),
            "n_other_events": int(sub["n_other_events"].max()),
            "self_fr_hz": float(self_fr),
            "other_fr_hz": float(other_fr),
            "other_minus_self_fr_hz": float(other_fr - self_fr),
        })
    pooled = pd.DataFrame(pooled_rows)
    return per_region, per_neuron, pooled


def paired_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (window_label, region), sub in df.groupby(["window_label", "region"], sort=False):
        sub = sub.dropna(subset=["self_fr_hz", "other_fr_hz"])
        t_res = stats.ttest_rel(sub["other_fr_hz"], sub["self_fr_hz"])
        diffs = sub["other_minus_self_fr_hz"].to_numpy()
        rows.append({
            "window_label": window_label,
            "region": region,
            "n_patients": int(len(sub)),
            "total_neurons": int(sub["n_neurons"].sum()),
            "mean_self_fr_hz": float(sub["self_fr_hz"].mean()),
            "sem_self_fr_hz": float(stats.sem(sub["self_fr_hz"])) if len(sub) > 1 else np.nan,
            "mean_other_fr_hz": float(sub["other_fr_hz"].mean()),
            "sem_other_fr_hz": float(stats.sem(sub["other_fr_hz"])) if len(sub) > 1 else np.nan,
            "mean_other_minus_self_fr_hz": float(diffs.mean()),
            "sem_other_minus_self_fr_hz": float(stats.sem(diffs)) if len(sub) > 1 else np.nan,
            "paired_t_other_vs_self": float(t_res.statistic),
            "paired_p_other_vs_self": float(t_res.pvalue),
        })
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    windows = [
        (
            "tshift-150_tlen500_oshift+200_olen500",
            500.0,
            "typical_self_m150_len500_other_p200_len500",
        ),
        (
            "tshift+0_tlen500_oshift+0_olen500",
            500.0,
            "shared_onset_0_len500",
        ),
    ]

    all_region_frames = []
    all_neuron_frames = []
    all_pooled_frames = []
    for window_tag, window_length_ms, label in windows:
        per_region, per_neuron, pooled = compute_for_window(window_tag, window_length_ms, label)
        if per_region.empty:
            continue
        all_region_frames.append(per_region)
        all_neuron_frames.append(per_neuron)
        all_pooled_frames.append(pooled)

    per_region = pd.concat(all_region_frames, ignore_index=True)
    per_neuron = pd.concat(all_neuron_frames, ignore_index=True)
    pooled = pd.concat(all_pooled_frames, ignore_index=True)
    patient_region = pd.concat([per_region, pooled], ignore_index=True)
    summary = paired_summary(patient_region)

    per_neuron.to_csv(OUT_DIR / "speaking_listening_firing_rates_per_neuron.csv", index=False)
    patient_region.to_csv(OUT_DIR / "speaking_listening_firing_rates_patient_region.csv", index=False)
    summary.to_csv(OUT_DIR / "speaking_listening_firing_rates_paired_ttests.csv", index=False)

    print(summary.to_string(index=False))
    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
