#!/usr/bin/env python3
"""Joint speaker-identity and semantic encoding analysis.

For each hippocampal neuron, fit three nested Poisson-ridge models to spike
counts from one common fixed 0--500 ms window used for every speaker:

    semantic: semantic PCs
    speaker:  categorical individual-speaker indicators
    full:     semantic PCs + speaker indicators

Unique speaker encoding is held-out LL(full) - LL(semantic).
Unique semantic encoding is held-out LL(full) - LL(speaker).

Significance uses circular outcome permutations independently within each
held-out temporal fold. Because every permuted outcome remains in the fold
that was excluded from model training, the null does not leak training
responses into test responses. No refit is needed: this tests whether the
incremental out-of-fold prediction is aligned with the held-out neural data.
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import gammaln
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


PROJECT = Path("/scratch/aniluchavez/hippocampal-speaker-semantics")
EMBED_ROOT = Path("/scratch/aniluchavez/ConvoDATAS/EmbedCache")
SPIKE_ROOT = Path("/scratch/aniluchavez/ConvoDATAS/SpikeWindows")
RESULT_ROOT = Path("/scratch/aniluchavez/ConvoDATAS/SpeakerSemanticJoint")
DEFAULT_SPIKE_TAG = "fixed_self0_other0_len500"

PATIENTS = {
    "PTYEU_task147": "ptYEU_task147",
    "PTYEV_task37": "ptYEV_task37",
    "PTYEY_task86": "ptYEY_task86",
    "PTYEZ_task60": "ptYEZ_task60",
    "PTYFA_task25": "ptYFA_task25",
    "PTYFC_task28": "ptYFC_task28",
    "PTYFF_task17": "ptYFF_task17",
    "PTYFG_task18": "ptYFG_task18",
    "PTYFI_task81": "ptYFI_task81",
    "PTYFK_task40": "ptYFK_task40",
    "PTYFM_task104": "ptYFM_task104",
    "PTYFP_task88": "ptYFP_task88",
    "PTYFR_task91": "ptYFR_task91",
    "PTYFS_task95": "ptYFS_task95",
    "PTYFU_task224": "ptYFU_task224",
}

ALPHAS = np.logspace(-2, 4, 20)
N_OUTER = 5
N_INNER = 3
ETA_CLIP = 20.0
LBFGS_ITER = 30
FDR_ALPHA = 0.05


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient", required=True, choices=sorted(PATIENTS))
    parser.add_argument("--model", default="llama-3.1-8b")
    parser.add_argument("--context_tag", default="_ctx200")
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--n_components", type=int, default=50)
    parser.add_argument("--n_perm", type=int, default=1000)
    parser.add_argument("--spike_tag", default=DEFAULT_SPIKE_TAG)
    parser.add_argument(
        "--window_type", choices=["fixed", "varwin", "worddur"], default="fixed"
    )
    parser.add_argument(
        "--timing_controls",
        action="store_true",
        help="Add duration, pre/post gap, and session-time nuisances to every model",
    )
    parser.add_argument("--min_speaker_words", type=int, default=100)
    parser.add_argument("--min_spikes", type=int, default=20)
    parser.add_argument(
        "--cv_mode",
        choices=["block", "shuffle", "purged_shuffle"],
        default="block",
    )
    parser.add_argument(
        "--embargo_ms",
        type=float,
        default=500.0,
        help="Onset gap around test words for purged_shuffle CV.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def temporal_splits(n: int, k: int):
    """Return contiguous train/test index pairs covering every row once."""
    folds = np.array_split(np.arange(n), k)
    all_idx = np.arange(n)
    return [(np.setdiff1d(all_idx, test), test) for test in folds]


def shuffled_splits(n: int, k: int, seed: int):
    """Random K-fold splits returned as integer index pairs."""
    return list(
        KFold(n_splits=k, shuffle=True, random_state=seed).split(np.arange(n))
    )


def purged_shuffle_splits(
    onset: np.ndarray,
    k: int,
    embargo_ms: float,
    seed: int,
):
    """Random test folds with temporally adjacent training rows removed."""
    onset = np.asarray(onset, dtype=float)
    splits = []
    for train, test in shuffled_splits(len(onset), k, seed):
        test_onsets = np.sort(onset[test])
        candidate = onset[train]
        positions = np.searchsorted(test_onsets, candidate)
        left = np.clip(positions - 1, 0, len(test_onsets) - 1)
        right = np.clip(positions, 0, len(test_onsets) - 1)
        nearest = np.minimum(
            np.abs(candidate - test_onsets[left]),
            np.abs(candidate - test_onsets[right]),
        )
        splits.append((train[nearest > embargo_ms], test))
    return splits


def make_splits(
    onset: np.ndarray,
    k: int,
    mode: str,
    embargo_ms: float,
    seed: int,
):
    if mode == "shuffle":
        return shuffled_splits(len(onset), k, seed)
    if mode == "purged_shuffle":
        return purged_shuffle_splits(onset, k, embargo_ms, seed)
    return temporal_splits(len(onset), k)


def fdr_bh(pvalues: np.ndarray):
    pvalues = np.asarray(pvalues, dtype=float)
    valid = np.isfinite(pvalues)
    adjusted = np.full(len(pvalues), np.nan)
    rejected = np.zeros(len(pvalues), dtype=bool)
    if not valid.any():
        return adjusted, rejected
    p = pvalues[valid]
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1].clip(0, 1)
    restored = np.empty_like(q)
    restored[order] = q
    adjusted[valid] = restored
    rejected[valid] = restored < FDR_ALPHA
    return adjusted, rejected


@torch.no_grad()
def gpu_pca(x: np.ndarray, n_components: int, device: torch.device):
    tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
    mean = tensor.mean(0)
    centered = tensor - mean
    rank = min(n_components + 10, min(centered.shape))
    projection = centered @ torch.randn(
        centered.shape[1], rank, device=device
    )
    for _ in range(2):
        projection = centered @ (centered.T @ projection)
    q, _ = torch.linalg.qr(projection)
    _, _, vt = torch.linalg.svd(q.T @ centered, full_matrices=False)
    return (centered @ vt[:n_components].T).cpu().numpy().astype(np.float32)


def gpu_fit(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha,
    device: torch.device,
    offset: torch.Tensor | None = None,
):
    """Fit batches of Poisson-ridge models.

    x: (batch, observations, predictors)
    y: (batch, observations, neurons)
    alpha: scalar or one value per batch
    """
    batch, n_obs, _ = x.shape
    n_neurons = y.shape[2]
    design = torch.cat(
        [torch.ones(batch, n_obs, 1, device=device), x], dim=2
    )
    beta = torch.zeros(
        batch, n_neurons, design.shape[2], device=device, requires_grad=True
    )
    optimizer = torch.optim.LBFGS(
        [beta], max_iter=LBFGS_ITER, line_search_fn="strong_wolfe"
    )
    if isinstance(alpha, torch.Tensor):
        penalty = alpha.to(device).view(batch, 1, 1)
    else:
        penalty = float(alpha)
    normalizer = max(float(y.sum().item()), 1.0)

    def closure():
        optimizer.zero_grad()
        eta = torch.bmm(design, beta.transpose(1, 2))
        if offset is not None:
            eta = eta + offset
        eta = torch.clamp(eta, -ETA_CLIP, ETA_CLIP)
        loss = (
            (torch.exp(eta) - y * eta).sum()
            + (penalty * beta[:, :, 1:].pow(2)).sum()
        ) / normalizer
        loss.backward()
        return loss

    optimizer.step(closure)
    fitted = beta.detach()
    return fitted[:, :, 1:], fitted[:, :, :1]


def fit_oof_eta(
    designs: dict[str, np.ndarray],
    y: np.ndarray,
    device: torch.device,
    offset: np.ndarray | None = None,
    onset: np.ndarray | None = None,
    cv_mode: str = "block",
    embargo_ms: float = 500.0,
    seed: int = 42,
):
    """Nested CV with independent alpha selection for each model."""
    n_rows, n_neurons = y.shape
    if onset is None:
        onset = np.arange(n_rows, dtype=float)
    outer = make_splits(
        onset, N_OUTER, cv_mode, embargo_ms, seed
    )
    eta_oof = {
        name: np.full((n_rows, n_neurons), np.nan, dtype=np.float32)
        for name in designs
    }
    best_alpha_rows = []

    for fold, (train, test) in enumerate(outer):
        y_train = torch.as_tensor(y[train], dtype=torch.float32, device=device)
        for name, design_np in designs.items():
            x_train = design_np[train]
            inner_scores = np.zeros((len(ALPHAS), n_neurons), dtype=np.float64)
            inner_splits = make_splits(
                onset[train],
                N_INNER,
                cv_mode,
                embargo_ms,
                seed + 1000 * fold,
            )
            for inner_train, inner_valid in inner_splits:
                x_inner = torch.as_tensor(
                    x_train[inner_train], dtype=torch.float32, device=device
                )
                y_inner = y_train[inner_train]
                x_batches = x_inner.unsqueeze(0).expand(
                    len(ALPHAS), -1, -1
                ).contiguous()
                y_batches = y_inner.unsqueeze(0).expand(
                    len(ALPHAS), -1, -1
                ).contiguous()
                beta, bias = gpu_fit(
                    x_batches,
                    y_batches,
                    torch.as_tensor(ALPHAS, dtype=torch.float32),
                    device,
                    (
                        torch.as_tensor(
                            offset[train][inner_train],
                            dtype=torch.float32,
                            device=device,
                        ).view(1, -1, 1)
                        if offset is not None
                        else None
                    ),
                )
                with torch.no_grad():
                    x_valid = torch.as_tensor(
                        x_train[inner_valid],
                        dtype=torch.float32,
                        device=device,
                    )
                    eta = torch.einsum("np,amp->anm", x_valid, beta)
                    eta = eta + bias.transpose(1, 2)
                    if offset is not None:
                        eta = eta + torch.as_tensor(
                            offset[train][inner_valid],
                            dtype=torch.float32,
                            device=device,
                        ).view(1, -1, 1)
                    eta = torch.clamp(eta, -ETA_CLIP, ETA_CLIP)
                    y_valid = y_train[inner_valid].unsqueeze(0)
                    inner_scores += (
                        y_valid * eta - torch.exp(eta)
                    ).sum(1).cpu().numpy()

            best_index = np.argmax(inner_scores, axis=0)
            x_train_t = torch.as_tensor(
                x_train[None], dtype=torch.float32, device=device
            )
            beta_final = torch.zeros(
                1, n_neurons, design_np.shape[1], device=device
            )
            bias_final = torch.zeros(1, n_neurons, 1, device=device)
            for alpha_index in np.unique(best_index):
                neuron_ids = np.flatnonzero(best_index == alpha_index)
                beta, bias = gpu_fit(
                    x_train_t,
                    y_train[None, :, neuron_ids],
                    float(ALPHAS[alpha_index]),
                    device,
                    (
                        torch.as_tensor(
                            offset[train],
                            dtype=torch.float32,
                            device=device,
                        ).view(1, -1, 1)
                        if offset is not None
                        else None
                    ),
                )
                beta_final[0, neuron_ids] = beta[0]
                bias_final[0, neuron_ids] = bias[0]

            with torch.no_grad():
                x_test = torch.as_tensor(
                    design_np[test], dtype=torch.float32, device=device
                )
                eta = x_test @ beta_final[0].T + bias_final[0, :, 0]
                if offset is not None:
                    eta = eta + torch.as_tensor(
                        offset[test], dtype=torch.float32, device=device
                    )[:, None]
                eta_oof[name][test] = (
                    torch.clamp(eta, -ETA_CLIP, ETA_CLIP).cpu().numpy()
                )
            best_alpha_rows.extend(
                {
                    "fold": fold,
                    "model": name,
                    "neuron_idx": int(neuron),
                    "alpha": float(ALPHAS[index]),
                }
                for neuron, index in enumerate(best_index)
            )
            del beta_final, bias_final
            torch.cuda.empty_cache()
        del y_train
    return eta_oof, outer, pd.DataFrame(best_alpha_rows)


def poisson_ll(y: np.ndarray, eta: np.ndarray):
    mu = np.exp(np.clip(eta, -ETA_CLIP, ETA_CLIP))
    return (y * eta - mu - gammaln(y + 1)).sum(axis=0)


def delta_ll(y: np.ndarray, eta_full: np.ndarray, eta_baseline: np.ndarray):
    """Per-neuron LL(full)-LL(baseline); factorial terms cancel."""
    mu_full = np.exp(np.clip(eta_full, -ETA_CLIP, ETA_CLIP))
    mu_base = np.exp(np.clip(eta_baseline, -ETA_CLIP, ETA_CLIP))
    return (
        y * (eta_full - eta_baseline) - (mu_full - mu_base)
    ).sum(axis=0)


def fold_circular_permutation(
    y: np.ndarray,
    outer_splits,
    rng: np.random.Generator,
):
    permuted = np.empty_like(y)
    for _, test in outer_splits:
        n = len(test)
        lower = max(1, n // 4)
        upper = max(lower + 1, 3 * n // 4)
        lag = int(rng.integers(lower, upper))
        permuted[test] = np.roll(y[test], lag, axis=0)
    return permuted


def load_aligned_data(args):
    patient_stub = PATIENTS[args.patient]
    spike_dir = (
        SPIKE_ROOT
        / f"output_{patient_stub}_english_only_{args.spike_tag}"
    )
    xlsx_files = list(spike_dir.glob("*_with_regress_dur.xlsx"))
    if not xlsx_files:
        raise FileNotFoundError(f"No transcript metadata in {spike_dir}")
    transcript = pd.read_excel(xlsx_files[0], keep_default_na=False)
    speaker_cols = sorted(
        [c for c in transcript if str(c).startswith("Speaker")],
        key=lambda c: int(str(c).replace("Speaker", "") or 999),
    )

    def nonempty(value):
        # Match semantic_glm.py's established alignment convention. Some
        # historical transcripts use "xxx" as a speaker-column placeholder,
        # and those rows were included when the spike matrices were generated.
        return str(value).strip().lower() not in ("", "nan")

    membership = {
        speaker: transcript[speaker].map(nonempty).to_numpy()
        for speaker in speaker_cols
    }
    assignment = np.full(len(transcript), "", dtype=object)
    for speaker in speaker_cols:
        assignment[(assignment == "") & membership[speaker]] = speaker

    matrices = {}
    for speaker in speaker_cols:
        filename = (
            "hippocampus_worddur_spike_counts.npy"
            if args.window_type == "worddur"
            else "hippocampus_spike_counts.npy"
        )
        path = spike_dir / speaker / filename
        if path.exists():
            matrices[speaker] = np.load(path)
    if not matrices:
        raise FileNotFoundError(f"No hippocampal spike matrices in {spike_dir}")

    n_neurons = next(iter(matrices.values())).shape[1]
    y_all = np.full((len(transcript), n_neurons), np.nan, dtype=np.float32)
    for speaker, matrix in matrices.items():
        row_ids = np.flatnonzero(membership[speaker])
        if len(row_ids) != len(matrix):
            raise ValueError(
                f"{speaker} row mismatch: transcript={len(row_ids)}, spikes={len(matrix)}"
            )
        # A few historical rows populate more than one speaker column. The
        # canonical assignment takes the first populated column. Keep matrix
        # cursors aligned to every membership row, but only write rows assigned
        # to this speaker.
        assigned_positions = np.flatnonzero(assignment[row_ids] == speaker)
        assigned_rows = row_ids[assigned_positions]
        y_all[assigned_rows] = matrix[assigned_positions]

    embed_path = (
        EMBED_ROOT
        / f"{args.patient}_{args.model}{args.context_tag}_word_emb_layers.npy"
    )
    raw_embedding = np.load(embed_path, mmap_mode="r")[args.layer].astype(
        np.float32
    )
    if len(raw_embedding) != len(transcript):
        raise ValueError(
            f"Embedding/transcript mismatch: {len(raw_embedding)} vs {len(transcript)}"
        )

    speaker_counts = pd.Series(assignment).value_counts()
    retained_speakers = sorted(
        speaker_counts[
            (speaker_counts.index != "")
            & (speaker_counts >= args.min_speaker_words)
        ].index
    )
    keep = (
        np.isin(assignment, retained_speakers)
        & np.isfinite(y_all).all(axis=1)
    )
    if len(retained_speakers) < 2:
        raise ValueError(
            f"Only {len(retained_speakers)} speakers have at least "
            f"{args.min_speaker_words} words"
        )
    onset = pd.to_numeric(transcript["onset"], errors="coerce").to_numpy(float)
    acoustic_offset = pd.to_numeric(
        transcript["offset"], errors="coerce"
    ).to_numpy(float)
    word_duration = np.clip(acoustic_offset - onset, 1.0, None)
    duration_column = (
        transcript.get("word_dur")
        if args.window_type == "worddur" and "word_dur" in transcript
        else transcript.get(
            "regress_dur", pd.Series(np.full(len(transcript), 500.0))
        )
    )
    window_duration = pd.to_numeric(
        duration_column,
        errors="coerce",
    ).to_numpy(float)
    window_duration = np.where(
        np.isfinite(window_duration) & (window_duration > 0),
        window_duration,
        500.0,
    )
    previous_offset = np.r_[acoustic_offset[0], acoustic_offset[:-1]]
    next_onset = np.r_[onset[1:], onset[-1]]
    pre_gap = np.clip(onset - previous_offset, 0, 5000)
    post_gap = np.clip(next_onset - acoustic_offset, 0, 5000)
    session_time = (onset - np.nanmin(onset)) / max(
        np.nanmax(onset) - np.nanmin(onset), 1.0
    )
    timing = np.column_stack(
        [
            np.log1p(word_duration),
            np.log1p(pre_gap),
            np.log1p(post_gap),
            session_time,
            session_time**2,
        ]
    ).astype(np.float32)
    timing[~np.isfinite(timing)] = 0.0
    log_exposure = (
        np.log(window_duration / 1000.0).astype(np.float32)
        if args.window_type in ("varwin", "worddur")
        else None
    )
    return (
        raw_embedding[keep],
        y_all[keep],
        assignment[keep],
        retained_speakers,
        timing[keep],
        log_exposure[keep] if log_exposure is not None else None,
        onset[keep],
    )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_tag = (
        f"{args.model}{args.context_tag}_L{args.layer:02d}_pc{args.n_components}"
        f"_{args.spike_tag}_{args.cv_mode}"
        f"{f'_emb{args.embargo_ms:g}ms' if args.cv_mode == 'purged_shuffle' else ''}"
        f"_yfoldcirc_perm{args.n_perm}"
        f"{'_timingctrl' if args.timing_controls else ''}"
    )
    out_dir = RESULT_ROOT / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{args.patient}_joint_results.csv"
    pkl_path = out_dir / f"{args.patient}_joint_nulls.pkl"
    alpha_path = out_dir / f"{args.patient}_alphas.csv"
    if csv_path.exists() and not args.force:
        print(f"[SKIP] {csv_path}")
        return

    print(f"device={device} patient={args.patient} run={run_tag}", flush=True)
    start = time.time()
    (
        raw_embedding,
        y,
        speakers,
        retained,
        timing_raw,
        log_exposure,
        onset,
    ) = load_aligned_data(args)
    spike_ok = y.sum(axis=0) >= args.min_spikes
    y = y[:, spike_ok].astype(np.float32)
    original_neuron_ids = np.flatnonzero(spike_ok)
    print(
        f"rows={len(y)} neurons={y.shape[1]} speakers={retained}",
        flush=True,
    )

    semantic_pcs = gpu_pca(raw_embedding, args.n_components, device)
    semantic = StandardScaler().fit_transform(semantic_pcs).astype(np.float32)
    onehot = pd.get_dummies(
        pd.Categorical(speakers, categories=retained), drop_first=True
    ).to_numpy(dtype=np.float32)
    speaker = StandardScaler().fit_transform(onehot).astype(np.float32)
    if args.timing_controls:
        timing = StandardScaler().fit_transform(timing_raw).astype(np.float32)
    else:
        timing = np.zeros((len(y), 0), dtype=np.float32)
    semantic_design = np.hstack([timing, semantic]).astype(np.float32)
    speaker_design = np.hstack([timing, speaker]).astype(np.float32)
    full = np.hstack([timing, semantic, speaker]).astype(np.float32)
    designs = {
        "semantic": semantic_design,
        "speaker": speaker_design,
        "full": full,
    }
    if args.timing_controls:
        designs["timing"] = timing

    eta, outer, alpha_df = fit_oof_eta(
        designs,
        y,
        device,
        offset=log_exposure,
        onset=onset,
        cv_mode=args.cv_mode,
        embargo_ms=args.embargo_ms,
        seed=args.seed,
    )
    eta_null = np.full_like(y, np.nan, dtype=np.float32)
    for train, test in outer:
        if log_exposure is None:
            train_mean = np.clip(y[train].mean(axis=0), 1e-8, None)
            eta_null[test] = np.log(train_mean)
        else:
            exposure = np.exp(log_exposure[train]).sum()
            train_rate = np.clip(y[train].sum(axis=0) / exposure, 1e-8, None)
            eta_null[test] = np.log(train_rate)[None, :] + log_exposure[test, None]

    ll_null = poisson_ll(y, eta_null)
    ll_semantic = poisson_ll(y, eta["semantic"])
    ll_speaker = poisson_ll(y, eta["speaker"])
    ll_full = poisson_ll(y, eta["full"])
    ll_timing = (
        poisson_ll(y, eta["timing"]) if "timing" in eta else ll_null.copy()
    )
    unique_speaker = delta_ll(y, eta["full"], eta["semantic"])
    unique_semantic = delta_ll(y, eta["full"], eta["speaker"])

    rng = np.random.default_rng(args.seed)
    null_speaker = np.zeros((args.n_perm, y.shape[1]), dtype=np.float32)
    null_semantic = np.zeros_like(null_speaker)
    for permutation in range(args.n_perm):
        y_perm = fold_circular_permutation(y, outer, rng)
        null_speaker[permutation] = delta_ll(
            y_perm, eta["full"], eta["semantic"]
        )
        null_semantic[permutation] = delta_ll(
            y_perm, eta["full"], eta["speaker"]
        )

    p_speaker = (
        1 + (null_speaker >= unique_speaker).sum(axis=0)
    ) / (args.n_perm + 1)
    p_semantic = (
        1 + (null_semantic >= unique_semantic).sum(axis=0)
    ) / (args.n_perm + 1)
    q_speaker, fdr_speaker = fdr_bh(p_speaker)
    q_semantic, fdr_semantic = fdr_bh(p_semantic)
    raw_speaker = (p_speaker < 0.05) & (unique_speaker > 0)
    raw_semantic = (p_semantic < 0.05) & (unique_semantic > 0)
    fdr_speaker &= unique_speaker > 0
    fdr_semantic &= unique_semantic > 0

    results = pd.DataFrame(
        {
            "patient": args.patient,
            "neuron_idx": original_neuron_ids,
            "n_words": len(y),
            "n_speakers": len(retained),
            "speakers": "|".join(retained),
            "r2_semantic": 1 - ll_semantic / ll_null,
            "r2_speaker": 1 - ll_speaker / ll_null,
            "r2_full": 1 - ll_full / ll_null,
            "r2_timing": 1 - ll_timing / ll_null,
            "unique_speaker_delta_ll": unique_speaker,
            "unique_semantic_delta_ll": unique_semantic,
            "unique_speaker_delta_ll_per_word": unique_speaker / len(y),
            "unique_semantic_delta_ll_per_word": unique_semantic / len(y),
            "p_speaker": p_speaker,
            "q_speaker": q_speaker,
            "speaker_raw_sig": raw_speaker,
            "speaker_fdr_sig": fdr_speaker,
            "p_semantic": p_semantic,
            "q_semantic": q_semantic,
            "semantic_raw_sig": raw_semantic,
            "semantic_fdr_sig": fdr_semantic,
            "overlap_raw": raw_speaker & raw_semantic,
            "overlap_fdr": fdr_speaker & fdr_semantic,
        }
    )
    results.to_csv(csv_path, index=False)
    alpha_df.to_csv(alpha_path, index=False)
    with pkl_path.open("wb") as handle:
        pickle.dump(
            {
                "null_speaker": null_speaker,
                "null_semantic": null_semantic,
                "retained_speakers": retained,
                "timing_feature_names": [
                    "log_word_duration",
                    "log_pre_gap",
                    "log_post_gap",
                    "session_time",
                    "session_time_squared",
                ],
                "outer_test_indices": [test for _, test in outer],
                "args": vars(args),
            },
            handle,
        )
    print(
        f"speaker raw/FDR={raw_speaker.sum()}/{fdr_speaker.sum()} "
        f"semantic raw/FDR={raw_semantic.sum()}/{fdr_semantic.sum()} "
        f"overlap raw/FDR={(raw_speaker & raw_semantic).sum()}/"
        f"{(fdr_speaker & fdr_semantic).sum()}",
        flush=True,
    )
    print(f"saved={csv_path} elapsed={time.time()-start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
