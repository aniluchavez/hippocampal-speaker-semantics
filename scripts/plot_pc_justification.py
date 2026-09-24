#!/usr/bin/env python3
"""
Generates a three-panel figure justifying the choice of N_COMPONENTS=30 PCs.

Panel 1: Cumulative variance explained by PCA on Llama 3.1-8B layer-18 embeddings
Panel 2: Observations-to-parameters ratio per patient/condition at each N_PCs
Panel 3: Cross-validated Pearson r (test set) vs N_PCs on one representative patient

Usage:
    cd /scratch/aniluchavez/hippocampal-speaker-semantics
    python3 -u scripts/plot_pc_justification.py
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import KFold
def poisson_pseudo_r2(y_true, y_pred, y_train_mean):
    """Fraction of Poisson deviance explained (D²): 1 - D_model/D_null.
    Uses saturated model as ceiling so result is bounded [−∞, 1].
    Clips to [-1, 1] to suppress numerical blow-ups when y_bar≈0."""
    lam  = np.clip(y_pred, 1e-10, None)
    lam0 = max(y_train_mean, 1e-10)
    # Poisson deviance: 2*(y*log(y/mu) - (y-mu)); term is 0 when y==0
    safe_log_ratio_model = np.where(y_true > 0, np.log(y_true / lam),  0.0)
    safe_log_ratio_null  = np.where(y_true > 0, np.log(y_true / lam0), 0.0)
    d_model = 2.0 * np.sum(y_true * safe_log_ratio_model - (y_true - lam))
    d_null  = 2.0 * np.sum(y_true * safe_log_ratio_null  - (y_true - lam0))
    if d_null < 1e-8:
        return np.nan
    return float(np.clip(1.0 - d_model / d_null, -1.0, 1.0))

warnings.filterwarnings("ignore")

# ── config ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT    = "/scratch/aniluchavez/hippocampal-speaker-semantics"
EMBED_CACHE_DIR = "/scratch/aniluchavez/ConvoDATAS/EmbedCache"
BERT_EMBED_DIR  = "/scratch/aniluchavez/ConvoDATAS/BERTEmbeds"
SPIKE_WINDOW_OUTPUT_ROOT = "/scratch/aniluchavez/ConvoDATAS/SpikeWindows"

MODEL_TAG    = "llama-3.1-8b"
LLAMA_LAYER  = 18
TARGET_SPEAKER = "SPK1"
CURRENT_N_PCS  = 30
PC_SWEEP       = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 300, 455]

# Representative patient for CV panel (biggest self-condition sample)
CV_PATIENT    = "PTYFF_task17"
CV_PATIENT_PT = "ptYFF_task17"
CV_N_NEURONS  = 10   # randomly sample this many neurons for speed
CV_ALPHAS     = np.logspace(-3, 2, 8)   # inner-CV alpha grid (same as main analysis)
CV_N_FOLDS    = 5

PATIENTS = [
    {"patient_ID": "PTYEU_task147", "patient": "ptYEU_task147"},
    {"patient_ID": "PTYFF_task17",  "patient": "ptYFF_task17"},
    {"patient_ID": "PTYFG_task18",  "patient": "ptYFG_task18"},
    {"patient_ID": "PTYFI_task81",  "patient": "ptYFI_task81"},
    {"patient_ID": "PTYFA_task25",  "patient": "ptYFA_task25"},
    {"patient_ID": "PTYFK_task40",  "patient": "ptYFK_task40"},
    {"patient_ID": "PTYEY_task86",  "patient": "ptYEY_task86"},
    {"patient_ID": "PTYEV_task37",  "patient": "ptYEV_task37"},
    {"patient_ID": "PTYEZ_task60",  "patient": "ptYEZ_task60"},
    {"patient_ID": "PTYFC_task28",  "patient": "ptYFC_task28"},
]

RESULTS_ROOT = "/projects/bhayden/anilu/Language_docs/RegressionRESULTSLLAMA31_L18/allwords"
if RESULTS_ROOT.startswith("/projects") and not os.path.isdir("/projects"):
    RESULTS_ROOT = "/scratch/aniluchavez/RegressionRESULTSLLAMA31_L18/allwords"

# ── helpers ────────────────────────────────────────────────────────────────────

def get_spike_window_dir(patient):
    return os.path.join(
        SPIKE_WINDOW_OUTPUT_ROOT,
        f"output_{patient}_english_only_tshift-150_tlen500_oshift+200_olen500"
    )

def load_metadata(patient_ID):
    csv = os.path.join(BERT_EMBED_DIR, f"{patient_ID}_words_english_only",
                       f"{patient_ID}_aligned_embeddings_withNP.csv")
    return pd.read_csv(csv)[["Word", "Speaker", "RowIndex"]].reset_index(drop=True)

def load_embeddings(patient_ID):
    npy = os.path.join(EMBED_CACHE_DIR, f"{patient_ID}_{MODEL_TAG}_word_emb_layers.npy")
    return np.load(npy)[LLAMA_LAYER].astype(np.float32)

def get_condition_counts(patient_ID):
    """Return (n_self, n_other) word counts from metadata."""
    df = load_metadata(patient_ID)
    return (df["Speaker"] == TARGET_SPEAKER).sum(), (df["Speaker"] != TARGET_SPEAKER).sum()

def build_X(emb, df_meta, n_pcs, cond):
    pca = PCA(n_components=n_pcs)
    pcs = pca.fit_transform(emb)
    mask = df_meta["Speaker"] == TARGET_SPEAKER if cond == "self" else df_meta["Speaker"] != TARGET_SPEAKER
    X = pcs[mask.values]
    return StandardScaler().fit_transform(X)

def load_Y_for_cv(patient_ID, patient, cond, n_neurons=CV_N_NEURONS):
    """Load spike matrix for one condition, randomly sample neurons."""
    spike_dir = get_spike_window_dir(patient)
    df_meta   = load_metadata(patient_ID)
    dur_file  = os.path.join(spike_dir, f"{patient}_with_regress_dur.xlsx")
    dur_df    = pd.read_excel(dur_file)
    df_meta["regress_dur"] = dur_df["regress_dur"].values

    spk_folder = "Speaker1" if cond == "self" else None
    if cond == "other":
        # find largest other-speaker folder
        other_spks = [f for f in os.listdir(spike_dir)
                      if os.path.isdir(os.path.join(spike_dir, f)) and f != "Speaker1"]
        if not other_spks:
            return None, None
        spk_folder = max(other_spks,
                         key=lambda s: os.path.getsize(
                             os.path.join(spike_dir, s,
                                          next(iter(os.listdir(os.path.join(spike_dir, s))), ""))))

    spk_path = os.path.join(spike_dir, spk_folder)
    npy_files = [f for f in os.listdir(spk_path)
                 if f.lower().startswith("hippocampus") and f.endswith(".npy")]
    if not npy_files:
        return None, None

    Y_full = np.load(os.path.join(spk_path, npy_files[0]))
    # drop NaN rows
    valid = ~np.isnan(Y_full).any(axis=1)
    Y_full = Y_full[valid]

    # sample neurons
    rng = np.random.default_rng(42)
    n_neurons = min(n_neurons, Y_full.shape[1])
    neuron_idx = rng.choice(Y_full.shape[1], n_neurons, replace=False)
    return Y_full[:, neuron_idx], valid

# ── panel 1: scree ─────────────────────────────────────────────────────────────

def compute_scree(patient_ID, max_pcs=500):
    emb  = load_embeddings(patient_ID)
    pca  = PCA(n_components=min(max_pcs, emb.shape[0]-1, emb.shape[1])).fit(emb)
    return np.cumsum(pca.explained_variance_ratio_) * 100

# ── panel 2: obs/param ratio ───────────────────────────────────────────────────

def compute_obs_param_ratios():
    rows = []
    for cfg in PATIENTS:
        pid = cfg["patient_ID"]
        try:
            n_self, n_other = get_condition_counts(pid)
            for n_pcs in PC_SWEEP:
                rows.append({"patient": pid, "condition": "self",  "n_pcs": n_pcs,
                              "ratio": n_self / n_pcs})
                rows.append({"patient": pid, "condition": "other", "n_pcs": n_pcs,
                              "ratio": n_other / n_pcs})
        except Exception as e:
            print(f"  {pid}: {e}")
    return pd.DataFrame(rows)

# ── panel 3: CV r vs N_PCs ─────────────────────────────────────────────────────

def cv_r_vs_npcs(patient_ID, patient):
    emb     = load_embeddings(patient_ID)
    df_meta = load_metadata(patient_ID)

    results = {}
    for cond in ["self", "other"]:
        print(f"  CV sweep: {cond}", flush=True)
        Y, valid_mask = load_Y_for_cv(patient_ID, patient, cond)
        if Y is None:
            continue

        r_by_npcs = []
        for n_pcs in PC_SWEEP:
            X = build_X(emb, df_meta, n_pcs, cond)
            # align rows if valid_mask dropped some
            if valid_mask is not None and X.shape[0] != Y.shape[0]:
                X = X[valid_mask[:X.shape[0]]] if len(valid_mask) >= X.shape[0] else X[:Y.shape[0]]
            min_rows = min(X.shape[0], Y.shape[0])
            X, Y_use = X[:min_rows], Y[:min_rows]

            kf = KFold(n_splits=CV_N_FOLDS, shuffle=True, random_state=0)
            neuron_r2s = []
            for nid in range(Y_use.shape[1]):
                y = Y_use[:, nid].astype(float)
                if np.std(y) == 0:
                    continue
                fold_r2 = []
                for tr, va in kf.split(X):
                    try:
                        # inner CV: pick best alpha on training fold
                        inner_kf = KFold(n_splits=3, shuffle=True, random_state=1)
                        best_alpha, best_ll = CV_ALPHAS[0], -np.inf
                        for a in CV_ALPHAS:
                            lls = []
                            for itr, iva in inner_kf.split(X[tr]):
                                try:
                                    mi = PoissonRegressor(alpha=a, max_iter=300).fit(X[tr][itr], y[tr][itr])
                                    pi = mi.predict(X[tr][iva])
                                    ll = np.sum(y[tr][iva] * np.log(np.clip(pi, 1e-10, None)) - pi)
                                    if np.isfinite(ll):
                                        lls.append(ll)
                                except Exception:
                                    pass
                            if lls and np.mean(lls) > best_ll:
                                best_ll = np.mean(lls)
                                best_alpha = a
                        m = PoissonRegressor(alpha=best_alpha, max_iter=500).fit(X[tr], y[tr])
                        pred = m.predict(X[va])
                        r2 = poisson_pseudo_r2(y[va], pred, y[tr].mean())
                        if np.isfinite(r2):
                            fold_r2.append(r2)
                    except Exception:
                        pass
                if fold_r2:
                    neuron_r2s.append(np.mean(fold_r2))
            r_by_npcs.append(np.mean(neuron_r2s) if neuron_r2s else np.nan)
            print(f"    {n_pcs} PCs: mean pseudo-R² = {r_by_npcs[-1]:.4f}", flush=True)
        results[cond] = r_by_npcs
    return results


# ── main ────────────────────────────────────────────────────────────────────────

def main():
    out_dir = os.path.join(RESULTS_ROOT)
    os.makedirs(out_dir, exist_ok=True)

    print("Panel 1: scree curve...", flush=True)
    cumvar = compute_scree("PTYEU_task147", max_pcs=200)

    print("Panel 2: obs/param ratios...", flush=True)
    ratio_df = compute_obs_param_ratios()

    print("Panel 3: CV r sweep...", flush=True)
    cv_results = cv_r_vs_npcs(CV_PATIENT, CV_PATIENT_PT)

    # ── plot ──
    fig = plt.figure(figsize=(16, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # --- Panel 1 ---
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(np.arange(1, len(cumvar)+1), cumvar, color="#4C78A8", lw=2)
    ax1.axvline(CURRENT_N_PCS, color="crimson", lw=1.5, ls="--",
                label=f"N={CURRENT_N_PCS} ({cumvar[CURRENT_N_PCS-1]:.1f}%)")
    ax1.set_xlabel("Number of PCs")
    ax1.set_ylabel("Cumulative variance explained (%)")
    ax1.set_title("Llama 3.1-8B Layer 18\nEmbedding variance")
    ax1.legend(fontsize=9)
    ax1.set_xlim(0, 200)
    ax1.set_ylim(40, 85)
    ax1.grid(alpha=0.3)

    # --- Panel 2 ---
    ax2 = fig.add_subplot(gs[1])
    colors = {"self": "#4C78A8", "other": "#F58518"}
    for cond, grp in ratio_df.groupby("condition"):
        med = grp.groupby("n_pcs")["ratio"].median()
        lo  = grp.groupby("n_pcs")["ratio"].quantile(0.25)
        hi  = grp.groupby("n_pcs")["ratio"].quantile(0.75)
        ax2.plot(med.index, med.values, color=colors[cond], lw=2,
                 label=f"{cond.capitalize()}")
        ax2.fill_between(med.index, lo.values, hi.values,
                         color=colors[cond], alpha=0.2)
    ax2.axvline(CURRENT_N_PCS, color="crimson", lw=1.5, ls="--", label=f"N={CURRENT_N_PCS}")
    ax2.axhline(10, color="gray", lw=1, ls=":", label="10:1 threshold")
    ax2.set_xlabel("Number of PCs")
    ax2.set_ylabel("Observations per parameter")
    ax2.set_title("Sample size / parameter ratio\n(median ± IQR across patients)")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    # --- Panel 3 ---
    ax3 = fig.add_subplot(gs[2])
    if cv_results:
        for cond, rs in cv_results.items():
            ax3.plot(PC_SWEEP, rs, marker="o", lw=2, color=colors.get(cond, "gray"),
                     label=f"{cond.capitalize()}")
        ax3.axvline(CURRENT_N_PCS, color="crimson", lw=1.5, ls="--", label=f"N={CURRENT_N_PCS}")
        ax3.set_xlabel("Number of PCs")
        ax3.set_ylabel("Mean CV pseudo-R² (test set)")
        ax3.set_title(f"Encoding model performance\n({CV_PATIENT}, hippocampus)")
        ax3.legend(fontsize=9)
        ax3.grid(alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "CV data unavailable", ha="center", va="center",
                 transform=ax3.transAxes)

    fig.suptitle("Justification for N_COMPONENTS = 30", fontsize=13, y=1.02)
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"pc_justification_llama31_L{LLAMA_LAYER}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}", flush=True)
    plt.show()


if __name__ == "__main__":
    main()
