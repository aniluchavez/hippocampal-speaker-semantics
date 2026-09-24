"""
GPU-accelerated LLaMA layer sweep for hippocampal encoding.

For each of the 33 LLaMA 3.1-8B layers, fits a semantic-only Poisson ridge
encoding model with per-neuron alpha selection and records cross-validated
pseudo-R² per neuron.

Key design:
  - GPU randomised PCA (all 33 layers, ~2s per patient vs 68s on CPU)
  - L-BFGS optimiser batched over all 33 layers × all neurons simultaneously
    (each layer's problem is independent, so joint L-BFGS = L independent L-BFGS)
  - Per-neuron alpha selection: alpha grid of 7 values, inner CV builds
    inner_ll[A, L, m] and selects best alpha per (layer, neuron)
  - Outer fit: group neurons by best alpha per layer → one L-BFGS per group
  - Layer sweep produces mean R² per layer (across all neurons); relative
    ranking reflects semantic encoding quality even when mean R² < 0 due to
    temporal non-stationarity in the grand-mean null model

Usage:
    cd /scratch/aniluchavez/hippocampal-speaker-semantics
    nohup conda run -n gpt2_embed --no-capture-output \
        python3 -u scripts/layer_sweep.py > /tmp/layer_sweep.log 2>&1 &
"""

import os, sys, pickle, glob, warnings
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# ── CONFIG ────────────────────────────────────────────────────────────────────

EMBED_CACHE_DIR   = "/scratch/aniluchavez/ConvoDATAS/EmbedCache"
BERT_EMBED_DIR    = "/scratch/aniluchavez/ConvoDATAS/BERTEmbeds"
SPIKE_WINDOW_ROOT = "/scratch/aniluchavez/ConvoDATAS/SpikeWindows"
RESULTS_ROOT      = "/scratch/aniluchavez/ConvoDATAS/LayerSweep"

MODEL_TAG    = "llama-3.1-8b"
N_LAYERS     = 33
N_COMPONENTS = 30
N_SPLITS     = 5
N_INNER      = 3
TARGET_SPEAKER = "SPK1"
ETA_CLIP     = 20.0
LBFGS_ITER   = 20

# Alpha grid — 7 values; start at 0.1 to prevent per-neuron inner-CV from
# selecting catastrophically low regularisation for sparse neurons
ALPHAS_NP = np.logspace(-1, 4, 7)
N_ALPHAS  = len(ALPHAS_NP)
PCA_OVERSAMPL  = 10
PCA_POWER_ITER = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {DEVICE}")
if DEVICE.type == "cuda":
    p = torch.cuda.get_device_properties(0)
    print(f"  {p.name}  |  {p.total_memory//1024**2} MB VRAM")

PATIENTS = [
    {"patient_ID": "PTYEU_task147", "patient": "ptYEU_task147",
     "region_ranges": {"hippocampus": [(1,16),(25,40)], "ACC": [(17,24),(41,48)]}},
    {"patient_ID": "PTYFF_task17",  "patient": "ptYFF_task17",
     "region_ranges": {"hippocampus": [(9,16),(25,40)], "ACC": [(17,24),(41,48)]}},
    {"patient_ID": "PTYFG_task18",  "patient": "ptYFG_task18",
     "region_ranges": {"hippocampus": [(9,16)],         "ACC": [(25,56)]}},
    {"patient_ID": "PTYFI_task81",  "patient": "ptYFI_task81",
     "region_ranges": {"hippocampus": [(1,8),(25,40)],  "ACC": [(9,16)]}},
    {"patient_ID": "PTYFA_task25",  "patient": "ptYFA_task25",
     "region_ranges": {"hippocampus": [(1,16),(25,40)], "ACC": [(17,24)]}},
    {"patient_ID": "PTYFK_task40",  "patient": "ptYFK_task40",
     "region_ranges": {"hippocampus": [(1,16),(25,40)], "ACC": [(49,56)]}},
    {"patient_ID": "PTYEY_task86",  "patient": "ptYEY_task86",
     "region_ranges": {"hippocampus": [(1,16)]}},
    {"patient_ID": "PTYEV_task37",  "patient": "ptYEV_task37",
     "region_ranges": {"hippocampus": [(1,16),(25,40)], "ACC": [(17,24),(41,48)]}},
    {"patient_ID": "PTYEZ_task60",  "patient": "ptYEZ_task60",
     "region_ranges": {"hippocampus": [(1,16)],         "ACC": [(17,24)]}},
    {"patient_ID": "PTYFC_task28",  "patient": "ptYFC_task28",
     "region_ranges": {"hippocampus": [(1,8),(33,48)],  "ACC": [(17,32),(49,64)]}},
    {"patient_ID": "PTYFM_task104", "patient": "ptYFM_task104",
     "region_ranges": {"hippocampus": [(33,48)]}},
    {"patient_ID": "PTYFP_task88",  "patient": "ptYFP_task88",
     "region_ranges": {"hippocampus": [(17,24),(25,32),(49,56),(57,64)]}},
    {"patient_ID": "PTYFR_task91",  "patient": "ptYFR_task91",
     "region_ranges": {"hippocampus": [(1,16),(41,56)]}},
    {"patient_ID": "PTYFS_task95",  "patient": "ptYFS_task95",
     "region_ranges": {"hippocampus": [(1,24)]}},
    {"patient_ID": "PTYFU_task224", "patient": "ptYFU_task224",
     "region_ranges": {"hippocampus": [(17,32),(41,56)]}},
]

os.makedirs(RESULTS_ROOT, exist_ok=True)

# ── GPU PCA ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def gpu_randomized_pca(X_np, n_components=N_COMPONENTS):
    """
    Randomised GPU PCA.  ~300× faster than sklearn for (4000+, 4096) matrices.
    Returns (n, n_components) float32 numpy array.
    """
    n, d = X_np.shape
    X_t  = torch.tensor(X_np, dtype=torch.float32, device=DEVICE)
    X_t  = X_t - X_t.mean(0, keepdim=True)
    k    = min(n_components + PCA_OVERSAMPL, min(n, d))
    Y    = X_t @ torch.randn(d, k, device=DEVICE)
    for _ in range(PCA_POWER_ITER):
        Y = X_t @ (X_t.T @ Y)
    Q, _ = torch.linalg.qr(Y)
    B    = Q.T @ X_t
    _, _, Vt = torch.linalg.svd(B, full_matrices=False)
    return (X_t @ Vt[:n_components].T).cpu().numpy()

# ── GPU L-BFGS POISSON RIDGE ─────────────────────────────────────────────────

def gpu_fit(X_t, Y_t, alpha, max_iter=LBFGS_ITER):
    """
    L-BFGS Poisson ridge, batched over k layers × m neurons.

    X_t : (k, n, p)  DEVICE float32 — features (intercept added internally)
    Y_t : (k, n, m)  DEVICE float32 — spike counts
    Ridge penalty applied only to feature coefficients, not to the intercept.

    Returns (beta_feat, bias):
      beta_feat : (k, m, p)   DEVICE detached
      bias      : (k, m, 1)   DEVICE detached
    """
    k, n, p = X_t.shape
    m = Y_t.shape[2]
    # Prepend intercept column
    Xi   = torch.cat([torch.ones(k, n, 1, device=DEVICE), X_t], dim=-1)  # (k,n,p+1)
    beta = torch.zeros(k, m, p+1, device=DEVICE, requires_grad=True)
    opt  = torch.optim.LBFGS([beta], max_iter=max_iter, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        eta  = torch.clamp(torch.bmm(Xi, beta.transpose(1, 2)), -ETA_CLIP, ETA_CLIP)
        loss = (torch.exp(eta) - Y_t * eta).sum() + alpha * (beta[:, :, 1:]**2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    b = beta.detach()
    return b[:, :, 1:], b[:, :, :1]   # (k,m,p), (k,m,1)


def block_splits(n, k):
    b = n // k
    for i in range(k):
        ts, te = i*b, (i*b+b if i < k-1 else n)
        m = np.zeros(n, dtype=bool); m[ts:te] = True
        yield ~m, m

# ── DATA LOADERS ──────────────────────────────────────────────────────────────

def load_speaker_masks(patient_ID):
    bert_csv = os.path.join(
        BERT_EMBED_DIR, f"{patient_ID}_words_english_only",
        f"{patient_ID}_aligned_embeddings_withNP.csv")
    if os.path.exists(bert_csv):
        df = pd.read_csv(bert_csv)
        return (df["Speaker"] == TARGET_SPEAKER).values, (df["Speaker"] != TARGET_SPEAKER).values
    import re as _re
    tx_files = [f for f in glob.glob(
        f"/scratch/aniluchavez/ConvoDATAS/Transcripts/{patient_ID}*.xlsx")
        if not os.path.basename(f).startswith(("._","~$"))]
    if not tx_files:
        raise FileNotFoundError(f"No transcript for {patient_ID}")
    tx = pd.read_excel(tx_files[0])
    spk_cols = [c for c in tx.columns if c.startswith("Speaker")]
    def _nn(col):
        return tx[col].apply(lambda v: pd.notna(v) and str(v).strip() not in ("","nan")).values
    spk1  = [c for c in spk_cols if c.replace("Speaker","").strip()=="1"]
    other = [c for c in spk_cols if c.replace("Speaker","").strip()!="1"]
    m_s, m_o = np.zeros(len(tx), bool), np.zeros(len(tx), bool)
    for c in spk1:  m_s |= _nn(c)
    for c in other: m_o |= _nn(c)
    return m_s, m_o


def find_spike_window_dir(patient):
    tag = "tshift-150_tlen500_oshift+200_olen500"
    d = os.path.join(SPIKE_WINDOW_ROOT, f"output_{patient}_english_only_{tag}")
    return d if os.path.isdir(d) else None


def load_spike_matrix(spike_dir, speaker, region):
    spk_dir = os.path.join(spike_dir, speaker)
    if not os.path.isdir(spk_dir):
        return None
    cands = [f for f in os.listdir(spk_dir)
             if f.lower().startswith(region.lower()) and f.endswith(".npy")]
    return np.load(os.path.join(spk_dir, cands[0])) if cands else None

# ── MAIN ──────────────────────────────────────────────────────────────────────

import time as _time

all_rows = []

for cfg in PATIENTS:
    patient_ID    = cfg["patient_ID"]
    patient       = cfg["patient"]
    region_ranges = cfg["region_ranges"]

    out_path = os.path.join(RESULTS_ROOT, f"{patient_ID}_layer_sweep.pkl")
    if os.path.exists(out_path):
        print(f"[SKIP] {patient_ID}")
        with open(out_path, "rb") as f:
            all_rows.append(pickle.load(f))
        continue

    print(f"\n{'='*60}\n  {patient_ID}", flush=True)

    npy_path  = os.path.join(EMBED_CACHE_DIR, f"{patient_ID}_{MODEL_TAG}_word_emb_layers.npy")
    spike_dir = find_spike_window_dir(patient)
    if not os.path.exists(npy_path) or spike_dir is None:
        print(f"  SKIP — emb={os.path.exists(npy_path)} spike={spike_dir is not None}")
        continue

    try:
        mask_self, mask_other = load_speaker_masks(patient_ID)
        arr = np.load(npy_path)           # load fully into RAM (~2 GB)
        assert arr.shape[0] == N_LAYERS
        print(f"  words={arr.shape[1]}  layers={N_LAYERS}  dim={arr.shape[2]}", flush=True)
    except Exception as e:
        print(f"  SKIP — {e}"); continue

    # GPU randomised PCA for all 33 layers (< 3 s total vs 68 s with sklearn)
    _t0 = _time.time()
    print("  GPU PCA ...", end=" ", flush=True)
    X_pca_all = np.zeros((N_LAYERS, arr.shape[1], N_COMPONENTS), dtype=np.float32)
    for l in range(N_LAYERS):
        X_pca_all[l] = gpu_randomized_pca(arr[l].astype(np.float32))
    del arr   # free ~2 GB RAM
    print(f"done ({_time.time()-_t0:.1f}s)", flush=True)

    patient_rows = []

    for region, rranges in region_ranges.items():
        for cond, mask in [("self", mask_self), ("other", mask_other)]:

            if cond == "self":
                Y_mat = load_spike_matrix(spike_dir, "Speaker1", region)
            else:
                parts = [load_spike_matrix(spike_dir, spk, region)
                         for spk in sorted(os.listdir(spike_dir))
                         if spk.lower() != "speaker1"]
                parts = [p for p in parts if p is not None]
                Y_mat = np.vstack(parts) if parts else None

            if Y_mat is None or Y_mat.ndim < 2 or Y_mat.shape[0] == 0:
                continue
            if Y_mat.shape[0] != int(mask.sum()):
                print(f"  {region}/{cond}: row mismatch {Y_mat.shape[0]} vs {mask.sum()} — skip")
                continue

            valid  = ~np.isnan(Y_mat).any(axis=1)
            Y_v    = Y_mat[valid].astype(np.float32)

            # Drop silent neurons (< 3 spikes total): their ll_null ≈ 0 → R² explodes
            spike_ok = Y_v.sum(axis=0) >= 3
            Y_v = Y_v[:, spike_ok]
            n_w, n_m = Y_v.shape
            print(f"  {region}/{cond}: {n_w} words × {n_m} neurons "
                  f"(dropped {spike_ok.size - n_m} silent)", flush=True)

            if n_m == 0 or n_w // N_SPLITS < N_COMPONENTS + 2:
                print("    too few words/fold or no neurons — skip"); continue

            # Standardise PCA features for this condition (L, n_w, p)
            X_cond = np.zeros((N_LAYERS, n_w, N_COMPONENTS), dtype=np.float32)
            for l in range(N_LAYERS):
                X_cond[l] = StandardScaler().fit_transform(X_pca_all[l][mask][valid])

            # Per-neuron grand mean for null (constant-rate) predictor
            gm     = Y_v.mean(axis=0).clip(min=1e-10)   # (m,)
            log_gm = np.log(gm)

            # Accumulators: (L, m)
            ll_model = np.zeros((N_LAYERS, n_m), dtype=np.float64)
            ll_null  = np.zeros((N_LAYERS, n_m), dtype=np.float64)
            n_folds  = 0

            _t_cond = _time.time()

            for tr_m, te_m in block_splits(n_w, N_SPLITS):
                n_tr, n_te = int(tr_m.sum()), int(te_m.sum())
                if n_tr < N_COMPONENTS + 2 or n_te < 2: continue

                X_tr_t = torch.tensor(X_cond[:, tr_m, :], device=DEVICE)  # (L, n_tr, p)
                X_te_t = torch.tensor(X_cond[:, te_m, :], device=DEVICE)  # (L, n_te, p)
                Y_tr_t = torch.tensor(Y_v[tr_m], device=DEVICE)           # (n_tr, m)
                Y_te_t = torch.tensor(Y_v[te_m], device=DEVICE)           # (n_te, m)
                Y_tr_L = Y_tr_t.unsqueeze(0).expand(N_LAYERS, -1, -1)     # (L, n_tr, m)
                Y_te_L = Y_te_t.unsqueeze(0).expand(N_LAYERS, -1, -1)     # (L, n_te, m)

                # ── Inner CV: inner_ll[A, L, m] per-neuron held-out LL ───────
                inner_ll  = np.zeros((N_ALPHAS, N_LAYERS, n_m), dtype=np.float64)
                inner_cnt = 0

                for itr_m, iva_m in block_splits(n_tr, N_INNER):
                    ni_tr, ni_va = int(itr_m.sum()), int(iva_m.sum())
                    if ni_tr < N_COMPONENTS + 2 or ni_va < 2: continue

                    Xii = X_tr_t[:, itr_m, :]    # (L, ni_tr, p)
                    Yii = Y_tr_L[:, itr_m, :]    # (L, ni_tr, m)
                    Xiv = X_tr_t[:, iva_m, :]    # (L, ni_va, p)
                    Yiv = Y_tr_L[:, iva_m, :]    # (L, ni_va, m)

                    for ai, alpha in enumerate(ALPHAS_NP):
                        # L-BFGS — must be OUTSIDE no_grad
                        bf, bi = gpu_fit(Xii, Yii, alpha)
                        with torch.no_grad():
                            eta_iv = torch.clamp(
                                torch.bmm(Xiv, bf.transpose(1,2)) + bi.transpose(1,2),
                                -ETA_CLIP, ETA_CLIP)   # (L, ni_va, m)
                            # per-neuron LL: sum over ni_va → (L, m)
                            ll_fold = (Yiv * eta_iv - torch.exp(eta_iv)).sum(dim=1)
                            inner_ll[ai] += ll_fold.cpu().numpy()
                    inner_cnt += 1

                if inner_cnt == 0:
                    best_ai = np.full((N_LAYERS, n_m), N_ALPHAS // 2, dtype=int)
                else:
                    inner_ll /= inner_cnt
                    best_ai = np.argmax(inner_ll, axis=0)   # (L, m)

                # ── Outer fit: group by (layer, alpha) ───────────────────────
                beta_f = torch.zeros(N_LAYERS, n_m, N_COMPONENTS, device=DEVICE)
                bias_f = torch.zeros(N_LAYERS, n_m, 1,           device=DEVICE)

                for l in range(N_LAYERS):
                    for ai in np.unique(best_ai[l]):
                        m_ids = np.where(best_ai[l] == ai)[0]
                        X_b = X_tr_t[l:l+1]                         # (1, n_tr, p)
                        Y_b = Y_tr_L[l:l+1, :, :][:, :, m_ids]      # (1, n_tr, k)
                        bf, bi = gpu_fit(X_b, Y_b, float(ALPHAS_NP[ai]))
                        beta_f[l, m_ids] = bf[0]
                        bias_f[l, m_ids] = bi[0]

                # ── Test-fold log-likelihoods ─────────────────────────────────
                with torch.no_grad():
                    eta_te = torch.clamp(
                        torch.bmm(X_te_t, beta_f.transpose(1, 2))
                        + bias_f.transpose(1, 2),   # (L, n_te, m)
                        -ETA_CLIP, ETA_CLIP)
                    ll_r = (Y_te_L * eta_te - torch.exp(eta_te)).sum(dim=1).cpu().numpy()  # (L,m)

                # Null LL per neuron (same for all layers)
                ll_n_row = (Y_v[te_m] * log_gm - gm).sum(axis=0)   # (m,)
                ll_null  += np.tile(ll_n_row, (N_LAYERS, 1))
                ll_model += ll_r
                n_folds  += 1

                del X_tr_t, X_te_t, Y_tr_t, Y_te_t, Y_tr_L, Y_te_L, beta_f, bias_f, eta_te
                torch.cuda.empty_cache()

            if n_folds == 0: continue

            with np.errstate(divide="ignore", invalid="ignore"):
                # require |ll_null| > 0.5 nat; then clip to [-1, 1] to prevent
                # low-alpha overfitting from dominating the layer-ranking mean
                r2 = np.where(np.abs(ll_null) > 0.5, 1.0 - ll_model / ll_null, np.nan)
                r2 = np.clip(r2, -1.0, 1.0)

            mean_r2 = np.nanmean(r2, axis=1)
            bl = int(np.nanargmax(mean_r2))
            print(f"    {_time.time()-_t_cond:.1f}s  best_layer={bl} "
                  f"mean_R²={mean_r2[bl]:.4f}  L18={mean_r2[18]:.4f}", flush=True)

            for layer in range(N_LAYERS):
                for nid in range(n_m):
                    patient_rows.append({
                        "layer": layer, "patient": patient_ID,
                        "region": region, "condition": cond,
                        "neuron_idx": nid, "r2": float(r2[layer, nid]),
                    })

    if patient_rows:
        df_p = pd.DataFrame(patient_rows)
        with open(out_path, "wb") as f:
            pickle.dump(df_p, f)
        print(f"  Saved → {out_path}")
        all_rows.append(df_p)

    torch.cuda.empty_cache()

# ── AGGREGATE & PLOT ──────────────────────────────────────────────────────────

if not all_rows:
    print("No results."); sys.exit(0)

data = pd.concat(all_rows, ignore_index=True)
agg_path = os.path.join(RESULTS_ROOT, "layer_sweep_all.pkl")
with open(agg_path, "wb") as f:
    pickle.dump(data, f)
print(f"\nAggregated → {agg_path}")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid", font_scale=1.15)

regions   = [r for r in ["hippocampus", "ACC"] if r in data["region"].values]
fig, axes = plt.subplots(1, len(regions), figsize=(7*len(regions), 5), squeeze=False)
fig.suptitle("LLaMA 3.1-8B Layer Sweep — Cross-validated Semantic Encoding R²",
             fontsize=13, fontweight="bold")

for ax, region in zip(axes[0], regions):
    for cond, color, ls in [("self","#4C78A8","-"), ("other","#E45756","--")]:
        sub = data[(data["region"]==region) & (data["condition"]==cond)]
        if sub.empty: continue
        lm  = sub.groupby("layer")["r2"].mean()
        sem = sub.groupby("layer")["r2"].sem()
        ax.plot(lm.index, lm.values, color=color, ls=ls, lw=2,
                label=f"{cond}  peak=L{lm.idxmax()} ({lm.max():.4f})")
        ax.fill_between(lm.index, lm-sem, lm+sem, color=color, alpha=0.15)
    ax.axvline(18, color="gray", lw=1, ls=":", label="L18 (geometry paper)")
    ax.set_xlabel("LLaMA Layer", fontsize=11)
    ax.set_ylabel("Mean pseudo-R²", fontsize=11)
    ax.set_title(region.upper(), fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)

plt.tight_layout()
fig_path = os.path.join(RESULTS_ROOT, "layer_sweep.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"Plot → {fig_path}")

print("\n=== PEAK LAYERS ===")
for region in data["region"].unique():
    for cond in ["self", "other"]:
        sub = data[(data["region"]==region) & (data["condition"]==cond)]
        if sub.empty: continue
        lm = sub.groupby("layer")["r2"].mean()
        print(f"  {region:12s} {cond:5s}: L{lm.idxmax():02d}  mean_R²={lm.max():.5f}")

print("\nDone.")
