"""
Decisive check for the worddur ~80%-flat-across-layers finding: is the
duration-offset + semantic-only test (semantic_glm.py's actual worddur
pipeline, no confound controls) genuinely miscalibrated, or is the real-data
inflation coming from embeddings correlating with confounds duration alone
doesn't remove?

Method (same logic as scripts/null_calibration_check.py, but testing
semantic_glm.py's own simpler test instead of variance_partitioning.py's
controls-partialled one):
  1. Compute a per-neuron null rate from the REAL data: lambda_hat = sum(Y)/sum(duration).
     This has ZERO functional dependence on the semantic embeddings by construction.
  2. Simulate Y_sim ~ Poisson(lambda_hat * duration_i) -- fake spike counts with
     exactly the real duration-driven rate structure, but NO real semantic
     effect injected at all.
  3. Run the exact same worddur+offset+xcirc pipeline (5-fold CV, per-fold PCA,
     Poisson ridge with log-duration offset, xcirc circular-shift permutation)
     on (real X_sem, real duration, Y_sim).

If genuinely zero true effect gives ~5% significant, the offset/null-model
machinery is calibrated -- the real inflated ~80% on real data must come from
embeddings correlating with confounds (frequency, word length, etc.) that
duration-as-offset alone doesn't remove. If it's still inflated on this
known-null simulated data, that IS a real bug in the offset/permutation code.

One patient (PTYEU_task147, hippocampus, self), layer 0 (rawest / most
suspicious case), matching the project convention of prototyping on this
patient/region/condition.
"""
import os, time
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

PATIENT_ID = "PTYEU_task147"
REGION = "hippocampus"
LAYER = 0
N_COMPONENTS = 20
N_PERM = 100
SEED = 0
ALPHAS = np.logspace(-2, 4, 20)
N_OUTER = 5
MIN_SPIKES = 20
ETA_CLIP = 20.0
LBFGS_ITER = 200

EMBED_DIR = "/scratch/aniluchavez/ConvoDATAS/EmbedCache"
SPIKE_ROOT = "/scratch/aniluchavez/ConvoDATAS/SpikeWindows"
MODEL_TAG, CONTEXT_TAG, WINDOW_TYPE = "gpt2-xl", "_ctx200", "worddur"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={DEVICE}", flush=True)


def block_splits(n, k):
    b = n // k
    for i in range(k):
        s, e = i * b, (i * b + b if i < k - 1 else n)
        mask = np.zeros(n, bool); mask[s:e] = True
        yield ~mask, mask


def gpu_fit(X_t, Y_t, alpha, offset_t):
    k, n, p = X_t.shape
    m = Y_t.shape[2]
    n_eff = float(Y_t.sum().clamp(min=1).item())
    Xi = torch.cat([torch.ones(k, n, 1, device=DEVICE), X_t], dim=-1)
    beta = torch.zeros(k, m, p + 1, device=DEVICE, requires_grad=True)
    opt = torch.optim.LBFGS([beta], max_iter=LBFGS_ITER, line_search_fn='strong_wolfe')
    a = alpha.to(DEVICE).view(k, 1, 1) if isinstance(alpha, torch.Tensor) else float(alpha)
    off = offset_t.view(1, n, 1) if offset_t is not None else 0.0
    def closure():
        opt.zero_grad()
        eta = torch.clamp(torch.bmm(Xi, beta.transpose(1, 2)) + off, -ETA_CLIP, ETA_CLIP)
        loss = ((torch.exp(eta) - Y_t * eta).sum()
                + (a * beta[:, :, 1:].pow(2)).sum()) / n_eff
        loss.backward()
        return loss
    opt.step(closure)
    b = beta.detach()
    return b[:, :, 1:], b[:, :, :1]


def fit_and_score(X_raw, Y_v, dur_s, n_comp, rng):
    """Full semantic_glm.py-style worddur pipeline: per-fold PCA, Poisson ridge
    with log-duration offset, rate-based null, xcirc permutation p-values."""
    n_w, n_m = Y_v.shape
    log_dur = np.log(dur_s).astype(np.float32)
    N_A = len(ALPHAS)
    alpha_t = torch.tensor(ALPHAS, dtype=torch.float32)

    ll_real = np.zeros(n_m, np.float64)
    ll_null = np.zeros(n_m, np.float64)
    fold_meta = []
    for tr_m, te_m in block_splits(n_w, N_OUTER):
        pca = PCA(n_components=n_comp)
        X_tr = pca.fit_transform(X_raw[tr_m]).astype(np.float32)
        X_te = pca.transform(X_raw[te_m]).astype(np.float32)
        sc = StandardScaler().fit(X_tr)
        X_tr, X_te = sc.transform(X_tr).astype(np.float32), sc.transform(X_te).astype(np.float32)

        Y_tr_t = torch.tensor(Y_v[tr_m], device=DEVICE)
        off_tr_t = torch.tensor(log_dur[tr_m], device=DEVICE)
        Xii = torch.tensor(X_tr, device=DEVICE).unsqueeze(0).expand(N_A, -1, -1).contiguous()
        Yii = Y_tr_t.unsqueeze(0).expand(N_A, -1, -1).contiguous()
        bf, bi = gpu_fit(Xii, Yii, alpha_t, off_tr_t)
        # pick best alpha per neuron via in-sample-ish quick proxy: reuse held-in fit likelihood on train
        with torch.no_grad():
            eta_tr = torch.clamp(torch.bmm(Xii, bf.transpose(1, 2)) + bi.transpose(1, 2)
                                  + off_tr_t.view(1, -1, 1), -ETA_CLIP, ETA_CLIP)
            ll_tr = (Yii * eta_tr - torch.exp(eta_tr)).sum(1).cpu().numpy()
        best_ai = np.argmax(ll_tr, axis=0)

        X_tr_t = torch.tensor(X_tr[None], device=DEVICE)
        beta_f = torch.zeros(1, n_m, n_comp, device=DEVICE)
        bias_f = torch.zeros(1, n_m, 1, device=DEVICE)
        for ai in np.unique(best_ai):
            ids = np.where(best_ai == ai)[0]
            bf2, bi2 = gpu_fit(X_tr_t, Y_tr_t[None, :, ids], float(ALPHAS[ai]), off_tr_t)
            beta_f[0, ids] = bf2[0]; bias_f[0, ids] = bi2[0]

        X_te_t = torch.tensor(X_te[None], device=DEVICE)
        off_te_t = torch.tensor(log_dur[te_m], device=DEVICE)
        Y_te_t = torch.tensor(Y_v[te_m], device=DEVICE)
        with torch.no_grad():
            eta_te = torch.clamp(torch.bmm(X_te_t, beta_f.transpose(1, 2)) + bias_f.transpose(1, 2)
                                  + off_te_t.view(1, -1, 1), -ETA_CLIP, ETA_CLIP)
            ll_fold = (Y_te_t * eta_te[0] - torch.exp(eta_te[0])).sum(0).cpu().numpy()
        ll_real += ll_fold

        rate_k = Y_v[tr_m].sum(0) / dur_s[tr_m].sum()
        mu_null_te = rate_k[None, :] * dur_s[te_m][:, None]
        ll_null += (Y_v[te_m] * np.log(mu_null_te.clip(1e-10)) - mu_null_te).sum(0)

        fold_meta.append(dict(tr_m=tr_m, te_m=te_m, alpha_vec=[float(ALPHAS[a]) for a in best_ai],
                               X_raw_tr=X_raw[tr_m], X_raw_te=X_raw[te_m],
                               Y_te=Y_v[te_m], off_te=off_te_t, best_ai=best_ai, n_comp=n_comp))
    r2_real = 1.0 - ll_real / ll_null.clip(max=-0.1)

    # xcirc permutation: circularly shift RAW X (pre-PCA), refit per fold at each neuron's chosen alpha
    rng_local = np.random.default_rng(SEED)
    min_lag = max(5, n_w // (N_OUTER * 2))
    ll_perms = np.zeros((N_PERM, n_m), np.float64)
    for pi in range(N_PERM):
        lag = int(rng_local.integers(min_lag, n_w - min_lag))
        X_perm_full = np.roll(X_raw, lag, axis=0)
        ll_perm_accum = np.zeros(n_m, np.float64)
        for fd in fold_meta:
            pca = PCA(n_components=fd["n_comp"])
            X_tr_p = pca.fit_transform(X_perm_full[fd["tr_m"]]).astype(np.float32)
            X_te_p = pca.transform(X_perm_full[fd["te_m"]]).astype(np.float32)
            sc = StandardScaler().fit(X_tr_p)
            X_tr_p, X_te_p = sc.transform(X_tr_p).astype(np.float32), sc.transform(X_te_p).astype(np.float32)
            Y_tr_t = torch.tensor(Y_v[fd["tr_m"]], device=DEVICE)
            off_tr_t = torch.tensor(log_dur[fd["tr_m"]], device=DEVICE)
            X_tr_t = torch.tensor(X_tr_p[None], device=DEVICE)
            beta_f = torch.zeros(1, n_m, fd["n_comp"], device=DEVICE)
            bias_f = torch.zeros(1, n_m, 1, device=DEVICE)
            best_ai = fd["best_ai"]
            for ai in np.unique(best_ai):
                ids = np.where(best_ai == ai)[0]
                bf2, bi2 = gpu_fit(X_tr_t, Y_tr_t[None, :, ids], float(ALPHAS[ai]), off_tr_t)
                beta_f[0, ids] = bf2[0]; bias_f[0, ids] = bi2[0]
            X_te_t = torch.tensor(X_te_p[None], device=DEVICE)
            Y_te_t = torch.tensor(fd["Y_te"], device=DEVICE)
            with torch.no_grad():
                eta_te = torch.clamp(torch.bmm(X_te_t, beta_f.transpose(1, 2)) + bias_f.transpose(1, 2)
                                      + fd["off_te"].view(1, -1, 1), -ETA_CLIP, ETA_CLIP)
                ll_fold_p = (Y_te_t * eta_te[0] - torch.exp(eta_te[0])).sum(0).cpu().numpy()
            ll_perm_accum += ll_fold_p
        ll_perms[pi] = ll_perm_accum
        if (pi + 1) % 20 == 0:
            print(f"  perm {pi+1}/{N_PERM}", flush=True)

    p_vals = (np.sum(ll_perms >= ll_real[None, :], axis=0) + 1) / (N_PERM + 1)
    return r2_real, p_vals


# ── DATA LOADING (mirrors semantic_glm.py's worddur loader for one patient) ──
def find_spike_dir(patient):
    d = os.path.join(SPIKE_ROOT, f"output_{patient}_english_only_{WINDOW_TYPE}")
    return d if os.path.isdir(d) else None


def load_speaker_assignment(spike_dir):
    cands = [f for f in os.listdir(spike_dir) if f.endswith("_with_regress_dur.xlsx")]
    tx = pd.read_excel(os.path.join(spike_dir, cands[0]))
    spk_cols = sorted([c for c in tx.columns if str(c).startswith("Speaker")],
                       key=lambda c: int(c.replace("Speaker", "").strip())
                       if c.replace("Speaker", "").strip().isdigit() else 999)
    def _nn(v): return pd.notna(v) and str(v).strip() not in ("", "nan")
    dm = {c: np.array([_nn(v) for v in tx[c]], dtype=bool) for c in spk_cols}
    mask_self = dm.get("Speaker1", np.zeros(len(tx), bool))
    return mask_self


def load_word_dur_ms(spike_dir):
    cands = [f for f in os.listdir(spike_dir) if f.endswith("_with_regress_dur.xlsx")]
    tx = pd.read_excel(os.path.join(spike_dir, cands[0]))
    for col in ("word_dur", "word_dur_ms", "duration_ms"):
        if col in tx.columns:
            return tx[col].to_numpy(dtype=float)
    raise KeyError("no word_dur column found")


def load_spike_matrix(spike_dir, speaker, region):
    spk_dir = os.path.join(spike_dir, speaker)
    cands = [f for f in os.listdir(spk_dir) if f.lower().startswith(region.lower())
             and f.endswith("_worddur_spike_counts.npy")]
    return np.load(os.path.join(spk_dir, cands[0]))


patient = "pt" + PATIENT_ID[2:]
spike_dir = find_spike_dir(patient)
mask_self = load_speaker_assignment(spike_dir)
dur_ms = load_word_dur_ms(spike_dir)
dur_ms = np.where(np.isnan(dur_ms) | (dur_ms <= 0), 500.0, dur_ms)

npy_path = os.path.join(EMBED_DIR, f"{PATIENT_ID}_{MODEL_TAG}{CONTEXT_TAG}_word_emb_layers.npy")
X_layer_raw = np.load(npy_path, mmap_mode='r')[LAYER].astype(np.float32)

Y_mat = load_spike_matrix(spike_dir, "Speaker1", REGION)
valid = ~np.isnan(Y_mat).any(axis=1)
Y_v_raw = Y_mat[valid].astype(np.float32)
spike_ok = Y_v_raw.sum(0) >= MIN_SPIKES
Y_v_real = Y_v_raw[:, spike_ok]
X_raw = X_layer_raw[mask_self][valid].astype(np.float32)
dur_s = (dur_ms[mask_self][valid] / 1000.0).astype(np.float32)
n_w, n_m = Y_v_real.shape
print(f"{REGION}/self layer={LAYER}: {n_w}w x {n_m}n", flush=True)

# ── Simulate Y under KNOWN-NULL (duration-only rate, zero semantic effect) ──
rate_hat = Y_v_real.sum(0) / dur_s.sum()
lam_null = rate_hat[None, :] * dur_s[:, None]
rng_sim = np.random.default_rng(SEED)
Y_sim = rng_sim.poisson(lam_null).astype(np.float32)
print(f"Y_sim ~ Poisson(rate_hat * duration), mean_sim={Y_sim.mean():.3f} vs mean_real={Y_v_real.mean():.3f}"
      f" -- ground truth: ZERO true semantic effect", flush=True)

t0 = time.time()
r2_sim, p_vals_sim = fit_and_score(X_raw, Y_sim, dur_s, N_COMPONENTS, rng_sim)
print(f"done in {time.time()-t0:.0f}s", flush=True)

sig_sim = p_vals_sim < 0.05
print(f"\n=== ON KNOWN-NULL SIMULATED DATA (should be ~5% if calibrated) ===")
print(f"  raw significant: {sig_sim.sum()}/{n_m} ({100*sig_sim.mean():.1f}%)")
print(f"  raw p-value mean={p_vals_sim.mean():.3f} (expect ~0.5)  min={p_vals_sim.min():.3f}")
hist, edges = np.histogram(p_vals_sim, bins=10, range=(0, 1))
print(f"  histogram (10 bins, 0->1): {hist.tolist()}  (flat ~{n_m/10:.1f}/bin if uniform)")
