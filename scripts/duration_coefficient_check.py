"""
Duration-coefficient check — is the hard Poisson offset (coefficient on
log(duration) fixed at exactly 1) the right assumption, or does firing RATE
itself genuinely vary with word duration (beyond pure window-length mechanics)?

Fits, per neuron, on the FULL data (no held-out split — this is a diagnostic
about the data's intrinsic relationship, not a predictive-accuracy question):
  Y ~ Poisson(exp(beta0 + beta1 * log(duration)))
beta1 freely estimated (no ridge — single covariate, well-identified).

If median(beta1) ~= 1: the hard offset is the right call, mechanics explain it.
If median(beta1) << 1: rate itself doesn't scale linearly with duration, offset
  may be partially over-correcting (but see below — even beta1 substantially
  below 1 is still consistent with a near-pure mechanical effect, see note).
If median(beta1) >> 1: rate increases faster than purely mechanical sub-linear
  exposure scaling, also worth flagging.

Usage: python3 -u scripts/duration_coefficient_check.py
"""
import os, warnings
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"dev={DEVICE}", flush=True)

PATIENT_ID = "PTYEU_task147"
PATIENT    = "ptYEU_task147"
REGION     = "hippocampus"
COND       = "self"
REGION_RANGES = {"hippocampus": [(1, 16), (25, 40)], "ACC": [(17, 24), (41, 48)]}

SPIKE_ROOT = "/scratch/aniluchavez/ConvoDATAS/SpikeWindows"
MIN_SPIKES = 5
ETA_CLIP   = 20.0
LBFGS_ITER = 50


def load_speaker_assignment(spike_dir):
    cands = [f for f in os.listdir(spike_dir) if f.endswith("_with_regress_dur.xlsx")]
    tx = pd.read_excel(os.path.join(spike_dir, cands[0]))
    spk_cols = sorted(
        [c for c in tx.columns if str(c).startswith("Speaker")],
        key=lambda c: int(c.replace("Speaker", "").strip())
                      if c.replace("Speaker", "").strip().isdigit() else 999,
    )
    def _nn(val):
        return pd.notna(val) and str(val).strip() not in ("", "nan")
    dir_membership = {col: np.array([_nn(v) for v in tx[col]], dtype=bool) for col in spk_cols}
    n = len(tx)
    assign = np.array([None] * n, dtype=object)
    for i in range(n):
        for col in spk_cols:
            if dir_membership[col][i]:
                assign[i] = col; break
    mask_self = assign == "Speaker1"
    return mask_self


def load_word_dur_ms(spike_dir):
    cands = [f for f in os.listdir(spike_dir) if f.endswith("_with_regress_dur.xlsx")]
    df = pd.read_excel(os.path.join(spike_dir, cands[0]))
    col = "word_dur" if "word_dur" in df.columns else "Duration"
    return df[col].values.astype(np.float64)


def load_spike_matrix(spike_dir, speaker, region):
    spk_dir = os.path.join(spike_dir, speaker)
    cands = [f for f in os.listdir(spk_dir)
             if f.lower().startswith(region.lower()) and f.endswith(".npy")]
    return np.load(os.path.join(spk_dir, cands[0]))


spike_dir = os.path.join(SPIKE_ROOT, f"output_{PATIENT}_english_only_worddur")
mask_self = load_speaker_assignment(spike_dir)
all_word_dur = load_word_dur_ms(spike_dir)

Y_mat = load_spike_matrix(spike_dir, "Speaker1", REGION)
valid = ~np.isnan(Y_mat).any(axis=1)
Y_v   = Y_mat[valid].astype(np.float32)
dur_v = all_word_dur[mask_self][valid].astype(np.float64).clip(1e-3, None)
logdur_v = np.log(dur_v).astype(np.float32)

spike_ok = Y_v.sum(0) >= MIN_SPIKES
Y_v = Y_v[:, spike_ok]
n_w, n_m = Y_v.shape
print(f"{REGION}/{COND}: {n_w}w x {n_m}n", flush=True)

# standardize log-duration for numerical stability, then convert beta back to
# the original (unstandardized log-ms) scale at the end
ld_mean, ld_std = logdur_v.mean(), logdur_v.std()
logdur_sc = ((logdur_v - ld_mean) / ld_std).astype(np.float32)

Y_t  = torch.tensor(Y_v, device=DEVICE)
ld_t = torch.tensor(logdur_sc, device=DEVICE).view(-1, 1)
Xi   = torch.cat([torch.ones(n_w, 1, device=DEVICE), ld_t], dim=-1)  # (n_w, 2)

beta = torch.zeros(n_m, 2, device=DEVICE, requires_grad=True)
opt  = torch.optim.LBFGS([beta], max_iter=LBFGS_ITER, line_search_fn='strong_wolfe')

def closure():
    opt.zero_grad()
    eta  = torch.clamp(Xi @ beta.T, -ETA_CLIP, ETA_CLIP)   # (n_w, n_m)
    loss = (torch.exp(eta) - Y_t * eta).sum()
    loss.backward()
    return loss

opt.step(closure)
b = beta.detach().cpu().numpy()
beta1_scaled = b[:, 1]                  # coefficient on standardized log(dur)
beta1_raw    = beta1_scaled / ld_std     # coefficient on raw log(dur) (ms)

print(f"\nFree-covariate coefficient on log(duration), per neuron (n={n_m}):")
print(f"  median beta1 = {np.median(beta1_raw):.3f}")
print(f"  mean   beta1 = {np.mean(beta1_raw):.3f}")
print(f"  std    beta1 = {np.std(beta1_raw):.3f}")
print(f"  [25th, 75th] pct = [{np.percentile(beta1_raw,25):.3f}, {np.percentile(beta1_raw,75):.3f}]")
print(f"  fraction with beta1 in [0.8, 1.2]: {100*np.mean((beta1_raw>=0.8)&(beta1_raw<=1.2)):.1f}%")
print(f"  fraction with beta1 < 0.5: {100*np.mean(beta1_raw<0.5):.1f}%")
print(f"  fraction with beta1 > 1.5: {100*np.mean(beta1_raw>1.5):.1f}%")
print(f"\nFull per-neuron beta1 values: {np.round(beta1_raw,3).tolist()}")

out = pd.DataFrame({"neuron_idx": np.arange(n_m), "beta1_logdur": beta1_raw})
out_path = "/scratch/aniluchavez/ConvoDATAS/VPResults/duration_coefficient_check_PTYEU_hippo_self.pkl"
out.to_pickle(out_path)
print(f"\nSaved -> {out_path}")
