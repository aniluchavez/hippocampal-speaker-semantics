# poisson_xy_utils.py

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def extract_XY_from_metadata(X_meta, Y_meta, spike_data, region, target_speaker="SPK1"):
    emb_list, dur_list, spikes_list, speaker_list = [], [], [], []
    speaker_counters = {spk: 0 for spk in spike_data}

    for x_i, y_i in zip(X_meta.index, Y_meta.index):
        spk = Y_meta.loc[y_i, "Speaker"]
        speaker_name = f"Speaker{spk.replace('SPK','')}"
        if speaker_name not in spike_data or region not in spike_data[speaker_name]:
            continue
        spikes = spike_data[speaker_name][region]
        idx = speaker_counters[speaker_name]
        if idx >= spikes.shape[0]:
            continue

        emb = np.array(X_meta.loc[x_i, "Parsed_Embedding"])
        dur = Y_meta.loc[y_i, "regress_dur"]
        spike_row = spikes[idx]

        emb_list.append(emb)
        dur_list.append(dur)
        spikes_list.append(spike_row)
        speaker_list.append(spk)
        speaker_counters[speaker_name] += 1

    return emb_list, dur_list, spikes_list, speaker_list

def build_design_matrix(embeddings, durations, n_components=30, device="cpu", return_shuffles=False, n_shuffles=100,
                         fixed_pca=None, fixed_scaler=None):
    def prepare_features(pcs, durations):
        interactions = pcs * durations
        return np.hstack([pcs, durations, interactions])

    def shuffle_X(X, n):
        return [X[torch.randperm(X.shape[0])] for _ in range(n)]

    if fixed_pca is None:
        pca = PCA(n_components=n_components)
        pcs = pca.fit_transform(np.vstack(embeddings))
    else:
        pca = fixed_pca
        pcs = pca.transform(np.vstack(embeddings))

    durations = np.array(durations).reshape(-1, 1)
    X_raw = prepare_features(pcs, durations)

    if fixed_scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)
    else:
        scaler = fixed_scaler
        X_scaled = scaler.transform(X_raw)

    X_scaled_torch = torch.tensor(X_scaled, dtype=torch.float32, device=device)
    Xs_list = shuffle_X(X_scaled_torch, n_shuffles) if return_shuffles else []

    return X_scaled_torch, X_raw, scaler, Xs_list, prepare_features, pca

def build_XY_continuous(X_meta, Y_meta, spike_data, region, target_speaker="SPK1",
                        n_components=30, device=None, return_weights=True,
                        return_shuffles=False, n_shuffles=100,
                        fixed_pca=None, fixed_scaler=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    emb_list, dur_list, spikes_list, speaker_list = extract_XY_from_metadata(
        X_meta, Y_meta, spike_data, region, target_speaker
    )

    if not emb_list:
        return {
            "X_scaled": torch.empty((0, n_components * 2 + 1), device=device),
            "X_raw": np.empty((0, n_components * 2 + 1)),
            "Y": torch.empty((0, 1), device=device),
            "weights": torch.empty((0,), device=device) if return_weights else None,
            "Xs_list": [],
            "speaker_labels": [],
            "is_self": torch.empty((0,), dtype=torch.float32, device=device),
            "scaler": None,
            "prepare_features": None,
            "pca": None
        }

    X_scaled, X_raw, scaler, Xs_list, prepare_features, pca = build_design_matrix(
        emb_list, dur_list, n_components=n_components, device=device,
        return_shuffles=return_shuffles, n_shuffles=n_shuffles,
        fixed_pca=fixed_pca, fixed_scaler=fixed_scaler
    )

    Y = torch.tensor(np.vstack(spikes_list), dtype=torch.float32, device=device)
    is_self = torch.tensor([1 if spk == target_speaker else 0 for spk in speaker_list], dtype=torch.float32, device=device)

    weights = None
    if return_weights:
        weights = torch.tensor(np.array([
            1.0 / (np.sum(np.array(speaker_list) == spk) * len(np.unique(speaker_list)))
            for spk in speaker_list
        ]), dtype=torch.float32, device=device)

    return {
        "X_scaled": X_scaled,
        "X_raw": X_raw,
        "Y": Y,
        "weights": weights,
        "Xs_list": Xs_list,
        "scaler": scaler,
        "prepare_features": prepare_features,
        "speaker_labels": speaker_list,
        "is_self": is_self,
        "pca": pca
    }

def build_XY_by_speaker_split(X_meta, Y_meta, spike_data, region, target_speaker="SPK1",
                              n_components=30, device=None, return_weights=True,
                              fixed_pca=None, fixed_scaler=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    emb_list, dur_list, spikes_list, speaker_list = extract_XY_from_metadata(
        X_meta, Y_meta, spike_data, region, target_speaker
    )

    if not emb_list:
        return {
            "self": {},
            "other": {}
        }

    X_scaled, X_raw, scaler, Xs_list, prepare_features, pca = build_design_matrix(
        emb_list, dur_list, n_components=n_components, device=device,
        fixed_pca=fixed_pca, fixed_scaler=fixed_scaler
    )

    Y_all = torch.tensor(np.vstack(spikes_list), dtype=torch.float32, device=device)
    is_self = np.array([1 if spk == target_speaker else 0 for spk in speaker_list])
    speaker_arr = np.array(speaker_list)

    output = {}
    for kind in ["self", "other"]:
        mask = (is_self == 1) if kind == "self" else (is_self == 0)
        if not np.any(mask):
            output[kind] = {
                "X_scaled": torch.empty((0, n_components * 2 + 1), device=device),
                "X_raw": np.empty((0, n_components * 2 + 1)),
                "Y": torch.empty((0, 1), device=device),
                "weights": torch.empty((0,), device=device) if return_weights else None,
                "Xs_list": [],
                "scaler": scaler,
                "prepare_features": prepare_features,
                "pca": pca
            }
            continue

        X_sub = X_scaled[mask]
        Y_sub = Y_all[mask]
        speaker_sub = speaker_arr[mask]

        weights = None
        if return_weights:
            weights = torch.tensor(np.array([
                1.0 / (np.sum(speaker_sub == spk) * len(np.unique(speaker_sub)))
                for spk in speaker_sub
            ]), dtype=torch.float32, device=device)

        output[kind] = {
            "X_scaled": X_sub,
            "X_raw": X_raw[mask],
            "Y": Y_sub,
            "weights": weights,
            "Xs_list": [],
            "scaler": scaler,
            "prepare_features": prepare_features,
            "pca": pca
        }

    return output

def build_XY_shifted_by_speaker_split(X_meta, Y_meta, spike_data, region, target_speaker="SPK1",
                                      n_components=30, device=None, return_weights=True,
                                      fixed_pca=None, fixed_scaler=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    emb_list, dur_list, spikes_list, speaker_list = extract_XY_from_metadata(
        X_meta, Y_meta, spike_data, region, target_speaker
    )

    if not emb_list:
        return {"self": {}, "other": {}}

    X_scaled, X_raw, scaler, Xs_list, prepare_features, pca = build_design_matrix(
        emb_list, dur_list, n_components=n_components, device=device,
        fixed_pca=fixed_pca, fixed_scaler=fixed_scaler
    )

    Y_all = torch.tensor(np.vstack(spikes_list), dtype=torch.float32, device=device)
    is_self = np.array([1 if spk == target_speaker else 0 for spk in speaker_list])
    speaker_arr = np.array(speaker_list)

    output = {}
    for kind in ["self", "other"]:
        mask = (is_self == 1) if kind == "self" else (is_self == 0)
        if not np.any(mask):
            output[kind] = {
                "X_scaled": torch.empty((0, n_components * 2 + 1), device=device),
                "X_raw": np.empty((0, n_components * 2 + 1)),
                "Y": torch.empty((0, 1), device=device),
                "weights": torch.empty((0,), device=device) if return_weights else None,
                "Xs_list": [],
                "scaler": scaler,
                "prepare_features": prepare_features,
                "pca": pca
            }
            continue

        X_sub = X_scaled[mask]
        Y_sub = Y_all[mask]
        speaker_sub = speaker_arr[mask]

        weights = None
        if return_weights:
            weights = torch.tensor(np.array([
                1.0 / (np.sum(speaker_sub == spk) * len(np.unique(speaker_sub)))
                for spk in speaker_sub
            ]), dtype=torch.float32, device=device)

        output[kind] = {
            "X_scaled": X_sub,
            "X_raw": X_raw[mask],
            "Y": Y_sub,
            "weights": weights,
            "Xs_list": [],
            "scaler": scaler,
            "prepare_features": prepare_features,
            "pca": pca
        }

    return output