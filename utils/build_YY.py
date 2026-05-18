import os, ast
import numpy as np
import pandas as pd

def build_cleanX_by_speaker_condition(meta_csv_path, meta_xlsx_path, spike_base_dir,
                                      region="hippocampus", target_speaker="SPK1",
                                      embedding_col="Embedding",
                                      separate_self_other=False):

    # === Load metadata and durations
    meta_df = pd.read_csv(meta_csv_path)
    dur_df = pd.read_excel(meta_xlsx_path)
    meta_df["regress_dur"] = dur_df["regress_dur"]

    # === Load available spike data by speaker
    speaker_folders = sorted([f for f in os.listdir(spike_base_dir) if f.startswith("Speaker")])
    spike_data = {}
    for speaker in speaker_folders:
        speaker_path = os.path.join(spike_base_dir, speaker)
        spike_data[speaker] = {}
        for fname in os.listdir(speaker_path):
            if fname.endswith("_spike_rates.npy"):
                region_name = fname.replace("_spike_rates.npy", "")
                full_path = os.path.join(speaker_path, fname)
                try:
                    spike_data[speaker][region_name] = np.load(full_path)
                except Exception as e:
                    print(f"[WARN] Failed to load {full_path}: {e}")

    # === Initialize containers
    def init(): return [], [], []
    data_self_X, data_self_Y, data_self_dur = init()
    data_other_X, data_other_Y, data_other_dur = init()
    all_X, all_Y, all_dur = init()
    kept_indices = []

    # === Track spike row indices per speaker/region
    speaker_counters = {
        spk: {reg: 0 for reg in spike_data[spk]} for spk in spike_data
    }

    # === Iterate through meta rows
    for i, row in meta_df.iterrows():
        # Normalize speaker column to match folder names
        speaker_raw = str(row["Speaker"]).strip().upper().replace("SPK", "")
        try:
            speaker_folder = f"Speaker{int(speaker_raw)}"
        except ValueError:
            continue  # skip rows with invalid speaker ID

        # Skip if no matching speaker folder or region
        if speaker_folder not in spike_data or region not in spike_data[speaker_folder]:
            continue

        spikes = spike_data[speaker_folder][region]
        idx = speaker_counters[speaker_folder][region]
        if idx >= spikes.shape[0]:
            continue

        # Parse embedding
        try:
            emb = np.array(ast.literal_eval(row[embedding_col]), dtype=float)
            if emb.ndim != 1 or emb.size == 0:
                raise ValueError("Empty or malformed embedding")
        except Exception as e:
            print(f"[SKIP] Bad embedding at row {i}: {e}")
            continue

        dur = row["regress_dur"]
        spike_row = spikes[idx]

        # Route by speaker identity
        if separate_self_other:
            if row["Speaker"] == target_speaker:
                data_self_X.append(emb)
                data_self_Y.append(spike_row)
                data_self_dur.append(dur)
            else:
                data_other_X.append(emb)
                data_other_Y.append(spike_row)
                data_other_dur.append(dur)
        else:
            all_X.append(emb)
            all_Y.append(spike_row)
            all_dur.append(dur)
            kept_indices.append(i)

        # Move pointer forward
        speaker_counters[speaker_folder][region] += 1

    # === Finalize output
    def finalize(Xs, Ys, Ds):
        if Xs:
            return np.vstack(Xs), np.vstack(Ys), np.array(Ds).reshape(-1, 1)
        return np.empty((0,)), np.empty((0,)), np.empty((0, 1))

    if separate_self_other:
        X_self, Y_self, dur_self = finalize(data_self_X, data_self_Y, data_self_dur)
        X_other, Y_other, dur_other = finalize(data_other_X, data_other_Y, data_other_dur)
        print(f"\n✅ Loaded self: {X_self.shape[0]} | other: {X_other.shape[0]}")
        print(f"X_self.shape: {X_self.shape}, Y_self.shape: {Y_self.shape}, dur_self.shape: {dur_self.shape}")
        print(f"X_other.shape: {X_other.shape}, Y_other.shape: {Y_other.shape}, dur_other.shape: {dur_other.shape}")
        return (X_self, Y_self, dur_self), (X_other, Y_other, dur_other)

    else:
        X, Y, dur = finalize(all_X, all_Y, all_dur)
        print(f"\n✅ Loaded {X.shape[0]} total samples.")
        return X, Y, dur, kept_indices


def build_cleanX_from_spike_dict(meta_csv_path, meta_xlsx_path, spike_data,
                                  region="hippocampus", target_speaker="SPK1",
                                  embedding_col="Embedding", separate_self_other=False):
    import ast
    meta_df = pd.read_csv(meta_csv_path)
    dur_df = pd.read_excel(meta_xlsx_path)
    meta_df["regress_dur"] = dur_df["regress_dur"]

    def init(): return [], [], []
    data_self_X, data_self_Y, data_self_dur = init()
    data_other_X, data_other_Y, data_other_dur = init()
    all_X, all_Y, all_dur = init()
    kept_indices = []

    speaker_counters = {spk: {reg: 0 for reg in spike_data[spk]} for spk in spike_data}

    for i, row in meta_df.iterrows():
        speaker_raw = str(row["Speaker"]).strip().upper().replace("SPK", "")
        try:
            speaker_folder = f"Speaker{int(speaker_raw)}"
        except ValueError:
            continue

        if speaker_folder not in spike_data or region not in spike_data[speaker_folder]:
            continue

        spikes = spike_data[speaker_folder][region]
        idx = speaker_counters[speaker_folder][region]
        if idx >= spikes.shape[0]:
            continue

        try:
            emb = np.array(ast.literal_eval(row[embedding_col]), dtype=float)
            if emb.ndim != 1 or emb.size == 0:
                raise ValueError()
        except:
            continue

        dur = row["regress_dur"]
        spike_row = spikes[idx]

        if separate_self_other:
            if row["Speaker"] == target_speaker:
                data_self_X.append(emb)
                data_self_Y.append(spike_row)
                data_self_dur.append(dur)
            else:
                data_other_X.append(emb)
                data_other_Y.append(spike_row)
                data_other_dur.append(dur)
        else:
            all_X.append(emb)
            all_Y.append(spike_row)
            all_dur.append(dur)
            kept_indices.append(i)

        speaker_counters[speaker_folder][region] += 1

    def finalize(Xs, Ys, Ds):
        if Xs:
            return np.vstack(Xs), np.vstack(Ys), np.array(Ds).reshape(-1, 1)
        return np.empty((0,)), np.empty((0,)), np.empty((0, 1))

    if separate_self_other:
        return (
            finalize(data_self_X, data_self_Y, data_self_dur),
            finalize(data_other_X, data_other_Y, data_other_dur)
        )
    else:
        return finalize(all_X, all_Y, all_dur), kept_indices


def clean_inputs(X_embed, Y, dur):
    dur = dur.reshape(-1, 1)
    mask = ~(
        np.isnan(X_embed).any(axis=1) |
        np.isnan(Y).any(axis=1) |
        np.isnan(dur).ravel()
    )
    return X_embed[mask], Y[mask], dur[mask]  