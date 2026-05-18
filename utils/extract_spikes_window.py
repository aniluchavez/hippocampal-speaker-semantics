import os
import h5py
import numpy as np
import scipy.sparse
import pandas as pd

def load_mat_data(file_path):
    with h5py.File(file_path, 'r') as mat_file:
        spike_group = mat_file['spikes']
        data = spike_group['data'][:]
        ir = spike_group['ir'][:]
        jc = spike_group['jc'][:]
        spikes = scipy.sparse.csr_matrix((data, ir, jc)).toarray()
        qual = mat_file['qual'][:].flatten()
        chan = mat_file['chan'][:].flatten()
    print(f"Loaded spikes shape: {spikes.shape}")
    print(f"Qual shape: {qual.shape}, Chan shape: {chan.shape}")
    return spikes, qual, chan

def get_cells_by_region(chan, qual, region_ranges):
    region_cells = {}
    for region, ranges in region_ranges.items():
        selected_cells = np.array([], dtype=int)
        for min_chan, max_chan in ranges:
            region_chan = np.where((chan >= min_chan) & (chan <= max_chan))[0]
            region_cells_in_range = region_chan[(qual[region_chan] == 4) | (qual[region_chan] == 5)]
            selected_cells = np.concatenate((selected_cells, region_cells_in_range))
        region_cells[region] = selected_cells
    return region_cells

# def extract_speaker_events(file_path, speaker_window_modes, default_mode="onset_to_offset", default_pre=0, default_post=500):
#     import pandas as pd
#     import numpy as np

#     df = pd.read_excel(file_path, sheet_name="Sheet1")
#     df.columns = [col.strip() for col in df.columns]  # strip whitespace
#     df.columns = [col.replace("’", "").replace("‘", "").replace("“", "").replace("”", "") for col in df.columns]  # clean quotes

#     speaker_columns = [col for col in df.columns if col.lower().startswith("speaker")]

#     print(f"📝 Detected speaker columns in Excel: {speaker_columns}")
#     print(f"📦 Provided speaker_window_modes: {list(speaker_window_modes.keys())}")

#     speaker_events = {}

#     for speaker in speaker_columns:
#         print(f"🔍 Attempting to process speaker: {speaker}")
#         if speaker not in speaker_window_modes:
#             print(f"  ⛔ Skipping {speaker} — not in speaker_window_modes")
#             continue

#         cfg = speaker_window_modes.get(speaker, {"mode": default_mode, "pre": default_pre, "post": default_post})
#         mode = cfg.get("mode", default_mode)
#         pre = cfg.get("pre", default_pre)
#         post = cfg.get("post", default_post)

#         # Find rows where this speaker actually said a word
#         valid_rows = df[speaker].notna()
#         if valid_rows.sum() == 0:
#             print(f"  ⚠️ No valid rows for {speaker}")
#             continue

#         speaker_words = df.loc[valid_rows, speaker]
#         onsets = df.loc[valid_rows, "onset"].values
#         offsets = df.loc[valid_rows, "offset"].values

#         if mode == "onset_to_offset":
#             starts = onsets
#             ends = offsets
#         elif mode == "pre_to_offset":
#             starts = onsets - pre
#             ends = offsets
#         elif mode == "onset_to_offset_plus":
#             starts = onsets
#             ends = offsets + post
#         elif mode == "onset_to_offset_plusminus":
#             starts = onsets - pre
#             ends = offsets + post
#         elif mode == "onset_to_onset_plus":
#             starts = onsets
#             ends = onsets + post
#         else:
#             raise ValueError(f"Unknown window mode: {mode}")

#         # Remove any NaNs
#         mask = ~(np.isnan(starts) | np.isnan(ends))
#         starts, ends = starts[mask], ends[mask]
#         words = speaker_words.values[mask]

#         print(f"  ✅ {speaker}: {len(words)} usable words with window mode '{mode}'")

#         speaker_events[speaker] = np.stack([starts, ends, onsets, offsets, words], axis=1)

#     print(f"🧠 Final speaker_events keys: {list(speaker_events.keys())}")
#     return speaker_events


def extract_speaker_events(file_path, speaker_window_modes,
                           default_mode="onset_to_offset", default_pre=0, default_post=500):
    import pandas as pd
    import numpy as np

    df = pd.read_excel(file_path, sheet_name="Sheet1")
    df.columns = [col.strip() for col in df.columns]
    df.columns = [col.replace("’", "").replace("‘", "").replace("“", "").replace("”", "") for col in df.columns]

    # Detect speakers
    speaker_columns = [col for col in df.columns if col.lower().startswith("speaker")]
    print(f"📝 Detected speaker columns in Excel: {speaker_columns}")
    print(f"📦 Provided speaker_window_modes: {list(speaker_window_modes.keys())}")

    speaker_events = {}

    for speaker in speaker_columns:
        print(f"🔍 Attempting to process speaker: {speaker}")
        if speaker not in speaker_window_modes:
            print(f"  ⛔ Skipping {speaker} — not in speaker_window_modes")
            continue

        valid_rows = df[speaker].notna()
        if valid_rows.sum() == 0:
            print(f"  ⚠️ No valid rows for {speaker}")
            continue

        onset = df.loc[valid_rows, "onset"]
        offset = df.loc[valid_rows, "offset"]
        duration = offset - onset

        cfg = speaker_window_modes.get(speaker, {})
        mode = cfg.get("mode", default_mode)
        pre = cfg.get("pre", default_pre)
        post = cfg.get("post", default_post)

        if mode == "pre_to_onset":
            eventTime = onset
            pre_onset = onset - pre
            post_offset = onset
        elif mode == "pre_to_offset":
            eventTime = onset
            pre_onset = onset - pre
            post_offset = offset
        elif mode == "onset_to_offset":
            eventTime = onset
            pre_onset = onset
            post_offset = offset
        elif mode == "onset_to_offset_plus":
            eventTime = onset
            pre_onset = onset
            post_offset = offset + post
        elif mode == "onset_to_onset_plus":
            eventTime = onset
            pre_onset = onset
            post_offset = onset + post
        elif mode == "pre_to_offset_plus":
            eventTime = onset
            pre_onset = onset - pre
            post_offset = offset + post
        elif mode == "onset_to_offset_plusminus":
            eventTime = onset
            pre_onset = onset - pre
            post_offset = offset + post
        else:
            raise ValueError(f"Unsupported mode: {mode} for {speaker}")

        # Safety checks
        if np.any(pre_onset < 0):
            print(f"⚠️ WARNING: {mode} — pre_onset has negative values!")
        if np.any(pd.isna(pre_onset)) or np.any(pd.isna(post_offset)):
            print("🚨 NaNs in window computation — might drop trials during cleaning.")

        result = np.vstack([eventTime, pre_onset, post_offset, duration]).T
        print(f"  ✅ {speaker}: {len(result)} usable words with window mode '{mode}'")
        speaker_events[speaker] = result

    print(f"🧠 Final speaker_events keys: {list(speaker_events.keys())}")
    return speaker_events

def extract_sliding_window_spikes(spikes, region_cells, speaker_events,
                                   shift_ms=0, sample_rate=1000):
    """
    Extract spike rates using fixed-duration windows (equal to word duration),
    shifted by a constant shift_ms relative to the window start.

    Parameters:
    - spikes: ndarray [T x N] spike matrix (time x all neurons)
    - region_cells: dict of region -> list of neuron indices
    - speaker_events: output from extract_speaker_events()
    - shift_ms: temporal shift applied to window start (e.g., -1000, +500)
    - sample_rate: Hz (default 1000 = 1 ms bins)

    Returns:
    - result: nested dict result[speaker][region] = spike rate matrix [words x neurons]
    """
    import numpy as np
    T = spikes.shape[0]
    result = {}

    for speaker, events in speaker_events.items():
        word_onsets = events[:, 0]
        pre_onsets = events[:, 1]
        durations = events[:, 3].astype(int)
        num_words = len(word_onsets)

        speaker_result = {}

        for region, neuron_indices in region_cells.items():
            if len(neuron_indices) == 0:
                continue

            region_spikes = spikes[:, neuron_indices]
            n_neurons = region_spikes.shape[1]
            spike_matrix = np.full((num_words, n_neurons), np.nan)

            for i in range(num_words):
                dur = durations[i]
                start = int(round(pre_onsets[i] + shift_ms))
                end = start + dur
                if dur <= 0 or start < 0 or end > T:
                    continue  # skip invalid window

                segment = region_spikes[start:end, :]
                rates = np.sum(segment, axis=0) * (sample_rate / dur)
                spike_matrix[i, :] = rates

            speaker_result[region] = spike_matrix

        result[speaker] = speaker_result

    return result



def process_and_save_spikes(spikes, region_cells, speaker_events, binSize, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    T = spikes.shape[0]
    for speaker, event_times in speaker_events.items():
        speaker_output_dir = os.path.join(output_dir, speaker)
        os.makedirs(speaker_output_dir, exist_ok=True)
        print(f"\nProcessing spikes for {speaker}...")
        word_onsets = event_times[:, 0]
        pre_onsets = event_times[:, 1]
        post_offsets = event_times[:, 2]
        num_events = len(word_onsets)
        for region, neuron_indices in region_cells.items():
            if len(neuron_indices) == 0:
                print(f"Skipping {region} for {speaker} (no neurons).")
                continue
            region_spikes = spikes[:, neuron_indices]
            n_neurons = region_spikes.shape[1]
            region_result = []
            for neuron_idx in range(n_neurons):
                neuron_spike_train = region_spikes[:, neuron_idx]
                neuron_event_bins = []
                for j in range(num_events):
                    start_idx = int(round(pre_onsets[j]))
                    end_idx = int(round(post_offsets[j])) + 1
                    start_idx = max(0, start_idx)
                    end_idx = min(T, end_idx)
                    segment = neuron_spike_train[start_idx:end_idx]
                    L = len(segment)
                    nBins = L // binSize
                    if nBins == 0:
                        binned = np.array([])
                    else:
                        binned = np.array([
                            np.nansum(segment[binSize * b:binSize * (b + 1)]) for b in range(nBins)
                        ])
                    neuron_event_bins.append(binned)
                max_bins = max(len(arr) for arr in neuron_event_bins) if neuron_event_bins else 0
                padded_bins = np.full((num_events, max_bins), np.nan)
                for j, arr in enumerate(neuron_event_bins):
                    padded_bins[j, :len(arr)] = arr
                neuron_rates = np.nanmean(padded_bins, axis=1) * (1000 / binSize)
                region_result.append(neuron_rates)
            region_result = np.column_stack(region_result)
            save_path = os.path.join(speaker_output_dir, f"{region}_spike_rates.npy")
            np.save(save_path, region_result)
            print(f"Saved {region} spike rates for {speaker} | Shape: {region_result.shape}")

def process_and_save_spike_sum(spikes, region_cells, speaker_events, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    T = spikes.shape[0]
    for speaker, event_times in speaker_events.items():
        speaker_output_dir = os.path.join(output_dir, speaker)
        os.makedirs(speaker_output_dir, exist_ok=True)
        print(f"\nProcessing spike sums for {speaker}...")
        word_onsets = event_times[:, 0]
        pre_onsets = event_times[:, 1]
        post_offsets = event_times[:, 2]
        num_events = len(word_onsets)
        for region, neuron_indices in region_cells.items():
            if len(neuron_indices) == 0:
                print(f"Skipping {region} for {speaker} (no neurons).")
                continue
            region_spikes = spikes[:, neuron_indices]
            n_neurons = region_spikes.shape[1]
            region_result = []
            for neuron_idx in range(n_neurons):
                neuron_spike_train = region_spikes[:, neuron_idx]
                spike_counts = []
                for j in range(num_events):
                    start_idx = int(round(pre_onsets[j]))
                    end_idx = int(round(post_offsets[j])) + 1
                    start_idx = max(0, start_idx)
                    end_idx = min(T, end_idx)
                    spike_sum = np.sum(neuron_spike_train[start_idx:end_idx])
                    spike_counts.append(spike_sum)
                region_result.append(spike_counts)
            region_result = np.column_stack(region_result)
            save_path = os.path.join(speaker_output_dir, f"{region}_spike_sum.npy")
            np.save(save_path, region_result)
            print(f"Saved {region} spike sums for {speaker} | Shape: {region_result.shape}")

def add_regress_dur_column(df, speaker_window_modes, default_mode="onset_to_offset", default_pre=0, default_post=500):
    speaker_columns = [col for col in df.columns if col.lower().startswith("speaker")]
    durations = []
    for idx, row in df.iterrows():
        onset = row.get("onset")
        offset = row.get("offset")
        speaker = next((spk for spk in speaker_columns if pd.notna(row.get(spk, np.nan))), None)
        if pd.isna(onset) or pd.isna(offset) or speaker is None:
            durations.append(np.nan)
            continue
        config = speaker_window_modes.get(speaker, {})
        mode = config.get("mode", default_mode)
        pre = config.get("pre", default_pre)
        post = config.get("post", default_post)

        if mode == "pre_to_onset":
            pre_onset = onset - pre
            post_offset = onset
        elif mode == "pre_to_offset":
            pre_onset = onset - pre
            post_offset = offset
        elif mode == "onset_to_offset":
            pre_onset = onset
            post_offset = offset
        elif mode == "onset_to_offset_plus":
            pre_onset = onset
            post_offset = offset + post
        elif mode == "onset_to_onset_plus":
            pre_onset = onset
            post_offset = onset + post
        elif mode == "pre_to_offset_plus":
            pre_onset = onset - pre
            post_offset = offset + post
        elif mode == "onset_to_offset_plusminus":
            pre_onset = onset - pre
            post_offset = offset + post
        else:
            durations.append(np.nan)
            continue

        durations.append(post_offset - pre_onset)

    df["regress_dur"] = durations
    return df

