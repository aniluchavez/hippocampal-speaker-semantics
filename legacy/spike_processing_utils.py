# spike_processing_utils.py
import os
import h5py
import numpy as np
import pandas as pd
import scipy.sparse
import re
from pathlib import Path

# === CORE UTILITIES ===

def load_mat_data(file_path):
    with h5py.File(file_path, 'r') as mat_file:
        spike_group = mat_file['spikes']
        data = spike_group['data'][:]
        ir = spike_group['ir'][:]
        jc = spike_group['jc'][:]
        spikes = scipy.sparse.csr_matrix((data, ir, jc)).toarray()
        qual = mat_file['qual'][:].flatten()
        chan = mat_file['chan'][:].flatten()
    return spikes, qual, chan


def get_cells_by_region(chan, qual, region_ranges, accepted_qual=(4, 5)):
    """
    Given arrays of channel numbers and quality ratings,
    return a dict mapping region name to list of indices in spike matrix
    """
    region_cells = {}
    chan = np.array(chan).flatten()
    qual = np.array(qual).flatten()

    for region, ranges in region_ranges.items():
        indices = []
        for (start, end) in ranges:
            # Find neurons whose channels are within the given range and quality is acceptable
            match = np.where((chan >= start) & (chan <= end) & np.isin(qual, accepted_qual))[0]
            indices.extend(match.tolist())
        region_cells[region] = indices
    return region_cells


# === WORD EVENT EXTRACTION ===

def extract_speaker_events(file_path, speaker_of_interest=None, interest_pre_window=0, interest_post_window=0,
                           other_pre_window=0, other_post_window=0, interest_shift=0, mode="fixed",
                           target_start_ref="onset", target_start_shift=0, target_end_ref="offset", target_end_shift=0,
                           other_start_ref="onset", other_start_shift=0, other_end_ref="offset", other_end_shift=0,
                           target_ref_point="onset", target_shift=250, target_window_length=500,
                           other_ref_point="offset", other_shift=-500, other_window_length=300):
    df = pd.read_excel(file_path, sheet_name="Sheet1", keep_default_na=False)
    speaker_columns = [col for col in df.columns if col.lower().startswith("speaker")]
    speaker_events = {}

    for speaker in speaker_columns:
        df[speaker] = df[speaker].astype(str).str.strip().replace({"": "xxx"})

    for speaker in speaker_columns:
        speaker_df = df[df[speaker] != "xxx"][["onset", "offset"]].copy()
        if speaker_df.empty: continue
        speaker_df["onset"] = pd.to_numeric(speaker_df["onset"], errors="coerce")
        speaker_df["offset"] = pd.to_numeric(speaker_df["offset"], errors="coerce")
        speaker_df = speaker_df.dropna(subset=["onset", "offset"])
        if speaker_df.empty: continue
        speaker_df["duration"] = np.round(speaker_df["offset"] - speaker_df["onset"])

        if mode == "fixed":
            speaker_df["eventTime"] = speaker_df["onset"] + 80
            if speaker == speaker_of_interest:
                speaker_df["pre_onset"] = speaker_df["eventTime"] - interest_pre_window
                speaker_df["post_offset"] = speaker_df["eventTime"] + interest_post_window
            else:
                speaker_df["pre_onset"] = speaker_df["eventTime"] - other_pre_window
                speaker_df["post_offset"] = speaker_df["eventTime"] + other_post_window

        elif mode == "regression":
            speaker_df["eventTime"] = speaker_df["onset"]
            speaker_df["pre_onset"] = speaker_df["onset"]
            speaker_df["post_offset"] = speaker_df["offset"] + 500

        elif mode == "target_vs_other":
            speaker_df["eventTime"] = speaker_df["onset"]
            if speaker == speaker_of_interest:
                speaker_df["pre_onset"] = speaker_df["onset"] - 500
                speaker_df["post_offset"] = speaker_df["onset"]
            else:
                speaker_df["pre_onset"] = speaker_df["onset"]
                speaker_df["post_offset"] = speaker_df["offset"] + 500

        elif mode == "onset_offset_plus_prepost":
            speaker_df["eventTime"] = speaker_df["onset"]
            if speaker == speaker_of_interest:
                speaker_df["pre_onset"] = speaker_df["onset"] - interest_pre_window
                speaker_df["post_offset"] = speaker_df["offset"] + interest_post_window
            else:
                speaker_df["pre_onset"] = speaker_df["onset"] - other_pre_window
                speaker_df["post_offset"] = speaker_df["offset"] + other_post_window

        elif mode == "duration_shifted_window":
            speaker_df["eventTime"] = speaker_df["onset"]
            if speaker == speaker_of_interest:
                shift = interest_shift
                speaker_df["pre_onset"] = speaker_df["onset"] - shift
                speaker_df["post_offset"] = speaker_df["offset"] - shift
            else:
                shift = other_shift
                speaker_df["pre_onset"] = speaker_df["onset"] + shift
                speaker_df["post_offset"] = speaker_df["offset"] + shift
            speaker_df = speaker_df[speaker_df["pre_onset"] < speaker_df["post_offset"]]

        elif mode == "onset_offset_append_window":
            speaker_df["eventTime"] = speaker_df["onset"]
            if speaker == speaker_of_interest:
                speaker_df["pre_onset"] = speaker_df["onset"] - interest_pre_window
                speaker_df["post_offset"] = speaker_df["offset"] + interest_post_window
            else:
                speaker_df["pre_onset"] = speaker_df["onset"] - other_pre_window
                speaker_df["post_offset"] = speaker_df["offset"] + other_post_window

        elif mode == "target_vs_other_custom_bounds":
            speaker_df["eventTime"] = speaker_df["onset"]
            if speaker == speaker_of_interest:
                speaker_df["pre_onset"] = speaker_df[target_start_ref] + target_start_shift
                speaker_df["post_offset"] = speaker_df[target_end_ref] + target_end_shift
            else:
                speaker_df["pre_onset"] = speaker_df[other_start_ref] + other_start_shift
                speaker_df["post_offset"] = speaker_df[other_end_ref] + other_end_shift

        elif mode == "target_vs_other_fixed_window_from_ref":
            speaker_df["eventTime"] = speaker_df["onset"]
            if speaker == speaker_of_interest:
                anchor = speaker_df[target_ref_point] + target_shift
                speaker_df["pre_onset"] = anchor
                speaker_df["post_offset"] = anchor + target_window_length
            else:
                anchor = speaker_df[other_ref_point] + other_shift
                speaker_df["pre_onset"] = anchor
                speaker_df["post_offset"] = anchor + other_window_length

        else:
            raise ValueError(f"Unsupported mode: {mode}")

        speaker_events[speaker] = speaker_df[["eventTime", "pre_onset", "post_offset", "duration"]].values

    print("\n✅ Word counts after extraction:")
    for speaker in speaker_columns:
        print(f"{speaker}: {len(speaker_events.get(speaker, []))}")
    return speaker_events

# === TRIAL CUTTING AND SPIKE PROCESSING ===

def trial_cut_fixations(spike_train, events, binSize, preDur=None, postDur=None, pre_onsets=None, post_offsets=None, mode="fixed_event_plus_prepost"):
    tbSpikes_list = []
    if mode == "fixed_event_plus_prepost":
        if preDur is None or postDur is None:
            raise ValueError("Missing preDur or postDur.")
        for i, event in enumerate(events):
            pre_val = preDur[i] if hasattr(preDur, '__len__') else preDur
            post_val = postDur[i] if hasattr(postDur, '__len__') else postDur
            start_idx = max(0, int(round(event - pre_val)))
            end_idx = min(len(spike_train), int(round(event + post_val)) + 1)
            segment = spike_train[start_idx:end_idx]
            nBins = len(segment) // binSize
            binned = np.array([np.nansum(segment[binSize * b: binSize * (b + 1)]) for b in range(nBins)]) if nBins > 0 else np.array([])
            tbSpikes_list.append(binned)

    elif mode == "explicit_bounds":
        if pre_onsets is None or post_offsets is None:
            raise ValueError("Missing pre_onsets or post_offsets.")
        for i in range(len(pre_onsets)):
            start_idx = max(0, int(round(pre_onsets[i])))
            end_idx = min(len(spike_train), int(round(post_offsets[i])) + 1)
            segment = spike_train[start_idx:end_idx]
            nBins = len(segment) // binSize
            binned = np.array([np.nansum(segment[binSize * b: binSize * (b + 1)]) for b in range(nBins)]) if nBins > 0 else np.array([])
            tbSpikes_list.append(binned)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return tbSpikes_list

def process_and_save_spikes(spikes, region_cells, speaker_events, speaker_windows=None, binSize=20, output_dir="output", mode="fixed_speaker_windows"):
    """
    Processes spikes separately for each speaker and brain region.

    Modes:
    - "fixed_speaker_windows": Uses speaker-level pre/post values (old behavior).
    - "explicit_event_bounds": Uses event-specific pre_onset/post_offset columns (new behavior).

    Saves resulting matrices to disk.
    """
    import os
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)
    T = spikes.shape[0]  # total timepoints

    for speaker, event_times in speaker_events.items():
        speaker_output_dir = os.path.join(output_dir, speaker)
        os.makedirs(speaker_output_dir, exist_ok=True)
        print(f"\nProcessing spikes for {speaker}...")

        if mode == "fixed_speaker_windows":
            if speaker_windows is None:
                raise ValueError("speaker_windows must be provided for fixed_speaker_windows mode.")

            if speaker in speaker_windows:
                pre_window = speaker_windows[speaker]['pre']
                post_window = speaker_windows[speaker]['post']
            else:
                pre_window = 0
                post_window = 500

            word_onsets = event_times[:, 0]
            num_events = len(word_onsets)

            for region, neuron_indices in region_cells.items():
                if len(neuron_indices) == 0:
                    print(f"Skipping {region} for {speaker} (no neurons found).")
                    continue

                region_spikes = spikes[:, neuron_indices]
                n_neurons = region_spikes.shape[1]
                region_result = []

                for neuron_idx in range(n_neurons):
                    neuron_spike_train = region_spikes[:, neuron_idx]
                    neuron_event_bins = []

                    for j, event in enumerate(word_onsets):
                        if event + post_window > T:
                            current_post = max(T - event, 0)
                        else:
                            current_post = post_window
                        current_pre = pre_window

                        tbSpikes = trial_cut_fixations(
                            neuron_spike_train,
                            np.array([event]),
                            binSize,
                            np.array([current_pre]),
                            np.array([current_post]),
                            mode="fixed_event_plus_prepost"
                        )
                        tbSpikes = tbSpikes[0]
                        neuron_event_bins.append(tbSpikes)

                    max_bins = max(len(arr) for arr in neuron_event_bins) if neuron_event_bins else 0
                    padded_bins = np.full((num_events, max_bins), np.nan)
                    for j, arr in enumerate(neuron_event_bins):
                        nBins = len(arr)
                        padded_bins[j, :nBins] = arr

                    neuron_rates = np.nanmean(padded_bins, axis=1) * (1000 / binSize)
                    region_result.append(neuron_rates)

                region_result = np.column_stack(region_result)
                file_path = os.path.join(speaker_output_dir, f"{region}_spike_rates.npy")
                np.save(file_path, region_result)
                print(f"Saved {region} spike rates for {speaker} | Shape: {region_result.shape} (words x neurons)")

        elif mode == "explicit_event_bounds":
            word_pres = event_times[:, 1]
            word_posts = event_times[:, 2]
            num_events = len(word_pres)

            for region, neuron_indices in region_cells.items():
                if len(neuron_indices) == 0:
                    print(f"Skipping {region} for {speaker} (no neurons found).")
                    continue

                region_spikes = spikes[:, neuron_indices]
                n_neurons = region_spikes.shape[1]
                region_result = []

                for neuron_idx in range(n_neurons):
                    neuron_spike_train = region_spikes[:, neuron_idx]
                    neuron_event_bins = []

                    for j in range(num_events):
                        tbSpikes = trial_cut_fixations(
                            neuron_spike_train,
                            None,
                            binSize,
                            pre_onsets=np.array([word_pres[j]]),
                            post_offsets=np.array([word_posts[j]]),
                            mode="explicit_bounds"
                        )
                        tbSpikes = tbSpikes[0]
                        neuron_event_bins.append(tbSpikes)

                    max_bins = max(len(arr) for arr in neuron_event_bins) if neuron_event_bins else 0
                    padded_bins = np.full((num_events, max_bins), np.nan)
                    for j, arr in enumerate(neuron_event_bins):
                        nBins = len(arr)
                        padded_bins[j, :nBins] = arr

                    neuron_rates = np.nanmean(padded_bins, axis=1) * (1000 / binSize)
                    region_result.append(neuron_rates)

                region_result = np.column_stack(region_result)
                file_path = os.path.join(speaker_output_dir, f"{region}_spike_rates.npy")
                np.save(file_path, region_result)
                print(f"Saved {region} spike rates for {speaker} | Shape: {region_result.shape} (words x neurons)")

        else:
            raise ValueError(f"Unsupported mode: {mode}")
        

def process_and_save_spike_sum(spikes, region_cells, speaker_events, speaker_windows=None, output_dir="output", mode="fixed_speaker_windows"):
    """
    Computes spike sums using two modes:

    - "fixed_speaker_windows": old behavior using speaker-level pre/post.
    - "explicit_event_bounds": new behavior using event-level pre_onset and post_offset.

    Saves resulting matrices to disk.
    """
    import os
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)
    T = spikes.shape[0]

    for speaker, event_times in speaker_events.items():
        speaker_output_dir = os.path.join(output_dir, speaker)
        os.makedirs(speaker_output_dir, exist_ok=True)
        print(f"\nProcessing spike sums for {speaker}...")

        if mode == "fixed_speaker_windows":
            if speaker_windows is None:
                raise ValueError("speaker_windows must be provided for fixed_speaker_windows mode.")

            if speaker in speaker_windows:
                pre_window = speaker_windows[speaker]['pre']
                post_window = speaker_windows[speaker]['post']
            else:
                pre_window = 0
                post_window = 500

            word_onsets = event_times[:, 0]
            num_events = len(word_onsets)

            for region, neuron_indices in region_cells.items():
                if len(neuron_indices) == 0:
                    print(f"Skipping {region} for {speaker} (no neurons found).")
                    continue

                region_spikes = spikes[:, neuron_indices]
                n_neurons = region_spikes.shape[1]
                region_result = []

                for neuron_idx in range(n_neurons):
                    neuron_spike_train = region_spikes[:, neuron_idx]
                    neuron_event_counts = []

                    for event in word_onsets:
                        if event + post_window > T:
                            current_post = max(T - event, 0)
                        else:
                            current_post = post_window
                        current_pre = pre_window

                        start_idx = int(round(event - current_pre))
                        end_idx = int(round(event + current_post)) + 1

                        start_idx = max(0, start_idx)
                        end_idx = min(T, end_idx)

                        spike_sum = np.sum(neuron_spike_train[start_idx:end_idx])
                        neuron_event_counts.append(spike_sum)

                    region_result.append(neuron_event_counts)

                region_result = np.column_stack(region_result)
                file_path = os.path.join(speaker_output_dir, f"{region}_spike_sum.npy")
                np.save(file_path, region_result)
                print(f"Saved {region} spike sums for {speaker} | Shape: {region_result.shape} (words x neurons)")

        elif mode == "explicit_event_bounds":
            word_pres = event_times[:, 1]
            word_posts = event_times[:, 2]
            num_events = len(word_pres)

            for region, neuron_indices in region_cells.items():
                if len(neuron_indices) == 0:
                    print(f"Skipping {region} for {speaker} (no neurons found).")
                    continue

                region_spikes = spikes[:, neuron_indices]
                n_neurons = region_spikes.shape[1]
                region_result = []

                for neuron_idx in range(n_neurons):
                    neuron_spike_train = region_spikes[:, neuron_idx]
                    neuron_event_counts = []

                    for i in range(num_events):
                        start_idx = int(round(word_pres[i]))
                        end_idx = int(round(word_posts[i])) + 1

                        clipped_start = max(0, start_idx)
                        clipped_end = min(T, end_idx)

                        if clipped_end <= clipped_start:
                            print(
                                f"❗⚠️ Skipped Speaker={speaker} Event={i}: "
                                f"original start={start_idx}, end={end_idx}, "
                                f"clipped start={clipped_start}, clipped end={clipped_end}"
                            )
                            continue

                        spike_sum = np.sum(neuron_spike_train[clipped_start:clipped_end])
                        neuron_event_counts.append(spike_sum)


                    region_result.append(neuron_event_counts)

                region_result = np.column_stack(region_result)
                file_path = os.path.join(speaker_output_dir, f"{region}_spike_sum.npy")
                np.save(file_path, region_result)
                print(f"Saved {region} spike sums for {speaker} | Shape: {region_result.shape} (words x neurons)")

        else:
            raise ValueError(f"Unsupported mode: {mode}")

def compute_spike_sums(spikes, region_cells, event_times, mode="explicit_event_bounds"):
    """
    Compute spike *sums* per word event per neuron, for a single speaker.
    Returns: dict of {region: (words x neurons) spike sum matrix}
    """
    T = spikes.shape[0]
    result_by_region = {}

    if mode == "explicit_event_bounds":
        word_pres = event_times[:, 1]
        word_posts = event_times[:, 2]
        num_events = len(word_pres)

        for region, neuron_indices in region_cells.items():
            if len(neuron_indices) == 0:
                continue

            region_spikes = spikes[:, neuron_indices]
            n_neurons = region_spikes.shape[1]
            region_result = []

            for neuron_idx in range(n_neurons):
                neuron_spike_train = region_spikes[:, neuron_idx]
                spike_counts = []

                for i in range(num_events):
                    start_idx = int(round(word_pres[i]))
                    end_idx = int(round(word_posts[i])) + 1

                    clipped_start = max(0, start_idx)
                    clipped_end = min(T, end_idx)

                    if clipped_end <= clipped_start:
                        spike_counts.append(np.nan)
                    else:
                        spike_sum = np.sum(neuron_spike_train[clipped_start:clipped_end])
                        spike_counts.append(spike_sum)

                region_result.append(spike_counts)

            region_result = np.column_stack(region_result)
            result_by_region[region] = region_result

    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return result_by_region


def compute_spike_sums_for_all_speakers(spikes, region_cells, speaker_events):
    result = {}

    for spk, events in speaker_events.items():
        result[spk] = compute_spike_sums(
            spikes=spikes,
            region_cells=region_cells,
            event_times=events,
            mode="explicit_event_bounds"
        )

    return result


# === DURATION ANNOTATION ===
def add_regress_dur_column(
    df,
    speaker_of_interest="Speaker1",
    mode="prepost",  # or "fixed_window"
    interest_pre_window=0,
    interest_post_window=0,
    other_pre_window=0,
    other_post_window=0,
    fixed_window_length=500,
):
    """
    Adds a 'regress_dur' column to the DataFrame.

    Modes:
    - "prepost": Uses onset/offset + pre/post windows
    - "fixed_window": Uses fixed_window_length as duration for all words
    """
    speaker_columns = [col for col in df.columns if col.lower().startswith("speaker")]
    durations = []

    for idx, row in df.iterrows():
        onset = row["onset"]
        offset = row["offset"]
        speaker = next((spk for spk in speaker_columns if pd.notna(row.get(spk, np.nan))), None)

        if pd.isna(onset) or pd.isna(offset) or speaker is None:
            durations.append(np.nan)
            continue

        if mode == "fixed_window":
            durations.append(fixed_window_length)
        else:  # prepost mode
            base_dur = offset - onset
            if speaker == speaker_of_interest:
                dur = base_dur + interest_pre_window + interest_post_window
            else:
                dur = base_dur + other_pre_window + other_post_window
            durations.append(dur)

    df["regress_dur"] = durations
    return df


# ALIAS for compatibility
bin_spikes_from_events = compute_spike_sums
