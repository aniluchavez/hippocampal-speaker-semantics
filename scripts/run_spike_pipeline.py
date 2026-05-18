import os
from pathlib import Path
import pandas as pd
from extract_spikes_window import (
    load_mat_data,
    get_cells_by_region,
    extract_speaker_events,
    process_and_save_spikes,
    process_and_save_spike_sum,
    add_regress_dur_column,
    extract_sliding_window_spikes
)

def run_patient_pipeline(patient_id, patient_prefix, mat_base, excel_base, region_ranges,
                         speaker_window_modes, binSize=20, output_suffix="w2v",
                         spike_base_dir="/Users/aniluchavez/Documents/Language/Python/spikesforw2v",
                         use_sliding_window=False, shift_ms=0, return_spike_data=False):

    from pathlib import Path
    import os
    import pandas as pd
    from extract_spikes_window import (
        load_mat_data, get_cells_by_region, extract_speaker_events,
        process_and_save_spikes, process_and_save_spike_sum, add_regress_dur_column,
        extract_sliding_window_spikes
    )

    file_path_mat = f"{mat_base}/{patient_id}_new_spikes.mat"
    excel_dir = Path(excel_base) / f"{patient_prefix}_words_word2vec"
    file_path_xlsx = excel_dir / f"{patient_prefix}_filtered_used_rows_word2vec.xlsx"
    df = pd.read_excel(file_path_xlsx, sheet_name="Sheet1")
    df.columns = [col.strip().replace("’", "").replace("‘", "").replace("“", "").replace("”", "") for col in df.columns]

    actual_speakers = [col for col in df.columns if str(col).startswith("Speaker") and df[col].notna().any()]
    requested_speakers = set(speaker_window_modes.keys())
    missing = requested_speakers - set(actual_speakers)
    if missing:
        print(f"⚠️ Dropped {len(missing)} speaker(s) not in Excel: {missing}")
    speaker_window_modes = {spk: cfg for spk, cfg in speaker_window_modes.items() if spk in actual_speakers}

    output_dir = os.path.join(spike_base_dir, f"output_{patient_prefix}_english_only_{output_suffix}")
    os.makedirs(output_dir, exist_ok=True)

    spikes, qual, chan = load_mat_data(file_path_mat)
    region_cells = get_cells_by_region(chan, qual, region_ranges)

    print("🧩 Final speaker_window_modes keys:", list(speaker_window_modes.keys()))
    speaker_events = extract_speaker_events(file_path_xlsx, speaker_window_modes)
    print("Speakers found in this file:", list(speaker_events.keys()))

    if use_sliding_window:
        print(f"🪟 Using SLIDING window spikes with shift={shift_ms}ms")
        spike_data = extract_sliding_window_spikes(
            spikes=spikes,
            region_cells=region_cells,
            speaker_events=speaker_events,
            shift_ms=shift_ms
        )
    else:
        process_and_save_spikes(spikes, region_cells, speaker_events, binSize, output_dir)
        process_and_save_spike_sum(spikes, region_cells, speaker_events, output_dir)
        spike_data = None

    df_with_dur = add_regress_dur_column(df, speaker_window_modes)
    output_excel_path = os.path.join(output_dir, f"{patient_id}_with_regress_dur_{output_suffix}.xlsx")
    df_with_dur.to_excel(output_excel_path, index=False)

    if return_spike_data:
        return spike_data, output_dir, output_excel_path
    else:
        print(f"✅ Annotated Excel saved to: {output_excel_path}")
        return output_dir, output_excel_path


##### This is an OLD version (June 2025) ######
# def run_patient_pipeline(patient_id, patient_prefix, mat_base, excel_base, region_ranges,
#                          speaker_window_modes, binSize=20, output_suffix="w2v",
#                          spike_base_dir="/Users/aniluchavez/Documents/Language/Python/spikesforw2v"):
    
#     from pathlib import Path
#     import os
#     import pandas as pd

#     # === Paths ===
#     file_path_mat = f"{mat_base}/{patient_id}_new_spikes.mat"
#     excel_dir = Path(excel_base) / f"{patient_prefix}_words_word2vec"
#     file_path_xlsx = excel_dir / f"{patient_prefix}_filtered_used_rows_word2vec.xlsx"
#     df = pd.read_excel(file_path_xlsx, sheet_name="Sheet1")
#     df.columns = [col.strip() for col in df.columns] 
#     df.columns = [col.strip().replace("’", "").replace("‘", "").replace("“", "").replace("”", "") for col in df.columns]
#  # ✅ Add this

#     # Get speaker columns that actually have non-NaN values
#     actual_speakers = [col for col in df.columns if str(col).startswith("Speaker") and df[col].notna().any()]

#     # Warn about requested speakers not found in Excel
#     requested_speakers = set(speaker_window_modes.keys())
#     missing = requested_speakers - set(actual_speakers)
#     if missing:
#         print(f"⚠️ Dropped {len(missing)} speaker(s) not in Excel: {missing}")

#     # Filter to valid speakers only
#     speaker_window_modes = {spk: cfg for spk, cfg in speaker_window_modes.items() if spk in actual_speakers}


#     output_dir = os.path.join(spike_base_dir, f"output_{patient_prefix}_english_only_{output_suffix}")
#     os.makedirs(output_dir, exist_ok=True)

#     # === Load spike data ===
#     spikes, qual, chan = load_mat_data(file_path_mat)
#     region_cells = get_cells_by_region(chan, qual, region_ranges)

#     # === Get event windows ===
#     print("🧩 Final speaker_window_modes keys:", list(speaker_window_modes.keys()))

#     speaker_events = extract_speaker_events(file_path_xlsx, speaker_window_modes)
#     print("Speakers found in this file:", list(speaker_events.keys()))

#     # === Process spikes ===
#     process_and_save_spikes(spikes, region_cells, speaker_events, binSize, output_dir)
#     process_and_save_spike_sum(spikes, region_cells, speaker_events, output_dir)

#     # === Save annotated Excel with regress_dur ===
#     df = pd.read_excel(file_path_xlsx, sheet_name="Sheet1")
#     df_with_dur = add_regress_dur_column(df, speaker_window_modes)
#     output_excel_path = os.path.join(output_dir, f"{patient_id}_with_regress_dur_{output_suffix}.xlsx")
#     df_with_dur.to_excel(output_excel_path, index=False)
#     print(f"✅ Annotated Excel saved to: {output_excel_path}")

