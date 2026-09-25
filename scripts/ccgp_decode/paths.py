# ccgp_decode/paths.py
from __future__ import annotations
import os, re
from dataclasses import dataclass

_TAG_RE = re.compile(r"^PT([A-Z]{3})_TASK0*(\d+)$", re.IGNORECASE)

@dataclass(frozen=True)
class PatientTag:
    short_id: str     # YFF
    task_int: int     # 17
    task_unpadded: str  # task17
    task_padded3: str   # task017
    canonical: str      # PTYFF_task17

def parse_patient_tag(patient_tag: str) -> PatientTag:
    s = patient_tag.strip()
    m = _TAG_RE.match(s.upper())
    if not m:
        raise ValueError(f"Bad patient_tag: {patient_tag} (expected 'PTYFF_task17' or 'PTYFF_task017')")
    short_id = m.group(1)
    task_int = int(m.group(2))
    return PatientTag(
        short_id=short_id,
        task_int=task_int,
        task_unpadded=f"task{task_int}",
        task_padded3=f"task{task_int:03d}",
        canonical=f"PT{short_id}_task{task_int}",
    )

@dataclass(frozen=True)
class PatientPaths:
    spikes_mat: str
    excel: str
    embeddings: str
    canonical: str

def build_paths(patient_tag: str, spikes_root: str, excel_root: str, embeddings_root: str) -> PatientPaths:
    t = parse_patient_tag(patient_tag)

    spikes_candidates = [
        os.path.join(spikes_root, t.short_id, f"pt{t.short_id}_{t.task_unpadded}_new_spikes.mat"),
        os.path.join(spikes_root, t.short_id, f"pt{t.short_id}_{t.task_padded3}_new_spikes.mat"),
    ]
    spikes_mat = next((p for p in spikes_candidates if os.path.exists(p)), spikes_candidates[0])

    folder_candidates = [f"{t.canonical}_words_english_only", f"{patient_tag}_words_english_only"]
    file_candidates = [
        f"{t.canonical}_filtered_used_rows_withNP_withClusterIDNew.xlsx",
        f"{patient_tag}_filtered_used_rows_withNP_withClusterIDNew.xlsx",
    ]

    excel = None
    for fold in folder_candidates:
        for fn in file_candidates:
            p = os.path.join(excel_root, fold, fn)
            if os.path.exists(p):
                excel = p
                break
        if excel:
            break

    if excel is None:
        excel = os.path.join(excel_root, f"{t.canonical}_words_english_only",
                             f"{t.canonical}_filtered_used_rows_withNP_withClusterIDNew.xlsx")
        
    # ---------------- EMBEDDINGS ----------------
    emb_folder_candidates = [
        f"{t.canonical}_words_english_only",
        f"{patient_tag}_words_english_only",
    ]

    emb_file_candidates = [
        f"{t.canonical}_aligned_embeddings_withNP.csv",
        f"{patient_tag}_aligned_embeddings_withNP.csv",
    ]

    embeddings = None
    for fold in emb_folder_candidates:
        for fn in emb_file_candidates:
            p = os.path.join(embeddings_root, fold, fn)
            if os.path.exists(p):
                embeddings = p
                break
        if embeddings:
            break

    if embeddings is None:
        embeddings = os.path.join(
            embeddings_root,
            f"{t.canonical}_words_english_only",
            f"{t.canonical}_aligned_embeddings_withNP.csv",
        )

    return PatientPaths(spikes_mat=spikes_mat, excel=excel, embeddings=embeddings, canonical=t.canonical)