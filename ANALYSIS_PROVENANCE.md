# Analysis provenance

This document maps each figure/table in the manuscript and supplementary
material to the script(s) and pipeline stage that generated it, so the
analysis can be re-run or audited without re-deriving it from scratch. It
intentionally contains no data or rendered figures/tables (all outputs are
git-ignored: `figures/`, `results/`, `figshare/`, `*.pkl`, `*.csv`, `*.xlsx`,
`*.docx`, etc.) — only the code paths and key parameters.

Where a figure/table's generating script could not be located during the
audit that produced this document, it is marked **[undocumented]** rather
than guessed at.

## Pipeline stages (shared across figures)

1. **Spike windows** — `scripts/generate_fixedwindow_spikes.py`,
   `scripts/generate_varwindow_spikes.py`, `scripts/generate_worddur_spikes.py`
   build per-patient spike-count windows from raw event/spike data, keyed by a
   `--out-tag`/window name (e.g. `bestfixed_selfm300_len500_otherp20_len500`,
   `tshift-150_tlen500_oshift+200_olen500`, `fixed_selfm200_otherp100_len500`,
   `worddur`).
2. **Embeddings** — `scripts/extract_embeddings_ctx200_speakertag.py` caches
   contextual embeddings (BERT-base, BERT-base-causal, GPT2-XL) per patient.
3. **Control features** — `scripts/extract_control_features.py` builds the
   lexical/syntactic/acoustic feature sets used as confounds.
4. **Encoding model** — `scripts/semantic_glm.py` is the main Poisson-ridge
   encoding pipeline: nested CV (`--outer_cv block|shuffle`), permutation
   nulls (`--perm_type xcirc|xshuffle`), optional beta-reliability /
   cross-condition r_cross (`--reliability`), optional R²-only mode
   (`--r2_only`, which skips permutation testing entirely — caches built this
   way have `p_perm` hardcoded to 1.0 and cannot be used for significance
   claims).
5. **Variance partitioning** — `scripts/variance_partitioning.py` decomposes
   encoding R² into unique semantic/lexical/syntactic/acoustic contributions;
   output lives under
   `VPResults/<model>_ctx200_<window>_<null>[_symperm]/pc<N>/`.
6. **Residualized cross-condition correlation** —
   `scripts/compute_residualized_rcross.py` (self-other) and
   `scripts/compute_residualized_rcross_other_other.py` (other-other) remove
   lexical/syntactic/acoustic variance from encoding betas before computing
   r_cross; output nested under the VP directory's
   `residualized_rcross(_other_other)/` subfolder.

## Main figures

| Figure | Script(s) | Notes |
|---|---|---|
| Fig 1J | `semantic_glm.py --window_type fixed --spike_tag tshift-150_tlen500_oshift+200_olen500 --model bert-base --layer 12` | −150/+200 ms window described in Methods; distinct from the "bestfixed" window used elsewhere. |
| Fig 1J semantic summary | `scripts/generate_reliability_tex_tables.py` (patient-level mean/SD/SEM aggregation) | |
| Fig 1K, 1L | `scripts/plot_fig1l_bestfixed_dots_overlay.py`, `scripts/overlay_fig1l_patient_dots.py` | "bestfixed" window (self −300/+200 ms, other +20/+520 ms), BERT L12, PC50. |
| Fig 2J, 2K | `semantic_glm.py --reliability` (bestfixed window, BERT L12) | Per-neuron self/other reliability and r_cross. |
| Fig 3B | **[undocumented]** | Category count filtering (raw vs. retained). |
| Fig 3C | Newclustering pipeline (script not identified by name in this audit; regenerated data matches `Fig3C_NEWCLUSTERING_MATCHED_unclipped.pdf`) | |
| Fig 3D | Per-patient one-way ANOVA on cosine distance by semantic category — same underlying analysis as Table S7 | |
| Fig 3E | Pooled neuron-category cosine distances across 15 patients (same pipeline as Fig 3C, pooled) | |
| Fig 4B, 4C | Same reliability pipeline as Fig 2J/K, filtered to the Fig 4 patient subset | |
| Fig 4D | Matched raw-count category means (`Fig4D_MATCHED_samecolor.pdf`) | |
| Fig 4E | Per-patient ANOVA F/p, no-tag / excl-YEZ / speaker-tag variants | |
| Fig 4F | Pooled neuron-category cosine distances, 12-patient subset | |
| Fig 4I | `scripts/three_way_r_cross.py` | Self/other1/other2 pairwise r_cross, BERT L12, PC30, `fixed_selfm200_otherp100_len500` window. |
| Fig 5B, 5C | Per-token word2vec RSA scripts (word-level extraction; script not renamed/identified precisely in this audit) | |
| Fig 5D | Patient-level Spearman RSA between speaking/listening word-word RDMs — same analysis as Table S8 | |
| Fig 5F | 11-class decoding — output consumed from `results500_10/` (`PTY*_hippocampus_timecourse_plotdata.npz`) | Generating decoding script not part of this repo audit; only the plotted/consumed output was verified. |
| Fig 5G | 4-class (grouped) decoding — output consumed from `resultssuper4/` | Same caveat as Fig 5F. |

## Supplementary figures

| Figure | Script(s) | Key parameters |
|---|---|---|
| Supp Fig 1 | `scripts/variance_partitioning.py` → `scripts/make_suppfig1_stacked_vp.py` | VP dirs: `bert-base_ctx200_worddur_xshuffle_cvshuffle_symperm/pc30` (L12) and `gpt2-xl_ctx200_worddur_xshuffle_cvshuffle_symperm/pc30` (L36) — trial-shuffle null, shuffled CV (same null as Table S4). Neuron selection: `p_perm<0.05 & unique_semantic>0 & r2_full>=0.02`, hippocampus only, ranked descending by `unique_semantic`. Reproduces published panel counts exactly (BERT self=45/other=75, GPT2-XL self=63/other=109). |
| Supp Fig 2 | `scripts/plot_rcross_bert_pc30_sigVP.py` | VP dir: `bert-base_ctx200_worddur_xcirc_symperm/pc30` (**circular-shift null, block CV** — deliberately different from Supp Fig 1/Table S4's shuffle null). Reliability dir: `SemanticGLM/bert-base_ctx200_bestfixed_selfm300_len500_otherp20_len500_notebookexact_shuf_xcirc_r2only/pc50`. Selection: `sig_either` = union of per-condition (`p_perm<0.05 & unique_semantic>0`) sets. n=118 neurons, 14 patients. |

## Supplementary tables

| Table | Script(s) | Notes |
|---|---|---|
| Table S1 | `semantic_glm.py` (bestfixed vs. worddur window comparison, BERT L12) | |
| Table S3 | `scripts/generate_reliability_tex_tables.py` and/or `scripts/compute_residualized_rcross.py` aggregation | Patient-level bootstrap CI on r_cross above null / below ceiling (Aarts et al. 2014 pseudoreplication correction — neurons nested within patients). |
| Table S4 | `scripts/compute_table_s4_stats.py` (this session) | Same VP dirs as Supp Fig 1 (`*_xshuffle_cvshuffle_symperm/pc30`). Adds patient-level one-sided Wilcoxon signed-rank test of % significant neurons against the nominal 5% chance rate, with rank-biserial r effect size, per (model, condition, family). |
| Table S5 | `scripts/run_selfother_reliability_bestfixed_multipc_parallel.sbatch` (+ `semantic_glm.py --reliability`) | BERT L12, bestfixed window, PC sweep {5,10,20,50,100,200,300,500}; r_cross vs. permutation null, Wilcoxon W / rank-biserial r / p. |
| Table S6 | `scripts/run_table_s6_cv_null_sweep.sbatch` (+ `semantic_glm.py`) | BERT L12, bestfixed window, PC50, 3 CV/null combinations (Block+circular, Block+shuffled, Shuffle+shuffled) — run with permutation testing actually enabled (no `--r2_only`); the previously cached bestfixed pkl had `p_perm` hardcoded to 1.0 and could not be used to verify this table. |
| Table S7 | Per-patient one-way ANOVA (cosine distance by semantic category) — same analysis as Fig 3D | |
| Table S8 | Per-patient Spearman RSA (speaking vs. listening word-word RDMs) — same analysis as Fig 5D | |

## Known pipeline gotchas (see git history / commit messages for fixes)

- `semantic_glm.py`'s `--force_all` flag does not force recomputation of a
  cached per-patient pkl if that cache already contains `reliability` /
  `perm_nulls` / cosine-bin keys — stale caches must be renamed/removed by
  hand to force a genuine rerun.
- `--r2_only` silently disables permutation testing (`p_perm` is hardcoded to
  `1.0` for every neuron); such caches cannot be used for any significance
  claim.
- Output aggregate pickles (`L<NN>_*_all.pkl`) are rebuilt from *all*
  per-patient files found on disk at run time — running the aggregation step
  with `--patient <ONE_PATIENT>` overwrites the aggregate with just that one
  patient's rows. If this happens, rebuild the aggregate by iterating over
  the intact per-patient pkl files rather than rerunning every patient.
- Patient `PTYFU_task224`'s cached embeddings/spike-windows/control-features
  have repeatedly gone stale relative to a canonical transcript update;
  always check row counts against other patients' caches after any
  transcript-adjacent change.
- Population-level inference must resample **patients**, not neurons
  (neurons are nested within patients — Aarts et al. 2014); several tables
  (S3, S4, S5) use patient-level bootstrap/Wilcoxon tests for this reason.
