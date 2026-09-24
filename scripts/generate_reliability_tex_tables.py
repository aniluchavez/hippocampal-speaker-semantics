#!/usr/bin/env python3
"""Create publication-ready LaTeX tables for reliability results, at the
neuron level, matching the methodology of notebooks/08_self_other_beta_
reliability.ipynb (Table 1 in the paper):

  - "ceiling" table: is r_cross below the lower within-condition ceiling
    min(r_self, r_other)? One-sample paired t-test on the per-neuron margin
    (r_cross - ceiling), pooled across all patients (neurons are NOT
    restricted to a significance-filtered subset here -- all hippocampus
    neurons are used, unlike the original Table 1's unique_semantic-filtered
    n=602).
  - "cross" table: is r_cross above a shuffle-null permutation baseline?
    Each neuron has a 100-draw null distribution of r_cross (row-permuted
    "other"-condition regressor, refit, from semantic_glm.py --reliability).
    Pool by averaging across neurons within each of the 100 draws to get a
    single null distribution of the pooled mean r_cross, then compare the
    observed pooled mean r_cross against it (one-sided permutation p, floored
    at 1/101).

Both tables draw directly from the semantic_glm.py --reliability pickle
output (SemanticGLM/<model>_<tag>_notebookexact_shuf_xcirc_r2only/pc50/
PTY*_L<layer>_sem.pkl), not from the intermediate violin-plot CSVs, because
those don't carry the per-neuron null_distribution needed for the
permutation test.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


SEMGLM_ROOT = Path("/scratch/aniluchavez/ConvoDATAS/SemanticGLM")
RESULTS = Path("/scratch/aniluchavez/hippocampal-speaker-semantics/results")
REGION = "hippocampus"
N_BOOT = 10000

RUNS = [
    ("BERT no-tag", "bert-base_ctx200", 12),
    ("BERT spktag", "bert-base-causal_ctx200spktag", 12),
    ("LLaMA-3.1-8B", "llama-3.1-8b_ctx200", 14),
    ("GPT2-XL", "gpt2-xl_ctx200", 45),
]

# Best-layer encoding R^2 (independently-optimized fixed-window McFadden
# pseudo-R^2, median across 15 patients), from encoding_performance_best_layers.csv.
# Pooled across all neurons, not restricted to a significance-filtered subset
# (no per-neuron significance data exists for this table's varwin windows for
# these models). No entry exists for "BERT spktag" in that source table.
BEST_LAYER_R2_SOURCE = RESULTS / "encoding_performance_best_layers.csv"
BEST_LAYER_R2_MODEL_MAP = {
    "BERT no-tag": "BERT-base",
    "LLaMA-3.1-8B": "Llama 3.1 8B",
    "GPT2-XL": "GPT-2 XL",
}


SPKTAG_R2_SOURCE = Path(
    "/scratch/aniluchavez/ConvoDATAS/SemanticGLM/plots/"
    "bert_causal_spktag_tshiftm150_otherp200_layer_llh_patient.csv"
)


def load_spktag_best_layer_r2() -> dict:
    df = pd.read_csv(SPKTAG_R2_SOURCE)
    agg = df.groupby(["condition", "layer"])["patient_median_r2"].median()
    out = {}
    for cond, key in [("self", "self"), ("other", "other")]:
        sub = agg.loc[cond]
        best_layer = int(sub.idxmax())
        out[f"{key}_layer"] = best_layer
        out[f"{key}_r2"] = float(sub.loc[best_layer])
    return out


def load_best_layer_r2() -> dict:
    df = pd.read_csv(BEST_LAYER_R2_SOURCE).set_index("model")
    out = {}
    for model_name, source_name in BEST_LAYER_R2_MODEL_MAP.items():
        row = df.loc[source_name]
        out[model_name] = {
            "self_layer": int(row["best_self_layer"]),
            "self_r2": float(row["best_self_median_r2"]),
            "other_layer": int(row["best_other_layer"]),
            "other_r2": float(row["best_other_median_r2"]),
        }
    out["BERT spktag"] = load_spktag_best_layer_r2()
    return out


BEST_LAYER_R2 = load_best_layer_r2()

ANALYSES = [
    {
        "prefix": "fixed",
        "tag": "bestfixed_selfm300_len500_otherp20_len500",
        "window_desc": "fixed windows (self: $-300$ to $+200$ ms; other: $+20$ to $+520$ ms)",
        "window_note": (
            "Fixed windows were self: $-300$ to $+200$ ms and other: "
            "$+20$ to $+520$ ms."
        ),
    },
    {
        "prefix": "varwin",
        "tag": "varwin_selfm300tooffset0_other0tooffsetp100",
        "window_desc": (
            "variable word-duration windows (self: onset$-300$ ms to offset; "
            "other: onset to offset$+100$ ms)"
        ),
        "window_note": (
            "Variable windows were self: onset$-300$ ms to natural offset and "
            "other: onset to natural offset$+100$ ms."
        ),
    },
]


def fmt_p(value: float) -> str:
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.3f}"


def load_neurons(folder_prefix: str, layer: int, tag: str) -> pd.DataFrame:
    folder = (
        SEMGLM_ROOT / f"{folder_prefix}_{tag}_notebookexact_shuf_xcirc_r2only" / "pc50"
    )
    rows = []
    for path in sorted(folder.glob(f"PTY*_L{layer:02d}_sem.pkl")):
        obj = pickle.load(path.open("rb"))
        for r in obj.get("reliability", {}).get(REGION, []):
            rows.append(
                {
                    "patient": path.name.split(f"_L{layer:02d}")[0],
                    "neuron": r["neuron"],
                    "r_cross": float(r["r_cross"]),
                    "self_reliability": float(r["self_reliability_mean"]),
                    "other_reliability": float(r["other_reliability_mean"]),
                    "null_distribution": np.asarray(r["null_distribution"], dtype=float),
                }
            )
    return pd.DataFrame(rows)


def ceiling_test(neurons: pd.DataFrame, ceiling_col: str) -> dict:
    diff = (neurons["r_cross"] - neurons[ceiling_col]).to_numpy()
    diff = diff[np.isfinite(diff)]
    n = len(diff)
    df = n - 1
    mean_ceiling = float(neurons[ceiling_col].mean())
    mean_cross = float(neurons["r_cross"].mean())
    mean_diff = float(np.mean(diff))
    sd_diff = float(np.std(diff, ddof=1))
    sem = sd_diff / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df)
    ci_low, ci_high = mean_diff - t_crit * sem, mean_diff + t_crit * sem
    t_stat, p_two = stats.ttest_1samp(diff, 0.0)
    p_one = p_two / 2 if t_stat < 0 else 1 - p_two / 2
    cohens_d = mean_diff / sd_diff
    return {
        "n": n,
        "df": df,
        "mean_ceiling": mean_ceiling,
        "mean_cross": mean_cross,
        "mean_diff": mean_diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "t": float(t_stat),
        "d": cohens_d,
        "p": float(p_one),
    }


def null_test(neurons: pd.DataFrame, rng: np.random.Generator) -> dict:
    keep = neurons["r_cross"].apply(np.isfinite) & neurons["null_distribution"].apply(
        lambda a: bool(np.all(np.isfinite(a)))
    )
    neurons = neurons.loc[keep]
    null_matrix = np.vstack(neurons["null_distribution"].to_numpy())  # (n_neurons, n_null)
    r_null = null_matrix.mean(axis=0)  # pooled null, one value per draw
    r_obs = float(neurons["r_cross"].mean())
    z = (r_obs - r_null.mean()) / r_null.std()
    p_one = (np.sum(r_null >= r_obs) + 1) / (len(r_null) + 1)

    # bootstrap CI for the observed pooled mean r_cross (resample neurons)
    values = neurons["r_cross"].to_numpy()
    draws = rng.choice(values, size=(N_BOOT, len(values)), replace=True)
    ci_low, ci_high = np.quantile(draws.mean(axis=1), [0.025, 0.975])
    return {
        "n": len(values),
        "n_null": null_matrix.shape[1],
        "r_obs": r_obs,
        "mean_null": float(r_null.mean()),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "z": float(z),
        "p": float(p_one),
    }


def write_ceiling_tex(rows: list[dict], analysis: dict, out_path: Path) -> None:
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{Hippocampal beta reliability with {analysis['window_desc']}: "
        rf"$r_\mathrm{{cross}}$ vs. each within-condition reliability ceiling. "
        rf"One-sample paired $t$-test per neuron.}}",
        rf"\label{{tab:{analysis['prefix']}_reliability_ceiling_hippocampus}}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l l r r r l r r l}",
        r"\toprule",
        (
            r"Model & Ceiling & $n$ & Mean cross & Mean ceiling & 95\% CI (diff.) & "
            r"$t$ ($df$) & Cohen's $d$ & $p$ \\"
        ),
        r"\midrule",
    ]
    previous_model = None
    for row in rows:
        model = row["Model"] if row["Model"] != previous_model else ""
        if previous_model is not None and model:
            lines.append(r"\addlinespace")
        lines.append(
            f"{model} & {row['Ceiling']} & {row['n']} & "
            f"{row['mean_cross']:.3f} & {row['mean_ceiling']:.3f} & "
            f"$[{row['ci_low']:.3f}, {row['ci_high']:.3f}]$ & "
            f"${row['t']:.2f}$ (${row['df']}$) & ${row['d']:.2f}$ & "
            f"{fmt_p(row['p'])} \\\\"
        )
        previous_model = row["Model"]
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            (
                r"\begin{flushleft}\footnotesize "
                r"Each row pools all hippocampus neurons across 15 patients "
                r"(neurons are not restricted to any significance-filtered "
                r"subset). Mean cross and mean ceiling are $r_\mathrm{cross}$ "
                r"and the named within-condition split-half reliability "
                r"(Speaking $=r_\mathrm{self}$, Listening $=r_\mathrm{other}$), "
                r"averaged over neurons. The $t$-test, CI, $d$, and $p$ are for "
                r"the per-neuron difference $r_\mathrm{cross}-\mathrm{ceiling}$; "
                r"$p$ is one-sided (difference $<0$). "
                rf"{analysis['window_note']} "
                r"Layers: BERT L12, LLaMA-3.1-8B L14, GPT2-XL L45."
                r"\end{flushleft}"
            ),
            r"\end{table}",
            "",
        ]
    )
    out_path.write_text("\n".join(lines))


def write_cross_tex(rows: list[dict], analysis: dict, out_path: Path) -> None:
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{Hippocampal beta $r_\mathrm{{cross}}$ reliability with "
        rf"{analysis['window_desc']}: tested against a shuffle-null "
        rf"permutation baseline, and against the lower of the two "
        rf"within-condition reliability ceilings.}}",
        rf"\label{{tab:{analysis['prefix']}_reliability_cross_hippocampus}}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l l r r l l r r l l}",
        r"\toprule",
        (
            r"Model & Metric & $n$ & Mean cross & "
            r"Mean ceiling (Speak/Listen) & 95\% CI & "
            r"Statistic & Cohen's $d$ & $p$ & "
            r"Encoding $R^2$ (best layer, Self/Other) \\"
        ),
        r"\midrule",
    ]
    previous_model = None
    for row in rows:
        model = row["Model"] if row["Model"] != previous_model else ""
        if previous_model is not None and model:
            lines.append(r"\addlinespace")
        best_r2 = BEST_LAYER_R2.get(row["Model"])
        if best_r2 is None:
            r2_str = "---"
        else:
            marker = "$^\\ddagger$" if row["Model"] == "BERT spktag" else ""
            r2_str = (
                f"{best_r2['self_r2']:.4f} (L{best_r2['self_layer']}) / "
                f"{best_r2['other_r2']:.4f} (L{best_r2['other_layer']}){marker}"
            )
        if row["kind"] == "null":
            floor = 1 / (row["n_null"] + 1)
            p_str = f"{fmt_p(row['p'])}$^\\dagger$" if row["p"] <= floor else fmt_p(row["p"])
            lines.append(
                f"{model} & above null & {row['n']} & "
                f"{row['r_obs']:.3f} & --- & "
                f"$[{row['ci_low']:.3f}, {row['ci_high']:.3f}]$ & "
                f"$z={row['z']:.2f}$ & --- & {p_str} & {r2_str} \\\\"
            )
        else:
            lines.append(
                f"{model} & below lower ceiling & {row['n']} & "
                f"{row['mean_cross']:.3f} & "
                f"{row['mean_self']:.3f}/{row['mean_other']:.3f} & "
                f"$[{row['ci_low']:.3f}, {row['ci_high']:.3f}]$ & "
                f"$t({row['df']})={row['t']:.2f}$ & ${row['d']:.2f}$ & "
                f"{fmt_p(row['p'])} & \\\\"
            )
        previous_model = row["Model"]
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            (
                r"\begin{flushleft}\footnotesize "
                r"Each row pools all hippocampus neurons across 15 patients "
                r"(neurons are not restricted to any significance-filtered "
                r"subset). Mean cross is $r_\mathrm{cross}$ averaged over "
                r"neurons. "
                r"For ``above null,'' each neuron's split-half "
                r"$r_\mathrm{cross}$ was recomputed on 100 row-permuted "
                r"(``other''-condition) draws, and the null distribution is "
                r"the per-draw mean across neurons; 95\% CI is a percentile "
                r"bootstrap interval (10,000 neuron resamples) for mean "
                r"cross (not the null), and $z$/one-sided $p$ compare mean "
                r"cross to that null mean "
                r"($\approx0$; not shown as a column). "
                r"$^\dagger$ no null draw reached the observed value; $p$ is "
                r"reported at the floor of the permutation's resolution, "
                r"$1/(n_\mathrm{null}+1)\approx0.0099$; Cohen's $d$ is not "
                r"applicable. "
                r"For ``below lower ceiling,'' mean ceiling (Speak/Listen) "
                r"gives the two within-condition split-half reliabilities "
                r"($r_\mathrm{self}$ during Speaking, $r_\mathrm{other}$ "
                r"during Listening) separately, for reference only -- not "
                r"separately tested here; the test itself is against their "
                r"per-neuron minimum $\min(r_\mathrm{self},r_\mathrm{other})$. "
                r"95\% CI, $t$, $d$, and one-sided $p$ (difference $<0$) are "
                r"from a one-sample paired $t$-test on the per-neuron margin "
                r"$r_\mathrm{cross}-\min(r_\mathrm{self},r_\mathrm{other})$. "
                rf"{analysis['window_note']} "
                r"Layers: BERT L12, LLaMA-3.1-8B L14, GPT2-XL L45. "
                r"Encoding $R^2$ (best layer, Self/Other) is each model's "
                r"independently-optimized fixed-window (self $-300$ to "
                r"$+200$ ms; other $+20$ to $+520$ ms) cross-validated "
                r"McFadden pseudo-$R^2$, median across 15 patients, at that "
                r"model's own best self layer and best other layer; pooled "
                r"across all neurons, not restricted to a "
                r"significance-filtered subset, and not from the varwin "
                r"window used elsewhere in this table. "
                r"$^\ddagger$BERT spktag uses a different fixed window "
                r"(self shift $-150$ ms, other shift $+200$ ms) from a "
                r"separate source (bert\_causal\_spktag\_tshiftm150\_"
                r"otherp200\_layer\_llh\_patient.csv), not the "
                r"self $-300$/$+200$ ms, other $+20$/$+520$ ms window used "
                r"for the other three models."
                r"\end{flushleft}"
            ),
            r"\end{table}",
            "",
        ]
    )
    out_path.write_text("\n".join(lines))


def main() -> None:
    for index, analysis in enumerate(ANALYSES):
        rng = np.random.default_rng(2026 + index)
        ceiling_rows, cross_rows = [], []
        for model_name, folder_prefix, layer in RUNS:
            neurons = load_neurons(folder_prefix, layer, analysis["tag"])
            neurons["lower_ceiling"] = neurons[
                ["self_reliability", "other_reliability"]
            ].min(axis=1)

            for ceiling_col, label in [
                ("self_reliability", "Speaking"),
                ("other_reliability", "Listening"),
            ]:
                stat = ceiling_test(neurons, ceiling_col)
                stat["Model"] = model_name
                stat["Ceiling"] = label
                ceiling_rows.append(stat)

            null_stat = null_test(neurons, rng)
            null_stat["Model"] = model_name
            null_stat["kind"] = "null"
            best_r2 = BEST_LAYER_R2.get(model_name)
            if best_r2 is not None:
                null_stat["encoding_self_layer"] = best_r2["self_layer"]
                null_stat["encoding_self_r2"] = best_r2["self_r2"]
                null_stat["encoding_other_layer"] = best_r2["other_layer"]
                null_stat["encoding_other_r2"] = best_r2["other_r2"]
            cross_rows.append(null_stat)

            lower_stat = ceiling_test(neurons, "lower_ceiling")
            lower_stat["Model"] = model_name
            lower_stat["kind"] = "lower_ceiling"
            lower_stat["mean_self"] = float(neurons["self_reliability"].mean())
            lower_stat["mean_other"] = float(neurons["other_reliability"].mean())
            cross_rows.append(lower_stat)

        cross_tex = RESULTS / f"{analysis['prefix']}_reliability_cross_hippocampus_table.tex"
        cross_csv = RESULTS / f"{analysis['prefix']}_reliability_cross_hippocampus_table.csv"
        ceiling_tex = RESULTS / f"{analysis['prefix']}_reliability_ceiling_hippocampus_table.tex"
        ceiling_csv = RESULTS / f"{analysis['prefix']}_reliability_ceiling_hippocampus_table.csv"

        pd.DataFrame(cross_rows).to_csv(cross_csv, index=False)
        pd.DataFrame(ceiling_rows).to_csv(ceiling_csv, index=False)
        write_cross_tex(cross_rows, analysis, cross_tex)
        write_ceiling_tex(ceiling_rows, analysis, ceiling_tex)

        for p in (cross_tex, cross_csv, ceiling_tex, ceiling_csv):
            print(p)


if __name__ == "__main__":
    main()
