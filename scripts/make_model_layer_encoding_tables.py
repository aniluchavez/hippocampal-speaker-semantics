#!/usr/bin/env python3
"""Make LaTeX tables for semantic encoding across models and layers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SOURCE = Path(
    "/scratch/aniluchavez/ConvoDATAS/SemanticGLM/plots/"
    "all15_bestfixed_model_layer_performance_summary.csv"
)
OUT = Path("/scratch/aniluchavez/hippocampal-speaker-semantics/results")
MODEL_ORDER = [
    "BERT-base",
    "GPT-2 Large",
    "GPT-2 XL",
    "Llama 3.1 8B",
    "fastText static",
]
MODEL_TEX = {
    "BERT-base": "BERT-base",
    "GPT-2 Large": "GPT-2 Large",
    "GPT-2 XL": "GPT-2 XL",
    "Llama 3.1 8B": "LLaMA-3.1-8B",
    "fastText static": "fastText static",
}


def value_iqr(row: pd.Series) -> str:
    return (
        f"{row['median_r2']:.4f} "
        f"$[{row['q25_r2']:.4f},\\,{row['q75_r2']:.4f}]$"
    )


def main() -> None:
    data = pd.read_csv(SOURCE)
    data["model_display"] = pd.Categorical(
        data["model_display"], MODEL_ORDER, ordered=True
    )
    data = data.sort_values(["model_display", "layer", "condition"])

    wide_rows = []
    for (model, layer), group in data.groupby(
        ["model_display", "layer"], observed=True, sort=False
    ):
        by_condition = group.set_index("condition")
        self_row = by_condition.loc["self"]
        other_row = by_condition.loc["other"]
        wide_rows.append(
            {
                "model": str(model),
                "layer": int(layer),
                "relative_depth": self_row["relative_depth"],
                "self_median_r2": self_row["median_r2"],
                "self_q25_r2": self_row["q25_r2"],
                "self_q75_r2": self_row["q75_r2"],
                "other_median_r2": other_row["median_r2"],
                "other_q25_r2": other_row["q25_r2"],
                "other_q75_r2": other_row["q75_r2"],
                "n_patients": int(self_row["n_patients"]),
            }
        )
    wide = pd.DataFrame(wide_rows)
    wide.to_csv(
        OUT / "encoding_performance_across_models_layers.csv", index=False
    )

    best_self = (
        wide.loc[wide.groupby("model")["self_median_r2"].idxmax()]
        .set_index("model")
    )
    best_other = (
        wide.loc[wide.groupby("model")["other_median_r2"].idxmax()]
        .set_index("model")
    )

    # Full layer-by-layer multipage table.
    full = [
        r"\begin{longtable}{l r r l l}",
        (
            r"\caption{Semantic encoding performance across models and layers "
            r"in hippocampus. Values are median cross-validated McFadden "
            r"pseudo-$R^2$ across 15 patient-level medians, with "
            r"$[\mathrm{Q1},\mathrm{Q3}]$.}"
            r"\label{tab:model_layer_encoding_all}\\"
        ),
        r"\toprule",
        (
            r"Model & Layer & Relative depth & Speaking/self & "
            r"Listening/other \\"
        ),
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{5}{c}{\tablename\ \thetable\ (continued)} \\",
        r"\toprule",
        (
            r"Model & Layer & Relative depth & Speaking/self & "
            r"Listening/other \\"
        ),
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{5}{r}{Continued on next page} \\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for model in MODEL_ORDER:
        model_rows = wide.loc[wide["model"].eq(model)].sort_values("layer")
        first = True
        for _, row in model_rows.iterrows():
            model_cell = MODEL_TEX[model] if first else ""
            depth = (
                f"{row['relative_depth']:.2f}"
                if np.isfinite(row["relative_depth"])
                else "--"
            )
            self_value = (
                f"{row['self_median_r2']:.4f} "
                f"$[{row['self_q25_r2']:.4f},\\,"
                f"{row['self_q75_r2']:.4f}]$"
            )
            other_value = (
                f"{row['other_median_r2']:.4f} "
                f"$[{row['other_q25_r2']:.4f},\\,"
                f"{row['other_q75_r2']:.4f}]$"
            )
            if int(row["layer"]) == int(best_self.loc[model, "layer"]):
                self_value = rf"\textbf{{{self_value}}}"
            if int(row["layer"]) == int(best_other.loc[model, "layer"]):
                other_value = rf"\textbf{{{other_value}}}"
            full.append(
                f"{model_cell} & {int(row['layer'])} & {depth} & "
                f"{self_value} & {other_value} \\\\"
            )
            first = False
        full.append(r"\addlinespace")
    full.extend(
        [
            r"\end{longtable}",
            (
                r"\noindent\footnotesize\textit{Note.} Bold denotes the "
                r"highest median within each model and condition. All fits use "
                r"50 fold-wise PCs, five-fold shuffled outer CV, and the "
                r"selected fixed windows (self: $-300$ to $+200$ ms; other: "
                r"$+20$ to $+520$ ms). Layer 0 is the cached embedding/output "
                r"before the first transformer block; the final indices are "
                r"BERT 12, GPT-2 Large 36, GPT-2 XL 48, and LLaMA 32. "
                r"fastText is a static, non-layered baseline. This sweep "
                r"contains real-fit performance only; no permutation test was "
                r"run for these layer comparisons."
            ),
            "",
        ]
    )
    (OUT / "encoding_performance_across_models_layers.tex").write_text(
        "\n".join(full)
    )

    # Compact best-layer table.
    compact_rows = []
    compact_csv = []
    for model in MODEL_ORDER:
        self_row = best_self.loc[model]
        other_row = best_other.loc[model]
        compact_rows.append(
            f"{MODEL_TEX[model]} & {int(self_row['layer'])} & "
            f"{self_row['self_median_r2']:.4f} "
            f"$[{self_row['self_q25_r2']:.4f},\\,"
            f"{self_row['self_q75_r2']:.4f}]$ & "
            f"{int(other_row['layer'])} & "
            f"{other_row['other_median_r2']:.4f} "
            f"$[{other_row['other_q25_r2']:.4f},\\,"
            f"{other_row['other_q75_r2']:.4f}]$ \\\\"
        )
        compact_csv.append(
            {
                "model": model,
                "best_self_layer": int(self_row["layer"]),
                "best_self_median_r2": self_row["self_median_r2"],
                "best_self_q25_r2": self_row["self_q25_r2"],
                "best_self_q75_r2": self_row["self_q75_r2"],
                "best_other_layer": int(other_row["layer"]),
                "best_other_median_r2": other_row["other_median_r2"],
                "best_other_q25_r2": other_row["other_q25_r2"],
                "best_other_q75_r2": other_row["other_q75_r2"],
            }
        )
    compact = [
        r"\begin{table}[ht]",
        r"\centering",
        (
            r"\caption{Best semantic-encoding layer for each model and "
            r"condition in hippocampus. Values are median cross-validated "
            r"McFadden pseudo-$R^2$ across 15 patients "
            r"$[\mathrm{Q1},\mathrm{Q3}]$.}"
        ),
        r"\label{tab:model_best_layer_encoding}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l r l r l}",
        r"\toprule",
        (
            r"Model & Best self layer & Self performance & "
            r"Best other layer & Other performance \\"
        ),
        r"\midrule",
        *compact_rows,
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        (
            r"\begin{flushleft}\footnotesize Fixed windows: self $-300$ to "
            r"$+200$ ms; other $+20$ to $+520$ ms. Fits use 50 fold-wise PCs "
            r"and five-fold shuffled outer CV. Layers are selected separately "
            r"for self and other; these maxima are descriptive and are not "
            r"corrected for layer selection.\end{flushleft}"
        ),
        r"\end{table}",
        "",
    ]
    (OUT / "encoding_performance_best_layers.tex").write_text(
        "\n".join(compact)
    )
    pd.DataFrame(compact_csv).to_csv(
        OUT / "encoding_performance_best_layers.csv", index=False
    )

    print(OUT / "encoding_performance_across_models_layers.tex")
    print(OUT / "encoding_performance_best_layers.tex")


if __name__ == "__main__":
    main()
