"""Rebuild Figure 1L with patient dots and place it over the PDF artwork."""

from pathlib import Path
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from openpyxl import load_workbook
from pypdf import PdfReader, PdfWriter, Transformation


ROOT = Path(__file__).resolve().parents[1] / "figshare" / "SourceData"
SOURCE = ROOT / "ChavezFigure1_V4.ai"
WORKBOOK = ROOT / "Source Data.xlsx"
PANEL_SVG = ROOT / "ChavezFigure1L_patient_dots.svg"
OUTPUT_PDF = ROOT / "ChavezFigure1_V4_patient_dots.pdf"

# Coordinates in the original A4 PDF, in points from its lower-left corner.
PANEL_X, PANEL_Y = 498, 0
PANEL_WIDTH, PANEL_HEIGHT = 97, 242


def main():
    sheet = load_workbook(WORKBOOK, read_only=True, data_only=True)["Fig1l"]
    rows = list(sheet.values)
    columns = {name: index for index, name in enumerate(rows[0])}
    groups = {
        condition: [
            row[columns["patient_median_ll_diff"]]
            for row in rows[1:]
            if row[columns["condition"]] == condition
        ]
        for condition in ("self", "other")
    }
    assert all(len(values) == 15 for values in groups.values())

    fig = plt.figure(figsize=(PANEL_WIDTH / 72, PANEL_HEIGHT / 72), facecolor="white")
    ax = fig.add_axes([0.27, 0.244, 0.71, 0.587])
    colors = ("#f01816", "#1717f7")
    for x, (condition, color) in enumerate(zip(groups, colors)):
        values = groups[condition]
        sem = stdev(values) / np.sqrt(len(values))
        ax.bar(x, mean(values), width=0.54, color=color, zorder=1)
        ax.errorbar(x, mean(values), yerr=sem, fmt="none", color="black", lw=0.9, capsize=2, zorder=3)
        offsets = np.linspace(-0.20, 0.20, len(values))
        ax.scatter(x + offsets, values, s=6, color="black", alpha=0.9, linewidths=0, zorder=4)

    ax.set_xlim(-0.48, 1.48)
    ax.set_ylim(0, 200)
    ax.set_yticks([0, 50, 100, 150, 200])
    ax.set_xticks([0, 1], ["self", "other"])
    ax.set_ylabel("log likelihood (LLH)", fontsize=6, labelpad=1)
    ax.tick_params(axis="both", labelsize=6, length=2, pad=1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(0.65)
    ax.plot([0, 0, 1, 1], [175, 183, 183, 175], color="black", lw=0.75, clip_on=False)
    ax.text(0.5, 187, "p=0.03", ha="center", va="bottom", fontsize=6)
    fig.text(0.01, 0.967, "L", fontsize=21, va="top", ha="left")
    fig.text(0.98, 0.847, "n=15 patients", fontsize=5, ha="right", va="bottom")

    fig.savefig(PANEL_SVG, format="svg")
    from io import BytesIO

    panel_pdf = BytesIO()
    fig.savefig(panel_pdf, format="pdf")
    plt.close(fig)
    panel_pdf.seek(0)

    original_page = PdfReader(SOURCE).pages[0]
    overlay_page = PdfReader(panel_pdf).pages[0]
    original_page.merge_transformed_page(
        overlay_page,
        Transformation().translate(PANEL_X, PANEL_Y),
        over=True,
    )
    writer = PdfWriter()
    writer.add_page(original_page)
    with OUTPUT_PDF.open("wb") as output:
        writer.write(output)
    print(f"Wrote {OUTPUT_PDF}")
    print(f"Wrote {PANEL_SVG}")
    for condition, values in groups.items():
        print(f"{condition}: n={len(values)}, mean={mean(values):.3f}, max={max(values):.3f}")


if __name__ == "__main__":
    main()
