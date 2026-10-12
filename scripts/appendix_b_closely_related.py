#!/usr/bin/env python3
"""Appendix B: supporting analysis for the closely related article line.

Research question
-----------------
The Method anchors "closely related" to the data: an article is closely related to a title when
their cosine similarity is at least the 99th percentile of similarity over all distinct pairs of
titles (.525). This appendix shows (a) a histogram of the exact number of closely related
articles per title under that line, with the cut of fewer than 10 and the median marked, and (b)
how the counts and the share of titles with fewer than 10 change if the line is drawn at other
percentiles (95th to 99.9th).

Data
----
All 2016-2026 titles (data/titles), all-MiniLM-L6-v2, unit length, read through
scripts/screen.records() so that the titles match the analysis.

Outputs
-------
* figures/appendix_b_closely_related.png (300 dpi) and .pdf
* results/appendix_b_lines.csv: per percentile, the similarity line, the median and
  interquartile range of closely related articles per title, the share with none, and the share
  with fewer than 10.

Reproducibility
---------------
    python scripts/appendix_b_closely_related.py
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib
import matplotlib.ticker
import numpy as np
import pandas as pd

from presented_published.neighborhood import article_count, similarity
from presented_published.style import PublicationPlotStyle
from screen import N0, Q, records

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path("figures/appendix_b_closely_related")
TABLE = Path("results/appendix_b_lines.csv")
SCREEN = Path("results/neighborhood_screen.parquet")
PERCENTILES = (0.95, 0.975, 0.99, 0.995, 0.999)
S = PublicationPlotStyle()
INK, MID, LIGHT = "#111111", "#888888", "#BBBBBB"


def style() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix", "axes.labelsize": S.axis_label_fontsize,
        "axes.labelpad": S.labelpad, "xtick.labelsize": S.tick_fontsize,
        "ytick.labelsize": S.tick_fontsize, "legend.fontsize": S.legend_fontsize,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False})


def main() -> None:
    V, rec = records()
    pres = rec["presented"].to_numpy()
    Sm = similarity(V)
    pairs = (V @ V.T)[np.triu_indices(len(V), 1)]
    rows = []
    counts = {}
    for q in PERCENTILES:
        line = float(np.quantile(pairs, q))
        c = article_count(Sm, pres, line)
        counts[q] = c
        rows.append({"percentile": q * 100, "line": line, "median": float(np.median(c)),
                     "q25": float(np.percentile(c, 25)), "q75": float(np.percentile(c, 75)),
                     "none": float((c == 0).mean()), "fewer_than_cut": float((c < N0).mean())})
    tab = pd.DataFrame(rows)
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    tab.to_csv(TABLE, index=False)
    c = counts[Q]
    arch = pd.read_parquet(SCREEN)["closely_related_articles"].to_numpy()
    if not np.array_equal(c, arch):
        raise SystemExit("counts at the 99th percentile differ from the archived screen")

    style()
    fig, ax = plt.subplots(figsize=(10, 6))
    n = np.bincount(c)
    x = np.arange(len(n))
    ax.bar(x, n, width=1.0, color=[INK if v < N0 else MID for v in x], linewidth=0)
    ax.axvline(N0 - 0.5, color=INK, ls="--", lw=1)
    ax.text(60, n.max() * 0.97, f"Dashed line: fewer than {N0} to its left", va="top",
            fontsize=S.annotation_fontsize + 2)
    med = float(np.median(c))
    ax.axvline(med, color=INK, ls=":", lw=1)
    ax.text(60, n.max() * 0.89, f"Dotted line: median = {med:g}", va="top",
            fontsize=S.annotation_fontsize + 2)
    ax.set_xlim(-2, len(n) + 2)
    ax.set_ylim(0, n.max() * 1.03)
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
    ax.set_xlabel("Closely related articles per title")
    ax.set_ylabel("Titles")
    fig.tight_layout()
    meta = f"Generated {date.today().isoformat()} by scripts/appendix_b_closely_related.py"
    fig.savefig(OUT.with_suffix(".png"), dpi=300, metadata={"Description": meta})
    fig.savefig(OUT.with_suffix(".pdf"), metadata={"Subject": meta})
    print(f"{len(V):,} titles; counts 0-{len(n) - 1}; titles with 0-4: {n[:5].tolist()}")
    print(tab.to_string(index=False))


if __name__ == "__main__":
    main()
