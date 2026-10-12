#!/usr/bin/env python3
"""Appendix A figure: how similar titles are to one another in the corpus.

Research question
-----------------
None. Reference distributions for reading cosine similarities in the Method section ("Defining
Presentation-Heavy Areas"). Top: similarity over all distinct pairs of titles (median about .18),
with the 99th percentile that defines a closely related article (.525). Bottom: each title's
similarity to its single most similar title (median about .74) and to its 75th most similar
title, the edge of its neighborhood (median about .50).

Data
----
All 2016-2026 titles (data/titles), all-MiniLM-L6-v2, unit length, read through
scripts/screen.records() so that the titles match the analysis.

Reproducibility
---------------
    python scripts/appendix_a_similarity.py

Writes figures/appendix_a_similarity.png (300 dpi) and .pdf, and prints the medians.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib
import numpy as np

from presented_published.neighborhood import nearest, similarity
from presented_published.style import PublicationPlotStyle
from screen import K0, Q, records

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path("figures/appendix_a_similarity")
S = PublicationPlotStyle()
INK, MID, LIGHT = "#111111", "#888888", "#BBBBBB"
BINS = np.linspace(-0.2, 1.0, 121)


def apa(v: float, d: int = 2) -> str:
    """No leading zero, as for values that cannot exceed 1 (APA style)."""
    return f"{v:.{d}f}".replace("0.", ".", 1)


def style() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix", "axes.labelsize": S.axis_label_fontsize,
        "axes.labelpad": S.labelpad, "xtick.labelsize": S.tick_fontsize,
        "ytick.labelsize": S.tick_fontsize, "legend.fontsize": S.legend_fontsize,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False})


def main() -> None:
    V, _ = records()
    Sm = similarity(V)
    nn = nearest(Sm, K0)
    first = Sm[np.arange(len(V)), nn[:, 0]]
    kth = Sm[np.arange(len(V)), nn[:, K0 - 1]]
    pairs = (V @ V.T)[np.triu_indices(len(V), 1)]
    line = float(np.quantile(pairs, Q))
    med = {"pairs": float(np.median(pairs)), "first": float(np.median(first)),
           "kth": float(np.median(kth))}

    style()
    fig, (a, b) = plt.subplots(2, 1, figsize=(9, 10), sharex=True)
    a.hist(pairs, bins=BINS, color=MID, weights=np.full(len(pairs), 1 / len(pairs)))
    b.hist(first, bins=BINS, histtype="step", lw=2, color=INK,
           weights=np.full(len(first), 1 / len(first)),
           label=f"Most similar title (median {apa(med['first'])})")
    b.hist(kth, bins=BINS, histtype="step", lw=2, color=MID, ls="--",
           weights=np.full(len(kth), 1 / len(kth)),
           label=f"{K0}th most similar title (median {apa(med['kth'])})")
    for ax in (a, b):
        ax.axvline(line, color=INK, lw=1, ls=":")
        ax.set_ylabel("Proportion")
    a.text(line + 0.015, a.get_ylim()[1] * 0.95, f"99th percentile = {apa(line, 3)}",
           fontsize=S.annotation_fontsize + 2, va="top")
    a.text(line + 0.015, a.get_ylim()[1] * 0.80,
           f"All {len(pairs):,} pairs of titles\n(median {apa(med['pairs'])})",
           fontsize=S.annotation_fontsize + 2, va="top")
    b.legend(frameon=False, loc="upper left")
    b.set_xlabel("Cosine similarity")
    b.set_xlim(-0.2, 1.0)
    b.set_xticks(np.arange(-0.2, 1.01, 0.2),
                 ["−.2", "0", ".2", ".4", ".6", ".8", "1"])
    fig.tight_layout(h_pad=2)
    meta = f"Generated {date.today().isoformat()} by scripts/appendix_a_similarity.py"
    fig.savefig(OUT.with_suffix(".png"), dpi=300, metadata={"Description": meta})
    fig.savefig(OUT.with_suffix(".pdf"), metadata={"Subject": meta})
    print(f"most similar title at or above .99: {(first >= 0.99).sum():,} titles")
    print(f"{len(V):,} titles; medians: pairs {med['pairs']:.3f}, most similar "
          f"{med['first']:.3f}, {K0}th {med['kth']:.3f}; 99th percentile {line:.4f}")


if __name__ == "__main__":
    main()
