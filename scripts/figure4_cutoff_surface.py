#!/usr/bin/env python3
# ruff: noqa: E501  (figure note text)
"""Figure 4: flag rates over the full grid of the two cutoffs, drawn as 3D surfaces.

Research question
-----------------
How does the proportion of titles flagged change with the number of nearest titles that define a
neighborhood (every K from 50 to 100) and with the closely related article cutoff (fewer than
1, i.e. none, up to fewer than 20)? Extends Table 3, which reports K = 50, 60, 70, 75, 80, 90, 100
and cutoffs of 1, 5, and 10.

Independent variables: K = 50..100; cutoff N = 1..20 (flagged when fewer than N closely related
articles; closely related line fixed at the 99th percentile of all-pair similarity).
Dependent variables: percent of titles, presentations, and articles flagged.

Data: scripts/screen.records(). Each K gets its own nearest() call, as in the screen, so the
grid reproduces Table 3 (checked against results/table3_cutoff_sweep.csv).

Known limitation: the surfaces zigzag by about half a percentage point along K because the
minimum number of presentations needed for the lower bound to exceed 0 rises by one at every
second K, so the required proportion alternates slightly.

Outputs
-------
* results/figure4_cutoff_grid.csv (computed once; delete it to recompute)
* figures/figure4_cutoff_surface.png (150 dpi)
Run:  python scripts/figure4_cutoff_surface.py   (after scripts/table3_cutoff_sweep.py)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from presented_published.neighborhood import (
    article_count,
    nearest,
    pair_quantile,
    similarity,
    skew_scores,
)
from screen import K0, N0, Q, records

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

GRID = Path("results/figure4_cutoff_grid.csv")
TABLE3 = Path("results/table3_cutoff_sweep.csv")
OUT = Path("figures/figure4_cutoff_surface.png")
KS = np.arange(50, 101)
CUTS = np.arange(1, 21)
AX = 20


def grid() -> pd.DataFrame:
    if GRID.exists():
        g = pd.read_csv(GRID)
        if set(g["cut"]) >= set(CUTS) and set(g["k"]) >= set(KS):
            return g
    V, rec = records()
    pres = rec["presented"].to_numpy()
    S = similarity(V)
    count = article_count(S, pres, pair_quantile(V, Q))
    rows = []
    for k in KS:
        sk = skew_scores(nearest(S, int(k)), pres, int(k))["skewed"]
        for n in CUTS:
            f = sk & (count < n)
            rows.append({"k": int(k), "cut": int(n), "titles_pct": f.mean() * 100,
                         "pres_pct": f[pres].mean() * 100, "pub_pct": f[~pres].mean() * 100})
        print(f"K = {k}", flush=True)
    g = pd.DataFrame(rows)
    t = pd.read_csv(TABLE3).rename(columns={"n": "cut"})
    m = t.merge(g, on=["k", "cut"], suffixes=("_t", "_g"))
    if len(m) != len(t) or (m.titles_pct_t - m.titles_pct_g).abs().max() > 1e-9:
        raise SystemExit("grid does not reproduce Table 3")
    GRID.parent.mkdir(parents=True, exist_ok=True)
    g.to_csv(GRID, index=False)
    return g


def main() -> None:
    g = grid()
    g = g[g["cut"].isin(CUTS) & g["k"].isin(KS)]
    ZMAX = float(np.ceil(g[["titles_pct", "pres_pct", "pub_pct"]].to_numpy().max() / 5) * 5)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"]})
    fig = plt.figure(figsize=(21, 8.5))
    for j, (col, title) in enumerate([("titles_pct", "Titles Flagged"), ("pres_pct", "Presentations Flagged"),
                                       ("pub_pct", "Articles Flagged")]):
        P = g.pivot(index="cut", columns="k", values=col).loc[CUTS, KS]
        X, Y = np.meshgrid(KS, CUTS)
        ax = fig.add_subplot(1, 3, j + 1, projection="3d", computed_zorder=False)
        ax.plot_surface(X, Y, P.to_numpy(), cmap="Greys", vmin=-0.2 * ZMAX, vmax=ZMAX, edgecolor="#555",
                        linewidth=0.15, rstride=1, cstride=2, antialiased=True, zorder=1)
        lift = 0.15   # traces sit just on top of the surface
        k_in = KS[KS <= K0]
        c_in = CUTS[CUTS <= N0]
        ax.plot(k_in, np.full(len(k_in), N0), P.loc[N0, k_in].to_numpy() + lift,
                color="#d00000", lw=2, ls="--", zorder=5)
        ax.plot(np.full(len(c_in), K0), c_in, P.loc[c_in, K0].to_numpy() + lift,
                color="#d00000", lw=2, ls="--", zorder=5)
        ax.scatter([K0], [N0], [float(P.loc[N0, K0]) + lift], color="#d00000", edgecolor="white",
                   linewidth=1.5, s=200, depthshade=False, zorder=6)
        ax.set_xlabel("Neighbors", fontsize=AX, labelpad=14)
        ax.set_ylabel("Closely Related\nArticles Cutoff", fontsize=AX, labelpad=20)
        ax.zaxis.set_rotate_label(False)
        ax.set_zlabel("Percent Flagged", fontsize=AX, labelpad=12, rotation=90)
        ax.tick_params(labelsize=12)
        ax.set_xticks([50, 60, 70, 75, 80, 90, 100])
        ax.set_yticks([1, 5, 10, 15, 20])
        ax.set_zlim(0, ZMAX)
        ax.set_title(title, fontsize=22, y=0.86)
        ax.view_init(elev=24, azim=-125)
        ax.set_box_aspect(None, zoom=0.80)
        for a in (ax.xaxis, ax.yaxis, ax.zaxis):
            a.pane.set_facecolor("white")
            a.pane.set_edgecolor("#ddd")
    fig.text(0.5, 0.025, "Red: The Definition Used in the Main Analysis (75 Neighbors, Fewer Than 10 Closely Related "
             "Articles), With Dashed Lines Along the Surface at Each Value. Same Vertical Scale in All Panels.",
             ha="center", fontsize=13)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.08, wspace=0.14)
    fig.savefig(OUT, dpi=150)
    main_vals = g[(g.k == K0) & (g.cut == N0)].iloc[0]
    print(f"wrote {OUT}; main analysis: titles {main_vals.titles_pct:.1f}%, presentations "
          f"{main_vals.pres_pct:.1f}%, articles {main_vals.pub_pct:.1f}%")


if __name__ == "__main__":
    main()
