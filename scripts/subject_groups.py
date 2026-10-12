#!/usr/bin/env python3
# ruff: noqa: E501  (report template)
"""Describe the flagged titles by subject area (Table 2, Figures 2 and 3).

Titles are sorted into 200 subject-area groups (presented_published.groups: k-means, seed 0,
fit on the ABAI and general-journal fit titles; "challenging behavior" and "problem behavior"
groups merged), and each group's proportion of flagged titles is reported. The groups are a
reporting device: no title's flag depends on its group. Labels and definitions are the archived
language-model output (data/labels/group_labels.csv; scripts/label_groups.py).

Stability: for every title, the proportion flagged in its group under the primary grouping is
correlated (Spearman, across titles) with the same quantity under seven other random starts
(k = 200) and under k = 100 and k = 300, for each definition of few closely related articles.

Figure 3 places the group centers in two dimensions by metric multidimensional scaling (SMACOF;
sklearn.manifold.MDS, cosine distances between centers, 8 random starts of up to 900
iterations, lowest stress kept), and reports the correlation between plotted and full-space
distances.

Outputs
-------
* results/subject_summary.csv: per group, counts, proportion flagged, label, definition (Table 2)
* results/group_map_coords.csv: group map coordinates
* results/subject_groups.md: summary, stability, and the map's distance correlation
* figures/figure2_titles.png: every title by its two measures
* figures/figure3_group_map.png: the group map shaded by proportion flagged
Run:  python scripts/subject_groups.py   (after scripts/screen.py)
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_distances

from presented_published.groups import (
    apply_synonym_merges,
    fit_kmeans,
    fit_matrix,
    merged_centers,
)
from presented_published.style import PublicationPlotStyle
from screen import K0, N0, NS, records

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MIN_TITLES = 40
ALT = [(200, s) for s in range(1, 8)] + [(100, 0), (300, 0)]
SCREEN = Path("results/neighborhood_screen.parquet")
OUT = Path("results/subject_summary.csv")
REPORT = Path("results/subject_groups.md")
COORDS = Path("results/group_map_coords.csv")
LABELS = Path("data/labels/group_labels.csv")
MDS_KW = {"n_components": 2, "dissimilarity": "precomputed", "random_state": 0,
          "normalized_stress": "auto", "n_init": 8, "max_iter": 900, "init": "random"}
FIG_Q = Path("figures/figure2_titles.png")
FIG_MAP = Path("figures/figure3_group_map.png")
GROUPS = Path("cache/groupings.npz")


def groupings(V: np.ndarray) -> dict[str, np.ndarray]:
    """Primary grouping (k = 200, seed 0, synonym merge) and the alternatives, cached."""
    if GROUPS.exists():
        g = dict(np.load(GROUPS))
        if len(g["k200_s0"]) == len(V):
            return g
    X = fit_matrix()
    g = {"k200_s0": apply_synonym_merges(KMeans(200, random_state=0, n_init=10).fit(X).predict(V))}
    for k, s in ALT:
        g[f"k{k}_s{s}"] = KMeans(k, random_state=s, n_init=10).fit(X).predict(V)
        print(f"grouping k={k} seed={s}", flush=True)
    GROUPS.parent.mkdir(parents=True, exist_ok=True)
    np.savez(GROUPS, **g)
    return g


def group_map() -> tuple[pd.DataFrame, float]:
    """Two-dimensional coordinates of the group centers, and the distance correlation."""
    import warnings
    X = fit_matrix()
    C, active = merged_centers(fit_kmeans(X), X)
    D = cosine_distances(C[active])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        P = MDS(**MDS_KW).fit_transform(D)
    r = float(pearsonr(squareform(D, checks=False), pdist(P))[0])
    return pd.DataFrame({"group": active, "x": P[:, 0], "y": P[:, 1]}), r


def group_share(groups: np.ndarray, flag: np.ndarray) -> np.ndarray:
    """Each title's value is the flagged share of the group it falls in."""
    s = pd.Series(flag.astype(float)).groupby(groups).mean()
    return s.reindex(groups).to_numpy()


def style() -> None:
    S = PublicationPlotStyle()
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix", "axes.labelsize": S.axis_label_fontsize,
        "axes.labelpad": S.labelpad, "xtick.labelsize": S.tick_fontsize,
        "ytick.labelsize": S.tick_fontsize, "legend.fontsize": S.tick_fontsize,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False})


def main() -> None:
    V, _ = records()
    rec = pd.read_parquet(SCREEN)
    assert len(rec) == len(V)
    flag = rec[f"flag_k{K0}_lt{N0}"].to_numpy()
    pres = rec["presented"].to_numpy()

    G = groupings(V)
    g0 = G["k200_s0"]
    stab = []
    for n in NS:
        f = rec[f"flag_k{K0}_lt{n}"].to_numpy()
        v0 = group_share(g0, f)
        for k, seed in ALT:
            stab.append({"n": n, "k": k, "seed": seed,
                         "rho": spearmanr(v0, group_share(G[f"k{k}_s{seed}"], f))[0]})
    st = pd.DataFrame(stab)

    names = pd.read_csv(LABELS).set_index("topic")
    df = (pd.DataFrame({"group": g0, "presented": pres, "flag": flag})
          .groupby("group").agg(titles=("flag", "size"), presentations=("presented", "sum"),
                                flagged=("flag", "sum")))
    df["articles"] = df.titles - df.presentations
    df["flagged_share"] = df.flagged / df.titles
    df["tested"] = df.titles >= MIN_TITLES
    df = df.join(names[["label", "definition"]])
    df = df.sort_values("flagged_share", ascending=False)
    df.to_csv(OUT)
    t = df[df.tested]

    # ---- Figure 2: every title ----
    style()
    lo = rec[f"log_odds_k{K0}"].to_numpy()
    rng = np.random.default_rng(0)
    jit = rng.uniform(-0.04, 0.04, len(lo))       # log odds take K + 1 values; jitter for display
    cnt = rec["closely_related_articles"].to_numpy().astype(float)
    yj = cnt * np.exp(rng.uniform(-0.06, 0.06, len(cnt)))   # counts are integers; jitter for display
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(lo[~flag] + jit[~flag], yj[~flag], s=4, c="#bdbdbd", lw=0, rasterized=True,
               label="Not flagged")
    ax.scatter(lo[flag] + jit[flag], yj[flag], s=4, c="#202020", lw=0, rasterized=True,
               label="Flagged")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylim(-0.3, 1000)
    ax.set_yticks([0, 1, 10, 100, 1000])
    ax.set_yticklabels(["0", "1", "10", "100", "1,000"])
    ax.axvline(0, color="#777", ls="--", lw=1)
    ax.axhline(N0 - 0.5, color="#777", ls="--", lw=1)
    ax.set_xlabel(f"Presentation log odds among the {K0} nearest titles")
    ax.set_ylabel("Closely related articles")
    ax.legend(loc="upper left", markerscale=4, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_Q, dpi=300)
    plt.close(fig)

    # ---- Figure 3: group centers, shaded by proportion flagged ----
    coords, map_r = group_map()
    coords.to_csv(COORDS, index=False)
    xy = coords.set_index("group")
    m = df.join(xy[["x", "y"]], how="inner")
    fig, ax = plt.subplots(figsize=(9, 8))
    # Full 0-1 scale so the shading reads as a proportion of the group's titles.
    sc = ax.scatter(m.x, m.y, s=m.titles * 1.2, c=m.flagged_share, cmap="Greys", vmin=0, vmax=1.0,
                    edgecolors="#555", linewidths=0.4)
    cb = fig.colorbar(sc, ax=ax, shrink=0.8, ticks=np.arange(0, 1.01, 0.2))
    cb.ax.set_yticklabels([f"{v:.1f}" for v in np.arange(0, 1.01, 0.2)])
    cb.set_label("Proportion of Titles Flagged")
    cb.ax.tick_params(labelsize=12)
    ax.set_xlabel("Semantic dimension 1")
    ax.set_ylabel("Semantic dimension 2")
    fig.tight_layout()
    fig.savefig(FIG_MAP, dpi=300)
    plt.close(fig)

    top = t.head(20)
    rows = "\n".join(f"| {r.label} | {int(r.presentations)} | {int(r.articles)} | {r.flagged_share:.0%} |"
                     for r in top.itertuples())
    bottom = "\n".join(f"| {r.label} | {int(r.presentations)} | {int(r.articles)} | {r.flagged_share:.0%} |"
                       for r in t.tail(10).itertuples())
    sline = "\n".join(
        f"- {'none' if n == 1 else f'fewer than {n}'}: median {g.rho.median():.2f}, range {g.rho.min():.2f}-{g.rho.max():.2f}"
        f" (k = 200 other seeds {g[g.k == 200].rho.min():.2f}-{g[g.k == 200].rho.max():.2f}; k = 100 {g[g.k == 100].rho.iloc[0]:.2f}; k = 300 {g[g.k == 300].rho.iloc[0]:.2f})"
        for n, g in st.groupby("n"))
    REPORT.write_text(f"""# Flagged titles by subject area

Generated by `scripts/subject_groups.py` on {date.today().isoformat()}. Flag: K = {K0},
fewer than {N0} closely related articles. Groups: k-means k = 200, seed 0, synonym merge applied; groups with at least
{MIN_TITLES} titles ({len(t)} of {len(df)}) are summarised.

Proportion flagged across summarised groups: median {t.flagged_share.median():.0%}, IQR
{t.flagged_share.quantile(.25):.0%}-{t.flagged_share.quantile(.75):.0%}; {int((t.flagged_share >= .5).sum())}
groups have at least half their titles flagged; {int((t.flagged_share == 0).sum())} have none.

## Stability of the subject-level summary

Spearman correlation, across titles, of the proportion flagged in each title's group under the
primary grouping and under seven other seeds (k = 200) and k = 100 and 300, by definition of few
closely related articles:

{sline}

Flagged titles in summarised groups: {int(t.flagged.sum()):,} of {int(flag.sum()):,} ({t.flagged.sum() / flag.sum():.0%}).

## Groups with the highest proportion flagged (Table 2: at least .25)

| Group | Presentations | Articles | Flagged |
|---|---:|---:|---:|
{rows}

## Groups with the lowest proportion flagged

| Group | Presentations | Articles | Flagged |
|---|---:|---:|---:|
{bottom}

Group map (Figure 3): metric MDS of the cosine distances between the {len(xy)} group centers;
correlation between plotted and full-space distances r = {map_r:.3f}. Coordinates: `{COORDS}`.

Figures: `{FIG_Q}`, `{FIG_MAP}`.
""", encoding="utf-8")
    print(REPORT.read_text())


if __name__ == "__main__":
    main()
