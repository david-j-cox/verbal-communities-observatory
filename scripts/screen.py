#!/usr/bin/env python3
# ruff: noqa: E501  (report template)
"""Flag each title by its neighbors (the main analysis).

For every title from 2016 to 2026 (presented: ABAI, APBA, SQAB; published: fourteen journals):
* Presentation-heavy neighborhood: among the K = 75 most similar other titles, presentations
  are more common than the corpus-wide share, judged by the lower bound of the 95% confidence
  interval on the log odds (0.5 added to each count) lying above 0. K = 50 also reported.
* Closely related article: cosine similarity at least the 99th percentile of similarity over all
  distinct pairs of titles.
* Few closely related articles: fewer than N = 10 (N = 1, i.e. none, and N = 5 also reported).
* Flagged: both hold.

Resampling: 10 random 80% subsets of the titles; the neighbors, the 99th-percentile line, and
the counts are recomputed in each, and N is scaled by 0.8 because a subset holds 80% of the
articles.

Outputs
-------
* results/neighborhood_screen.parquet: per title, both conditions at each K and N.
* results/neighborhood_stability.csv: flags kept (all, presentations, articles) and Jaccard per
  subsample, K, and N.
* results/neighborhood_screen.md: flag rates, sensitivity, resampling, and reference values.
Run:  python scripts/screen.py   (after scripts/embed_titles.py)
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from presented_published.groups import TITLES, load
from presented_published.neighborhood import (
    article_count,
    nearest,
    pair_quantile,
    similarity,
    skew_scores,
)

KS, NS, K0, N0, Q = (50, 75), (1, 5, 10), 75, 10, 0.99
N_SUB, FRAC, SEED, WINDOW = 10, 0.8, 0, (2016, 2026)
SOURCES = [("conference", "year", True), ("apba", "year", True), ("sqab", "year", True),
           ("journal", "publication_year", False), ("specialty", "publication_year", False)]
OUT = Path("results/neighborhood_screen.parquet")
STAB = Path("results/neighborhood_stability.csv")
REPORT = Path("results/neighborhood_screen.md")


def records() -> tuple[np.ndarray, pd.DataFrame]:
    """Unit-length title vectors and their records, presentations first, 2016-2026."""
    vecs, rows = [], []
    for name, col, pres in SOURCES:
        t = pd.read_parquet(TITLES / f"{name}.parquet")
        m = pd.to_numeric(t[col], errors="coerce").between(*WINDOW).to_numpy()
        vecs.append(load(name)[m])
        rows.append(pd.DataFrame({"title": t["title"].fillna("").astype(str)[m].to_numpy(),
                                  "source": name, "presented": pres}))
    V = np.vstack(vecs)
    return V / np.linalg.norm(V, axis=1, keepdims=True), pd.concat(rows, ignore_index=True)


def run(S: np.ndarray, V: np.ndarray, pres: np.ndarray, scale: float = 1.0) -> dict:
    nn = nearest(S, max(KS))
    line = pair_quantile(V, Q)
    count = article_count(S, pres, line)
    out = {"line": line, "count": count}
    for k in KS:
        out[k] = skew_scores(nn, pres, k)
        for n in NS:
            out[(k, n)] = out[k]["skewed"] & (count < scale * n)
    return out


def main() -> None:
    V, rec = records()
    pres = rec["presented"].to_numpy()
    S = similarity(V)
    res = run(S, V, pres)
    rec["closely_related_articles"] = res["count"]
    for k in KS:
        for key in ("share", "log_odds", "ci_lo", "skewed"):
            rec[f"{key}_k{k}"] = res[k][key]
        for n in NS:
            rec[f"flag_k{k}_lt{n}"] = res[(k, n)]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rec.to_parquet(OUT, index=False)
    flag0 = rec[f"flag_k{K0}_lt{N0}"].to_numpy()
    print(f"{len(rec):,} titles; line {res['line']:.3f}; flagged: {flag0.mean():.1%}", flush=True)

    rng = np.random.default_rng(SEED)
    stab = []
    for b in range(N_SUB):
        idx = np.sort(rng.choice(len(V), int(FRAC * len(V)), replace=False))
        r = run(S[np.ix_(idx, idx)], V[idx], pres[idx], scale=FRAC)
        for k in KS:
            for n in NS:
                ff, fb = rec[f"flag_k{k}_lt{n}"].to_numpy()[idx], r[(k, n)]
                pp = pres[idx]
                stab.append({"sub": b, "k": k, "n": n,
                             "kept": (ff & fb).sum() / max(1, ff.sum()),
                             "kept_pres": (ff & fb & pp).sum() / max(1, (ff & pp).sum()),
                             "kept_pub": (ff & fb & ~pp).sum() / max(1, (ff & ~pp).sum()),
                             "jaccard": (ff & fb).sum() / max(1, (ff | fb).sum()),
                             "line": r["line"]})
        print(f"subsample {b + 1}/{N_SUB}", flush=True)
    pd.DataFrame(stab).to_csv(STAB, index=False)
    st = pd.DataFrame(stab).drop(columns="sub").groupby(["k", "n"]).median()

    # Reference values for reading similarities.
    iu = np.triu_indices(len(V), 1)
    allp = (V @ V.T)[iu]
    srt = -np.partition(-S, K0 - 1, axis=1)[:, :K0]
    nn1, nnk = srt.max(axis=1), srt.min(axis=1)
    c = res["count"]

    def label(n: int) -> str:
        return "none" if n == 1 else f"fewer than {n}"

    def rate(f: np.ndarray) -> str:
        return f"{f.sum():,} ({f.mean():.1%}) | {f[pres].sum():,} ({f[pres].mean():.1%}) | {f[~pres].sum():,} ({f[~pres].mean():.1%})"

    grid = "\n".join(
        f"| {k} | {label(n)} | {rate(rec[f'flag_k{k}_lt{n}'].to_numpy())} | {st.loc[(k, n), 'kept']:.0%} | {st.loc[(k, n), 'jaccard']:.2f} |"
        for k in KS for n in NS)
    by_source = "; ".join(f"{s}: {g[f'flag_k{K0}_lt{N0}'].mean():.1%} of {len(g):,}"
                          for s, g in rec.groupby("source", sort=False))
    sk = rec[f"skewed_k{K0}"].to_numpy()
    few = c < N0
    k_min = int(np.round(np.min(rec.loc[rec[f"skewed_k{K0}"], f"share_k{K0}"]) * K0))

    # Worked-example candidates: presentations with the same skew share, many vs. few articles.
    sh = rec[f"share_k{K0}"].to_numpy()
    cand = rec[pres & (np.abs(sh - 0.72) < 0.015) & (rec.title.str.split().str.len() >= 5)]
    thick = cand.sort_values("closely_related_articles", ascending=False).head(5)
    thin = cand[cand.closely_related_articles < N0].sample(min(8, (cand.closely_related_articles < N0).sum()), random_state=0)
    ex = "\n".join(f"- {r.title} ({r.share_k75:.0%} presentations; {r.closely_related_articles} closely related articles)"
                   for r in pd.concat([thick, thin]).itertuples())

    REPORT.write_text(f"""# Neighbourhood screen (no topics)

Generated by `scripts/screen.py` on {date.today().isoformat()}. {len(rec):,} titles
from 2016-2026: {pres.sum():,} presentations ({pres.mean():.1%}) and {(~pres).sum():,} articles.

Definitions: skewed = among the {K0} nearest titles, presentations reliably above the corpus share
(lower 95% bound of the Haldane log odds > 0; at K = {K0} this needs at least {k_min} presentations);
closely related = similarity at least {res['line']:.3f}, the 99th percentile over all
{allp.size:,} distinct title pairs; flagged = skewed and fewer than {N0} closely related
articles.

## Flag rates, sensitivity, and stability

| K (nearest titles) | Closely related articles | Titles flagged | Presentations flagged | Articles flagged | Flags kept in 80% subsamples (median) | Jaccard |
|---:|---|---|---|---|---:|---:|
{grid}

Primary (K = {K0}, fewer than {N0}) by source: {by_source}.

Skewed alone: {sk.sum():,} titles ({sk.mean():.0%}). Fewer than {N0} closely related articles
alone: {few.sum():,} ({few.mean():.0%}). Closely related articles per title: median {np.median(c):.0f},
IQR {np.percentile(c, 25):.0f}-{np.percentile(c, 75):.0f}, 90th percentile {np.percentile(c, 90):.0f},
none for {np.mean(c == 0):.1%}. Median for presentations {np.median(c[pres]):.0f}, for articles
{np.median(c[~pres]):.0f}; for skewed titles {np.median(c[sk]):.0f}, others {np.median(c[~sk]):.0f}.
Spearman correlation of skew share with the count: {pd.Series(sh).corr(pd.Series(c), method='spearman'):.2f}.
Median 99th-percentile line across subsamples: {st['line'].median():.3f}.

## Reading similarities

All distinct title pairs: median {np.median(allp):.2f}, 99th percentile {np.quantile(allp, .99):.3f}.
Each title's nearest title: median {np.median(nn1):.2f}. Each title's {K0}th nearest title: median
{np.median(nnk):.2f}.

## Worked-example candidates (presentations, {K0}-title share near 72%)

{ex}
""", encoding="utf-8")
    print(REPORT.read_text())


if __name__ == "__main__":
    main()
