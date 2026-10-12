#!/usr/bin/env python3
"""For comparison: how often a screen that flags whole groups flags the same titles again.

Reported in the Checks tab of the interactive supplement, beside the title-level resampling in
scripts/screen.py. Six random 80% subsets of the fit titles are drawn; in each, k-means (k = 200,
n_init = 10, random_state = subset index) sorts them into groups, every 2016-2026 title is
assigned to the nearest group mean (cosine), and a group is flagged when (a) it has at least 40
titles, (b) its titles are presentations more often than the corpus, judged by the lower bound
of the 95% confidence interval on the log odds (0.5 added to each count) lying above 0, and (c)
its article count is at or below the median of the groups meeting (a). Every title in a flagged
group is flagged.

Measure, over all ordered pairs of subsets: of the titles (presentations, articles) flagged in
subset i, the proportion flagged again in subset j.

Outputs: results/group_screen_comparison.csv (pairs), results/group_screen_comparison_runs.csv,
         results/group_screen_comparison.md
Run:     python scripts/group_screen_comparison.py   (after scripts/embed_titles.py)
"""
from __future__ import annotations

import platform
import time
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans

from presented_published.groups import TITLES, fit_matrix, load

K, WINDOW, MIN_TOTAL, H = 200, (2016, 2026), 40, 0.5
N_SUB, FRAC = 6, 0.8
OUT_MD = Path("results/group_screen_comparison.md")
OUT_CSV = Path("results/group_screen_comparison.csv")


def unit(X: np.ndarray) -> np.ndarray:
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def years(name: str, col: str) -> np.ndarray:
    y = pd.to_numeric(pd.read_parquet(TITLES / f"{name}.parquet")[col],
                      errors="coerce").to_numpy()
    return (y >= WINDOW[0]) & (y <= WINDOW[1])


def screen(P: np.ndarray, J: np.ndarray) -> set[int]:
    Pc, Jc = np.bincount(P, minlength=K), np.bincount(J, minlength=K)
    nP, nJ = Pc.sum(), Jc.sum()
    a, b = Pc + H, Jc + H
    lor = np.log(a * (nJ - Jc + H) / (b * (nP - Pc + H)))
    se = np.sqrt(1 / a + 1 / b + 1 / (nP - Pc + H) + 1 / (nJ - Jc + H))
    viable = (Pc + Jc) >= MIN_TOTAL
    cut = np.median(Jc[viable])
    return set(np.flatnonzero(viable & (lor - 1.96 * se > 0) & (Jc <= cut)).tolist())


def fit_labels(method: str, X: np.ndarray, seed: int) -> np.ndarray:
    return KMeans(n_clusters=K, random_state=seed, n_init=10).fit_predict(X)


def main() -> None:
    Xfit = unit(fit_matrix())
    P_v = unit(np.vstack([load("conference")[years("conference", "year")],
                          load("apba")[years("apba", "year")],
                          load("sqab")[years("sqab", "year")]]))
    J_v = unit(np.vstack([load("journal")[years("journal", "publication_year")],
                          load("specialty")[years("specialty", "publication_year")]]))
    V = np.vstack([P_v, J_v])
    pres = np.arange(len(V)) < len(P_v)
    rng = np.random.default_rng(0)
    subs = [rng.choice(len(Xfit), int(FRAC * len(Xfit)), replace=False) for _ in range(N_SUB)]

    runs, pairs = [], []
    for method in ("kmeans",):
        feet = []
        for i, idx in enumerate(subs):
            t0 = time.time()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lab = fit_labels(method, Xfit[idx], seed=i)
            C = unit(np.vstack([Xfit[idx][lab == t].mean(axis=0) for t in range(K)]))
            a = (V @ C.T).argmax(axis=1)               # nearest group mean, cosine
            flagged = screen(a[pres], a[~pres])
            foot = np.isin(a, list(flagged))
            feet.append(foot)
            sizes = np.bincount(lab, minlength=K)
            runs.append({"method": method, "sub": i, "flagged": len(flagged),
                         "flagged_titles": int(foot.sum()),
                         "flagged_articles": int((foot & ~pres).sum()),
                         "largest_group_share": sizes.max() / sizes.sum(),
                         "groups_under_10": int((sizes < 10).sum()),
                         "seconds": round(time.time() - t0, 1)})
            print(f"{method} sub {i}: flagged {len(flagged)}, largest group "
                  f"{sizes.max() / sizes.sum():.1%}, {time.time() - t0:.0f}s", flush=True)
        for i in range(N_SUB):
            for j in range(N_SUB):
                if i == j:
                    continue
                f_i, f_j = feet[i], feet[j]
                both = f_i & f_j
                pairs.append({
                    "method": method, "from": i, "to": j,
                    "retained": both.sum() / max(1, f_i.sum()),
                    "retained_pub": (both & ~pres).sum() / max(1, (f_i & ~pres).sum()),
                    "retained_pres": (both & pres).sum() / max(1, (f_i & pres).sum())})

    rn, pr = pd.DataFrame(runs), pd.DataFrame(pairs)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pr.to_csv(OUT_CSV, index=False)
    rn.to_csv(OUT_CSV.with_name(OUT_CSV.stem + "_runs.csv"), index=False)
    s = rn.groupby("method").agg(flagged=("flagged", "median"), fmin=("flagged", "min"),
                                 fmax=("flagged", "max"), articles=("flagged_articles", "median"),
                                 largest=("largest_group_share", "median"),
                                 tiny=("groups_under_10", "median"))
    g = pr.groupby("method")
    for c in ("retained", "retained_pub", "retained_pres"):
        s[c], s[c + "_q1"], s[c + "_q3"] = (g[c].median(), g[c].quantile(.25),
                                            g[c].quantile(.75))
    s = s.loc[["kmeans"]]
    print("\n" + s.round(3).to_string())
    lines = ["| Method | Flagged groups, median [range] | Articles in flagged groups | "
             "Articles retained, median [IQR] | Presentations retained | Largest group "
             "(share of titles) | Groups with <10 titles |",
             "|---|---|---:|---|---:|---:|---:|"]
    for m, x in s.iterrows():
        lines.append(f"| {m} | {x.flagged:.0f} [{x.fmin:.0f}-{x.fmax:.0f}] | {x.articles:.0f} "
                     f"| {x.retained_pub:.0%} [{x.retained_pub_q1:.0%}, "
                     f"{x.retained_pub_q3:.0%}] | {x.retained_pres:.0%} | {x.largest:.1%} | "
                     f"{x.tiny:.0f} |")
    OUT_MD.write_text(f"""# A screen that flags whole groups: how often its flags recur

Generated by `scripts/group_screen_comparison.py` on {date.today().isoformat()}. K-means is refit
on {N_SUB} random {FRAC:.0%} subsets of the {len(Xfit):,} fit titles; every 2016-2026 title
({len(V):,}) is assigned to the nearest group mean (cosine) and the group-level screen is run.
Retention is over all ordered pairs of subsets: titles retained {pr.retained.median():.0%},
presentations {pr.retained_pres.median():.0%}, articles {pr.retained_pub.median():.0%} (medians).

{chr(10).join(lines)}

Software: Python {platform.python_version()}, scikit-learn {sklearn.__version__},
numpy {np.__version__}, pandas {pd.__version__}.
""", encoding="utf-8")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
