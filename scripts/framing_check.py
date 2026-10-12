#!/usr/bin/env python3
# ruff: noqa: E501  (report template)
"""Does the wording of a title, rather than its subject, drive its neighborhood?

Presentation and article titles tend to open differently: presentations with "Recent Advances
in" or "Applications of", articles with "The Effects of" or "An Evaluation of". If those leading
phrases pulled presentations toward one another, part of a neighborhood's skew would reflect
phrasing instead of subject. The check removes them and recomputes every neighborhood.

Conditions (all 2016-2026 titles)
---------------------------------
* title:        titles as analyzed.
* framing:      the longest leading phrase in FRAMING removed (47 phrases).
* framing_stop: framing removed, then English stop words removed (scikit-learn list), as a
                reference for a much larger change in wording.
Each condition is embedded with all-MiniLM-L6-v2 and scored at the K nearest titles (proportion
that are presentations; presentation-heavy when the lower bound of the 95% confidence interval on
the log odds against the corpus proportion is above 0).

Outputs: results/framing_check.csv, results/framing_check.md
Run:     python scripts/framing_check.py   (downloads the model on first use)
"""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from presented_published.groups import TITLES

KS = (50, 75)
WINDOW, H = (2016, 2026), 0.5
SOURCES = [("conference", "year", True), ("apba", "year", True), ("sqab", "year", True),
           ("journal", "publication_year", False), ("specialty", "publication_year", False)]
REPORT = Path("results/framing_check.md")
OUT = Path("results/framing_check.csv")

FRAMING = sorted("""advancements in|recent advancements in|advances in|recent advances in|innovations in|
applications of|recent research on|how to|considerations for|toward a|the importance of|the role of|
effects of the|examining the|use of|evaluating the effects of|the use of|introduction to|
the impact of|using the|application of|exploring the|an examination of|effect of|an analysis of|
evaluating the|the influence of|effects of|the effect of|further evaluation of|effectiveness of|
evaluation of|comparing the|examination of|comparison of|the effects of|an evaluation of the|
using a|evaluation of a|an evaluation of|a review of|effects of a|the effects of a|
a comparison of|assessing the|systematic review of|a systematic review of""".replace("\n", "").split("|"),
                 key=lambda p: -len(p.split()))


def strip_framing(title: str) -> str:
    words = re.findall(r"[A-Za-z][A-Za-z']*|[^A-Za-z\s]+", title.replace("-", " "))
    low = [w.lower() for w in words]
    for ph in FRAMING:
        p = ph.split()
        if low[:len(p)] == p and len(words) > len(p):
            return " ".join(words[len(p):]).strip()
    return title


def strip_stop(title: str) -> str:
    kept = [w for w in re.findall(r"[A-Za-z][A-Za-z']*", title.replace("-", " "))
            if w.lower() not in ENGLISH_STOP_WORDS]
    return " ".join(kept) or title


def neighbours(V: np.ndarray, kmax: int) -> np.ndarray:
    S = V @ V.T
    np.fill_diagonal(S, -np.inf)
    nn = np.argpartition(-S, kmax, axis=1)[:, :kmax]
    order = np.argsort(-np.take_along_axis(S, nn, axis=1), axis=1)
    return np.take_along_axis(nn, order, axis=1)


def score(nn, pres, k, pi):
    p = pres[nn[:, :k]].sum(axis=1)
    lor = np.log((p + H) / (k - p + H)) - np.log(pi / (1 - pi))
    se = np.sqrt(1 / (p + H) + 1 / (k - p + H))
    return p / k, (lor - 1.96 * se) > 0


def main() -> None:
    rows = []
    for name, col, pres in SOURCES:
        t = pd.read_parquet(TITLES / f"{name}.parquet")
        m = pd.to_numeric(t[col], errors="coerce").between(*WINDOW)
        rows.append(pd.DataFrame({"title": t.loc[m, "title"].fillna("").astype(str).to_numpy(),
                                  "presented": pres}))
    rec = pd.concat(rows, ignore_index=True)
    rec["framing"] = rec.title.map(strip_framing)
    rec["framing_stop"] = rec.framing.map(strip_stop)
    changed = (rec.framing != rec.title)
    pres = rec.presented.to_numpy()
    pi = float(pres.mean())
    print(f"{len(rec):,} titles; framing removed from "
          f"{changed.sum():,} ({changed[pres].mean():.0%} of presentations, "
          f"{changed[~pres].mean():.0%} of articles)", flush=True)

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2",
                                revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41")
    res = {}
    for cond in ("title", "framing", "framing_stop"):
        V = model.encode(rec[cond].tolist(), batch_size=512, normalize_embeddings=True)
        nn = neighbours(V, max(KS))
        res[cond] = {k: score(nn, pres, k, pi) for k in KS}
        print(f"{cond} scored", flush=True)

    lines, table = [], []
    for k in KS:
        s0, f0 = res["title"][k]
        for cond, lab in (("title", "Original titles"), ("framing", "Framing removed"),
                          ("framing_stop", "Framing and stop words removed")):
            s, f = res[cond][k]
            rho = spearmanr(s0, s).statistic
            kept = (f0 & f).sum() / max(1, f0.sum())
            table.append({"k": k, "condition": cond, "label": lab, "titles": len(rec),
                          "changed": int(changed.sum()), "skewed": f.mean(),
                          "skewed_pres": f[pres].mean(), "skewed_pub": f[~pres].mean(),
                          "rho": rho, "kept": kept, "pres_lost": int((f0 & ~f & pres).sum())})
            lines.append(f"| {k} | {lab} | {f.mean():.0%} | {f[pres].mean():.0%} | {f[~pres].mean():.0%} | "
                         f"{rho:.3f} | {kept:.0%} | {(f0 & ~f & pres).sum():,} |")
    pd.DataFrame(table).to_csv(OUT, index=False)
    REPORT.write_text(f"""# Does the wording of a title drive its neighborhood?

Generated by `scripts/framing_check.py` on {date.today().isoformat()}. {len(rec):,} titles from
2016-2026 ({pi:.1%} presentations). A leading framing phrase was removed from {changed.sum():,} titles
({changed[pres].mean():.0%} of presentations, {changed[~pres].mean():.0%} of articles). Each version
was re-embedded (all-MiniLM-L6-v2) and every title rescored.

| K | Titles | Flagged (all) | Presentations flagged | Articles flagged | Score correlation with original | Original flags kept | Presentations that lose their flag |
|---:|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(lines)}

If framing drove the skew, removing it would lower the flag rates and break the correlation with
the original scores. Framing phrases removed: {", ".join(FRAMING)}.
""", encoding="utf-8")
    print(REPORT.read_text())


if __name__ == "__main__":
    main()
