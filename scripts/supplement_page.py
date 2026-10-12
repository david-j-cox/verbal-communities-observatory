#!/usr/bin/env python3
"""Build the interactive supplement (docs/index.html, served by GitHub Pages).

The page has three tabs. The Method tab takes any title through every step (collect the titles,
turn them into numbers, find the 75 nearest titles, count closely related articles, combine the
two conditions, and group and describe the titles). The Results tab carries Figures 2 and 3 as
linked, selectable plots with Table 2. The Checks tab reports how sharply subjects separate, the
sensitivity to the two cutoffs, resampling (with the group-level comparison), the wording check,
and the stability of the subject-level summary.

Every flag, count, and group value on the page is read from, or recomputed and checked against,
the outputs of the scripts below; the build stops if any value differs.

Inputs
------
* results/neighborhood_screen.parquet, results/neighborhood_stability.csv (scripts/screen.py)
* results/subject_summary.csv, results/group_map_coords.csv (scripts/subject_groups.py)
* results/framing_check.csv (scripts/framing_check.py)
* results/group_screen_comparison.csv (scripts/group_screen_comparison.py)
* data/labels (archived group labels, evidence, and prompts; scripts/label_groups.py)
* scripts/templates/supplement_page.html

Run:  python scripts/supplement_page.py   (after the scripts above)
"""
from __future__ import annotations

import base64
import json
import platform
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_samples

from presented_published.groups import TITLES
from presented_published.neighborhood import (
    HALDANE,
    article_count,
    nearest,
    pair_quantile,
    similarity,
    skew_scores,
)
from screen import K0, N0, NS, SOURCES, WINDOW, records
from subject_groups import MIN_TITLES, group_share, groupings

MODEL = "all-MiniLM-L6-v2"
N_ART = 10          # nearest articles stored per title
N_STRIP = 6         # nearest titles drawn as strips in Step 2
MAX_RANK = 300      # neighbor ranks in the similarity fall-off curve
SEED = 0
SCREEN = Path("results/neighborhood_screen.parquet")
STAB = Path("results/neighborhood_stability.csv")
SUBJECTS = Path("results/subject_summary.csv")
COORDS = Path("results/group_map_coords.csv")
FRAMING = Path("results/framing_check.csv")
GROUP_SCREEN = Path("results/group_screen_comparison.csv")
NAMES = Path("data/labels/group_labels.csv")
LABEL_INPUTS = Path("data/labels")   # evidence.jsonl and prompts/ (scripts/label_groups.py)
TEMPLATE = Path("scripts/templates/supplement_page.html")
OUT = Path("docs/index.html")

# One title per combination of the two conditions, guarded by its expected outcome so that a
# change upstream fails loudly instead of showing the wrong case.
EXAMPLES: tuple[tuple[str, str], ...] = (
    ("Opportunities and Issues in Practitioner Publication", "flagged"),
    ("Improving Social Functioning for Children With Autism", "skewed_many"),
    ("Randomization Procedures for Changing Criterion Designs", "few_not_skewed"),
    ("Technological Advancements in Stimulus Preference Assessments in Individuals With Autism",
     "neither"),
)
SOURCE_NAME = {"conference": "ABAI", "apba": "APBA", "sqab": "SQAB",
               "journal": "General journals", "specialty": "Specialty journals"}


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def venues_years() -> pd.DataFrame:
    """Venue and year of every record, in the order records() returns them."""
    rows = []
    for name, col, _ in SOURCES:
        t = pd.read_parquet(TITLES / f"{name}.parquet")
        y = pd.to_numeric(t[col], errors="coerce")
        m = y.between(*WINDOW).to_numpy()
        venue = (t["journal"].astype(str) if name in ("journal", "specialty")
                 else pd.Series(SOURCE_NAME[name], index=t.index))
        rows.append(pd.DataFrame({"title": t["title"].fillna("").astype(str)[m].to_numpy(),
                                  "venue": venue[m].to_numpy(),
                                  "year": y[m].astype(int).to_numpy()}))
    return pd.concat(rows, ignore_index=True)


def outcome(skewed: bool, count: int) -> str:
    if skewed and count < N0:
        return "flagged"
    if skewed:
        return "skewed_many"
    return "few_not_skewed" if count < N0 else "neither"


def main() -> None:
    V, rec = records()
    pres = rec["presented"].to_numpy()
    arch = pd.read_parquet(SCREEN)
    if not (arch["title"].to_numpy() == rec["title"].to_numpy()).all():
        raise SystemExit("record order differs from the archived screen; rerun "
                         "scripts/screen.py")
    vy = venues_years()
    if not (vy["title"].to_numpy() == rec["title"].to_numpy()).all():
        raise SystemExit("venue table is out of step with the embedding records")

    S = similarity(V)
    nn75 = nearest(S, K0)   # as in the screen; ties at rank K0 resolve the same way
    nn = nearest(S, MAX_RANK)
    sk = skew_scores(nn75, pres, K0)
    line = pair_quantile(V, 0.99)
    count = article_count(S, pres, line)
    flag = sk["skewed"] & (count < N0)
    checks = {"closely_related_articles": count, f"skewed_k{K0}": sk["skewed"],
              f"flag_k{K0}_lt{N0}": flag}
    for col, v in checks.items():
        if not np.array_equal(arch[col].to_numpy(), v):
            raise SystemExit(f"recomputed {col} differs from the archive")
    print(f"{len(rec):,} titles; line {line:.4f}; flagged {flag.sum():,}", flush=True)

    rank_sim = np.take_along_axis(S, nn, axis=1)
    cols = np.flatnonzero(~pres)
    an = nearest(S[:, cols], N_ART)
    art_sim = np.take_along_axis(S[:, cols], an, axis=1)
    art_idx = cols[an]
    if not np.array_equal((art_sim >= line).sum(axis=1), np.minimum(count, N_ART)):
        raise SystemExit("nearest-article lists disagree with the closely related count")

    iu = np.triu_indices(len(V), 1)
    pairs = (V @ V.T)[iu]
    hist, edges = np.histogram(pairs, bins=np.linspace(-0.3, 1.0, 131))
    pair_median = float(np.median(pairs))
    del pairs

    # Groups (reporting device) and the evidence on how sharply subjects separate.
    G = groupings(V)
    g0 = G["k200_s0"]
    sil = silhouette_samples(V, g0, metric="cosine")
    ari = [adjusted_rand_score(g0, G[f"k200_s{s}"]) for s in range(1, 8)]
    curve = {"median": np.median(rank_sim, axis=0), "p10": np.percentile(rank_sim, 10, axis=0),
             "p90": np.percentile(rank_sim, 90, axis=0)}

    summ = pd.read_csv(SUBJECTS).set_index("group")
    coords = pd.read_csv(COORDS).set_index("group")
    names = pd.read_csv(NAMES).set_index("topic")
    present = np.unique(g0)
    groups = []
    for g in present:
        r = summ.loc[g]
        if r["label"] != names.loc[g, "label"]:
            raise SystemExit(f"group {g}: summary label differs from the recorded LLM label")
        groups.append({"id": int(g), "label": r["label"], "def": r["definition"],
                       "n": int(r["titles"]), "p": int(r["presentations"]),
                       "a": int(r["articles"]), "f": int(r["flagged"]),
                       "x": round(float(coords.loc[g, "x"]), 4),
                       "y": round(float(coords.loc[g, "y"]), 4),
                       "tested": bool(r["tested"])})
    titles = rec["title"].tolist()
    # What the labeling model saw and returned, for Step 6 (scripts/ceu_topic_names.py).
    evidence = {}
    for line_ in (LABEL_INPUTS / "evidence.jsonl").read_text(encoding="utf-8").splitlines():
        ev = json.loads(line_)
        evidence[int(ev["topic"])] = ev
    prompt_head = None
    for gr in groups:
        g = gr["id"]
        if g not in evidence:
            raise SystemExit(f"no archived labeling evidence for group {g}")
        ev = evidence[g]
        text = (LABEL_INPUTS / "prompts" / f"{g}.txt").read_text(encoding="utf-8")
        head, _, tail = text.partition(f"TOPIC {g}\n")
        if prompt_head is None:
            prompt_head = head
        elif head != prompt_head:
            raise SystemExit(f"group {g}: prompt instructions differ from the others")
        gr.update({"terms": ev["terms"], "inP": ev["presented_titles"],
                   "inA": ev["published_titles"],
                   "distinct": names.loc[g, "distinct_from_neighbours"],
                   "nbs": [{"id": int(nb["topic"]), "cos": nb["cosine"], "terms": nb["terms"],
                            "titles": nb["titles"]} for nb in ev["neighbours"]],
                   "prompt": f"TOPIC {g}\n" + tail})
    # Step 6 map: each group's titles and those of its three nearest groups, projected to two
    # dimensions (PCA fit on those titles only); the titles the labeling model saw are marked.
    present_ids = {gr["id"] for gr in groups}
    for gr in groups:
        ids = [gr["id"]] + [nb["id"] for nb in gr["nbs"] if nb["id"] in present_ids]
        idx = np.concatenate([np.flatnonzero(g0 == k) for k in ids])
        slot = np.concatenate([np.full((g0 == k).sum(), j) for j, k in enumerate(ids)])
        pca = PCA(n_components=2, random_state=SEED).fit(V[idx])
        xy = pca.transform(V[idx])
        cen = pca.transform(np.vstack([V[g0 == k].mean(axis=0) for k in ids]))
        seen = {t.lower() for t in gr["inP"] + gr["inA"]}
        own = idx[slot == 0]
        gr["map"] = {"ids": ids, "x": [round(float(v), 3) for v in xy[:, 0]],
                     "y": [round(float(v), 3) for v in xy[:, 1]], "s": slot.tolist(),
                     "i": idx.tolist(),
                     "c": [[round(float(a), 3), round(float(b), 3)] for a, b in cen],
                     "central": [int(k) for k in own if titles[k].lower() in seen]}
    got = pd.Series(flag).groupby(g0).sum()
    for gr in groups:
        if int(got[gr["id"]]) != gr["f"]:
            raise SystemExit(f"group {gr['id']}: flagged count drifts from the summary")

    group_stab = []
    for n in NS:
        f = arch[f"flag_k{K0}_lt{n}"].to_numpy()
        v0 = group_share(g0, f)
        rhos = [spearmanr(v0, group_share(G[k], f))[0] for k in G if k != "k200_s0"]
        group_stab.append({"n": n, "median": float(np.median(rhos)), "min": float(min(rhos)),
                           "max": float(max(rhos))})

    # Sensitivity and resampling.
    st = pd.read_csv(STAB).drop(columns="sub").groupby(["k", "n"]).median()
    sens = []
    for (k, n), r in st.iterrows():
        f = arch[f"flag_k{k}_lt{n}"].to_numpy()
        sens.append({"k": int(k), "n": int(n), "all": float(f.mean()),
                     "pres": float(f[pres].mean()), "pub": float(f[~pres].mean()),
                     "kept": float(r["kept"]), "kept_pres": float(r["kept_pres"]),
                     "kept_pub": float(r["kept_pub"]), "jaccard": float(r["jaccard"])})
    gs = pd.read_csv(GROUP_SCREEN)
    gs = gs[gs["method"] == "kmeans"][["retained", "retained_pres", "retained_pub"]].median()
    framing = pd.read_csv(FRAMING)
    framing = framing[framing["k"] == K0].to_dict("records")

    # Worked examples.
    rng = np.random.default_rng(SEED)
    examples = []
    for t, want in EXAMPLES:
        hits = np.flatnonzero(rec["title"].to_numpy() == t)
        if len(hits) != 1:
            raise SystemExit(f"worked example not found exactly once: {t!r}")
        i = int(hits[0])
        got_o = outcome(bool(sk["skewed"][i]), int(count[i]))
        if got_o != want:
            raise SystemExit(f"worked example {t!r} is now {got_o}, expected {want}")
        far = int(rng.integers(len(V)))
        strip = [i, *nn75[i, :N_STRIP].tolist(), far]
        examples.append({"i": i, "kind": want, "strip": strip,
                         "vec": [[round(float(x), 3) for x in V[j]] for j in strip],
                         "cos": [None, *[round(float(S[i, j]), 3) for j in strip[1:]]]})

    venue_list = sorted(vy["venue"].unique())
    vindex = {v: k for k, v in enumerate(venue_list)}
    src_order = [s for s, _, _ in SOURCES]

    meta = {
        "n": len(rec), "nP": int(pres.sum()), "nA": int((~pres).sum()),
        "pi": float(pres.mean()), "k": K0, "cut": N0, "h": HALDANE,
        "line": round(line, 4), "pair_median": round(pair_median, 3),
        "n_pairs": int(len(iu[0])), "window": f"{WINDOW[0]}-{WINDOW[1]}", "model": MODEL,
        "n_flag": int(flag.sum()), "n_flag_p": int((flag & pres).sum()),
        "n_flag_a": int((flag & ~pres).sum()), "n_skew": int(sk["skewed"].sum()),
        "n_few": int((count < N0).sum()),
        "rho_skew_count": float(spearmanr(sk["share"], count)[0]),
        "count_median": float(np.median(count)),
        "count_q": [float(np.percentile(count, 25)), float(np.percentile(count, 75))],
        "count_zero": float((count == 0).mean()),
        "nn1": float(np.median(rank_sim[:, 0])), "nnk": float(np.median(rank_sim[:, K0 - 1])),
        "sil_neg": float((sil < 0).mean()), "sil_median": float(np.median(sil)),
        "ari": [float(min(ari)), float(np.median(ari)), float(max(ari))],
        "n_groups": len(groups), "min_titles": MIN_TITLES,
        "n_label_titles": len(groups[0]["inP"]) + len(groups[0]["inA"]),
        "n_tested": sum(g["tested"] for g in groups),
        "namer": str(names["model"].iloc[0]),
        "generated": date.today().isoformat(),
        "software": f"Python {platform.python_version()}, scikit-learn {sklearn.__version__}, "
                    f"numpy {np.__version__}, pandas {pd.__version__}",
    }
    data = {
        "meta": meta, "venues": venue_list, "titles": titles,
        "src": [src_order.index(s) for s in rec["source"]],
        "venue": [vindex[v] for v in vy["venue"]], "year": vy["year"].tolist(),
        "count": count.tolist(), "group": g0.tolist(),
        "nn": b64(nn75.astype("<u2")),
        "nns": b64(np.clip(np.round(np.take_along_axis(S, nn75, axis=1) * 250), 0, 255)
                   .astype("u1")),
        "art": b64(art_idx.astype("<u2")),
        "arts": b64(np.clip(np.round(art_sim * 10000), 0, 65535).astype("<u2")),
        "examples": examples, "groups": groups, "prompt_head": prompt_head,
        "curve": {k: [round(float(x), 4) for x in v] for k, v in curve.items()},
        "hist": {"counts": hist.tolist(), "edges": [round(float(e), 3) for e in edges]},
        "sens": sens, "group_screen": {k: float(v) for k, v in gs.items()},
        "framing": framing, "group_stab": group_stab,
    }
    html = TEMPLATE.read_text(encoding="utf-8").replace(
        "__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        .replace("</", "<\\/"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print(f"silhouette < 0: {meta['sil_neg']:.1%}; ARI {meta['ari']}; groups {len(groups)}")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
