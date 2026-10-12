#!/usr/bin/env python3
"""Label and define each subject-area group with a language model, and verify the archive.

For each of the 199 groups (presented_published.groups; k-means, seed 0, synonym merge
applied), the evidence given to the model is assembled from the 2016-2026 titles:
  * the group's key terms (class-based TF-IDF over the fit titles; presented_published.groups);
  * the 10 presentation titles and 10 article titles closest to the group center (cosine);
  * the 3 most similar groups by center cosine, each with its key terms and 3 central titles.
The evidence is rendered into a fixed prompt (presented_published.labeling) and sent to a pinned
model (claude-opus-5-5). The answer must be JSON with a label, a one-sentence definition, and a
sentence on what separates the group from its neighbors; an answer that breaks that format is
returned to the model with the validation error (``attempts`` records how many calls a group
took).

The model's responses cannot be regenerated exactly, so they are archived in data/labels:
evidence.jsonl, prompts/<group>.txt, responses/<sha256>.json (keyed by the SHA-256 of the model
and the exchange), and group_labels.csv. By default this script rebuilds the evidence and
prompts from the data, checks that they match the archive, replays every response from the
archive, and checks that the replayed labels and definitions match group_labels.csv. It calls
the model only with --call-model, and only for responses missing from the archive
(ANTHROPIC_API_KEY in the environment; pip install anthropic).

Outputs: results/label_check.md (and data/labels/responses/* when --call-model fetches new ones)
Run:     python scripts/label_groups.py   (after scripts/embed_titles.py)
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from presented_published.groups import (
    FIT_SETS,
    TITLES,
    apply_synonym_merges,
    class_tfidf_terms,
    fit_kmeans,
    fit_matrix,
    load,
    merged_centers,
    validate_synonym_merges,
)
from presented_published.labeling import (
    PROMPT_VERSION,
    NamingError,
    Neighbour,
    TopicEvidence,
    build_prompt,
    parse_response,
    sha256,
)

WINDOW = (2016, 2026)
MODEL = "claude-opus-5-5"
N_TITLES, N_NEIGHBOURS, N_NEIGHBOUR_TITLES = 10, 3, 3
MAX_ATTEMPTS = 3
LABELS = Path("data/labels")
REPORT = Path("results/label_check.md")
CORRECTION = ("Your answer broke the required format: {error}. Return the corrected JSON only, "
              "with the same three keys and the same meaning, within the stated limits.")
SOURCES = [("conference", "year", "presented"), ("journal", "publication_year", "published"),
           ("apba", "year", "presented"), ("sqab", "year", "presented"),
           ("specialty", "publication_year", "published")]


def load_records() -> tuple[np.ndarray, pd.DataFrame]:
    vecs = np.vstack([load(n) for n, _, _ in SOURCES])
    frames = []
    for n, col, side in SOURCES:
        t = pd.read_parquet(TITLES / f"{n}.parquet")
        frames.append(pd.DataFrame({"title": t["title"].fillna("").astype(str).str.strip(),
                                    "year": pd.to_numeric(t[col], errors="coerce"),
                                    "side": side}))
    rec = pd.concat(frames, ignore_index=True)
    if len(rec) != len(vecs):
        raise SystemExit(f"title table has {len(rec)} rows, vectors have {len(vecs)}")
    return vecs, rec


def central(idx: np.ndarray, vecs: np.ndarray, c: np.ndarray, titles: pd.Series,
            n: int) -> tuple[str, ...]:
    """The n distinct titles among ``idx`` closest to center ``c`` (cosine)."""
    v = vecs[idx]
    s = (v @ c) / (np.linalg.norm(v, axis=1) * np.linalg.norm(c) + 1e-12)
    out, seen = [], set()
    for i in idx[np.argsort(-s)]:
        t = titles.iloc[i]
        if t and t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
        if len(out) == n:
            break
    return tuple(out)


def gather() -> list[TopicEvidence]:
    vecs, rec = load_records()
    X = fit_matrix()
    km = fit_kmeans(X)
    fit_titles = pd.concat([pd.read_parquet(TITLES / f"{n}.parquet")["title"]
                            for n in FIT_SETS]).fillna("").astype(str).tolist()
    terms = class_tfidf_terms(fit_titles, km.labels_, len(km.cluster_centers_))
    validate_synonym_merges(terms)
    C, active = merged_centers(km, X)
    grp = apply_synonym_merges(km.predict(vecs))
    Cn = C / np.linalg.norm(C, axis=1, keepdims=True)
    S = Cn @ Cn.T
    win = rec["year"].between(*WINDOW).to_numpy()
    side = rec["side"].to_numpy()
    evidence = []
    for t in active.tolist():
        others = [o for o in active[np.argsort(-S[t, active])] if o != t][:N_NEIGHBOURS]
        nbs = tuple(Neighbour(int(o), round(float(S[t, o]), 3), terms[o],
                              central(np.flatnonzero(win & (grp == o)), vecs, C[o],
                                      rec["title"], N_NEIGHBOUR_TITLES)) for o in others)
        evidence.append(TopicEvidence(
            topic=int(t), terms=terms[t],
            presented_titles=central(np.flatnonzero(win & (grp == t) & (side == "presented")),
                                     vecs, C[t], rec["title"], N_TITLES),
            published_titles=central(np.flatnonzero(win & (grp == t) & (side == "published")),
                                     vecs, C[t], rec["title"], N_TITLES),
            neighbours=nbs))
    return evidence


def call_model(client, model: str, messages: list[dict]) -> dict:
    msg = client.messages.create(model=model, max_tokens=2000, messages=messages)
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    return {"model": msg.model, "text": text, "stop_reason": msg.stop_reason,
            "usage": {"input": msg.usage.input_tokens, "output": msg.usage.output_tokens},
            "generated_utc": datetime.now(UTC).isoformat(timespec="seconds")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--call-model", action="store_true",
                    help="fetch responses missing from the archive (needs ANTHROPIC_API_KEY)")
    args = ap.parse_args()

    evidence = gather()
    arch = [json.loads(x) for x in (LABELS / "evidence.jsonl").read_text("utf-8").splitlines()]
    ev_match = [json.loads(ev.to_json()) == a for ev, a in zip(evidence, arch, strict=True)]
    prompt_match = [build_prompt(ev) == (LABELS / "prompts" / f"{ev.topic}.txt").read_text("utf-8")
                    for ev in evidence]
    print(f"evidence matches archive: {sum(ev_match)}/{len(evidence)}; "
          f"prompts match: {sum(prompt_match)}/{len(evidence)}", flush=True)

    client, rows = None, []
    for ev in evidence:
        prompt = build_prompt(ev)
        messages = [{"role": "user", "content": prompt}]
        for attempt in range(1, MAX_ATTEMPTS + 1):
            key = sha256(f"{MODEL}\n{prompt}" if attempt == 1
                         else f"{MODEL}\n{json.dumps(messages, ensure_ascii=False)}")
            hit = LABELS / "responses" / f"{key}.json"
            if hit.exists():
                resp = json.loads(hit.read_text(encoding="utf-8"))
            elif args.call_model:
                if client is None:
                    import anthropic
                    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
                resp = call_model(client, MODEL, messages)
                hit.write_text(json.dumps(resp, ensure_ascii=False, indent=1), encoding="utf-8")
            else:
                raise SystemExit(f"group {ev.topic}: no archived response {key}; rerun with "
                                 "--call-model to fetch it")
            try:
                ans, err = parse_response(resp["text"]), ""
                break
            except NamingError as e:
                ans, err = {"label": "", "definition": "", "distinct_from_neighbours": ""}, str(e)
                messages = messages + [{"role": "assistant", "content": resp["text"]},
                                       {"role": "user", "content": CORRECTION.format(error=err)}]
        rows.append({"topic": ev.topic, "terms": ev.terms, **ans, "attempts": attempt,
                     "prompt_version": PROMPT_VERSION, "prompt_sha256": sha256(prompt),
                     "evidence_sha256": sha256(ev.to_json()), "error": err})

    new = pd.DataFrame(rows).set_index("topic")
    old = pd.read_csv(LABELS / "group_labels.csv").set_index("topic")
    cols = ["label", "definition", "distinct_from_neighbours"]
    same = (new[cols] == old.loc[new.index, cols]).all(axis=1)
    sha_same = new["evidence_sha256"] == old.loc[new.index, "evidence_sha256"]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(f"""# Group labels: rebuild and replay check

Generated by `scripts/label_groups.py`. Groups: {len(new)}.

- Evidence rebuilt from the data matches the archived evidence: {sum(ev_match)}/{len(new)}
  (evidence hash matches group_labels.csv: {int(sha_same.sum())}/{len(new)})
- Prompts rebuilt from the evidence match the archived prompts: {sum(prompt_match)}/{len(new)}
- Labels, definitions, and neighbor sentences replayed from the archived responses match
  group_labels.csv: {int(same.sum())}/{len(new)}
- Groups that needed a corrected second answer: {int((new.attempts > 1).sum())}
""", encoding="utf-8")
    print(REPORT.read_text())
    if not (all(ev_match) and all(prompt_match) and same.all()):
        raise SystemExit("rebuilt labels do not match the archive")


if __name__ == "__main__":
    main()
