"""Label and define each subject-area group with a language model, reproducibly.

Each group's evidence (its class-based TF-IDF key terms, the presentation and article titles
closest to its center, and its three most similar groups with their key terms and central titles)
is rendered into a fixed prompt, and the model's answer is validated. The prompt text below is
archived verbatim with every response (data/labels), so each label can be traced to exactly
what produced it. The prompt refers to groups as "topics"; it is kept as sent.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field

PROMPT_VERSION = "2026-10-10.1"

INSTRUCTIONS = """You are helping label topics found by clustering the titles of behavior-analytic \
conference presentations and journal articles (2016-2026). Clustering and term extraction were \
automatic. Your job is to describe one topic accurately for readers of a behavior-analytic \
journal who will see the label in a table.

Base your answer only on the evidence below. The titles closest to the topic's center are the \
best guide to what it contains; the key terms are word statistics and can be generic. The \
neighbouring topics are the most similar topics in the same model: say what separates this \
topic from them, so that neighbours never receive the same label.

Return JSON only, with exactly these keys:
  "label": a plain-language name, at most 6 words, no abbreviations a general behavior-analytic \
reader would not know, not ending in a period
  "definition": one sentence, at most 35 words, saying what work in this topic is about
  "distinct_from_neighbours": one sentence, at most 30 words, saying what separates it from \
the neighbouring topics
Do not mention conferences, journals, counts, or the clustering itself."""


@dataclass(frozen=True)
class Neighbour:
    topic: int
    cosine: float
    terms: str
    titles: tuple[str, ...]


@dataclass(frozen=True)
class TopicEvidence:
    topic: int
    terms: str
    presented_titles: tuple[str, ...]
    published_titles: tuple[str, ...]
    neighbours: tuple[Neighbour, ...] = field(default_factory=tuple)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)


def build_prompt(ev: TopicEvidence) -> str:
    """The full user message for one topic. Deterministic for a given evidence object."""
    lines = [INSTRUCTIONS, "", f"TOPIC {ev.topic}", f"Key terms: {ev.terms}", "",
             "Titles closest to the topic's center, conference presentations:"]
    lines += [f"- {t}" for t in ev.presented_titles] or ["- (none in window)"]
    lines += ["", "Titles closest to the topic's center, journal articles:"]
    lines += [f"- {t}" for t in ev.published_titles] or ["- (none in window)"]
    for nb in ev.neighbours:
        lines += ["", f"NEIGHBOURING TOPIC {nb.topic} (similarity {nb.cosine:.2f})",
                  f"Key terms: {nb.terms}"]
        lines += [f"- {t}" for t in nb.titles]
    return "\n".join(lines)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class NamingError(ValueError):
    """The model's answer did not meet the contract in INSTRUCTIONS."""


def parse_response(text: str) -> dict[str, str]:
    """Extract and validate the JSON answer; raise NamingError if it breaks the contract."""
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise NamingError(f"no JSON object in response: {text[:200]!r}")
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise NamingError(f"invalid JSON: {e}") from e
    keys = {"label", "definition", "distinct_from_neighbours"}
    if set(obj) != keys:
        raise NamingError(f"expected keys {sorted(keys)}, got {sorted(obj)}")
    out = {k: " ".join(str(obj[k]).split()) for k in keys}
    if not out["label"] or len(out["label"].split()) > 6 or out["label"].endswith("."):
        raise NamingError(f"label breaks the contract: {out['label']!r}")
    for k, cap in (("definition", 35), ("distinct_from_neighbours", 30)):
        n = len(out[k].split())
        if n == 0 or n > cap:
            raise NamingError(f"{k} has {n} words (limit {cap})")
    return out
