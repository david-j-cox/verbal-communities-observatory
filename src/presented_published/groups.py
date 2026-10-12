"""Subject-area groups used to describe the flagged titles.

The groups are a reporting device: no title's flag depends on its group. Titles are sorted into
200 groups by k-means (random_state = 0, n_init = 10) fit on a fixed set of title vectors: the
ABAI titles and the eight general journals' titles (all years) as they stood when the groups
were first fit (``conference_fit`` and ``journal_fit``). Every analyzed title is then assigned
to its nearest group center. Two groups that differ only in a synonymous term ("challenging
behavior" and "problem behavior") are merged.

The module also computes each group's key terms by class-based TF-IDF (Grootendorst, 2022),
which are part of the evidence given to the language model that labeled the groups.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

TITLES = Path("data/titles")        # title tables, one per source (shipped)
EMB = Path("data/embeddings")       # vectors written by scripts/embed_titles.py (regenerated)
SLUG = "all-minilm-l6-v2"
K, SEED = 200, 0
FIT_SETS = ("conference_fit", "journal_fit")


def load(name: str, emb: Path = EMB) -> np.ndarray:
    """The embedding matrix written by scripts/embed_titles.py for one title set."""
    return np.load(emb / f"{name}__{SLUG}.npy")


def fit_matrix(emb: Path = EMB) -> np.ndarray:
    """The vectors the group centers are fit on: ABAI fit titles, then journal fit titles."""
    return np.vstack([load(n, emb) for n in FIT_SETS])


def fit_kmeans(X: np.ndarray, k: int = K, seed: int = SEED) -> KMeans:
    return KMeans(n_clusters=k, random_state=seed, n_init=10).fit(X)


@dataclass(frozen=True)
class SynonymMerge:
    """Two groups that differ only in a synonymous term, merged into ``keep``."""

    keep: int
    absorb: int
    keep_term: str
    absorb_term: str


SYNONYM_MERGES: tuple[SynonymMerge, ...] = (
    SynonymMerge(keep=160, absorb=41, keep_term="challenging behavior",
                 absorb_term="problem behavior"),
)


def validate_synonym_merges(terms: Mapping[int, str],
                            merges: Sequence[SynonymMerge] = SYNONYM_MERGES) -> None:
    """Raise if a merged group does not carry its expected term (i.e., the fit has changed)."""
    for m in merges:
        for g, term in ((m.keep, m.keep_term), (m.absorb, m.absorb_term)):
            found = str(terms.get(g, ""))
            if term not in found.lower():
                raise ValueError(f"synonym merge expects group {g} to carry '{term}', found "
                                 f"'{found}'; the k-means fit has changed")


def apply_synonym_merges(labels: np.ndarray,
                         merges: Sequence[SynonymMerge] = SYNONYM_MERGES) -> np.ndarray:
    """A copy of ``labels`` with each absorbed group relabelled to its kept group."""
    out = np.asarray(labels).copy()
    for m in merges:
        out[out == m.absorb] = m.keep
    return out


def merged_centers(km: KMeans, X: np.ndarray,
                   merges: Sequence[SynonymMerge] = SYNONYM_MERGES,
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Group centers after the merge (a kept group's center is the mean of its fit rows) and
    the ids of the groups that remain."""
    C = km.cluster_centers_.copy()
    labels = apply_synonym_merges(km.labels_, merges)
    for m in merges:
        C[m.keep] = X[labels == m.keep].mean(axis=0)
    active = np.setdiff1d(np.arange(len(C)), [m.absorb for m in merges])
    return C, active


# Field-generic terms: true of essentially the whole corpus, so useless as labels.
GENERIC_TERMS = frozenset({
    "behavior", "behaviors", "behavioral", "analysis", "applied", "aba", "analytic",
    "behavior analysis", "applied behavior", "applied behavior analysis", "analysts",
    "behavior analysts", "behavior analytic", "effect", "effects", "study", "studies",
    "research", "using", "use", "based", "approach", "review", "assessment intervention",
})


def class_tfidf_terms(texts: list[str], labels: np.ndarray, k: int, top: int = 4,
                min_doc_frac: float = 0.08, min_docs: int = 4) -> dict[int, str]:
    """Label each cluster by class-based TF-IDF (c-TF-IDF), prevalence-gated.

    c-TF-IDF (IDF computed across the K clusters) down-weights field-wide terms so
    each label shows what is distinctive to a cluster. But distinctive is not the
    same as representative: a rare term mentioned in a handful of a cluster's
    titles can still rank top. So a term is only eligible if it is also PREVALENT
    in the cluster -- present in at least ``min_doc_frac`` of its titles (or
    ``min_docs`` of them). Field-generic terms are dropped and n-grams deduped.
    """

    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

    texts_arr = np.asarray(texts, dtype=object)
    counts = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=5,
                             token_pattern=r"[A-Za-z][A-Za-z-]{2,}")
    doc_term = counts.fit_transform(texts_arr)            # n_docs x V
    vocab = np.array(counts.get_feature_names_out())
    present = (doc_term > 0).astype(int)
    class_counts = np.vstack([
        np.asarray(doc_term[labels == t].sum(axis=0)).ravel() for t in range(k)
    ])                                                    # K x V dense counts
    ctfidf = TfidfTransformer(sublinear_tf=True).fit_transform(class_counts)

    out: dict[int, str] = {}
    for t in range(k):
        mask = labels == t
        n = int(mask.sum())
        in_docs = np.asarray(present[mask].sum(axis=0)).ravel()  # docs in cluster with term
        ctf = ctfidf[t].toarray().ravel()
        prevalent = (in_docs >= max(min_docs, min_doc_frac * n))

        def pick(allow_generic: bool, require_prevalent: bool, min_in: int) -> list[str]:
            chosen: list[str] = []
            order = (ctf * prevalent if require_prevalent else ctf).argsort()[::-1]
            for idx in order:
                if (ctf[idx] <= 0) or (in_docs[idx] < min_in):
                    continue
                term = vocab[idx]
                if (not allow_generic) and term in GENERIC_TERMS:
                    continue
                parts = term.split()
                if len(parts) == 2 and parts[0] == parts[1]:   # cross-title bigram artifact
                    continue
                if any(term in c or c in term for c in chosen):  # dedupe overlapping n-grams
                    continue
                chosen.append(term)
                if len(chosen) >= top:
                    break
            return chosen

        # tiers: strict (distinctive+prevalent+non-generic) -> relax prevalence ->
        # allow generic terms as a last resort, so every topic gets a real label.
        chosen = (pick(False, True, max(min_docs, int(min_doc_frac * n)))
                  or pick(False, False, 2)
                  or pick(True, True, max(min_docs, int(min_doc_frac * n)))
                  or pick(True, False, 2))
        out[t] = ", ".join(chosen) if chosen else f"topic-{t}"
    return out
