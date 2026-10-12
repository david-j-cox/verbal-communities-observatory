"""Judge each title by its neighbors: the two conditions behind every flag.

1. Presentation-heavy neighborhood. Of a title's K nearest other titles (cosine similarity),
   the share that are presentations, compared with the corpus-wide share as a log odds with
   0.5 added to each count (Haldane-Anscombe correction). The neighborhood is presentation-heavy
   when the lower bound of the 95% confidence interval (Woolf) is above 0. Neighborhoods
   overlap, so the intervals describe each title's own neighborhood; they are not independent
   tests.
2. Few closely related articles. An article is closely related to a title when their cosine
   similarity is at least the q-th quantile (default .99) of similarity over all distinct pairs
   of titles. A title has few closely related articles when it has fewer than N of them. An
   article never counts itself.

A title is flagged when both conditions hold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HALDANE = 0.5


def nearest(S: np.ndarray, k: int) -> np.ndarray:
    """Column indices of the k largest entries of each row of S, largest first."""
    idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(S, idx, axis=1), axis=1)
    return np.take_along_axis(idx, order, axis=1)


def similarity(V: np.ndarray) -> np.ndarray:
    """Cosine similarity of unit-length rows, with self-similarity masked to -inf."""
    S = V @ V.T
    np.fill_diagonal(S, -np.inf)
    return S


def skew_scores(nn: np.ndarray, presented: np.ndarray, k: int,
                corpus_share: float | None = None) -> dict[str, np.ndarray]:
    """Presentation share among the k nearest titles, its log odds against the corpus share,
    and whether the lower 95% bound of that log odds is above 0."""
    pi = float(presented.mean()) if corpus_share is None else corpus_share
    p = presented[nn[:, :k]].sum(axis=1)
    lor = np.log((p + HALDANE) / (k - p + HALDANE)) - np.log(pi / (1 - pi))
    se = np.sqrt(1 / (p + HALDANE) + 1 / (k - p + HALDANE))
    return {"share": p / k, "log_odds": lor, "ci_lo": lor - 1.96 * se,
            "skewed": (lor - 1.96 * se) > 0}


def pair_quantile(V: np.ndarray, q: float = 0.99) -> float:
    """The q-th quantile of cosine similarity over all distinct pairs of unit-length rows."""
    S = V @ V.T
    return float(np.quantile(S[np.triu_indices(len(V), 1)], q))


def article_count(S: np.ndarray, presented: np.ndarray, threshold: float) -> np.ndarray:
    """Number of articles (~presented columns) with similarity at or above threshold to each
    title. S must have its diagonal masked so an article does not count itself."""
    return (S[:, ~presented] >= threshold).sum(axis=1)


@dataclass(frozen=True)
class ScreenResult:
    share: np.ndarray
    log_odds: np.ndarray
    skewed: np.ndarray
    line: float
    closely_related: np.ndarray
    few: np.ndarray
    flagged: np.ndarray


def screen(V: np.ndarray, presented: np.ndarray, k: int = 75, q: float = 0.99,
           n: float = 10) -> ScreenResult:
    """Run both conditions on unit-length title vectors V."""
    S = similarity(V)
    sk = skew_scores(nearest(S, k), presented, k)
    line = pair_quantile(V, q)
    count = article_count(S, presented, line)
    few = count < n
    return ScreenResult(sk["share"], sk["log_odds"], sk["skewed"], line, count, few,
                        sk["skewed"] & few)
