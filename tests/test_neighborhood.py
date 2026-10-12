import numpy as np

from presented_published.neighborhood import (
    article_count,
    nearest,
    pair_quantile,
    screen,
    similarity,
    skew_scores,
)


def unit(X):
    X = np.asarray(X, float)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def test_nearest_excludes_self_and_orders():
    V = unit([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]])
    nn = nearest(similarity(V), 2)
    assert nn[0, 0] == 1 and 0 not in nn[0]
    assert nn[2, 0] == 3


def test_skew_flags_presentation_rich_neighbourhood():
    presented = np.array([True] * 30 + [False] * 30)
    nn = np.tile(np.arange(30), (60, 1))          # every title's neighbours: 30 presentations
    out = skew_scores(nn, presented, 30)
    assert np.allclose(out["share"], 1.0) and out["skewed"].all()
    nn2 = np.tile(np.arange(30, 60), (60, 1))     # all articles
    assert not skew_scores(nn2, presented, 30)["skewed"].any()


def test_screen_combines_both_conditions():
    rng = np.random.default_rng(0)
    V = unit(rng.normal(size=(200, 8)))
    presented = rng.random(200) < 0.45
    r = screen(V, presented, k=20, q=0.9, n=5)
    assert np.array_equal(r.flagged, r.skewed & r.few)
    assert np.array_equal(r.few, r.closely_related < 5)


def test_article_count_excludes_self_and_presentations():
    V = unit([[1, 0], [1, 0.05], [1, 0.1], [0, 1]])
    presented = np.array([True, False, False, False])
    S = similarity(V)
    c = article_count(S, presented, 0.99)
    assert c.tolist() == [2, 1, 1, 0]       # row 0 is a presentation and never counted


def test_pair_quantile_bounds():
    V = unit(np.random.default_rng(1).normal(size=(50, 4)))
    S = V @ V.T
    off = S[np.triu_indices(50, 1)]
    assert off.min() <= pair_quantile(V, 0.5) <= off.max()
