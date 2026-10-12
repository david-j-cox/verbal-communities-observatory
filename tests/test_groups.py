import numpy as np
import pytest

from presented_published.groups import (
    SYNONYM_MERGES,
    SynonymMerge,
    apply_synonym_merges,
    class_tfidf_terms,
    merged_centers,
    validate_synonym_merges,
)

MERGE = SynonymMerge(keep=5, absorb=2, keep_term="challenging behavior",
                     absorb_term="problem behavior")


def test_apply_relabels_absorbed_group_only():
    labels = np.array([0, 2, 5, 2, 3])
    out = apply_synonym_merges(labels, [MERGE])
    assert out.tolist() == [0, 5, 5, 5, 3]
    assert labels.tolist() == [0, 2, 5, 2, 3]  # input untouched


def test_validate_passes_on_expected_terms():
    terms = {5: "severe challenging, challenging behavior", 2: "severe problem, problem behavior"}
    validate_synonym_merges(terms, [MERGE])


def test_validate_raises_when_fit_changed():
    terms = {5: "severe problem, problem behavior", 2: "pediatric feeding"}
    with pytest.raises(ValueError, match="fit has changed"):
        validate_synonym_merges(terms, [MERGE])


def test_declared_merges_are_distinct_groups():
    for m in SYNONYM_MERGES:
        assert m.keep != m.absorb


def test_merged_center_is_mean_of_both_groups():
    class KM:
        cluster_centers_ = np.array([[0.0, 0.0], [9.0, 9.0], [1.0, 1.0]])
        labels_ = np.array([0, 0, 1, 2, 2])
    X = np.array([[0.0, 0.0], [0.0, 0.2], [4.0, 4.0], [2.0, 2.0], [2.0, 2.4]])
    merge = SynonymMerge(keep=2, absorb=1, keep_term="a", absorb_term="b")
    C, active = merged_centers(KM(), X, [merge])
    assert active.tolist() == [0, 2]
    assert np.allclose(C[2], X[2:].mean(axis=0))


def test_key_terms_pick_distinctive_words():
    texts = (["functional communication training for problem behavior"] * 6
             + ["delay discounting of monetary rewards in smokers"] * 6)
    labels = np.array([0] * 6 + [1] * 6)
    terms = class_tfidf_terms(texts, labels, 2, top=2)
    assert "discounting" in terms[1] and "discounting" not in terms[0]
