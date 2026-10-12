import json

import pytest

from presented_published.labeling import (
    NamingError,
    Neighbour,
    TopicEvidence,
    build_prompt,
    parse_response,
    sha256,
)

EV = TopicEvidence(
    topic=4, terms="autism service, autism spectrum",
    presented_titles=("Scaling Autism Service Delivery",),
    published_titles=("Defining Meaningful Outcomes",),
    neighbours=(Neighbour(131, 0.906, "autism, autistic", ("Supporting Autistic Adults",)),),
)


def answer(**kw):
    base = {"label": "Autism service delivery quality",
            "definition": "Work on organizing, measuring, and improving behavior-analytic "
                          "services for autistic clients.",
            "distinct_from_neighbours": "It concerns service systems rather than individual "
                                        "needs or the field as a whole."}
    base.update(kw)
    return json.dumps(base)


def test_prompt_is_deterministic_and_carries_evidence():
    p = build_prompt(EV)
    assert p == build_prompt(EV)
    assert sha256(p) == sha256(build_prompt(EV))
    for s in ("TOPIC 4", "Scaling Autism Service Delivery", "Defining Meaningful Outcomes",
              "NEIGHBOURING TOPIC 131 (similarity 0.91)", "Supporting Autistic Adults"):
        assert s in p


def test_prompt_marks_an_empty_side():
    ev = TopicEvidence(topic=1, terms="x", presented_titles=(), published_titles=("A",))
    assert "- (none in window)" in build_prompt(ev)


def test_parse_accepts_json_wrapped_in_prose():
    out = parse_response("Here it is:\n" + answer() + "\nDone.")
    assert out["label"] == "Autism service delivery quality"


@pytest.mark.parametrize("bad", [
    "no json here",
    answer(label="A label that is far too long to use"),
    answer(label="Ends with a period."),
    answer(definition=" ".join(["word"] * 36)),
    json.dumps({"label": "x", "definition": "y"}),
])
def test_parse_rejects_contract_breaks(bad):
    with pytest.raises(NamingError):
        parse_response(bad)
