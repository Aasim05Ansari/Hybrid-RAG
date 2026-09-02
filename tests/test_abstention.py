import pytest

from app.generation.abstention import AbstentionPolicy


def test_good_confidence_does_not_abstain():
    decision = AbstentionPolicy().decide(
        composite_confidence=0.80,
        retrieval_confidence=0.70,
    )

    assert decision.should_abstain is False
    assert decision.reason is None


def test_low_composite_confidence_abstains():
    decision = AbstentionPolicy().decide(
        composite_confidence=0.40,
        retrieval_confidence=0.80,
    )

    assert decision.should_abstain is True
    assert decision.reason == "composite_confidence_too_low"


def test_low_retrieval_confidence_abstains():
    decision = AbstentionPolicy().decide(
        composite_confidence=0.90,
        retrieval_confidence=0.10,
    )

    assert decision.should_abstain is True
    assert decision.reason == "retrieval_confidence_too_low"


def test_threshold_values_are_allowed():
    decision = AbstentionPolicy().decide(
        composite_confidence=0.50,
        retrieval_confidence=0.25,
    )

    assert decision.should_abstain is False


@pytest.mark.parametrize(
    "composite_confidence",
    [-0.1, 1.1],
)
def test_invalid_composite_confidence(composite_confidence):
    with pytest.raises(ValueError):
        AbstentionPolicy().decide(
            composite_confidence=composite_confidence,
            retrieval_confidence=0.80,
        )


@pytest.mark.parametrize(
    "retrieval_confidence",
    [-0.1, 1.1],
)
def test_invalid_retrieval_confidence(retrieval_confidence):
    with pytest.raises(ValueError):
        AbstentionPolicy().decide(
            composite_confidence=0.80,
            retrieval_confidence=retrieval_confidence,
        )


def test_custom_thresholds():
    policy = AbstentionPolicy(
        min_composite_confidence=0.70,
        min_retrieval_confidence=0.60,
    )

    decision = policy.decide(
        composite_confidence=0.65,
        retrieval_confidence=0.80,
    )

    assert decision.should_abstain is True
    assert decision.reason == "composite_confidence_too_low"
