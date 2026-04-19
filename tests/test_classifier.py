"""Tests for the deterministic intent classifier."""

from __future__ import annotations

from analyst.nodes.classifier import classify_intent
from analyst.schemas import AnalystState, Mode


def _state(q: str) -> AnalystState:
    return AnalystState(query=q, brand_id="HEC")


def test_forecast_cue():
    s = classify_intent(_state("will face wash stock out in mumbai?"))
    assert Mode.FORECAST in s.modes_triggered


def test_counterfactual_cue():
    s = classify_intent(_state("what would we have made if we had acted?"))
    assert Mode.COUNTERFACTUAL in s.modes_triggered


def test_causal_cue():
    s = classify_intent(_state("why is revenue down?"))
    assert Mode.CAUSAL in s.modes_triggered


def test_compound_query_triggers_multiple_modes():
    s = classify_intent(
        _state("why is revenue down and will it get worse next 7 days?")
    )
    assert Mode.CAUSAL in s.modes_triggered
    assert Mode.FORECAST in s.modes_triggered


def test_bare_query_defaults_to_forecast():
    s = classify_intent(_state(""))
    assert s.modes_triggered == [Mode.FORECAST]


def test_denoise_cue():
    s = classify_intent(_state("can you denoise the inventory series?"))
    assert Mode.DENOISE in s.modes_triggered


def test_anomaly_cue():
    s = classify_intent(_state("any unusual spikes in OSA last week?"))
    assert Mode.ANOMALY in s.modes_triggered


def test_changepoint_cue():
    s = classify_intent(_state("when did the OSA regime change?"))
    assert Mode.CHANGEPOINT in s.modes_triggered


def test_elasticity_cue():
    s = classify_intent(_state("what happens if we drop price by 10%?"))
    assert Mode.ELASTICITY in s.modes_triggered


def test_classifier_is_deterministic():
    q = "why did revenue drop and will it recover?"
    a = classify_intent(_state(q)).modes_triggered
    b = classify_intent(_state(q)).modes_triggered
    assert a == b
