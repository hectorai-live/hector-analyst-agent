"""Tests for compute_counterfactual."""

from __future__ import annotations

import pytest

from analyst.tools import compute_counterfactual


def test_missed_attack_window_in_bengaluru(store):
    r = compute_counterfactual(
        brand_id="HEC",
        sku_id="FW100-HEC",
        city="Bengaluru",
        scenario="missed_attack_window",
        store=store,
    )
    d = r.to_dict()
    assert d["tool"] == "compute_counterfactual"
    assert d["scenario"] == "missed_attack_window"
    assert d["confidence"] in {"high", "medium", "low"}
    assert isinstance(d["data_quality_flags"], list)
    # Planted attack window: delta should be meaningful
    assert r.delta_rs > 0
    assert r.window_start != ""
    assert r.window_end != ""


def test_cost_of_delay_for_mumbai_oos(store):
    r = compute_counterfactual(
        brand_id="HEC",
        sku_id="FW100-HEC",
        city="Mumbai",
        scenario="cost_of_delay",
        store=store,
    )
    d = r.to_dict()
    assert d["scenario"] == "cost_of_delay"
    assert d["confidence"] in {"high", "medium", "low"}
    assert isinstance(d["data_quality_flags"], list)


def test_what_if_osa(store):
    r = compute_counterfactual(
        brand_id="HEC",
        sku_id="SH200-HEC",
        city="Delhi",
        scenario="what_if_osa",
        target_osa=0.9,
        store=store,
    )
    assert r.scenario == "what_if_osa"
    assert r.counterfactual_revenue_rs >= r.actual_revenue_rs - 1e-6
    assert r.delta_rs >= 0.0


def test_unknown_scenario_raises(store):
    with pytest.raises(ValueError):
        compute_counterfactual(
            brand_id="HEC",
            sku_id="FW100-HEC",
            city="Mumbai",
            scenario="bogus",
            store=store,
        )


def test_counterfactual_always_has_confidence_and_flags(store):
    for scenario in ("missed_attack_window", "cost_of_delay", "what_if_osa"):
        r = compute_counterfactual(
            brand_id="HEC",
            sku_id="FW100-HEC",
            city="Mumbai",
            scenario=scenario,
            store=store,
        )
        assert r.confidence in {"high", "medium", "low"}
        assert isinstance(r.data_quality_flags, list)
