"""Tests for compute_forecast."""

from __future__ import annotations

from analyst.tools import compute_forecast


def test_forecast_happy_path(store):
    r = compute_forecast(sku_id="FW100-HEC", city="Mumbai", store=store)
    d = r.to_dict()
    assert d["tool"] == "compute_forecast"
    assert d["sku_id"] == "FW100-HEC"
    assert d["city"] == "Mumbai"
    assert d["confidence"] in {"high", "medium", "low"}
    assert isinstance(d["data_quality_flags"], list)
    assert d["total_stores"] > 0


def test_forecast_probabilities_are_bounded(store):
    r = compute_forecast(sku_id="SH200-HEC", city="Delhi", store=store)
    assert 0.0 <= r.probability_oos_within_7_days <= 1.0
    assert 0.0 <= r.probability_oos_within_14_days <= 1.0
    # p(14d) should be >= p(7d)
    assert r.probability_oos_within_14_days + 1e-6 >= r.probability_oos_within_7_days


def test_forecast_detects_planted_oos_event(store):
    """Face Wash HEC in Mumbai has a planted OOS event in last 5 days.
    Forecast should flag meaningful stock-out risk."""
    r = compute_forecast(sku_id="FW100-HEC", city="Mumbai", store=store)
    assert r.stores_at_risk >= 1
    assert r.probability_oos_within_7_days > 0.3


def test_forecast_psl_is_non_negative(store):
    r = compute_forecast(sku_id="BL250-HEC", city="Pune", store=store)
    assert r.forward_psl_rs >= 0.0


def test_forecast_always_has_confidence_and_flags(store):
    for sku in ("FW100-HEC", "SH200-HEC", "BL250-HEC"):
        for city in ("Mumbai", "Delhi", "Bengaluru", "Pune"):
            r = compute_forecast(sku_id=sku, city=city, store=store)
            assert r.confidence in {"high", "medium", "low"}
            assert isinstance(r.data_quality_flags, list)
