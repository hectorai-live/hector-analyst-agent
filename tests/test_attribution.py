"""Tests for compute_attribution."""

from __future__ import annotations

from datetime import timedelta

from analyst.data.mock import TODAY
from analyst.tools import compute_attribution


def _period(days_end: int, days_start: int) -> str:
    start = TODAY - timedelta(days=days_start)
    end = TODAY - timedelta(days=days_end)
    return f"{start}:{end}"


def test_attribution_decomposition_structure(store):
    r = compute_attribution(
        brand_id="HEC",
        category="Face Wash",
        city="Bengaluru",
        current_period=_period(1, 7),
        comparison_period=_period(8, 14),
        store=store,
    )
    d = r.to_dict()
    assert d["tool"] == "compute_attribution"
    attr = d["attribution"]
    for bucket in ("availability", "visibility", "pricing", "competitor"):
        assert bucket in attr
        assert "delta_rs" in attr[bucket]
        assert "share_pct" in attr[bucket]
    assert r.dominant_cause in {"availability", "visibility", "pricing", "competitor"}


def test_attribution_shares_sum_to_one(store):
    r = compute_attribution(
        brand_id="HEC",
        category="Shampoo",
        city="Mumbai",
        current_period=_period(1, 7),
        comparison_period=_period(8, 14),
        store=store,
    )
    share_sum = (
        r.availability.share_pct
        + r.visibility.share_pct
        + r.pricing.share_pct
        + r.competitor.share_pct
    )
    assert abs(share_sum - 1.0) < 0.02


def test_attribution_deltas_sum_to_total(store):
    r = compute_attribution(
        brand_id="HEC",
        category="Face Wash",
        city="Delhi",
        current_period=_period(1, 7),
        comparison_period=_period(8, 14),
        store=store,
    )
    partial_sum = (
        r.availability.delta_rs
        + r.visibility.delta_rs
        + r.pricing.delta_rs
        + r.competitor.delta_rs
    )
    # Partials renormalise to the observed total delta
    assert abs(partial_sum - r.total_delta_rs) < max(1.0, 0.02 * abs(r.total_delta_rs))


def test_attribution_always_has_confidence_and_flags(store):
    r = compute_attribution(
        brand_id="HEC",
        category="Face Wash",
        city="Pune",
        current_period=_period(1, 7),
        comparison_period=_period(8, 14),
        store=store,
    )
    assert r.confidence in {"high", "medium", "low"}
    assert isinstance(r.data_quality_flags, list)


def test_attribution_empty_periods_graceful(store):
    # Periods entirely outside history -> empty flag, still returns a result
    r = compute_attribution(
        brand_id="HEC",
        category="Face Wash",
        city="Pune",
        current_period="2030-01-01:2030-01-07",
        comparison_period="2030-01-08:2030-01-14",
        store=store,
    )
    assert r.confidence == "low"
    assert "one_or_both_periods_empty" in r.data_quality_flags
