"""Tests for detect_anomalies — STL-lite + robust Z + CUSUM."""

from __future__ import annotations

import numpy as np

from analyst.tools import detect_anomalies


def test_flags_large_drop():
    rng = np.random.default_rng(11)
    baseline = 85.0 + rng.normal(0, 1.5, 28)
    baseline[20] = 50.0  # large drop
    dates = [f"2026-03-{(i % 28) + 1:02d}" for i in range(28)]
    r = detect_anomalies(baseline, dates=dates, kind="osa", series_name="osa_delhi")
    assert r.n_points == 28
    assert any(e.label == "attack_window_start" and e.direction == "drop" for e in r.events)


def test_flags_spike():
    rng = np.random.default_rng(13)
    baseline = 500.0 + rng.normal(0, 10, 28)
    baseline[10] = 900.0  # spike
    r = detect_anomalies(baseline, kind="units")
    assert any(e.direction == "spike" and e.label == "promo_spike" for e in r.events)


def test_short_series_is_low_confidence():
    r = detect_anomalies([1.0, 2.0, 3.0], kind="other")
    assert r.confidence == "low"
    assert any("insufficient_points" in f for f in r.data_quality_flags)


def test_stealth_oos_label_when_osa_stable():
    rng = np.random.default_rng(17)
    units = 1000.0 + rng.normal(0, 30, 28)
    units[22] = 500.0  # unit drop not explained by OSA
    r = detect_anomalies(units, kind="units", osa_also_dropped=False)
    assert any(e.label == "stealth_oos" for e in r.events)


def test_stable_series_no_events():
    rng = np.random.default_rng(19)
    y = 100.0 + rng.normal(0, 0.5, 28)
    r = detect_anomalies(y, kind="osa")
    # Under N(0, 0.5) with |z|>3.5 threshold we expect zero events almost surely
    assert r.events == []
