"""Tests for detect_changepoints — PELT with BIC penalty."""

from __future__ import annotations

import numpy as np

from analyst.tools import detect_changepoints


def test_detects_mean_shift():
    rng = np.random.default_rng(23)
    seg_a = 100.0 + rng.normal(0, 2.0, 20)
    seg_b = 140.0 + rng.normal(0, 2.0, 20)
    series = np.concatenate([seg_a, seg_b])
    dates = [f"2026-01-{i + 1:02d}" for i in range(40)]
    r = detect_changepoints(series, dates=dates, series_name="unit_series")
    d = r.to_dict()
    assert d["tool"] == "detect_changepoints"
    assert len(d["changepoints"]) >= 1
    # Must find a cp near index 20 (i.e. date 2026-01-21)
    cp_ints = [dates.index(cp) for cp in d["changepoints"]]
    assert any(15 <= i <= 25 for i in cp_ints)
    assert len(d["segment_means"]) == len(d["changepoints"]) + 1


def test_stable_series_returns_no_changepoints():
    rng = np.random.default_rng(29)
    y = 100.0 + rng.normal(0, 1.0, 30)
    r = detect_changepoints(y)
    assert r.changepoints == []
    assert len(r.segment_means) == 1


def test_short_series_is_low_confidence():
    r = detect_changepoints([1.0, 2.0])
    assert r.confidence == "low"
