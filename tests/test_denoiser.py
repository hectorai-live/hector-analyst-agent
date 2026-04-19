"""Tests for denoise_signal — 2-component GMM over inventory deltas."""

from __future__ import annotations

import numpy as np

from analyst.tools import denoise_signal


def test_denoiser_separates_restock_from_consumption():
    rng = np.random.default_rng(7)
    # 28 days: mostly small negative consumption with 4 large positive restocks mixed in
    consumption = (-rng.uniform(3, 8, 24)).tolist()
    restocks = rng.uniform(80, 140, 4).tolist()
    series = np.array(consumption + restocks)
    rng.shuffle(series)

    r = denoise_signal(series.tolist(), series_name="test")
    d = r.to_dict()

    assert d["tool"] == "denoise_signal"
    assert d["restock_events"] >= 3  # should catch most of the 4 restocks
    assert d["denoised_mean"] > 0  # consumption magnitude is positive
    assert d["noise_share_pct"] > 50  # restocks dominate by mass
    assert d["confidence"] in {"high", "medium", "low"}
    assert isinstance(d["data_quality_flags"], list)


def test_denoiser_short_series_is_low_confidence():
    r = denoise_signal([1.0, -2.0, 3.0])
    assert r.confidence == "low"
    assert any("insufficient_points" in f for f in r.data_quality_flags)


def test_denoiser_handles_nans():
    r = denoise_signal([1.0, float("nan"), -2.0, -3.0, 50.0, -2.0, -1.0, 40.0])
    assert r.n_points == 7  # nan stripped
