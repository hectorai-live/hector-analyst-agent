"""Tests for the data-quality grader."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from analyst.data.quality import grade_quality


def _make_df(n_days: int, available_pct: float = 0.8) -> pd.DataFrame:
    base = date(2026, 4, 1)
    rows = []
    for i in range(n_days):
        for ds in range(3):
            rows.append(
                {
                    "snapshot_date": base + timedelta(days=i),
                    "dark_store_id": f"DS{ds}",
                    "inventory_level": 100 - i,
                    "units_consumed": 5,
                    "is_available": (ds + i) % 10 < int(10 * available_pct),
                }
            )
    return pd.DataFrame(rows)


def test_empty_df_is_low():
    conf, flags = grade_quality(pd.DataFrame(), tool="forecast")
    assert conf == "low"
    assert "empty_dataset" in flags


def test_short_history_flags_and_downgrades():
    df = _make_df(5)  # way below forecast minimum of 14
    conf, flags = grade_quality(df, tool="forecast")
    assert conf == "low"
    assert any("short_history" in f for f in flags)


def test_borderline_history_is_medium():
    df = _make_df(12)  # between 60% of 14 and 14 → medium
    conf, flags = grade_quality(df, tool="forecast")
    assert conf == "medium"


def test_healthy_history_is_high():
    df = _make_df(20, available_pct=0.9)
    conf, flags = grade_quality(df, tool="forecast")
    assert conf == "high"
    assert flags == [] or "short_history" not in " ".join(flags)


def test_low_coverage_flag():
    df = _make_df(20, available_pct=0.2)
    conf, flags = grade_quality(df, tool="forecast")
    assert any("low_store_coverage" in f for f in flags)
