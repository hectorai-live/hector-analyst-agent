"""Data quality grader — mirrors the analyst_data_quality BigQuery table.

A confident wrong number is worse than an honest uncertain one — so every tool
must call this before emitting a result.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

Confidence = Literal["high", "medium", "low"]

MIN_DAYS_FORECAST = 14
MIN_DAYS_COUNTERFACTUAL = 21
MIN_DAYS_ATTRIBUTION = 21


def grade_quality(
    df: pd.DataFrame,
    *,
    tool: Literal["forecast", "counterfactual", "attribution"],
    date_col: str = "snapshot_date",
) -> tuple[Confidence, list[str]]:
    """Grade a pulled dataframe and return (confidence, flags).

    Confidence descends on any of: insufficient history, sparse store coverage,
    or high restock/inwarding noise (proxied by day-over-day stock spikes).
    """
    flags: list[str] = []
    if df.empty:
        return "low", ["empty_dataset"]

    days = df[date_col].nunique()
    min_required = {
        "forecast": MIN_DAYS_FORECAST,
        "counterfactual": MIN_DAYS_COUNTERFACTUAL,
        "attribution": MIN_DAYS_ATTRIBUTION,
    }[tool]

    if days < min_required:
        flags.append(f"short_history_{days}d_need_{min_required}d")

    if "is_available" in df.columns:
        coverage = df["is_available"].mean()
        if coverage < 0.5:
            flags.append(f"low_store_coverage_{coverage:.0%}")

    if "units_consumed" in df.columns and len(df) > 2:
        # Detect restock / inwarding events as large positive jumps in inventory
        if "inventory_level" in df.columns:
            deltas = (
                df.sort_values(date_col)
                .groupby("dark_store_id")["inventory_level"]
                .diff()
            )
            positive_jumps = (deltas > 50).sum()
            if positive_jumps > len(df) * 0.08:
                flags.append("high_restock_noise")

    if days < min_required * 0.6:
        confidence: Confidence = "low"
    elif flags:
        confidence = "medium"
    else:
        confidence = "high"

    return confidence, flags
