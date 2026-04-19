"""compute_attribution — Tool C.

Partial-attribution decomposition. For each driver (availability, visibility,
pricing, competitor), compute what the period-over-period revenue would have
been if that driver had held at its comparison-period level while the others
moved as observed. The difference is that driver's attributed share.

This is a first-order (Shapley-style) approximation — not full Shapley — which
is what the deployment brief calls for at V1. The four partial deltas are
renormalised so shares sum to 1.
"""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd

from analyst.data import MockDataStore, get_store, grade_quality
from analyst.schemas import AttributionBucket, AttributionResult


def _parse_period(period: str) -> tuple[date, date]:
    """Parse 'YYYY-MM-DD:YYYY-MM-DD' into (start, end)."""
    start, end = period.split(":")
    return (
        datetime.strptime(start, "%Y-%m-%d").date(),
        datetime.strptime(end, "%Y-%m-%d").date(),
    )


def _slice(df: pd.DataFrame, period: str) -> pd.DataFrame:
    start, end = _parse_period(period)
    mask = (df["snapshot_date"] >= start) & (df["snapshot_date"] <= end)
    return df[mask]


def _signals(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {
            "revenue": 0.0,
            "osa": 0.0,
            "sov": 0.0,
            "price": 0.0,
            "competitor_osa": 0.0,
        }
    return {
        "revenue": float(df["revenue"].sum()),
        "osa": float(df["osa"].mean()),
        "sov": float(df["brand_sov"].mean()),
        "price": float(df["selling_price"].mean()),
        "competitor_osa": float(df["competitor_osa_avg"].mean()),
    }


def _revenue_under(
    sig: dict[str, float], driver_overrides: dict[str, float]
) -> float:
    """
    Simple multiplicative response model:
      revenue ≈ base × (osa / osa_ref) × (sov / sov_ref) × (price_ref / price) × f(competitor)

    where f(competitor) = 1 + 0.4 × (1 - competitor_osa) (competitor weak → brand up).

    Not meant as a causal truth — it is a consistent scoring function used to
    isolate each driver's partial contribution. Reasonable in aggregate when
    drivers are bounded and independent.
    """
    osa = driver_overrides.get("osa", sig["osa"])
    sov = driver_overrides.get("sov", sig["sov"])
    price = driver_overrides.get("price", sig["price"])
    comp = driver_overrides.get("competitor_osa", sig["competitor_osa"])

    osa_ref = max(sig["osa"], 0.01)
    sov_ref = max(sig["sov"], 0.01)
    price_ref = max(sig["price"], 0.01)

    base = sig["revenue"]
    adj = (
        (osa / osa_ref)
        * (sov / sov_ref)
        * (price_ref / max(price, 0.01))
        * (1 + 0.4 * (1 - comp))
        / (1 + 0.4 * (1 - sig["competitor_osa"]))
    )
    return base * adj


def compute_attribution(
    brand_id: str,
    category: str,
    city: str,
    *,
    current_period: str,
    comparison_period: str,
    store: MockDataStore | None = None,
) -> AttributionResult:
    """Decompose a period-over-period revenue delta into four buckets."""
    store = store or get_store()
    df = store.attribution_input(brand_id, category, city)
    confidence, flags = grade_quality(df, tool="attribution")

    curr = _slice(df, current_period)
    prev = _slice(df, comparison_period)
    if curr.empty or prev.empty:
        flags.append("one_or_both_periods_empty")
        return AttributionResult(
            brand_id=brand_id,
            category=category,
            city=city,
            current_period=current_period,
            comparison_period=comparison_period,
            total_delta_rs=0.0,
            availability=AttributionBucket(0.0, 0.0),
            visibility=AttributionBucket(0.0, 0.0),
            pricing=AttributionBucket(0.0, 0.0),
            competitor=AttributionBucket(0.0, 0.0),
            dominant_cause="none",
            competitor_caused_rs=0.0,
            confidence="low",
            data_quality_flags=flags,
        )

    s_curr = _signals(curr)
    s_prev = _signals(prev)

    total_delta = s_curr["revenue"] - s_prev["revenue"]

    # Partial: hold each driver at previous level, take delta vs current baseline
    partials: dict[str, float] = {}
    for driver in ("osa", "sov", "price", "competitor_osa"):
        revenue_if_held = _revenue_under(s_curr, {driver: s_prev[driver]})
        # If we hold driver at previous level and the other drivers move, the
        # attributed delta to this driver is: (current_revenue - revenue_if_held)
        partials[driver] = s_curr["revenue"] - revenue_if_held

    # Renormalise to sum to total_delta
    partial_sum = sum(partials.values())
    if abs(partial_sum) < 1e-6:
        # Fall back to equal split over non-zero delta
        partials = {k: total_delta / 4 for k in partials}
    else:
        scale = total_delta / partial_sum
        partials = {k: v * scale for k, v in partials.items()}

    total_abs = sum(abs(v) for v in partials.values()) or 1.0
    shares = {k: abs(v) / total_abs for k, v in partials.items()}

    dominant = max(partials, key=lambda k: abs(partials[k]))
    dominant_label = {
        "osa": "availability",
        "sov": "visibility",
        "price": "pricing",
        "competitor_osa": "competitor",
    }[dominant]

    return AttributionResult(
        brand_id=brand_id,
        category=category,
        city=city,
        current_period=current_period,
        comparison_period=comparison_period,
        total_delta_rs=round(total_delta, 2),
        availability=AttributionBucket(round(partials["osa"], 2), round(shares["osa"], 3)),
        visibility=AttributionBucket(round(partials["sov"], 2), round(shares["sov"], 3)),
        pricing=AttributionBucket(round(partials["price"], 2), round(shares["price"], 3)),
        competitor=AttributionBucket(
            round(partials["competitor_osa"], 2), round(shares["competitor_osa"], 3)
        ),
        dominant_cause=dominant_label,
        competitor_caused_rs=round(partials["competitor_osa"], 2),
        confidence=confidence,
        data_quality_flags=flags,
    )
