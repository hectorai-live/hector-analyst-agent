"""compute_counterfactual — Tool B.

Three scenarios (per the deployment brief):
  A. missed_attack_window — value of a competitor OOS window the brand didn't act on.
  B. cost_of_delay       — ₹ lost between detectable-OOS day and actual-action day.
  C. what_if_osa         — ₹ recovered if OSA had been held at a target level.
"""

from __future__ import annotations

from datetime import date, datetime


from analyst.data import MockDataStore, get_store, grade_quality
from analyst.data.mock import CATALOG
from analyst.schemas import CounterfactualResult

SOV_UPLIFT_WHEN_COMPETITOR_OOS = 0.35  # empirical prior; tune from portal history


def _parse(d: str | date) -> date:
    if isinstance(d, date):
        return d
    return datetime.strptime(d, "%Y-%m-%d").date()


def compute_counterfactual(
    brand_id: str,
    sku_id: str,
    city: str,
    *,
    scenario: str,
    window_start: str | date | None = None,
    window_end: str | date | None = None,
    target_osa: float = 0.85,
    store: MockDataStore | None = None,
) -> CounterfactualResult:
    if scenario not in {"missed_attack_window", "cost_of_delay", "what_if_osa"}:
        raise ValueError(f"Unknown scenario: {scenario}")

    store = store or get_store()
    df = store.counterfactual_input(brand_id, sku_id, city)
    confidence, flags = grade_quality(df, tool="counterfactual")
    sku = CATALOG[sku_id]

    if scenario == "missed_attack_window":
        return _scenario_missed_attack_window(
            brand_id, sku_id, city, df, flags, confidence, sku.selling_price
        )
    if scenario == "cost_of_delay":
        ws = _parse(window_start) if window_start else None
        we = _parse(window_end) if window_end else None
        return _scenario_cost_of_delay(
            brand_id, sku_id, city, df, flags, confidence, sku.selling_price, ws, we
        )
    return _scenario_what_if_osa(
        brand_id, sku_id, city, df, flags, confidence, sku.selling_price, target_osa
    )


def _scenario_missed_attack_window(
    brand_id, sku_id, city, df, flags, confidence, price
) -> CounterfactualResult:
    # Find contiguous days where competitor OSA < 0.40
    daily = (
        df.groupby("snapshot_date")
        .agg(
            competitor_osa=("competitor_osa", "mean"),
            brand_units=("units_consumed", "sum"),
            brand_osa=("is_available", "mean"),
        )
        .reset_index()
        .sort_values("snapshot_date")
    )
    daily["is_attack"] = daily["competitor_osa"] < 0.40
    if not daily["is_attack"].any():
        flags.append("no_attack_window_found")
        return _empty_cf(
            brand_id, sku_id, city, "missed_attack_window", flags, "low",
            "no attack window detected"
        )

    attack = daily[daily["is_attack"]]
    window_start = attack["snapshot_date"].min()
    window_end = attack["snapshot_date"].max()
    window_days = (window_end - window_start).days + 1

    # Baseline demand (non-attack days) for the brand
    baseline_units = daily.loc[~daily["is_attack"], "brand_units"].mean()
    actual_units_in_window = attack["brand_units"].sum()
    expected_units_if_acted = (
        baseline_units * window_days * (1 + SOV_UPLIFT_WHEN_COMPETITOR_OOS)
    )
    actual_revenue = actual_units_in_window * price
    counterfactual_revenue = expected_units_if_acted * price
    delta = counterfactual_revenue - actual_revenue
    delta_pct = delta / max(counterfactual_revenue, 1.0)

    return CounterfactualResult(
        brand_id=brand_id,
        sku_id=sku_id,
        city=city,
        scenario="missed_attack_window",
        window_start=str(window_start),
        window_end=str(window_end),
        actual_revenue_rs=round(actual_revenue, 2),
        counterfactual_revenue_rs=round(counterfactual_revenue, 2),
        delta_rs=round(delta, 2),
        delta_pct=round(delta_pct, 3),
        key_driver=f"competitor OSA collapsed to {attack['competitor_osa'].mean():.0%}",
        confidence=confidence,
        data_quality_flags=flags,
    )


def _scenario_cost_of_delay(
    brand_id, sku_id, city, df, flags, confidence, price, ws, we
) -> CounterfactualResult:
    daily = (
        df.groupby("snapshot_date")
        .agg(
            brand_osa=("is_available", "mean"),
            brand_units=("units_consumed", "sum"),
        )
        .reset_index()
        .sort_values("snapshot_date")
    )

    # Detectable day = first day OSA < 0.7 (significant city-wide availability drop)
    low = daily[daily["brand_osa"] < 0.7]
    if low.empty:
        flags.append("no_oos_episode_detected")
        return _empty_cf(
            brand_id, sku_id, city, "cost_of_delay", flags, "low",
            "no OOS episode detected"
        )
    detectable_day = low["snapshot_date"].iloc[0]

    # Action day = first day after detectable where OSA recovers >0.7, else last day
    recovered = daily[
        (daily["snapshot_date"] > detectable_day) & (daily["brand_osa"] > 0.7)
    ]
    action_day = (
        recovered["snapshot_date"].iloc[0]
        if not recovered.empty
        else daily["snapshot_date"].iloc[-1]
    )
    ws = ws or detectable_day
    we = we or action_day

    # Baseline demand from healthy days (OSA > 0.9)
    healthy = daily[daily["brand_osa"] > 0.9]
    baseline = healthy["brand_units"].mean() if not healthy.empty else daily["brand_units"].median()

    window = daily[(daily["snapshot_date"] >= ws) & (daily["snapshot_date"] <= we)]
    actual_units = window["brand_units"].sum()
    expected_units = baseline * len(window)
    actual_revenue = actual_units * price
    counterfactual_revenue = expected_units * price
    delta = counterfactual_revenue - actual_revenue

    delay_days = (action_day - detectable_day).days
    return CounterfactualResult(
        brand_id=brand_id,
        sku_id=sku_id,
        city=city,
        scenario="cost_of_delay",
        window_start=str(ws),
        window_end=str(we),
        actual_revenue_rs=round(actual_revenue, 2),
        counterfactual_revenue_rs=round(counterfactual_revenue, 2),
        delta_rs=round(delta, 2),
        delta_pct=round(delta / max(counterfactual_revenue, 1.0), 3),
        key_driver=f"{delay_days}-day delay between detectable-OOS and action",
        confidence=confidence,
        data_quality_flags=flags,
    )


def _scenario_what_if_osa(
    brand_id, sku_id, city, df, flags, confidence, price, target_osa
) -> CounterfactualResult:
    per_store = (
        df.groupby("dark_store_id")
        .agg(
            actual_osa=("is_available", "mean"),
            avg_demand_available_day=("units_consumed", lambda s: s[s > 0].median()),
            days=("snapshot_date", "nunique"),
        )
        .reset_index()
    )
    per_store["avg_demand_available_day"] = per_store["avg_demand_available_day"].fillna(0)
    under = per_store[per_store["actual_osa"] < target_osa].copy()
    # units that would have sold if OSA = target_osa
    under["osa_gap"] = (target_osa - under["actual_osa"]).clip(lower=0)
    under["missed_units"] = (
        under["osa_gap"] * under["days"] * under["avg_demand_available_day"]
    )
    missed_units = float(under["missed_units"].sum())

    actual_units = float(df["units_consumed"].sum())
    actual_revenue = actual_units * price
    counterfactual_revenue = actual_revenue + missed_units * price
    delta = counterfactual_revenue - actual_revenue

    period_start = df["snapshot_date"].min()
    period_end = df["snapshot_date"].max()

    return CounterfactualResult(
        brand_id=brand_id,
        sku_id=sku_id,
        city=city,
        scenario="what_if_osa",
        window_start=str(period_start),
        window_end=str(period_end),
        actual_revenue_rs=round(actual_revenue, 2),
        counterfactual_revenue_rs=round(counterfactual_revenue, 2),
        delta_rs=round(delta, 2),
        delta_pct=round(delta / max(counterfactual_revenue, 1.0), 3),
        key_driver=(
            f"{len(under)} stores under {target_osa:.0%} target OSA "
            f"(actual avg {per_store['actual_osa'].mean():.0%})"
        ),
        confidence=confidence,
        data_quality_flags=flags,
    )


def _empty_cf(brand_id, sku_id, city, scenario, flags, confidence, reason):
    return CounterfactualResult(
        brand_id=brand_id,
        sku_id=sku_id,
        city=city,
        scenario=scenario,
        window_start="",
        window_end="",
        actual_revenue_rs=0.0,
        counterfactual_revenue_rs=0.0,
        delta_rs=0.0,
        delta_pct=0.0,
        key_driver=reason,
        confidence=confidence,
        data_quality_flags=flags,
    )
