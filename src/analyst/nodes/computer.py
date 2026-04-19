"""Node 3 — Computation.

Dispatches to the three Python tools based on modes_triggered. Any tool that
fails is captured in state.errors; the rest still produce estimates.
"""

from __future__ import annotations

from datetime import timedelta

from analyst.data.mock import TODAY
from analyst.schemas import AnalystState, Mode
from analyst.tools import (
    compute_attribution,
    compute_counterfactual,
    compute_forecast,
    denoise_signal,
    detect_anomalies,
    detect_changepoints,
    estimate_elasticity,
)


def run_computation(state: AnalystState) -> AnalystState:
    if Mode.FORECAST in state.modes_triggered:
        try:
            result = compute_forecast(
                sku_id=state.sku_id or _require(state, "sku_id"),
                city=state.city or _require(state, "city"),
                brand_id=state.brand_id,
            )
            state.computation_results.append(result.to_dict())
        except Exception as e:
            state.errors.append(f"forecast_failed: {e!r}")

    if Mode.COUNTERFACTUAL in state.modes_triggered:
        try:
            result = compute_counterfactual(
                brand_id=state.brand_id,
                sku_id=state.sku_id or _require(state, "sku_id"),
                city=state.city or _require(state, "city"),
                scenario=state.scenario or "missed_attack_window",
                window_start=state.window_start,
                window_end=state.window_end,
                target_osa=state.target_osa or 0.85,
            )
            state.computation_results.append(result.to_dict())
        except Exception as e:
            state.errors.append(f"counterfactual_failed: {e!r}")

    if Mode.CAUSAL in state.modes_triggered:
        try:
            current_period = state.period or _default_period(7)
            comparison_period = state.comparison_period or _default_period(14, 8)
            result = compute_attribution(
                brand_id=state.brand_id,
                category=state.category or _require(state, "category"),
                city=state.city or _require(state, "city"),
                current_period=current_period,
                comparison_period=comparison_period,
            )
            state.computation_results.append(result.to_dict())
        except Exception as e:
            state.errors.append(f"attribution_failed: {e!r}")

    if Mode.DENOISE in state.modes_triggered:
        try:
            series = state.series_input or _series_from_state(state, kind="deltas")
            if series is None:
                raise ValueError("no series available to denoise")
            result = denoise_signal(series, series_name=f"{state.sku_id or state.brand_id}_deltas")
            state.computation_results.append(result.to_dict())
        except Exception as e:
            state.errors.append(f"denoise_failed: {e!r}")

    if Mode.ANOMALY in state.modes_triggered:
        try:
            series = state.series_input or _series_from_state(state, kind="osa_or_units")
            if series is None:
                raise ValueError("no series available for anomaly detection")
            result = detect_anomalies(
                series,
                dates=state.series_dates,
                series_name=f"{state.brand_id}_{state.city or 'all'}",
                kind="osa",
            )
            state.computation_results.append(result.to_dict())
        except Exception as e:
            state.errors.append(f"anomaly_failed: {e!r}")

    if Mode.CHANGEPOINT in state.modes_triggered:
        try:
            series = state.series_input or _series_from_state(state, kind="osa_or_units")
            if series is None:
                raise ValueError("no series available for changepoint detection")
            result = detect_changepoints(
                series,
                dates=state.series_dates,
                series_name=f"{state.brand_id}_{state.city or 'all'}",
            )
            state.computation_results.append(result.to_dict())
        except Exception as e:
            state.errors.append(f"changepoint_failed: {e!r}")

    if Mode.ELASTICITY in state.modes_triggered:
        try:
            from analyst.data import get_store

            store = get_store()
            panel = store.forecast_input(
                sku_id=state.sku_id or _require(state, "sku_id"),
                city=state.city or _require(state, "city"),
            )
            result = estimate_elasticity(
                panel,
                brand_id=state.brand_id,
                city=state.city or "",
            )
            state.computation_results.append(result.to_dict())
        except Exception as e:
            state.errors.append(f"elasticity_failed: {e!r}")

    return state


def _series_from_state(state: AnalystState, kind: str) -> list[float] | None:
    """When no explicit series is supplied, try to pull one from the mock
    store using brand_id / city context. Deterministic fallback so DENOISE /
    ANOMALY / CHANGEPOINT are usable against canned data in CI."""
    from analyst.data import get_store

    if not state.city or not state.sku_id:
        return None
    store = get_store()
    df = store.forecast_input(sku_id=state.sku_id, city=state.city)
    if df.empty:
        return None
    daily = (
        df.groupby("snapshot_date")
        .agg(units=("units_consumed", "sum"), osa=("is_available", "mean"))
        .reset_index()
        .sort_values("snapshot_date")
    )
    state.series_dates = [str(d) for d in daily["snapshot_date"].tolist()]
    if kind == "deltas":
        return daily["units"].diff().fillna(0).tolist()
    return daily["units"].tolist()


def _require(state: AnalystState, field: str) -> str:
    raise ValueError(f"Missing required field for computation: {field}")


def _default_period(lookback_end: int, lookback_start: int | None = None) -> str:
    end = TODAY - timedelta(days=lookback_end - 7)
    start = TODAY - timedelta(days=lookback_start or lookback_end)
    return f"{start}:{end}"
