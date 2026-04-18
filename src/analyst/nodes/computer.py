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

    return state


def _require(state: AnalystState, field: str) -> str:
    raise ValueError(f"Missing required field for computation: {field}")


def _default_period(lookback_end: int, lookback_start: int | None = None) -> str:
    end = TODAY - timedelta(days=lookback_end - 7)
    start = TODAY - timedelta(days=lookback_start or lookback_end)
    return f"{start}:{end}"
