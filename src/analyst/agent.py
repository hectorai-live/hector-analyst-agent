"""LangGraph-style agent runner.

Uses a simple deterministic node chain (classify → retrieve → compute →
narrate). The interface matches the LangGraph StateGraph contract so the whole
thing can be swapped for `langgraph.graph.StateGraph` in production without
touching the nodes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from analyst.nodes import (
    classify_intent,
    generate_narrative,
    retrieve_data,
    run_computation,
)
from analyst.schemas import AnalystState

NodeFn = Callable[[AnalystState], AnalystState]


class AnalystAgent:
    """Minimal graph runner: linear pipeline, deterministic order, observable state."""

    def __init__(self, nodes: list[tuple[str, NodeFn]] | None = None) -> None:
        self.nodes: list[tuple[str, NodeFn]] = nodes or [
            ("classify", classify_intent),
            ("retrieve", retrieve_data),
            ("compute", run_computation),
            ("narrate", generate_narrative),
        ]

    def invoke(self, state: AnalystState) -> AnalystState:
        for name, fn in self.nodes:
            try:
                state = fn(state)
            except Exception as e:
                state.errors.append(f"{name}_node_failed: {e!r}")
                break
        return state


def run(
    query: str,
    *,
    brand_id: str,
    sku_id: str | None = None,
    city: str | None = None,
    category: str | None = None,
    period: str | None = None,
    comparison_period: str | None = None,
    scenario: str | None = None,
    window_start: str | None = None,
    window_end: str | None = None,
    target_osa: float | None = None,
) -> dict[str, Any]:
    """Run a single turn of the Analyst Agent and return serialised state."""
    agent = AnalystAgent()
    state = AnalystState(
        query=query,
        brand_id=brand_id,
        sku_id=sku_id,
        city=city,
        category=category,
        period=period,
        comparison_period=comparison_period,
        scenario=scenario,
        window_start=window_start,
        window_end=window_end,
        target_osa=target_osa,
    )
    state = agent.invoke(state)
    return state.to_dict()
