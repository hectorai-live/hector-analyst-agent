"""Node 2 — Data Retrieval.

In production this node calls the existing MCP tools (get_inventory_snapshot,
get_competitor_visibility, get_sales_estimate, etc.). The contract is the same:
populate state.raw_context with whatever downstream computation needs.
"""

from __future__ import annotations

from analyst.data import get_store
from analyst.schemas import AnalystState, Mode


def retrieve_data(state: AnalystState) -> AnalystState:
    store = get_store()
    state.raw_context["as_of"] = str(store.as_of)
    state.raw_context["history_days"] = store.history_days

    # Data quality summary at state level (also attached per-tool)
    state.data_quality = {
        "history_days": store.history_days,
        "as_of": str(store.as_of),
        "psl_ready": store.history_days >= 7,
    }

    # Record which slices each triggered mode expects. The actual pull happens
    # inside each tool (they are self-contained) — this node validates the ask.
    needs: list[str] = []
    if Mode.FORECAST in state.modes_triggered:
        needs.append("forecast_input")
    if Mode.COUNTERFACTUAL in state.modes_triggered:
        needs.append("counterfactual_input")
    if Mode.CAUSAL in state.modes_triggered:
        needs.append("attribution_input")
    state.raw_context["views_required"] = needs

    return state
