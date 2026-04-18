"""Node 4 — LLM Narrator.

Reads computation_results and emits one EstimateObject per result. The LLM
never invents numbers — it only restates what the tool produced. When no
ANTHROPIC_API_KEY is present, falls back to a deterministic template narrator
so the agent remains runnable offline and in CI.
"""

from __future__ import annotations

import json
import os
from typing import Any

from analyst.schemas import AnalystState, EstimateObject

MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are the Hector Analyst Agent's narrator.
You are given a structured computation result from a statistical tool. Your job
is to produce a short, grounded estimate object. You MUST NOT invent any
numbers. Every ₹ figure, percentage, or date in your narrative must appear in
the input object. If a figure is missing, say so rather than estimating it.

Return ONLY valid JSON with these keys:
headline, evidence, estimate, confidence, action, methodology.

Keep each field to one or two short sentences. Currency is INR (₹).
"""


def generate_narrative(state: AnalystState) -> AnalystState:
    for result in state.computation_results:
        estimate = _narrate_one(result)
        state.estimate_objects.append(estimate)
    return state


def _narrate_one(result: dict[str, Any]) -> EstimateObject:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        try:
            return _narrate_with_llm(result, api_key)
        except Exception:
            pass  # fall through to template
    return _narrate_with_template(result)


def _narrate_with_llm(result: dict[str, Any], api_key: str) -> EstimateObject:
    import anthropic  # type: ignore

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=MODEL,
        max_tokens=600,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": json.dumps(result, default=str)}],
    )
    text = message.content[0].text  # type: ignore[attr-defined]
    payload = json.loads(text)
    return EstimateObject(
        headline=payload.get("headline", ""),
        evidence=payload.get("evidence", ""),
        estimate=payload.get("estimate", ""),
        confidence=payload.get("confidence", result.get("confidence", "unknown")),
        action=payload.get("action", ""),
        methodology=payload.get("methodology", ""),
        source_tool=result["tool"],
    )


def _narrate_with_template(result: dict[str, Any]) -> EstimateObject:
    tool = result["tool"]
    if tool == "compute_forecast":
        return _tpl_forecast(result)
    if tool == "compute_counterfactual":
        return _tpl_counterfactual(result)
    if tool == "compute_attribution":
        return _tpl_attribution(result)
    return EstimateObject(
        headline="Estimate computed",
        evidence=json.dumps(result, default=str)[:400],
        estimate="(unknown tool)",
        confidence=result.get("confidence", "unknown"),
        action="Review result manually.",
        methodology="(unknown)",
        source_tool=tool,
    )


def _rs(v: float) -> str:
    if v >= 10_00_000:
        return f"₹{v / 1_00_000:.1f}L"
    if v >= 1_000:
        return f"₹{v / 1_000:.1f}K"
    return f"₹{v:.0f}"


def _tpl_forecast(r: dict[str, Any]) -> EstimateObject:
    return EstimateObject(
        headline=(
            f"{r['sku_id']} in {r['city']} will likely stock out in "
            f"{r['days_to_oos_point_estimate']:.0f} days."
        ),
        evidence=(
            f"{r['stores_at_risk']} of {r['total_stores']} dark stores at risk. "
            f"P(OOS within 7d)={r['probability_oos_within_7_days']:.0%}, "
            f"P(OOS within 14d)={r['probability_oos_within_14_days']:.0%}. "
            f"Range: {r['days_to_oos_range'][0]}–{r['days_to_oos_range'][1]} days."
        ),
        estimate=f"Projected forward PSL if OOS holds: {_rs(r['forward_psl_rs'])}.",
        confidence=r["confidence"].capitalize(),
        action=(
            "Raise PO on warehouse and prioritise stock transfers to at-risk dark stores."
            if r["stores_at_risk"] > 0
            else "Maintain current replenishment cadence."
        ),
        methodology=(
            "Per-store demand = median(units on available days). "
            "days_to_oos = FE_stock / demand. Survival: P(OOS by t)=1-exp(-λt), "
            "λ = 1/mean(days_to_oos) scaled by city importance."
        ),
        source_tool="compute_forecast",
    )


def _tpl_counterfactual(r: dict[str, Any]) -> EstimateObject:
    scenario_labels = {
        "missed_attack_window": "Missed competitor attack window",
        "cost_of_delay": "Cost of delayed action",
        "what_if_osa": "What-if OSA simulation",
    }
    return EstimateObject(
        headline=(
            f"{scenario_labels.get(r['scenario'], r['scenario'])}: "
            f"{_rs(r['delta_rs'])} delta for {r['brand_id']} {r['sku_id']} in {r['city']}."
        ),
        evidence=(
            f"Window {r['window_start']} → {r['window_end']}. "
            f"Actual: {_rs(r['actual_revenue_rs'])}; "
            f"Counterfactual: {_rs(r['counterfactual_revenue_rs'])}. "
            f"Driver: {r['key_driver']}."
        ),
        estimate=f"Estimated delta: {_rs(r['delta_rs'])} ({r['delta_pct']:.0%}).",
        confidence=r["confidence"].capitalize(),
        action=(
            "Train ops to act on Hector alerts on detection day, not recovery day."
            if r["scenario"] == "cost_of_delay"
            else "Launch targeted ad push and stock transfer next time this pattern fires."
        ),
        methodology=(
            "Counterfactual revenue = baseline demand × window days × "
            "historical uplift prior, vs observed units × price."
        ),
        source_tool="compute_counterfactual",
    )


def _tpl_attribution(r: dict[str, Any]) -> EstimateObject:
    attr = r["attribution"]
    buckets = ", ".join(
        f"{k} {attr[k]['share_pct']:.0%} ({_rs(attr[k]['delta_rs'])})"
        for k in ("availability", "visibility", "pricing", "competitor")
    )
    return EstimateObject(
        headline=(
            f"Revenue moved {_rs(r['total_delta_rs'])} period-over-period. "
            f"Dominant driver: {r['dominant_cause']}."
        ),
        evidence=f"Decomposition — {buckets}.",
        estimate=(
            f"Competitor-attributed share: {_rs(r['competitor_caused_rs'])} "
            f"({attr['competitor']['share_pct']:.0%})."
        ),
        confidence=r["confidence"].capitalize(),
        action=(
            f"Focus recovery effort on {r['dominant_cause']} — it owns the largest "
            "share of the move."
        ),
        methodology=(
            "Partial attribution: hold each driver at prior-period level, "
            "recompute revenue under a multiplicative response model, normalise "
            "deltas to sum to total change."
        ),
        source_tool="compute_attribution",
    )
