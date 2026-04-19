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


_TOOL_SCHEMA = {
    "name": "emit_estimate",
    "description": (
        "Emit the final grounded estimate object. Every field must be "
        "derived from the input result — do not invent numbers or dates."
    ),
    "input_schema": {
        "type": "object",
        "required": [
            "headline",
            "evidence",
            "estimate",
            "confidence",
            "action",
            "methodology",
        ],
        "properties": {
            "headline": {"type": "string"},
            "evidence": {"type": "string"},
            "estimate": {"type": "string"},
            "confidence": {"type": "string"},
            "action": {"type": "string"},
            "methodology": {"type": "string"},
        },
    },
}


def _narrate_with_llm(result: dict[str, Any], api_key: str) -> EstimateObject:
    """Use Anthropic tool-use for robust structured output.

    Tool-use guarantees the model returns a well-typed payload instead of
    JSON-in-text which can be partial or malformed.
    """
    import anthropic  # type: ignore

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=MODEL,
        max_tokens=800,
        system=SYSTEM_PROMPT,
        tools=[_TOOL_SCHEMA],
        tool_choice={"type": "tool", "name": "emit_estimate"},
        messages=[{"role": "user", "content": json.dumps(result, default=str)}],
    )
    payload: dict[str, Any] | None = None
    for block in message.content:  # type: ignore[attr-defined]
        if getattr(block, "type", None) == "tool_use":
            payload = block.input  # type: ignore[assignment]
            break
    if payload is None:
        raise RuntimeError("narrator: tool_use block missing from response")
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
    if tool == "denoise_signal":
        return _tpl_denoise(result)
    if tool == "detect_anomalies":
        return _tpl_anomaly(result)
    if tool == "detect_changepoints":
        return _tpl_changepoint(result)
    if tool == "estimate_elasticity":
        return _tpl_elasticity(result)
    return EstimateObject(
        headline="Estimate computed",
        evidence=json.dumps(result, default=str)[:400],
        estimate="(unknown tool)",
        confidence=result.get("confidence", "unknown"),
        action="Review result manually.",
        methodology="(unknown)",
        source_tool=tool,
    )


def _tpl_denoise(r: dict[str, Any]) -> EstimateObject:
    return EstimateObject(
        headline=(
            f"Signal denoised: {r['restock_events']} restock events removed "
            f"from {r['n_points']}-point series."
        ),
        evidence=(
            f"Raw mean Δ={r['raw_mean']:.2f}; denoised demand mean={r['denoised_mean']:.2f}. "
            f"Restock mass {r['restock_mass']:.0f} ({r['noise_share_pct']:.1f}% of total). "
            f"SNR {r['signal_to_noise_db']:.1f} dB."
        ),
        estimate=f"True daily demand ≈ {r['denoised_mean']:.2f} units.",
        confidence=r["confidence"].capitalize(),
        action=(
            "Use denoised demand as input to forecast instead of the raw series."
        ),
        methodology=(
            "2-component Gaussian mixture fit by EM over day-over-day inventory "
            "deltas. Restock = component with higher mean. Hard MAP assignment."
        ),
        source_tool="denoise_signal",
    )


def _tpl_anomaly(r: dict[str, Any]) -> EstimateObject:
    events = r.get("events", [])
    by_label: dict[str, int] = {}
    for e in events:
        by_label[e["label"]] = by_label.get(e["label"], 0) + 1
    summary = (
        ", ".join(f"{k}×{v}" for k, v in by_label.items())
        if by_label
        else "no significant anomalies"
    )
    headline = (
        f"{len(events)} anomaly event(s) in {r['n_points']}-day window "
        f"(α={r['alpha']})."
    )
    most_recent = events[-1] if events else None
    estimate = (
        f"Most recent: {most_recent['date']} — {most_recent['label']} "
        f"(z={most_recent['z_score']:+.2f}, value={most_recent['value']:.2f})."
        if most_recent
        else "Series is within expected bounds."
    )
    return EstimateObject(
        headline=headline,
        evidence=f"Breakdown: {summary}. CUSUM drift statistic = {r['cusum_shift']:.2f}.",
        estimate=estimate,
        confidence=r["confidence"].capitalize(),
        action=(
            "Investigate flagged dates; cross-check with ad spend, competitor OSA, and PO activity."
            if events
            else "No action — series is stable."
        ),
        methodology=(
            "STL-lite (rolling-median trend + weekly seasonal) → robust Z on "
            "residuals (|z|>3.5) → Page's CUSUM for drift. Labels heuristic by series kind."
        ),
        source_tool="detect_anomalies",
    )


def _tpl_changepoint(r: dict[str, Any]) -> EstimateObject:
    cps = r.get("changepoints", [])
    means = r.get("segment_means", [])
    if cps:
        segments_str = " → ".join(f"{m:.2f}" for m in means)
        headline = f"{len(cps)} regime shift(s) detected."
        evidence = f"Changepoints at: {', '.join(cps)}. Segment means: {segments_str}."
        action = "Align ops review with changepoint dates to isolate root cause."
    else:
        headline = "No significant regime shift in window."
        evidence = f"Single-segment mean = {means[0]:.2f}." if means else "Series too short."
        action = "Series is stable — keep existing cadence."
    return EstimateObject(
        headline=headline,
        evidence=evidence,
        estimate=f"k={len(cps)} changepoints at BIC penalty β={r['penalty']}.",
        confidence=r["confidence"].capitalize(),
        action=action,
        methodology="PELT with Gaussian mean-shift cost and BIC penalty β = 2σ²log(n).",
        source_tool="detect_changepoints",
    )


def _tpl_elasticity(r: dict[str, Any]) -> EstimateObject:
    beta = r["elasticity"]
    low, high = r["elasticity_ci"]
    return EstimateObject(
        headline=(
            f"Price elasticity of demand = {beta:+.2f} "
            f"({r['interpretation']})."
        ),
        evidence=(
            f"95% CI [{low:+.2f}, {high:+.2f}]. R²={r['r_squared']:.2f} on "
            f"{r['n_observations']} obs. Price range ₹{r['price_range'][0]:.0f}–₹{r['price_range'][1]:.0f}."
        ),
        estimate=(
            f"A 10% price cut would shift units by ≈ {-beta * 10:+.1f}% "
            "(holding availability constant)."
        ),
        confidence=r["confidence"].capitalize(),
        action=(
            "Pricing lever is effective — test a limited discount."
            if r["interpretation"] == "elastic"
            else "Price cuts will not meaningfully grow volume; focus on availability/visibility."
        ),
        methodology=(
            "log-log OLS with store fixed effects via within-transform "
            "(de-meaning). 95% CI from analytic SE under i.i.d. residuals."
        ),
        source_tool="estimate_elasticity",
    )


def _rs(v: float) -> str:
    """Format INR with Indian lakh notation. Handles negatives cleanly."""
    sign = "-" if v < 0 else ""
    a = abs(v)
    if a >= 1_00_00_000:  # >= 1 crore
        return f"{sign}₹{a / 1_00_00_000:.2f}Cr"
    if a >= 10_00_000:  # >= 10 lakh
        return f"{sign}₹{a / 1_00_000:.1f}L"
    if a >= 1_000:
        return f"{sign}₹{a / 1_000:.1f}K"
    return f"{sign}₹{a:.0f}"


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
        f"{k} {attr[k]['share_pct']:.0%} ({_rs(float(attr[k]['delta_rs']))})"
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
