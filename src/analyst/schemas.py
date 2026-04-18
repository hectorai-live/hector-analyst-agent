"""Shared schemas for the Analyst Agent.

Every tool returns a structured result with a confidence grade and data-quality
flags. The LLM narrator never invents numbers — it only explains what these
objects contain. That invariant is the basis of the system's credibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


class Mode(str, Enum):
    FORECAST = "forecast"
    COUNTERFACTUAL = "counterfactual"
    CAUSAL = "causal"


Confidence = Literal["high", "medium", "low"]


@dataclass
class ForecastResult:
    sku_id: str
    city: str
    days_to_oos_point_estimate: float
    days_to_oos_range: tuple[float, float]
    probability_oos_within_7_days: float
    probability_oos_within_14_days: float
    forward_psl_rs: float
    stores_at_risk: int
    total_stores: int
    confidence: Confidence
    data_quality_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": "compute_forecast",
            "sku_id": self.sku_id,
            "city": self.city,
            "days_to_oos_point_estimate": self.days_to_oos_point_estimate,
            "days_to_oos_range": list(self.days_to_oos_range),
            "probability_oos_within_7_days": self.probability_oos_within_7_days,
            "probability_oos_within_14_days": self.probability_oos_within_14_days,
            "forward_psl_rs": self.forward_psl_rs,
            "stores_at_risk": self.stores_at_risk,
            "total_stores": self.total_stores,
            "confidence": self.confidence,
            "data_quality_flags": self.data_quality_flags,
        }


@dataclass
class CounterfactualResult:
    brand_id: str
    sku_id: str
    city: str
    scenario: Literal["missed_attack_window", "cost_of_delay", "what_if_osa"]
    window_start: str
    window_end: str
    actual_revenue_rs: float
    counterfactual_revenue_rs: float
    delta_rs: float
    delta_pct: float
    key_driver: str
    confidence: Confidence
    data_quality_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": "compute_counterfactual",
            "brand_id": self.brand_id,
            "sku_id": self.sku_id,
            "city": self.city,
            "scenario": self.scenario,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "actual_revenue_rs": self.actual_revenue_rs,
            "counterfactual_revenue_rs": self.counterfactual_revenue_rs,
            "delta_rs": self.delta_rs,
            "delta_pct": self.delta_pct,
            "key_driver": self.key_driver,
            "confidence": self.confidence,
            "data_quality_flags": self.data_quality_flags,
        }


@dataclass
class AttributionBucket:
    delta_rs: float
    share_pct: float

    def to_dict(self) -> dict[str, float]:
        return {"delta_rs": self.delta_rs, "share_pct": self.share_pct}


@dataclass
class AttributionResult:
    brand_id: str
    category: str
    city: str
    current_period: str
    comparison_period: str
    total_delta_rs: float
    availability: AttributionBucket
    visibility: AttributionBucket
    pricing: AttributionBucket
    competitor: AttributionBucket
    dominant_cause: str
    competitor_caused_rs: float
    confidence: Confidence
    data_quality_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": "compute_attribution",
            "brand_id": self.brand_id,
            "category": self.category,
            "city": self.city,
            "current_period": self.current_period,
            "comparison_period": self.comparison_period,
            "total_delta_rs": self.total_delta_rs,
            "attribution": {
                "availability": self.availability.to_dict(),
                "visibility": self.visibility.to_dict(),
                "pricing": self.pricing.to_dict(),
                "competitor": self.competitor.to_dict(),
            },
            "dominant_cause": self.dominant_cause,
            "competitor_caused_rs": self.competitor_caused_rs,
            "confidence": self.confidence,
            "data_quality_flags": self.data_quality_flags,
        }


@dataclass
class EstimateObject:
    """The final narrated output surfaced to brands."""

    headline: str
    evidence: str
    estimate: str
    confidence: str
    action: str
    methodology: str
    source_tool: str

    def to_dict(self) -> dict[str, str]:
        return {
            "headline": self.headline,
            "evidence": self.evidence,
            "estimate": self.estimate,
            "confidence": self.confidence,
            "action": self.action,
            "methodology": self.methodology,
            "source_tool": self.source_tool,
        }


@dataclass
class AnalystState:
    """Shared state object passed through the LangGraph nodes."""

    query: str
    brand_id: str
    sku_id: str | None = None
    category: str | None = None
    city: str | None = None
    period: str | None = None
    comparison_period: str | None = None
    scenario: str | None = None
    window_start: str | None = None
    window_end: str | None = None
    target_osa: float | None = None

    modes_triggered: list[Mode] = field(default_factory=list)
    raw_context: dict[str, Any] = field(default_factory=dict)
    computation_results: list[dict[str, Any]] = field(default_factory=list)
    estimate_objects: list[EstimateObject] = field(default_factory=list)
    data_quality: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "brand_id": self.brand_id,
            "sku_id": self.sku_id,
            "category": self.category,
            "city": self.city,
            "period": self.period,
            "modes_triggered": [m.value for m in self.modes_triggered],
            "computation_results": self.computation_results,
            "estimate_objects": [e.to_dict() for e in self.estimate_objects],
            "data_quality": self.data_quality,
            "errors": self.errors,
        }
