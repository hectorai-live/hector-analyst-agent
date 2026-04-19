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
    DENOISE = "denoise"
    ANOMALY = "anomaly"
    CHANGEPOINT = "changepoint"
    ELASTICITY = "elasticity"


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
class DenoisedSeriesResult:
    """HMM / robust-filter output that separates restock noise from true demand."""

    series_name: str
    n_points: int
    raw_mean: float
    denoised_mean: float
    restock_events: int
    restock_mass: float  # sum of magnitudes attributed to restock state
    noise_share_pct: float  # restock mass / raw mass
    signal_to_noise_db: float
    confidence: Confidence
    data_quality_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": "denoise_signal",
            "series_name": self.series_name,
            "n_points": self.n_points,
            "raw_mean": self.raw_mean,
            "denoised_mean": self.denoised_mean,
            "restock_events": self.restock_events,
            "restock_mass": self.restock_mass,
            "noise_share_pct": self.noise_share_pct,
            "signal_to_noise_db": self.signal_to_noise_db,
            "confidence": self.confidence,
            "data_quality_flags": self.data_quality_flags,
        }


@dataclass
class AnomalyEvent:
    date: str
    value: float
    expected: float
    residual: float
    z_score: float
    direction: Literal["spike", "drop"]
    label: str  # e.g. "attack_window_start", "stealth_oos", "promo_spike"

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "value": self.value,
            "expected": self.expected,
            "residual": self.residual,
            "z_score": self.z_score,
            "direction": self.direction,
            "label": self.label,
        }


@dataclass
class AnomalyReport:
    series_name: str
    n_points: int
    alpha: float  # nominal false-positive rate
    events: list[AnomalyEvent]
    cusum_shift: float  # most recent CUSUM statistic
    confidence: Confidence
    data_quality_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": "detect_anomalies",
            "series_name": self.series_name,
            "n_points": self.n_points,
            "alpha": self.alpha,
            "events": [e.to_dict() for e in self.events],
            "cusum_shift": self.cusum_shift,
            "confidence": self.confidence,
            "data_quality_flags": self.data_quality_flags,
        }


@dataclass
class ChangepointResult:
    series_name: str
    n_points: int
    changepoints: list[str]  # ISO dates
    segment_means: list[float]
    penalty: float
    confidence: Confidence
    data_quality_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": "detect_changepoints",
            "series_name": self.series_name,
            "n_points": self.n_points,
            "changepoints": self.changepoints,
            "segment_means": self.segment_means,
            "penalty": self.penalty,
            "confidence": self.confidence,
            "data_quality_flags": self.data_quality_flags,
        }


@dataclass
class ElasticityResult:
    brand_id: str
    city: str
    n_observations: int
    elasticity: float  # log-log β
    elasticity_ci_low: float
    elasticity_ci_high: float
    r_squared: float
    price_range: tuple[float, float]
    interpretation: str  # "elastic", "inelastic", "unit_elastic"
    confidence: Confidence
    data_quality_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": "estimate_elasticity",
            "brand_id": self.brand_id,
            "city": self.city,
            "n_observations": self.n_observations,
            "elasticity": self.elasticity,
            "elasticity_ci": [self.elasticity_ci_low, self.elasticity_ci_high],
            "r_squared": self.r_squared,
            "price_range": list(self.price_range),
            "interpretation": self.interpretation,
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
    series_input: list[float] | None = None  # generic series for denoise/anomaly/changepoint
    series_dates: list[str] | None = None

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
