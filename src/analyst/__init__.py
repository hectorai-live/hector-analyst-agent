"""Hector Q-Comm Analyst Agent.

Three-mode statistical computation engine that turns MCP data into probabilistic
estimates: forecasting, counterfactual valuation, and causal attribution.
"""

from analyst.agent import AnalystAgent, run
from analyst.schemas import (
    AnalystState,
    AttributionResult,
    CounterfactualResult,
    EstimateObject,
    ForecastResult,
    Mode,
)

__all__ = [
    "AnalystAgent",
    "AnalystState",
    "AttributionResult",
    "CounterfactualResult",
    "EstimateObject",
    "ForecastResult",
    "Mode",
    "run",
]

__version__ = "0.1.0"
