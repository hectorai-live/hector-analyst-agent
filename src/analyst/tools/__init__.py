"""Statistical computation tools — the three models behind the Analyst Agent.

Each tool is independently callable (agent-free) so it can be unit-tested and
invoked from other agents (Fixer, Finder) as the brief describes.
"""

from analyst.tools.anomaly import detect_anomalies
from analyst.tools.attribution import compute_attribution
from analyst.tools.changepoint import detect_changepoints
from analyst.tools.counterfactual import compute_counterfactual
from analyst.tools.denoiser import denoise_signal
from analyst.tools.elasticity import estimate_elasticity
from analyst.tools.forecast import compute_forecast

__all__ = [
    "compute_attribution",
    "compute_counterfactual",
    "compute_forecast",
    "denoise_signal",
    "detect_anomalies",
    "detect_changepoints",
    "estimate_elasticity",
]
