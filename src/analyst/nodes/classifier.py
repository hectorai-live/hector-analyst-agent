"""Node 1 — Intent Classifier.

Deterministic. Not LLM. The deployment brief is explicit that intent routing
must be stable and auditable; a mis-routed intent silently changes the class
of estimate a brand receives.
"""

from __future__ import annotations

import re

from analyst.schemas import AnalystState, Mode

FORECAST_CUES = [
    r"\bwill\b", r"\bgoing to\b", r"\bget worse\b", r"\bdays until\b",
    r"\bstock ?out\b", r"\bforecast\b", r"\bpredict\b", r"\bupcoming\b",
    r"\bnext \d+ (days|weeks)\b", r"\bimminent\b",
]
COUNTERFACTUAL_CUES = [
    r"\bwould have\b", r"\bwhat if\b", r"\bmissed\b", r"\bcost of delay\b",
    r"\bif we had\b", r"\bcould have\b", r"\bshould have\b", r"\bleft on the table\b",
    r"\battack window\b", r"\bopportunity lost\b",
    r"\bwould (?:we|they|I) have\b", r"\bcould (?:we|they|I) have\b",
    r"\bcost us\b", r"\bcould .* saved\b", r"\brevenue lost\b",
]
CAUSAL_CUES = [
    r"\bwhy\b", r"\bwhat caused\b", r"\battribution\b", r"\bdecompose\b",
    r"\bdriver\b", r"\bdue to\b", r"\bbreak ?down\b", r"\bresponsible for\b",
]
DENOISE_CUES = [
    r"\bdenoise\b", r"\bclean (?:the )?signal\b", r"\bfilter (?:out )?restocks?\b",
    r"\brestock noise\b", r"\btrue demand\b", r"\bhmm\b",
]
ANOMALY_CUES = [
    r"\banomal(?:y|ies|ous)\b", r"\bspike[s]?\b", r"\bdrop ?off\b",
    r"\battack window\b", r"\bstealth oos\b", r"\bpromo spike\b",
    r"\bunusual\b", r"\boutlier[s]?\b",
]
CHANGEPOINT_CUES = [
    r"\bchange ?point[s]?\b", r"\bregime change\b", r"\bwhen did .* (?:break|shift|change)\b",
    r"\bstructural break\b", r"\binflection\b",
]
ELASTICITY_CUES = [
    r"\belasticity\b", r"\bprice sensitiv(?:e|ity)\b",
    r"\bif we (?:drop|raise|cut) price\b", r"\bprice response\b",
]


def classify_intent(state: AnalystState) -> AnalystState:
    q = state.query.lower()
    triggered: list[Mode] = []

    if any(re.search(p, q) for p in FORECAST_CUES):
        triggered.append(Mode.FORECAST)
    if any(re.search(p, q) for p in COUNTERFACTUAL_CUES):
        triggered.append(Mode.COUNTERFACTUAL)
    if any(re.search(p, q) for p in CAUSAL_CUES):
        triggered.append(Mode.CAUSAL)
    if any(re.search(p, q) for p in DENOISE_CUES):
        triggered.append(Mode.DENOISE)
    if any(re.search(p, q) for p in ANOMALY_CUES):
        triggered.append(Mode.ANOMALY)
    if any(re.search(p, q) for p in CHANGEPOINT_CUES):
        triggered.append(Mode.CHANGEPOINT)
    if any(re.search(p, q) for p in ELASTICITY_CUES):
        triggered.append(Mode.ELASTICITY)

    # Fallback: if query is bare (e.g. from a scheduled trigger) use forecast.
    if not triggered:
        triggered = [Mode.FORECAST]

    state.modes_triggered = triggered
    return state
