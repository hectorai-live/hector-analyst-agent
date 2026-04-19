"""detect_anomalies — Tool E.

Flags statistically-significant departures in a daily time series (OSA, SOV,
units, revenue). Neither Claude nor raw MCP can do this — MCP returns a
series, Claude can eyeball a direction. This tool decomposes trend and weekly
seasonality, then runs a Median-Absolute-Deviation z-score on the residuals,
plus a two-sided CUSUM for drift detection.

Outputs
-------
- events: list of AnomalyEvent (date, value, expected, residual, |z|, direction,
  label). Label is one of:
    * attack_window_start  — large drop in OSA/SOV
    * stealth_oos          — sustained drop in units/revenue without OSA drop
    * promo_spike          — large positive spike
    * drift                — CUSUM drift event
- cusum_shift: most-recent one-sided CUSUM statistic

Method
------
1. STL-lite decomposition:
   - Trend  = rolling median (window = min(7, n // 3))
   - Season = weekly median of detrended values (mod 7), re-tiled
   - Residual = y - trend - season
2. Robust z-score: z_i = 0.6745 × (r_i − median(r)) / MAD(r); threshold |z| > 3.5
   gives ~α = 0.001 false-positive rate under normality.
3. CUSUM: Page's scheme S⁺_t = max(0, S⁺_{t-1} + (r_t − k)), with
   k = 0.5 σ̂_r, h = 5 σ̂_r.
4. Label heuristics:
   * OSA/SOV series with a big drop → attack_window_start
   * Units/revenue series with a big drop *without* corresponding OSA drop
     (when provided) → stealth_oos
   * Big positive spike → promo_spike
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from analyst.schemas import AnomalyEvent, AnomalyReport, Confidence

DEFAULT_Z_THRESHOLD = 3.5  # two-sided ~ α = 0.0009
SeriesKind = Literal["osa", "sov", "units", "revenue", "other"]


def _stl_lite(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(y)
    w = max(3, min(7, n // 3))
    # Rolling median trend
    trend = np.array(
        [np.median(y[max(0, i - w // 2) : min(n, i + w // 2 + 1)]) for i in range(n)]
    )
    detrended = y - trend
    season = np.zeros(n)
    for phase in range(7):
        idx = np.arange(phase, n, 7)
        if len(idx) >= 2:
            season[idx] = np.median(detrended[idx])
    residual = y - trend - season
    return trend, season, residual


def _robust_z(r: np.ndarray) -> np.ndarray:
    med = float(np.median(r))
    mad = float(np.median(np.abs(r - med)))
    if mad < 1e-9:
        return np.zeros_like(r)
    return 0.6745 * (r - med) / mad


def _cusum(r: np.ndarray) -> tuple[float, np.ndarray]:
    sigma = max(float(np.std(r)), 1e-6)
    k = 0.5 * sigma
    s_plus = np.zeros_like(r)
    s_minus = np.zeros_like(r)
    for i in range(1, len(r)):
        s_plus[i] = max(0.0, s_plus[i - 1] + (r[i] - k))
        s_minus[i] = min(0.0, s_minus[i - 1] + (r[i] + k))
    current_shift = float(s_plus[-1] + abs(s_minus[-1]))
    drift_flags = np.where(np.abs(s_plus) + np.abs(s_minus) > 5 * sigma)[0]
    return current_shift, drift_flags


def _label(
    kind: SeriesKind, direction: str, osa_context_drop: bool | None
) -> str:
    if direction == "drop" and kind in ("osa", "sov"):
        return "attack_window_start"
    if direction == "drop" and kind in ("units", "revenue"):
        if osa_context_drop is False:
            return "stealth_oos"
        return "availability_drop"
    if direction == "spike":
        return "promo_spike"
    return "drift"


def detect_anomalies(
    values: list[float] | np.ndarray,
    *,
    dates: list[str] | None = None,
    series_name: str = "series",
    kind: SeriesKind = "other",
    z_threshold: float = DEFAULT_Z_THRESHOLD,
    osa_also_dropped: bool | None = None,
) -> AnomalyReport:
    """Detect anomalies in a daily series.

    Parameters
    ----------
    values
        Daily observations. Must be strictly chronological.
    dates
        Optional ISO-date strings matching values. If omitted, events carry
        integer indices as dates.
    kind
        Domain hint for labelling: osa / sov / units / revenue / other.
    osa_also_dropped
        For units/revenue series, whether OSA also dropped. If False (OSA was
        fine), negative anomalies get labelled "stealth_oos".
    """
    flags: list[str] = []
    y = np.asarray(values, dtype=float)
    y = y[np.isfinite(y)]
    n = len(y)

    if n < 7:
        flags.append(f"insufficient_points_{n}_need_7")
        return AnomalyReport(
            series_name=series_name,
            n_points=n,
            alpha=0.0,
            events=[],
            cusum_shift=0.0,
            confidence="low",
            data_quality_flags=flags,
        )

    _, _, residual = _stl_lite(y)
    z = _robust_z(residual)
    cusum_shift, drift_idx = _cusum(residual)

    events: list[AnomalyEvent] = []
    for i in range(n):
        if abs(z[i]) < z_threshold and i not in drift_idx:
            continue
        direction = "spike" if residual[i] > 0 else "drop"
        label = (
            "drift"
            if abs(z[i]) < z_threshold
            else _label(kind, direction, osa_also_dropped)
        )
        events.append(
            AnomalyEvent(
                date=(dates[i] if dates and i < len(dates) else str(i)),
                value=float(y[i]),
                expected=float(y[i] - residual[i]),
                residual=float(residual[i]),
                z_score=round(float(z[i]), 3),
                direction=direction,  # type: ignore[arg-type]
                label=label,
            )
        )

    if n < 14:
        flags.append(f"short_series_{n}d")
    if np.std(y) < 1e-6:
        flags.append("degenerate_series_no_variance")

    confidence: Confidence = "high" if n >= 28 and not flags else "medium" if n >= 14 else "low"

    return AnomalyReport(
        series_name=series_name,
        n_points=n,
        alpha=round(2 * (1 - _cdf_abs(z_threshold)), 4),
        events=events,
        cusum_shift=round(cusum_shift, 3),
        confidence=confidence,
        data_quality_flags=flags,
    )


def _cdf_abs(z: float) -> float:
    """Standard normal CDF of |z|. Numerically stable approximation."""
    from math import erf, sqrt

    return 0.5 * (1 + erf(z / sqrt(2)))
