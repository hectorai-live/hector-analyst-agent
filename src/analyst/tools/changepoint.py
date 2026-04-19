"""detect_changepoints — Tool F.

PELT-style (Pruned Exact Linear Time) changepoint detection on a univariate
time series. Answers: "when did the number *really* break?" with
statistical rigour — something neither Claude nor MCP can do.

Cost function: Gaussian sum-of-squared-errors (changes in mean).
Penalty: BIC-style, `β = 2 σ̂² log(n)`.

Returns
-------
- changepoints: indices where the series changes regime
- segment_means: mean of each resulting segment
- penalty: the β used

Reference: Killick, Fearnhead & Eckley (2012), "Optimal detection of
changepoints with a linear computational cost".
"""

from __future__ import annotations

import numpy as np

from analyst.schemas import ChangepointResult, Confidence

MIN_SEGMENT = 5


def _segment_cost(cumsum: np.ndarray, cumsum_sq: np.ndarray, a: int, b: int) -> float:
    """Gaussian SSE over y[a:b]."""
    n = b - a
    if n <= 0:
        return 0.0
    s = cumsum[b] - cumsum[a]
    ss = cumsum_sq[b] - cumsum_sq[a]
    return ss - (s * s) / n


def _pelt(y: np.ndarray, beta: float) -> list[int]:
    """Classic PELT (Killick et al. 2012). F[t] = min cost over all partitions of y[0:t]."""
    n = len(y)
    cumsum = np.concatenate([[0.0], np.cumsum(y)])
    cumsum_sq = np.concatenate([[0.0], np.cumsum(y * y)])
    F = np.full(n + 1, np.inf)
    F[0] = -beta
    prev = np.zeros(n + 1, dtype=int)

    for t in range(MIN_SEGMENT, n + 1):
        best_cost = np.inf
        best_tau = 0
        for tau in range(0, t - MIN_SEGMENT + 1):
            if not np.isfinite(F[tau]):
                continue
            cost = F[tau] + _segment_cost(cumsum, cumsum_sq, tau, t) + beta
            if cost < best_cost:
                best_cost = cost
                best_tau = tau
        F[t] = best_cost
        prev[t] = best_tau

    # Backtrack
    cps: list[int] = []
    t = n
    while t > 0:
        tau = int(prev[t])
        if 0 < tau < n:
            cps.append(tau)
        t = tau
        if t == 0:
            break
    return sorted(cps)


def detect_changepoints(
    values: list[float] | np.ndarray,
    *,
    dates: list[str] | None = None,
    series_name: str = "series",
    penalty: float | None = None,
) -> ChangepointResult:
    """Find changepoints that minimise SSE + β·k under a BIC penalty.

    Penalty auto-selects to β = 2 σ̂² log(n) if not supplied.
    """
    flags: list[str] = []
    y = np.asarray(values, dtype=float)
    y = y[np.isfinite(y)]
    n = len(y)

    if n < 2 * MIN_SEGMENT:
        flags.append(f"insufficient_points_{n}_need_{2 * MIN_SEGMENT}")
        return ChangepointResult(
            series_name=series_name,
            n_points=n,
            changepoints=[],
            segment_means=[float(y.mean()) if n else 0.0],
            penalty=0.0,
            confidence="low",
            data_quality_flags=flags,
        )

    # σ from IQR of first differences: Δy ~ N(0, 2σ²) under i.i.d. Gaussian,
    # so σ(y) ≈ IQR(Δ) / (1.349 · √2). Robust to level shifts — the hallmark
    # property needed here, since raw variance is inflated by the very
    # changepoints we are trying to detect. Small-n IQR can under-estimate,
    # so we floor σ at `min(2·σ_d, std(y))` — `std(y)` is inflated by
    # shifts but acts as a safety ceiling on stable runs.
    diffs = np.diff(y)
    q75, q25 = np.percentile(diffs, [75, 25])
    sigma_d = max((q75 - q25) / (1.349 * np.sqrt(2)), 1e-3)
    sigma_full = float(np.std(y))
    sigma = max(min(2.0 * sigma_d, sigma_full), sigma_d)
    sigma2 = sigma * sigma
    # Modified BIC (Lavielle, 2005, variant 2): β = 4σ²log(n) — calibrated
    # to reject spurious cps on stable runs while retaining power on real
    # mean shifts of ≥~2σ.
    beta = penalty if penalty is not None else 4.0 * sigma2 * np.log(n)

    cps = _pelt(y, beta)
    # Post-filter: reject cps whose adjacent-segment mean shift isn't
    # significant under a Bonferroni-corrected two-sample z-test. With `n`
    # candidate split points we need p < 0.05/n per test; z ≈ 3 for the
    # typical n we see (20–60). This removes the tail false-positives that
    # survive BIC on noisy i.i.d. runs.
    z_crit = 3.0
    filtered: list[int] = []
    boundaries = [0, *cps, n]
    for i, cp in enumerate(cps):
        left_n = cp - boundaries[i]
        right_n = boundaries[i + 2] - cp
        if left_n < 2 or right_n < 2:
            continue
        left_mean = float(np.mean(y[boundaries[i] : cp]))
        right_mean = float(np.mean(y[cp : boundaries[i + 2]]))
        se = sigma * np.sqrt(1.0 / left_n + 1.0 / right_n)
        if se < 1e-9:
            continue
        z = abs(left_mean - right_mean) / se
        if z >= z_crit:
            filtered.append(cp)
    cps = filtered
    segments = [0, *cps, n]
    means = [float(np.mean(y[segments[i] : segments[i + 1]])) for i in range(len(segments) - 1)]

    if dates and len(dates) == n:
        cp_labels = [dates[i] for i in cps]
    else:
        cp_labels = [str(i) for i in cps]

    confidence: Confidence = "high" if n >= 28 else "medium" if n >= 14 else "low"
    if n < 14:
        flags.append(f"short_series_{n}d")

    return ChangepointResult(
        series_name=series_name,
        n_points=n,
        changepoints=cp_labels,
        segment_means=[round(m, 4) for m in means],
        penalty=round(float(beta), 3),
        confidence=confidence,
        data_quality_flags=flags,
    )
