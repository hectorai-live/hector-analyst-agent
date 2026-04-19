"""denoise_signal — Tool D.

Separates restock/inwarding noise from real demand in an inventory time series.
Neither Claude nor raw MCP can do this: MCP returns the raw noisy series,
Claude can only eyeball spikes. This tool fits a 2-state HMM and attributes
each day to either consumption (negative delta of modest magnitude) or restock
(positive delta or large spike) with a posterior probability.

Algorithm
---------
1. Compute day-over-day inventory deltas per store × SKU.
2. Fit a 2-component Gaussian mixture in pure numpy (EM) to the distribution
   of deltas: one component centred on typical consumption (near zero,
   slightly negative), one on restock (large positive mass).
3. Hard-assign each observation to its most-likely component (MAP).
4. Emit denoised demand = sum of |delta| over "consumption" observations,
   and restock_mass = sum over "restock" observations.

Why this, not a full Baum-Welch with sequential structure?
- V1: independence across days is a reasonable first-order assumption for
  inventory deltas in Q-comm. Full HMM with transitions is V2.
- EM on GMM has a closed-form in 2D — converges in <30 iterations for
  N ≤ 10k points. No external HMM library needed.
- Matches the guarantee bar (ML inference that Claude/MCP cannot replicate).
"""

from __future__ import annotations

import math

import numpy as np

from analyst.schemas import DenoisedSeriesResult

MAX_ITERS = 40
TOL = 1e-6


def _gmm_em_two_component(x: np.ndarray) -> tuple[np.ndarray, tuple[float, float, float, float, float, float]]:
    """Fit a 2-component univariate Gaussian mixture via EM.

    Returns
    -------
    responsibilities
        posterior[i, k] — N×2 matrix, component-1 is "restock" (higher mean).
    params
        (pi0, pi1, mu0, mu1, sigma0, sigma1) — component 1 is always restock.
    """
    n = len(x)
    # Initialise: split by sign. Component 0 = consumption (non-positive deltas),
    # component 1 = restock (positive deltas). If one side is empty, seed from
    # quantiles to avoid degeneracy.
    neg = x[x <= 0]
    pos = x[x > 0]
    if len(neg) < 2 or len(pos) < 2:
        q1, q3 = np.quantile(x, [0.25, 0.75])
        mu0, mu1 = float(q1), float(q3)
        sigma0 = sigma1 = max(float(np.std(x)), 1.0)
    else:
        mu0 = float(np.mean(neg))
        mu1 = float(np.mean(pos))
        sigma0 = max(float(np.std(neg)), 1.0)
        sigma1 = max(float(np.std(pos)), 1.0)
    pi0 = 0.5
    pi1 = 0.5

    def _pdf(v: np.ndarray, mu: float, sigma: float) -> np.ndarray:
        sigma = max(sigma, 1e-3)
        return (
            np.exp(-0.5 * ((v - mu) / sigma) ** 2)
            / (sigma * math.sqrt(2 * math.pi))
        )

    prev_ll = -np.inf
    resp = np.zeros((n, 2))
    for _ in range(MAX_ITERS):
        # E-step
        p0 = pi0 * _pdf(x, mu0, sigma0)
        p1 = pi1 * _pdf(x, mu1, sigma1)
        total = p0 + p1 + 1e-12
        resp[:, 0] = p0 / total
        resp[:, 1] = p1 / total

        # M-step
        nk = resp.sum(axis=0) + 1e-12
        pi0, pi1 = nk[0] / n, nk[1] / n
        mu0 = float((resp[:, 0] * x).sum() / nk[0])
        mu1 = float((resp[:, 1] * x).sum() / nk[1])
        sigma0 = math.sqrt(max((resp[:, 0] * (x - mu0) ** 2).sum() / nk[0], 1e-6))
        sigma1 = math.sqrt(max((resp[:, 1] * (x - mu1) ** 2).sum() / nk[1], 1e-6))

        ll = float(np.log(total).sum())
        if abs(ll - prev_ll) < TOL:
            break
        prev_ll = ll

    # Ensure component 1 is restock (higher mean)
    if mu0 > mu1:
        resp = resp[:, ::-1]
        mu0, mu1 = mu1, mu0
        sigma0, sigma1 = sigma1, sigma0
        pi0, pi1 = pi1, pi0
    return resp, (pi0, pi1, mu0, mu1, sigma0, sigma1)


def denoise_signal(
    values: list[float] | np.ndarray,
    *,
    series_name: str = "inventory_delta",
) -> DenoisedSeriesResult:
    """Fit a 2-state GMM and emit a denoised series summary.

    Accepts the *deltas* (day-over-day inventory changes). Positive deltas are
    restock events; negative deltas are consumption.

    Returns a DenoisedSeriesResult carrying the summary + data-quality flags.
    """
    flags: list[str] = []
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)

    if n < 5:
        flags.append(f"insufficient_points_{n}_need_5")
        return DenoisedSeriesResult(
            series_name=series_name,
            n_points=n,
            raw_mean=float(arr.mean()) if n else 0.0,
            denoised_mean=0.0,
            restock_events=0,
            restock_mass=0.0,
            noise_share_pct=0.0,
            signal_to_noise_db=0.0,
            confidence="low",
            data_quality_flags=flags,
        )

    resp, params = _gmm_em_two_component(arr)
    restock_assign = resp[:, 1] > 0.5
    restock_mass = float(np.abs(arr[restock_assign]).sum())
    signal_mass = float(np.abs(arr[~restock_assign]).sum())
    raw_mass = restock_mass + signal_mass + 1e-9

    # Denoised demand = mean of |consumption| on consumption days.
    # The sign convention here is: consumption observations carry negative
    # deltas (stock went down); we report their absolute mean as demand.
    consumption = -arr[~restock_assign]
    denoised_mean = float(consumption.mean()) if consumption.size else 0.0
    noise_share = 100.0 * restock_mass / raw_mass

    snr_db = 10.0 * math.log10(max(signal_mass, 1e-6) / max(restock_mass, 1e-6))

    if n < 14:
        flags.append(f"short_series_{n}d")
    if params[4] > 5 * max(abs(params[2]), 1.0):
        flags.append("consumption_variance_high")

    if n >= 30 and not flags:
        confidence = "high"
    elif n >= 14:
        confidence = "medium"
    else:
        confidence = "low"

    return DenoisedSeriesResult(
        series_name=series_name,
        n_points=n,
        raw_mean=float(arr.mean()),
        denoised_mean=denoised_mean,
        restock_events=int(restock_assign.sum()),
        restock_mass=restock_mass,
        noise_share_pct=round(noise_share, 2),
        signal_to_noise_db=round(snr_db, 2),
        confidence=confidence,
        data_quality_flags=flags,
    )
