"""estimate_elasticity — Tool G.

Log-log demand elasticity to price, with optional store fixed effects via
within-transform (de-meaning). Neither Claude nor MCP can run a regression
with confidence intervals.

Model
-----
    log(units_it) = α_i + β · log(price_it) + ε_it

where i = store, t = day, and α_i is a store fixed effect absorbed by
within-transform. β is the price elasticity of demand.

Returns a point estimate, 95% CI, R², and an `interpretation` label
(elastic / inelastic / unit_elastic).

Inputs
------
A DataFrame-like object (or list of records) with at least:
    - dark_store_id (optional; enables fixed effects)
    - selling_price (> 0)
    - units_consumed (>= 0)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from analyst.schemas import Confidence, ElasticityResult


def estimate_elasticity(
    df: pd.DataFrame | list[dict[str, Any]],
    *,
    brand_id: str = "",
    city: str = "",
    price_col: str = "selling_price",
    qty_col: str = "units_consumed",
    store_col: str = "dark_store_id",
) -> ElasticityResult:
    flags: list[str] = []
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    if df.empty or price_col not in df.columns or qty_col not in df.columns:
        flags.append("missing_price_or_qty")
        return ElasticityResult(
            brand_id=brand_id,
            city=city,
            n_observations=0,
            elasticity=0.0,
            elasticity_ci_low=0.0,
            elasticity_ci_high=0.0,
            r_squared=0.0,
            price_range=(0.0, 0.0),
            interpretation="undetermined",
            confidence="low",
            data_quality_flags=flags,
        )

    # Keep strictly positive rows (log requires > 0)
    mask = (df[price_col] > 0) & (df[qty_col] > 0)
    d = df.loc[mask, [price_col, qty_col] + ([store_col] if store_col in df.columns else [])].copy()
    n = len(d)
    if n < 10:
        flags.append(f"insufficient_observations_{n}_need_10")
        return ElasticityResult(
            brand_id=brand_id,
            city=city,
            n_observations=n,
            elasticity=0.0,
            elasticity_ci_low=0.0,
            elasticity_ci_high=0.0,
            r_squared=0.0,
            price_range=(float(d[price_col].min()), float(d[price_col].max())) if n else (0.0, 0.0),
            interpretation="undetermined",
            confidence="low",
            data_quality_flags=flags,
        )

    y = np.log(d[qty_col].astype(float).values)
    x = np.log(d[price_col].astype(float).values)

    # Within-transform on store fixed effects (if available & useful)
    if store_col in d.columns and d[store_col].nunique() > 1:
        d["_gx"] = x
        d["_gy"] = y
        g_mean_x = d.groupby(store_col)["_gx"].transform("mean")
        g_mean_y = d.groupby(store_col)["_gy"].transform("mean")
        x = x - g_mean_x.values
        y = y - g_mean_y.values
        # Net degrees of freedom after fixed effects
        dof = max(n - d[store_col].nunique() - 1, 1)
        used_fe = True
    else:
        # Center (remove intercept)
        x = x - x.mean()
        y = y - y.mean()
        dof = max(n - 1, 1)
        used_fe = False

    xx = float((x * x).sum())
    if xx < 1e-9:
        flags.append("price_variance_insufficient")
        return ElasticityResult(
            brand_id=brand_id,
            city=city,
            n_observations=n,
            elasticity=0.0,
            elasticity_ci_low=0.0,
            elasticity_ci_high=0.0,
            r_squared=0.0,
            price_range=(float(d[price_col].min()), float(d[price_col].max())),
            interpretation="undetermined",
            confidence="low",
            data_quality_flags=flags,
        )
    beta = float((x * y).sum() / xx)
    yhat = beta * x
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float((y * y).sum())
    r2 = 0.0 if ss_tot < 1e-12 else 1 - ss_res / ss_tot
    sigma2 = ss_res / dof
    se_beta = (sigma2 / xx) ** 0.5
    ci = 1.96 * se_beta

    abs_b = abs(beta)
    if abs_b > 1.1:
        interpretation = "elastic"
    elif abs_b < 0.9:
        interpretation = "inelastic"
    else:
        interpretation = "unit_elastic"

    if not used_fe:
        flags.append("no_store_fixed_effects_applied")
    if r2 < 0.05:
        flags.append("low_r_squared")

    confidence: Confidence = (
        "high" if n >= 50 and abs(beta) > 2 * se_beta and r2 >= 0.1
        else "medium" if n >= 20 and abs(beta) > se_beta
        else "low"
    )

    return ElasticityResult(
        brand_id=brand_id,
        city=city,
        n_observations=n,
        elasticity=round(beta, 4),
        elasticity_ci_low=round(beta - ci, 4),
        elasticity_ci_high=round(beta + ci, 4),
        r_squared=round(r2, 4),
        price_range=(round(float(d[price_col].min()), 2), round(float(d[price_col].max()), 2)),
        interpretation=interpretation,
        confidence=confidence,
        data_quality_flags=flags,
    )
