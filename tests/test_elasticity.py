"""Tests for estimate_elasticity — log-log OLS with store FE."""

from __future__ import annotations

import numpy as np
import pandas as pd

from analyst.tools import estimate_elasticity


def _synthetic_panel(beta: float = -1.5, n_stores: int = 5, n_days: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(37)
    rows = []
    for s in range(n_stores):
        base_units = 50.0 + 5 * s  # store fixed effect
        for t in range(n_days):
            price = 100.0 + rng.normal(0, 8)
            # log(q) = log(base) + β log(p) + ε
            log_q = np.log(base_units) + beta * (np.log(price) - np.log(100)) + rng.normal(0, 0.05)
            units = max(1, int(round(np.exp(log_q))))
            rows.append(
                {"dark_store_id": f"DS{s}", "selling_price": price, "units_consumed": units}
            )
    return pd.DataFrame(rows)


def test_recovers_true_elasticity():
    df = _synthetic_panel(beta=-1.5)
    r = estimate_elasticity(df, brand_id="HEC", city="Mumbai")
    assert abs(r.elasticity - (-1.5)) < 0.25
    assert r.elasticity_ci_low < -1.5 < r.elasticity_ci_high
    assert r.interpretation == "elastic"
    assert r.confidence in {"high", "medium"}


def test_recovers_inelastic():
    df = _synthetic_panel(beta=-0.3)
    r = estimate_elasticity(df, brand_id="HEC", city="Mumbai")
    assert abs(r.elasticity - (-0.3)) < 0.2
    assert r.interpretation == "inelastic"


def test_degenerate_price_variance_flagged():
    df = pd.DataFrame(
        [
            {"dark_store_id": "DS0", "selling_price": 100.0, "units_consumed": 10}
            for _ in range(15)
        ]
    )
    r = estimate_elasticity(df)
    assert "price_variance_insufficient" in r.data_quality_flags
    assert r.interpretation == "undetermined"


def test_missing_columns_graceful():
    r = estimate_elasticity(pd.DataFrame([{"foo": 1}]))
    assert "missing_price_or_qty" in r.data_quality_flags
    assert r.confidence == "low"
