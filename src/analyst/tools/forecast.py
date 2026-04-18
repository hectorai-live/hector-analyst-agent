"""compute_forecast — Tool A.

Algorithm (as specified in the deployment brief):
  1. Pull 21d inventory for SKU × city.
  2. Median demand rate on *available* days per dark store.
  3. Pull current DOI (warehouse FE stock).
  4. Linear depletion: days_to_oos_ds = current_fe_stock_ds / demand_rate_ds.
  5. Aggregate across stores weighted by importance.
  6. Survival model: P(OOS by t) = 1 - exp(-lambda * t).
  7. Forward PSL = demand_rate_city × expected_oos_days × selling_price.
  8. Attach data-quality confidence grade.
"""

from __future__ import annotations

import math

import numpy as np

from analyst.data import MockDataStore, get_store, grade_quality
from analyst.data.mock import CATALOG, CITIES
from analyst.schemas import ForecastResult


def compute_forecast(
    sku_id: str,
    city: str,
    *,
    brand_id: str | None = None,
    horizon_days: int = 14,
    store: MockDataStore | None = None,
) -> ForecastResult:
    """Forecast days-to-OOS and forward PSL for a SKU × city."""
    store = store or get_store()
    sku = CATALOG[sku_id]
    city_spec = CITIES[city]

    df = store.forecast_input(sku_id, city)
    confidence, flags = grade_quality(df, tool="forecast")

    # --- Step 2: demand rate per dark store on available days ---
    available = df[df["is_available"]]
    if available.empty:
        flags.append("no_available_days")
        demand_rate_per_store: dict[str, float] = {}
    else:
        demand_rate_per_store = (
            available.groupby("dark_store_id")["units_consumed"]
            .median()
            .to_dict()
        )

    # --- Step 3: current FE stock per dark store (most recent snapshot) ---
    latest = df.sort_values("snapshot_date").groupby("dark_store_id").tail(1)
    current_stock = latest.set_index("dark_store_id")["inventory_level"].to_dict()

    # --- Step 4: days-to-OOS per store ---
    days_to_oos: list[float] = []
    for ds_id, stock in current_stock.items():
        rate = demand_rate_per_store.get(ds_id)
        if rate is None or rate <= 0:
            continue
        days_to_oos.append(stock / rate)

    total_stores = len(current_stock)
    stores_at_risk = sum(1 for d in days_to_oos if d <= horizon_days)

    if not days_to_oos:
        return ForecastResult(
            sku_id=sku_id,
            city=city,
            days_to_oos_point_estimate=float("inf"),
            days_to_oos_range=(float("inf"), float("inf")),
            probability_oos_within_7_days=0.0,
            probability_oos_within_14_days=0.0,
            forward_psl_rs=0.0,
            stores_at_risk=0,
            total_stores=total_stores,
            confidence="low",
            data_quality_flags=[*flags, "no_demand_signal"],
        )

    # --- Step 5: importance-weighted aggregation ---
    # In the absence of per-store importance weights, use equal weights within city
    # and the city-level importance_weight to scale survival rate.
    mean_days = float(np.mean(days_to_oos))
    p10 = float(np.quantile(days_to_oos, 0.10))
    p90 = float(np.quantile(days_to_oos, 0.90))

    # --- Step 6: survival / CDF ---
    lam = 1.0 / max(mean_days, 0.5)
    # Scale by city importance — busier cities trend OOS faster in aggregate
    lam *= 0.6 + 0.8 * city_spec.importance_weight
    p_oos_7 = 1 - math.exp(-lam * 7)
    p_oos_14 = 1 - math.exp(-lam * 14)

    # --- Step 7: forward PSL ---
    total_demand_rate = sum(demand_rate_per_store.values())
    expected_oos_days = max(0.0, horizon_days - mean_days)
    stores_oos_fraction = stores_at_risk / max(total_stores, 1)
    forward_psl = (
        total_demand_rate
        * expected_oos_days
        * stores_oos_fraction
        * sku.selling_price
    )

    # Warehouse DOI sanity check — flag if FE stock low
    wh = store.warehouse_for(sku_id)
    if not wh.empty:
        latest_fe = wh.sort_values("snapshot_date").iloc[-1]["fe_stock"]
        if latest_fe < 1500:
            flags.append("warehouse_fe_stock_low")

    return ForecastResult(
        sku_id=sku_id,
        city=city,
        days_to_oos_point_estimate=round(mean_days, 2),
        days_to_oos_range=(round(p10, 2), round(p90, 2)),
        probability_oos_within_7_days=round(p_oos_7, 3),
        probability_oos_within_14_days=round(p_oos_14, 3),
        forward_psl_rs=round(forward_psl, 2),
        stores_at_risk=stores_at_risk,
        total_stores=total_stores,
        confidence=confidence,
        data_quality_flags=flags,
    )
