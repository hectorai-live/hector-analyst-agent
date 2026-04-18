"""Synthetic Asgard-style data.

Generates deterministic, realistic-looking inventory, portal sales, competitor,
and ad-spend time series. Shape mirrors the three BigQuery views described in
the deployment brief (analyst_forecast_input, analyst_counterfactual_input,
analyst_attribution_input).

Seeded so backtests and tests are reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from functools import lru_cache

import numpy as np
import pandas as pd

DEFAULT_SEED = 42
DEFAULT_HISTORY_DAYS = 45
TODAY = date(2026, 4, 19)


@dataclass(frozen=True)
class SkuSpec:
    sku_id: str
    brand_id: str
    category: str
    selling_price: float
    base_demand_per_store: float


@dataclass(frozen=True)
class CitySpec:
    city: str
    n_dark_stores: int
    importance_weight: float


CATALOG: dict[str, SkuSpec] = {
    "FW100-MAE": SkuSpec("FW100-MAE", "MAE", "Face Wash", 199.0, 12.0),
    "FW100-HEC": SkuSpec("FW100-HEC", "HEC", "Face Wash", 209.0, 11.0),
    "SH200-HEC": SkuSpec("SH200-HEC", "HEC", "Shampoo", 349.0, 8.0),
    "BL250-HEC": SkuSpec("BL250-HEC", "HEC", "Body Lotion", 299.0, 6.0),
}

CITIES: dict[str, CitySpec] = {
    "Mumbai": CitySpec("Mumbai", n_dark_stores=24, importance_weight=1.00),
    "Delhi": CitySpec("Delhi", n_dark_stores=22, importance_weight=0.92),
    "Bengaluru": CitySpec("Bengaluru", n_dark_stores=18, importance_weight=0.85),
    "Pune": CitySpec("Pune", n_dark_stores=12, importance_weight=0.64),
}


def _dow_factor(d: date) -> float:
    """Sunday sells more, Tuesday less — rough Q-comm pattern."""
    return {0: 1.00, 1: 0.92, 2: 0.98, 3: 1.02, 4: 1.08, 5: 1.15, 6: 1.22}[d.weekday()]


def _date_range(days: int, end: date) -> list[date]:
    return [end - timedelta(days=i) for i in range(days - 1, -1, -1)]


@dataclass
class MockDataStore:
    """In-memory stand-in for the three BigQuery analyst views."""

    inventory: pd.DataFrame
    portal_sales: pd.DataFrame
    competitor: pd.DataFrame
    warehouse: pd.DataFrame
    ad_spend: pd.DataFrame
    history_days: int
    as_of: date

    # ---------- forecast view ----------
    def forecast_input(self, sku_id: str, city: str) -> pd.DataFrame:
        mask = (self.inventory["sku_id"] == sku_id) & (self.inventory["city"] == city)
        return self.inventory.loc[mask].copy()

    def warehouse_for(self, sku_id: str) -> pd.DataFrame:
        return self.warehouse.loc[self.warehouse["sku_id"] == sku_id].copy()

    # ---------- counterfactual view ----------
    def counterfactual_input(
        self, brand_id: str, sku_id: str, city: str
    ) -> pd.DataFrame:
        inv = self.inventory.loc[
            (self.inventory["sku_id"] == sku_id) & (self.inventory["city"] == city)
        ]
        comp = self.competitor.loc[
            (self.competitor["city"] == city)
            & (self.competitor["category"] == CATALOG[sku_id].category)
        ]
        ad = self.ad_spend.loc[
            (self.ad_spend["brand_id"] == brand_id) & (self.ad_spend["city"] == city)
        ]
        return inv.merge(
            comp, on=["snapshot_date", "city", "category"], how="left"
        ).merge(ad, on=["snapshot_date", "brand_id", "city"], how="left")

    # ---------- attribution view ----------
    def attribution_input(
        self, brand_id: str, category: str, city: str
    ) -> pd.DataFrame:
        mask = (
            (self.portal_sales["brand_id"] == brand_id)
            & (self.portal_sales["category"] == category)
            & (self.portal_sales["city"] == city)
        )
        return self.portal_sales.loc[mask].copy()


def _build_inventory(
    rng: np.random.Generator, dates: list[date], inject_oos_event: bool
) -> pd.DataFrame:
    rows: list[dict] = []
    for sku_id, sku in CATALOG.items():
        for city_name, city in CITIES.items():
            for ds in range(city.n_dark_stores):
                store_id = f"{city_name[:3].upper()}-DS{ds:03d}"
                base = sku.base_demand_per_store * rng.uniform(0.7, 1.3)
                stock = rng.integers(80, 180)
                for d in dates:
                    seasonal = _dow_factor(d)
                    noise = rng.normal(1.0, 0.15)
                    consumed = max(0, int(base * seasonal * noise))

                    is_oos_event = (
                        inject_oos_event
                        and sku_id == "FW100-HEC"
                        and city_name == "Mumbai"
                        and ds < 8
                        and (TODAY - d).days < 5
                    )
                    if is_oos_event:
                        consumed = 0
                        stock = 0
                        is_available = False
                    else:
                        stock = max(0, stock - consumed)
                        if stock < base * 2 and rng.random() < 0.35:
                            stock += int(rng.integers(60, 140))
                        is_available = stock > 0 and consumed > 0

                    rows.append(
                        {
                            "snapshot_date": d,
                            "sku_id": sku_id,
                            "brand_id": sku.brand_id,
                            "category": sku.category,
                            "city": city_name,
                            "dark_store_id": store_id,
                            "inventory_level": int(stock),
                            "units_consumed": consumed,
                            "is_available": is_available,
                            "selling_price": sku.selling_price,
                        }
                    )
    return pd.DataFrame(rows)


def _build_warehouse(rng: np.random.Generator, dates: list[date]) -> pd.DataFrame:
    rows: list[dict] = []
    for sku_id in CATALOG:
        fe = rng.integers(4000, 12000)
        for d in dates:
            depletion = rng.integers(280, 620)
            fe = max(0, fe - depletion)
            if fe < 2000 and rng.random() < 0.25:
                fe += int(rng.integers(3000, 8000))
            be = int(fe * rng.uniform(0.3, 0.8))
            rows.append(
                {
                    "snapshot_date": d,
                    "sku_id": sku_id,
                    "fe_stock": int(fe),
                    "be_stock": be,
                    "doi_days": round(fe / max(depletion, 1), 2),
                }
            )
    return pd.DataFrame(rows)


def _build_portal_sales(
    rng: np.random.Generator, dates: list[date], inventory: pd.DataFrame
) -> pd.DataFrame:
    # Portal sales ≈ sum of units_consumed × selling_price per (brand, category, city, day)
    daily = (
        inventory.groupby(["snapshot_date", "brand_id", "category", "city"])
        .agg(
            units=("units_consumed", "sum"),
            selling_price=("selling_price", "mean"),
            osa=("is_available", "mean"),
        )
        .reset_index()
    )
    daily["revenue"] = daily["units"] * daily["selling_price"]
    # Competitor / SOV / ad noise
    daily["brand_sov"] = rng.uniform(0.18, 0.42, len(daily))
    daily["ad_spend"] = rng.uniform(800, 3500, len(daily))
    daily["roas"] = rng.uniform(2.5, 6.0, len(daily))
    daily["market_size_rs"] = daily["revenue"] / daily["brand_sov"].clip(lower=0.05)
    daily["competitor_osa_avg"] = rng.uniform(0.65, 0.95, len(daily))
    daily["competitor_sov_avg"] = rng.uniform(0.28, 0.55, len(daily))
    daily["selling_price"] = daily["selling_price"] + rng.normal(0, 2, len(daily))
    return daily


def _build_competitor(rng: np.random.Generator, dates: list[date]) -> pd.DataFrame:
    rows: list[dict] = []
    categories = sorted({s.category for s in CATALOG.values()})
    for city_name in CITIES:
        for cat in categories:
            for d in dates:
                # simulate a competitor attack window: Mamaearth Face Wash in Bengaluru
                is_attack = (
                    cat == "Face Wash"
                    and city_name == "Bengaluru"
                    and 8 <= (TODAY - d).days <= 10
                )
                comp_osa = rng.uniform(0.75, 0.95)
                if is_attack:
                    comp_osa = rng.uniform(0.15, 0.35)
                rows.append(
                    {
                        "snapshot_date": d,
                        "city": city_name,
                        "category": cat,
                        "competitor_osa": comp_osa,
                        "competitor_units": int(rng.integers(140, 420) * comp_osa),
                    }
                )
    return pd.DataFrame(rows)


def _build_ad_spend(rng: np.random.Generator, dates: list[date]) -> pd.DataFrame:
    rows: list[dict] = []
    for brand_id in {s.brand_id for s in CATALOG.values()}:
        for city_name in CITIES:
            for d in dates:
                rows.append(
                    {
                        "snapshot_date": d,
                        "brand_id": brand_id,
                        "city": city_name,
                        "ad_spend": float(rng.uniform(500, 4000)),
                        "roas": float(rng.uniform(2.5, 6.0)),
                    }
                )
    return pd.DataFrame(rows)


@lru_cache(maxsize=8)
def get_store(
    history_days: int = DEFAULT_HISTORY_DAYS,
    as_of: date = TODAY,
    seed: int = DEFAULT_SEED,
    inject_oos_event: bool = True,
) -> MockDataStore:
    """Return a cached MockDataStore. Seeded for reproducibility."""
    rng = np.random.default_rng(seed)
    dates = _date_range(history_days, as_of)
    inventory = _build_inventory(rng, dates, inject_oos_event)
    warehouse = _build_warehouse(rng, dates)
    competitor = _build_competitor(rng, dates)
    ad_spend = _build_ad_spend(rng, dates)
    portal_sales = _build_portal_sales(rng, dates, inventory)
    return MockDataStore(
        inventory=inventory,
        portal_sales=portal_sales,
        competitor=competitor,
        warehouse=warehouse,
        ad_spend=ad_spend,
        history_days=history_days,
        as_of=as_of,
    )
