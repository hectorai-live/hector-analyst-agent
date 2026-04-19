"""Real-MCP data adapter for the Hector Q-Comm MCP server.

Strictly read-only. Mirrors MockDataStore's surface so tools are backend-agnostic.

Design notes (validated against live payloads, 2026-04-19)
----------------------------------------------------------
Every Q-Comm MCP response is wrapped as:
    {"success": true, "tool": "<name>", "data": {<rows|items|series|brands>: [...]}}

Live schemas observed:
- get_inventory_snapshot      -> data.rows[] with {store_id, product_id, brand,
                                 city, inventory, sp, mrp, is_available, ...};
                                 single snapshot_date per call (no per-row date).
- get_doi_overview            -> data.items[] with {item_id, total_soh_units,
                                 daily_velocity_units, doi_days, risk_bucket, ...}.
- get_sales_estimate          -> data.rows[] with {snapshot_date, total_units_estimate,
                                 total_revenue_estimate_rs, store_count, ...}.
- get_osa_trend               -> data.series[] with {snapshot_date, value, store_count};
                                 wt_osa_pct by default.
- get_osa_darkstores          -> per-store OSA cross-section for a SKU.
- get_osa_competition         -> brand-vs-brand OSA.
- get_period_comparison       -> keyword-gain/loss between two dates (public SOV data).

This adapter assembles analyst-tool-ready DataFrames by calling multiple
endpoints and deriving day-level quantities (e.g. units_consumed from
inventory day-deltas when get_sales_estimate isn't grained enough).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd

MCPCaller = Callable[[str, dict[str, Any]], Any]


def _unwrap(resp: Any) -> Any:
    """Peel the standard {success, tool, data: {...}} envelope.
    Returns the inner list of rows/items/series/brands or [] if absent.
    """
    if resp is None:
        return []
    if isinstance(resp, list):
        return resp
    if not isinstance(resp, dict):
        return []
    payload = resp.get("data", resp)
    if not isinstance(payload, dict):
        return payload if isinstance(payload, list) else []
    for key in ("rows", "items", "series", "brands", "data", "store_list"):
        if key in payload and isinstance(payload[key], list):
            return payload[key]
    return []


def _rows_to_df(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "snapshot_date" in df.columns:
        df["snapshot_date"] = pd.to_datetime(df["snapshot_date"]).dt.date
    return df


def _daterange(start: date, end: date) -> list[date]:
    days = (end - start).days
    return [start + timedelta(days=i) for i in range(days + 1)]


@dataclass
class MCPDataStore:
    """Production DataStore backed by the Q-Comm MCP server.

    Parameters
    ----------
    caller
        Callable(tool_name, arguments) -> raw MCP response dict.
    as_of
        Freshness cut-off. Usually the latest available snapshot_date from
        get_data_status.
    default_lookback_days
        Lookback window for multi-day reconstructions.
    """

    caller: MCPCaller
    as_of: date
    default_lookback_days: int = 14

    @property
    def history_days(self) -> int:
        return self.default_lookback_days

    # ------------------------------------------------------------------ #
    # Inventory — per-day snapshot stitched across the lookback window   #
    # ------------------------------------------------------------------ #
    def _inventory_snapshots(
        self,
        *,
        brand: str | None = None,
        city: str | None = None,
        product_id: str | None = None,
        limit: int = 200,
    ) -> pd.DataFrame:
        """Stitch one snapshot per day across `default_lookback_days`."""
        frames: list[pd.DataFrame] = []
        start = self.as_of - timedelta(days=self.default_lookback_days - 1)
        for d in _daterange(start, self.as_of):
            args: dict[str, Any] = {"snapshot_date": d.isoformat(), "limit": limit}
            if brand:
                args["brand"] = brand
            if city:
                args["city"] = city
            if product_id:
                args["product_id"] = product_id
            rows = _unwrap(self.caller("get_inventory_snapshot", args))
            if not rows:
                continue
            df = _rows_to_df(rows)
            if df.empty:
                continue
            df["snapshot_date"] = d
            frames.append(df)
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True)
        return self._normalise_inventory(out)

    @staticmethod
    def _normalise_inventory(df: pd.DataFrame) -> pd.DataFrame:
        rename = {
            "store_id": "dark_store_id",
            "sp": "selling_price",
            "inventory": "inventory_level",
            "product_id": "sku_id",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        if "category" not in df.columns:
            df["category"] = None
        if "brand_id" not in df.columns and "brand" in df.columns:
            df["brand_id"] = df["brand"]
        # Derive units_consumed per (dark_store, sku) as lag-diff of inventory_level,
        # floored at 0 (negatives = restock events → excluded from demand signal).
        if {"dark_store_id", "sku_id", "snapshot_date", "inventory_level"}.issubset(
            df.columns
        ):
            df = df.sort_values(["dark_store_id", "sku_id", "snapshot_date"])
            df["_prev"] = df.groupby(["dark_store_id", "sku_id"])[
                "inventory_level"
            ].shift(1)
            df["units_consumed"] = (df["_prev"] - df["inventory_level"]).clip(lower=0)
            df["units_consumed"] = df["units_consumed"].fillna(0).astype(int)
            df = df.drop(columns=["_prev"])
        return df

    # ---------- forecast view ----------
    def forecast_input(self, sku_id: str, city: str) -> pd.DataFrame:
        return self._inventory_snapshots(product_id=sku_id, city=city)

    def warehouse_for(self, sku_id: str) -> pd.DataFrame:
        rows = _unwrap(self.caller("get_doi_overview", {"limit": 500}))
        df = _rows_to_df(rows)
        if df.empty:
            return df
        df = df.rename(
            columns={
                "item_id": "sku_id",
                "total_soh_units": "fe_stock",
                "total_frontend_units": "fe_frontend",
                "daily_velocity_units": "daily_velocity",
            }
        )
        df = df[df.get("sku_id", "").astype(str) == str(sku_id)].copy()
        if df.empty:
            return df
        df["snapshot_date"] = self.as_of
        if "fe_stock" in df.columns and "doi_days" in df.columns:
            df["be_stock"] = (df["fe_stock"] * 0.5).astype(int)  # proxy; BE not in DOI
        return df

    # ---------- counterfactual view ----------
    def counterfactual_input(
        self, brand_id: str, sku_id: str, city: str
    ) -> pd.DataFrame:
        """Inventory history merged with competitor and ads signals where available."""
        inv = self.forecast_input(sku_id, city)
        if inv.empty:
            return inv

        brand = inv["brand"].iloc[0] if "brand" in inv.columns else brand_id
        comp_osa = self._competitor_osa_series(
            brand=brand, city=city, start=inv["snapshot_date"].min(),
            end=inv["snapshot_date"].max(),
        )
        if not comp_osa.empty:
            inv = inv.merge(comp_osa, on="snapshot_date", how="left")
        return inv

    def _competitor_osa_series(
        self, brand: str, city: str, start: date, end: date
    ) -> pd.DataFrame:
        """Use get_osa_trend on the *opposite* scope as a competitor proxy when
        get_osa_competition requires an l1_category_id we may not know. We pull
        the brand's own OSA trend and synthesize a competitor_osa series as
        (pan-category mean − brand series) — a reasonable first-order proxy for
        V1. Replace with explicit competitor brands once a category is known.
        """
        own = _unwrap(
            self.caller(
                "get_osa_trend",
                {
                    "brand": brand,
                    "city": city,
                    "start_date": start.isoformat(),
                    "end_date": end.isoformat(),
                },
            )
        )
        if not own:
            return pd.DataFrame()
        df = _rows_to_df(own)
        if df.empty or "value" not in df.columns:
            return pd.DataFrame()
        # Proxy: competitor_osa = 1 - own OSA gap from a nominal 95% ceiling.
        df = df.rename(columns={"value": "own_osa_pct"})
        df["competitor_osa"] = ((95.0 - df["own_osa_pct"]) / 100.0).clip(lower=0.0, upper=1.0)
        return df[["snapshot_date", "competitor_osa"]]

    # ---------- attribution view ----------
    def attribution_input(
        self, brand_id: str, category: str, city: str
    ) -> pd.DataFrame:
        """Daily brand × category × city panel assembled from OSA trend + sales estimate."""
        start = self.as_of - timedelta(days=self.default_lookback_days * 2 - 1)
        osa_rows = _unwrap(
            self.caller(
                "get_osa_trend",
                {
                    "brand": brand_id,
                    "city": city,
                    "start_date": start.isoformat(),
                    "end_date": self.as_of.isoformat(),
                },
            )
        )
        osa = _rows_to_df(osa_rows).rename(columns={"value": "osa"})
        if not osa.empty:
            osa["osa"] = osa["osa"] / 100.0  # percent -> ratio

        sales_rows = _unwrap(
            self.caller(
                "get_sales_estimate",
                {
                    "brand": brand_id,
                    "city": city,
                    "start_date": start.isoformat(),
                    "end_date": self.as_of.isoformat(),
                    "group_by": "day",
                },
            )
        )
        sales = _rows_to_df(sales_rows).rename(
            columns={
                "total_revenue_estimate_rs": "revenue",
                "total_units_estimate": "units",
            }
        )

        if osa.empty and sales.empty:
            return pd.DataFrame()
        if osa.empty:
            panel = sales
        elif sales.empty:
            panel = osa
        else:
            panel = sales.merge(osa, on="snapshot_date", how="outer")

        panel["brand_id"] = brand_id
        panel["category"] = category
        panel["city"] = city
        panel["brand_sov"] = panel.get("brand_sov", pd.Series([0.30] * len(panel)))
        panel["selling_price"] = panel.get(
            "selling_price", pd.Series([None] * len(panel))
        )
        if "selling_price" in panel.columns and panel["selling_price"].isna().all():
            # Derive price from sales: rev / units when both present
            denom = panel["units"].where(panel["units"] > 0)
            panel["selling_price"] = (panel["revenue"] / denom).replace(
                [float("inf"), float("-inf")], pd.NA
            )
        panel["competitor_osa_avg"] = panel.get(
            "competitor_osa_avg",
            ((0.95 - panel.get("osa", 0.85)).clip(lower=0.0, upper=1.0)),
        )
        panel["competitor_sov_avg"] = panel.get(
            "competitor_sov_avg", pd.Series([0.4] * len(panel))
        )
        return panel

    # ------------------------------------------------------------------ #
    # Helpers for new tools (denoiser / anomaly / changepoint)           #
    # ------------------------------------------------------------------ #
    def osa_trend_series(
        self,
        *,
        brand: str,
        city: str | None = None,
        metric: str = "wt_osa_pct",
        days: int | None = None,
    ) -> pd.DataFrame:
        """Daily OSA time series for a brand, optionally scoped to a city."""
        days = days or self.default_lookback_days * 2
        start = self.as_of - timedelta(days=days - 1)
        args: dict[str, Any] = {
            "brand": brand,
            "start_date": start.isoformat(),
            "end_date": self.as_of.isoformat(),
            "metric": metric,
        }
        if city:
            args["city"] = city
        rows = _unwrap(self.caller("get_osa_trend", args))
        return _rows_to_df(rows)

    def sales_series(
        self,
        *,
        brand: str,
        city: str | None = None,
        days: int | None = None,
    ) -> pd.DataFrame:
        """Daily sales estimate series for a brand."""
        days = days or self.default_lookback_days * 2
        start = self.as_of - timedelta(days=days - 1)
        args: dict[str, Any] = {
            "brand": brand,
            "start_date": start.isoformat(),
            "end_date": self.as_of.isoformat(),
            "group_by": "day",
        }
        if city:
            args["city"] = city
        rows = _unwrap(self.caller("get_sales_estimate", args))
        return _rows_to_df(rows)


def build_mcp_store(caller: MCPCaller, as_of: date | str | None = None) -> MCPDataStore:
    """Convenience constructor. If as_of is None, calls get_data_status to pick
    the latest available snapshot_date."""
    if as_of is None:
        resp = caller("get_data_status", {})
        payload = (resp or {}).get("data", resp or {}) if isinstance(resp, dict) else {}
        as_of_s = (payload or {}).get("latest_snapshot_date")
        as_of_date = (
            datetime.strptime(as_of_s, "%Y-%m-%d").date() if as_of_s else date.today()
        )
    elif isinstance(as_of, str):
        as_of_date = datetime.strptime(as_of, "%Y-%m-%d").date()
    else:
        as_of_date = as_of
    return MCPDataStore(caller=caller, as_of=as_of_date)
