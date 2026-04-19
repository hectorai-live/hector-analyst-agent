"""Real-MCP data adapter for the Hector Q-Comm MCP server.

The MCP server at `12ce7690-7939-4322-83b5-5173bc48e3c9` exposes the Asgard
data surface. This adapter is the production path — it mirrors MockDataStore's
interface so tools are backend-agnostic.

Strictly read-only. This module never issues writes, mutations, or deletes.

Design notes
------------
- The MCP tools are invoked through a `caller` callable injected at construction
  time. In a real deployment, caller is bound to the MCP client (e.g. a wrapper
  over the Anthropic tool-use loop, or a LangGraph MCP node). For offline tests,
  caller can be a stub.
- Each method materialises the BigQuery-style row set into a pandas DataFrame
  whose columns match what `analyst.tools.*` consume — same schema as
  `MockDataStore` output.
- Short-circuits on failure: the caller may return None / empty, in which case
  the downstream quality grader flags `empty_dataset` and the tool emits a
  low-confidence result rather than crashing.

Why not call MCP tools directly inside `forecast.py` etc.?
- Separation of concerns: tools are pure numerical functions.
- Testability: tool code can run against any DataStore without a network mock.
- Pluggable backends: BigQuery, Snowflake, Redshift, or a warm cache can be
  dropped in behind the same Protocol.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import pandas as pd

MCPCaller = Callable[[str, dict[str, Any]], dict[str, Any] | list[dict[str, Any]] | None]


def _to_df(rows: Any) -> pd.DataFrame:
    if rows is None:
        return pd.DataFrame()
    if isinstance(rows, dict):
        # Some MCP responses wrap rows in {"data": [...]}; tolerate either shape.
        rows = rows.get("data") or rows.get("rows") or []
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "snapshot_date" in df.columns:
        df["snapshot_date"] = pd.to_datetime(df["snapshot_date"]).dt.date
    return df


@dataclass
class MCPDataStore:
    """Production DataStore backed by the Q-Comm MCP server.

    Parameters
    ----------
    caller
        A callable that takes (tool_name, arguments) and returns the MCP
        tool's raw response. Bind this to your MCP client at wire-up.
    as_of
        Freshness cutoff (usually today). Passed as an argument to MCP tools
        that accept one.
    default_lookback_days
        Lookback window for the "*_input" pulls. 30d is sensible for V1.
    """

    caller: MCPCaller
    as_of: date
    default_lookback_days: int = 30

    @property
    def history_days(self) -> int:
        return self.default_lookback_days

    # ---------- forecast view ----------
    def forecast_input(self, sku_id: str, city: str) -> pd.DataFrame:
        rows = self.caller(
            "get_inventory_snapshot",
            {
                "sku_id": sku_id,
                "city": city,
                "lookback_days": self.default_lookback_days,
                "as_of": self.as_of.isoformat(),
            },
        )
        df = _to_df(rows)
        if df.empty:
            return df
        # Normalise schema to match mock.MockDataStore output
        column_aliases = {
            "store_id": "dark_store_id",
            "darkstore_id": "dark_store_id",
            "available": "is_available",
            "units": "units_consumed",
            "stock": "inventory_level",
            "price": "selling_price",
        }
        df = df.rename(columns={k: v for k, v in column_aliases.items() if k in df.columns})
        return df

    def warehouse_for(self, sku_id: str) -> pd.DataFrame:
        rows = self.caller(
            "get_doi_overview",
            {"sku_id": sku_id, "as_of": self.as_of.isoformat()},
        )
        df = _to_df(rows)
        if df.empty:
            return df
        df = df.rename(
            columns={
                "fe": "fe_stock",
                "be": "be_stock",
                "doi": "doi_days",
            }
        )
        return df

    # ---------- counterfactual view ----------
    def counterfactual_input(
        self, brand_id: str, sku_id: str, city: str
    ) -> pd.DataFrame:
        inv = self.forecast_input(sku_id, city)
        comp = _to_df(
            self.caller(
                "get_competitor_visibility",
                {
                    "brand_id": brand_id,
                    "city": city,
                    "lookback_days": self.default_lookback_days,
                    "as_of": self.as_of.isoformat(),
                },
            )
        )
        ads = _to_df(
            self.caller(
                "get_ads_attribution",
                {
                    "brand_id": brand_id,
                    "city": city,
                    "lookback_days": self.default_lookback_days,
                    "as_of": self.as_of.isoformat(),
                },
            )
        )
        if inv.empty:
            return inv
        if not comp.empty:
            comp = comp.rename(columns={"osa": "competitor_osa"})
            keys = [c for c in ("snapshot_date", "city", "category") if c in comp.columns]
            if keys:
                inv = inv.merge(comp[[*keys, "competitor_osa"]], on=keys, how="left")
        if not ads.empty:
            keys = [c for c in ("snapshot_date", "brand_id", "city") if c in ads.columns]
            if keys:
                inv = inv.merge(ads, on=keys, how="left")
        return inv

    # ---------- attribution view ----------
    def attribution_input(
        self, brand_id: str, category: str, city: str
    ) -> pd.DataFrame:
        rows = self.caller(
            "get_period_comparison",
            {
                "brand_id": brand_id,
                "category": category,
                "city": city,
                "lookback_days": self.default_lookback_days * 2,
                "as_of": self.as_of.isoformat(),
            },
        )
        df = _to_df(rows)
        if df.empty:
            return df
        # Normalise to attribution-tool's expected columns
        aliases = {
            "avg_osa": "osa",
            "sov": "brand_sov",
            "price": "selling_price",
            "competitor_osa": "competitor_osa_avg",
            "competitor_sov": "competitor_sov_avg",
        }
        df = df.rename(columns={k: v for k, v in aliases.items() if k in df.columns})
        return df


def build_mcp_store(caller: MCPCaller, as_of: date | str | None = None) -> MCPDataStore:
    """Convenience constructor: `build_mcp_store(caller)` uses today as freshness."""
    if as_of is None:
        as_of_date = date.today()
    elif isinstance(as_of, str):
        as_of_date = datetime.strptime(as_of, "%Y-%m-%d").date()
    else:
        as_of_date = as_of
    return MCPDataStore(caller=caller, as_of=as_of_date)
