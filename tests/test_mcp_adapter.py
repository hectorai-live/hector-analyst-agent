"""Tests for the MCP data adapter using a stub caller.

No network. No real MCP server. Validates that shape-normalisation works and
that the store satisfies the DataStore protocol so tools can run against it.
"""

from __future__ import annotations

from datetime import date

from analyst.data import DataStore, build_mcp_store
from analyst.data.mcp_adapter import MCPDataStore


def _stub_caller(tool: str, args: dict):
    # Return minimally-shaped rows that mirror the live MCP payloads.
    if tool == "get_inventory_snapshot":
        return [
            {
                "snapshot_date": "2026-04-10",
                "sku_id": args["sku_id"],
                "brand_id": "HEC",
                "category": "Face Wash",
                "city": args["city"],
                "store_id": "MUM-DS000",
                "stock": 80,
                "units": 10,
                "available": True,
                "price": 209.0,
            },
            {
                "snapshot_date": "2026-04-11",
                "sku_id": args["sku_id"],
                "brand_id": "HEC",
                "category": "Face Wash",
                "city": args["city"],
                "store_id": "MUM-DS000",
                "stock": 70,
                "units": 10,
                "available": True,
                "price": 209.0,
            },
        ]
    if tool == "get_doi_overview":
        return [
            {
                "snapshot_date": "2026-04-11",
                "sku_id": args["sku_id"],
                "fe": 4200,
                "be": 1100,
                "doi": 12.5,
            }
        ]
    if tool == "get_competitor_visibility":
        return [
            {
                "snapshot_date": "2026-04-10",
                "city": args["city"],
                "category": "Face Wash",
                "osa": 0.8,
            }
        ]
    if tool == "get_ads_attribution":
        return [
            {
                "snapshot_date": "2026-04-10",
                "brand_id": args["brand_id"],
                "city": args["city"],
                "ad_spend": 1200.0,
                "roas": 3.1,
            }
        ]
    if tool == "get_period_comparison":
        return [
            {
                "snapshot_date": "2026-04-10",
                "brand_id": args["brand_id"],
                "category": args["category"],
                "city": args["city"],
                "revenue": 15000.0,
                "avg_osa": 0.82,
                "sov": 0.35,
                "price": 199.0,
                "competitor_osa": 0.7,
                "competitor_sov": 0.42,
            }
        ]
    return []


def test_adapter_satisfies_protocol():
    store = build_mcp_store(_stub_caller, as_of=date(2026, 4, 19))
    assert isinstance(store, DataStore)


def test_forecast_input_normalises_columns():
    store = MCPDataStore(caller=_stub_caller, as_of=date(2026, 4, 19))
    df = store.forecast_input("FW100-HEC", "Mumbai")
    assert not df.empty
    for col in ("dark_store_id", "is_available", "units_consumed", "inventory_level", "selling_price"):
        assert col in df.columns


def test_counterfactual_input_merges_competitor_and_ads():
    store = MCPDataStore(caller=_stub_caller, as_of=date(2026, 4, 19))
    df = store.counterfactual_input("HEC", "FW100-HEC", "Mumbai")
    assert not df.empty
    assert "competitor_osa" in df.columns


def test_attribution_input_normalises_columns():
    store = MCPDataStore(caller=_stub_caller, as_of=date(2026, 4, 19))
    df = store.attribution_input("HEC", "Face Wash", "Mumbai")
    assert not df.empty
    for col in ("osa", "brand_sov", "selling_price", "competitor_osa_avg", "competitor_sov_avg"):
        assert col in df.columns


def test_empty_payload_is_graceful():
    store = MCPDataStore(caller=lambda _t, _a: None, as_of=date(2026, 4, 19))
    assert store.forecast_input("FW100-HEC", "Mumbai").empty
    assert store.attribution_input("HEC", "Face Wash", "Mumbai").empty
