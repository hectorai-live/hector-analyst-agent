"""Tests for the MCP data adapter using a stub caller.

Stubs return payloads shaped exactly like the live Q-Comm MCP envelope
(`{success, tool, data: {rows|items|series|brands}}`) so normalization is
exercised end-to-end.
"""

from __future__ import annotations

from datetime import date, timedelta

from analyst.data import DataStore, build_mcp_store
from analyst.data.mcp_adapter import MCPDataStore

AS_OF = date(2026, 4, 18)


def _ok(payload: dict) -> dict:
    return {"success": True, "tool": "stub", "data": payload}


def _stub_caller(tool: str, args: dict):
    if tool == "get_data_status":
        return _ok({"latest_snapshot_date": AS_OF.isoformat(), "days_of_data": 18})
    if tool == "get_inventory_snapshot":
        # One row per call; adapter loops over dates and stacks
        d = args.get("snapshot_date", AS_OF.isoformat())
        return _ok(
            {
                "rows": [
                    {
                        "store_id": "29581",
                        "product_id": args.get("product_id", "503775"),
                        "sku_name": "Foxtale Something",
                        "brand": args.get("brand", "Foxtale"),
                        "city": args.get("city", "delhi"),
                        "is_available": True,
                        "inventory": 100 - (AS_OF - date.fromisoformat(d)).days,
                        "sp": 360,
                        "mrp": 449,
                    }
                ]
            }
        )
    if tool == "get_doi_overview":
        return _ok(
            {
                "items": [
                    {
                        "item_id": "503775",
                        "total_soh_units": 4200,
                        "total_frontend_units": 60,
                        "daily_velocity_units": 0.286,
                        "doi_days": 12.5,
                        "risk_bucket": "healthy",
                    }
                ]
            }
        )
    if tool == "get_osa_trend":
        # 10 days of OSA
        series = []
        for i in range(10):
            d = AS_OF - timedelta(days=9 - i)
            series.append(
                {"snapshot_date": d.isoformat(), "value": 85.0 - i * 0.5, "store_count": 170}
            )
        return _ok({"series": series})
    if tool == "get_sales_estimate":
        rows = []
        for i in range(10):
            d = AS_OF - timedelta(days=9 - i)
            rows.append(
                {
                    "snapshot_date": d.isoformat(),
                    "total_units_estimate": 2000 - i * 50,
                    "total_revenue_estimate_rs": 500000 - i * 10000,
                    "store_count": 1700,
                }
            )
        return _ok({"rows": rows})
    if tool == "get_period_comparison":
        return _ok({"rows": []})
    return _ok({"rows": []})


def test_adapter_satisfies_protocol():
    store = build_mcp_store(_stub_caller)
    assert isinstance(store, DataStore)
    assert store.as_of == AS_OF


def test_forecast_input_normalises_columns():
    store = MCPDataStore(caller=_stub_caller, as_of=AS_OF, default_lookback_days=5)
    df = store.forecast_input("503775", "delhi")
    assert not df.empty
    for col in (
        "dark_store_id",
        "is_available",
        "units_consumed",
        "inventory_level",
        "selling_price",
        "snapshot_date",
    ):
        assert col in df.columns


def test_warehouse_for_normalises_columns():
    store = MCPDataStore(caller=_stub_caller, as_of=AS_OF)
    df = store.warehouse_for("503775")
    assert not df.empty
    assert "fe_stock" in df.columns


def test_counterfactual_input_merges_competitor_signal():
    store = MCPDataStore(caller=_stub_caller, as_of=AS_OF, default_lookback_days=5)
    df = store.counterfactual_input("Foxtale", "503775", "delhi")
    assert not df.empty
    assert "competitor_osa" in df.columns


def test_attribution_input_assembles_panel():
    store = MCPDataStore(caller=_stub_caller, as_of=AS_OF, default_lookback_days=5)
    df = store.attribution_input("Foxtale", "Face Wash", "delhi")
    assert not df.empty
    for col in ("osa", "revenue", "brand_sov", "selling_price", "competitor_osa_avg"):
        assert col in df.columns


def test_empty_payload_is_graceful():
    store = MCPDataStore(caller=lambda _t, _a: None, as_of=AS_OF)
    assert store.forecast_input("503775", "delhi").empty
    assert store.attribution_input("Foxtale", "Face Wash", "delhi").empty


def test_osa_trend_series():
    store = MCPDataStore(caller=_stub_caller, as_of=AS_OF)
    df = store.osa_trend_series(brand="Foxtale", city="delhi", days=10)
    assert not df.empty
    assert "value" in df.columns
    assert "snapshot_date" in df.columns


def test_sales_series():
    store = MCPDataStore(caller=_stub_caller, as_of=AS_OF)
    df = store.sales_series(brand="Foxtale", city="delhi", days=10)
    assert not df.empty
    assert "total_units_estimate" in df.columns
