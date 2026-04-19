"""End-to-end tests for the AnalystAgent graph runner."""

from __future__ import annotations

from datetime import timedelta

from analyst import run
from analyst.agent import AnalystAgent
from analyst.data.mock import TODAY
from analyst.schemas import AnalystState


def test_invoke_never_raises_on_missing_fields():
    """The agent must route errors into state.errors rather than raising."""
    agent = AnalystAgent()
    state = AnalystState(query="will it stock out?", brand_id="HEC")
    # No sku_id / city — computation node should record an error, not raise.
    result = agent.invoke(state)
    assert isinstance(result.errors, list)
    assert any("forecast_failed" in e for e in result.errors)


def test_full_forecast_pipeline():
    result = run(
        query="will face wash stock out next 7 days?",
        brand_id="HEC",
        sku_id="FW100-HEC",
        city="Mumbai",
    )
    assert "forecast" in result["modes_triggered"]
    assert len(result["computation_results"]) == 1
    assert len(result["estimate_objects"]) == 1
    est = result["estimate_objects"][0]
    assert est["source_tool"] == "compute_forecast"
    assert est["headline"]
    assert est["evidence"]
    assert est["estimate"]
    assert est["confidence"]
    assert est["action"]
    assert est["methodology"]


def test_compound_query_produces_two_estimates():
    start = TODAY - timedelta(days=14)
    end_prev = TODAY - timedelta(days=8)
    end_curr = TODAY - timedelta(days=1)
    start_curr = TODAY - timedelta(days=7)
    result = run(
        query=(
            "why is face wash revenue down in bengaluru, "
            "and will it get worse next 7 days?"
        ),
        brand_id="HEC",
        sku_id="FW100-HEC",
        city="Bengaluru",
        category="Face Wash",
        period=f"{start_curr}:{end_curr}",
        comparison_period=f"{start}:{end_prev}",
    )
    assert "forecast" in result["modes_triggered"]
    assert "causal" in result["modes_triggered"]
    # Both estimate objects present
    tools = {e["source_tool"] for e in result["estimate_objects"]}
    assert "compute_forecast" in tools
    assert "compute_attribution" in tools


def test_every_estimate_has_confidence_grade():
    result = run(
        query="will it stock out?",
        brand_id="HEC",
        sku_id="FW100-HEC",
        city="Mumbai",
    )
    for est in result["estimate_objects"]:
        assert est["confidence"]
    for cr in result["computation_results"]:
        assert cr["confidence"] in {"high", "medium", "low"}
        assert isinstance(cr["data_quality_flags"], list)


def test_narrator_grounded_in_numbers():
    """Narrator must echo numbers present in the computation result — no hallucination.
    We verify this by ensuring the forecast headline contains the point estimate
    as an integer day count from the computation result."""
    result = run(
        query="will face wash stock out?",
        brand_id="HEC",
        sku_id="FW100-HEC",
        city="Mumbai",
    )
    cr = result["computation_results"][0]
    est = result["estimate_objects"][0]
    # template narrator formats with .0f — assert the integer is present in headline
    pt = f"{cr['days_to_oos_point_estimate']:.0f}"
    assert pt in est["headline"]
