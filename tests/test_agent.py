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


def test_anomaly_mode_via_agent():
    osa_vals = [
        86.16, 93.91, 90.34, 90.18, 89.18, 88.75, 88.45, 88.48, 88.26,
        88.14, 87.65, 89.82, 90.87, 90.79, 81.18, 83.03, 82.66, 79.94,
    ]
    osa_dates = [f"2026-04-{i + 1:02d}" for i in range(18)]
    from analyst.agent import AnalystAgent
    from analyst.schemas import AnalystState, Mode

    state = AnalystState(
        query="any anomalies in OSA?",
        brand_id="Foxtale",
        city="delhi",
        series_input=osa_vals,
        series_dates=osa_dates,
    )
    result = AnalystAgent().invoke(state)
    assert Mode.ANOMALY in result.modes_triggered
    assert any(cr["tool"] == "detect_anomalies" for cr in result.computation_results)
    assert any(e.source_tool == "detect_anomalies" for e in result.estimate_objects)


def test_changepoint_mode_via_agent():
    osa_vals = [
        86.16, 93.91, 90.34, 90.18, 89.18, 88.75, 88.45, 88.48, 88.26,
        88.14, 87.65, 89.82, 90.87, 90.79, 81.18, 83.03, 82.66, 79.94,
    ]
    osa_dates = [f"2026-04-{i + 1:02d}" for i in range(18)]
    from analyst.agent import AnalystAgent
    from analyst.schemas import AnalystState, Mode

    state = AnalystState(
        query="when did the regime change for our brand?",
        brand_id="Foxtale",
        city="delhi",
        series_input=osa_vals,
        series_dates=osa_dates,
    )
    result = AnalystAgent().invoke(state)
    assert Mode.CHANGEPOINT in result.modes_triggered
    cp_results = [cr for cr in result.computation_results if cr["tool"] == "detect_changepoints"]
    assert cp_results
    # Live data should surface at least one cp near 2026-04-14
    assert any("2026-04-1" in cp for cp in cp_results[0]["changepoints"])


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
