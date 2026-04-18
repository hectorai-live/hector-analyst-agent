# Hector Q-Comm Analyst Agent

Statistical computation engine for Hector's Q-Comm Brain. Turns MCP/Asgard data
into probabilistic estimates across three modes:

| Mode | Question it answers | Tool |
|---|---|---|
| **Forecast** | *Will this SKU go OOS? When? How much ₹ at risk?* | `compute_forecast` |
| **Counterfactual** | *What would revenue have been if we'd acted / held OSA / caught the competitor window?* | `compute_counterfactual` |
| **Causal** | *Why did revenue move? How much is availability vs visibility vs pricing vs competitor?* | `compute_attribution` |

The agent sits alongside the Fixer and Finder as a third brain — **statistical
computation + LLM reasoning**, not LLM-driven gate logic.

## Architecture

```
Entry
  ↓
Intent Classifier (deterministic)
  ↓
Data Retrieval (MCP tools)
  ↓
Computation Node (Python statistical tools)
  ↓
LLM Narrator (grounded — never invents numbers)
  ↓
EstimateObject { headline, evidence, estimate, confidence, action, methodology }
```

Four nodes, one shared state object (`AnalystState`). Any mode can fire
independently or together — compound questions like *"why did revenue drop and
is it going to get worse?"* run Causal + Forecast on shared state.

## Install

```bash
pip install -e '.[dev]'
```

Python 3.11+. Dependencies: numpy, pandas, scipy, pydantic, typer, rich,
anthropic (optional — narrator falls back to templates without an API key).

## Quick start

```bash
# See what data we have (mirrors MCP get_data_status)
analyst status

# See available SKUs/cities in the mock store
analyst catalog

# Run a forecast
analyst run "Will Face Wash stock out in Mumbai?" \
    --brand HEC --sku FW100-HEC --city Mumbai

# Counterfactual — missed attack window
analyst run "What did we leave on the table last week?" \
    --brand HEC --sku FW100-HEC --city Bengaluru \
    --scenario missed_attack_window

# Causal attribution — why did revenue move
analyst run "Why did Face Wash revenue drop?" \
    --brand HEC --city Mumbai --category "Face Wash"

# JSON output for piping into dashboards
analyst run "Forecast OOS" --sku FW100-HEC --city Mumbai --json
```

## Library use

```python
from analyst import run
from analyst.tools import compute_forecast

# Direct tool call
result = compute_forecast(sku_id="FW100-HEC", city="Mumbai")
print(result.to_dict())

# Full agent
response = run(
    "Will we stock out?",
    brand_id="HEC",
    sku_id="FW100-HEC",
    city="Mumbai",
)
for est in response["estimate_objects"]:
    print(est["headline"])
```

## Layout

```
src/analyst/
├── agent.py            # LangGraph-style runner
├── schemas.py          # AnalystState, *Result dataclasses
├── cli.py              # analyst CLI
├── nodes/
│   ├── classifier.py   # Deterministic intent classifier
│   ├── retriever.py    # MCP data retrieval (mock)
│   ├── computer.py     # Dispatch to the three tools
│   └── narrator.py     # LLM narrator + template fallback
├── tools/
│   ├── forecast.py         # compute_forecast (survival model)
│   ├── counterfactual.py   # compute_counterfactual (3 scenarios)
│   └── attribution.py      # compute_attribution (partial-attribution)
└── data/
    ├── mock.py         # Synthetic Asgard inventory/portal/competitor data
    └── quality.py      # Data quality grader (high / medium / low)
```

## Deployment brief

Follows the six-phase brief from the April 2026 Q-Comm Brain session:

1. Data foundation — BigQuery views + data quality tracker (Architect)
2. Python computation tools (Developer)
3. LangGraph agent wiring (Developer)
4. Cloud Run deployment (Architect + Developer)
5. Backtest validation before live brand exposure

**Hard rule:** the agent never surfaces a ₹ estimate without a confidence label
and a `data_quality_flags` list. Every tool consults
[`quality.grade_quality`](src/analyst/data/quality.py) before emitting a result.

## Roadmap

- [ ] Replace mock data store with real MCP tool calls against BigQuery
- [ ] Swap the in-house graph runner for `langgraph.graph.StateGraph`
- [ ] Add compound-mode orchestration (Causal + Forecast shared state)
- [ ] Backtest harness against 60+ day history with ±3 day accuracy gate
- [ ] Cloud Run deployment + Pub/Sub event trigger from Fixer OOS events

## License

Proprietary — Hector.
