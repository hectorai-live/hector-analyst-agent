# Hector Q-Comm Analyst Agent

Statistical computation engine for Hector's Q-Comm Brain. Turns MCP/Asgard data
into probabilistic estimates across three modes:

| Mode | Question it answers | Tool |
|---|---|---|
| **Forecast** | *Will this SKU go OOS? When? How much ₹ at risk?* | `compute_forecast` |
| **Counterfactual** | *What would revenue have been if we'd acted / held OSA / caught the competitor window?* | `compute_counterfactual` |
| **Causal** | *Why did revenue move? How much is availability vs visibility vs pricing vs competitor?* | `compute_attribution` |
| **Denoise** | *What's the true demand signal once restock noise is filtered?* | `denoise_signal` |
| **Anomaly** | *Which days are statistically-significant attack windows / stealth-OOS / promo spikes?* | `detect_anomalies` |
| **Changepoint** | *When did the regime actually shift?* | `detect_changepoints` |
| **Elasticity** | *How sensitive is demand to price (with 95% CI)?* | `estimate_elasticity` |

The last four are things **neither Claude nor MCP can do** standalone —
statistical inference (EM mixtures, robust z/CUSUM, PELT dynamic programming,
log-log OLS with fixed effects) that MCP doesn't expose and that Claude can't
reproduce in prose.

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
│   ├── retriever.py    # DataStore gate + quality attachment
│   ├── computer.py     # Dispatch to the three tools
│   └── narrator.py     # LLM narrator (tool-use) + template fallback
├── tools/
│   ├── forecast.py         # compute_forecast (survival model)
│   ├── counterfactual.py   # compute_counterfactual (3 scenarios)
│   └── attribution.py      # compute_attribution (partial-attribution)
└── data/
    ├── protocol.py     # DataStore Protocol (swap backends without touching tools)
    ├── mock.py         # Synthetic 45-day Asgard store (seed=42)
    ├── mcp_adapter.py  # Q-Comm MCP adapter (production path)
    └── quality.py      # Data-quality grader (high / medium / low)
scripts/
├── run_demo.py         # end-to-end demo across all three modes
└── backtest.py         # Phase 5 forecast calibration gate
tests/                  # 36+ pytest cases; pass-gate for every PR
docs/
└── ARCHITECTURE.md     # deeper dive for architect/developer audiences
```

## Development workflow

```bash
make install          # pip install -e '.[dev]'
make test             # pytest
make lint             # ruff check
make demo             # scripts/run_demo.py — prints all three modes
make backtest         # scripts/backtest.py — calibration gate (±3d in 4/5)
make live-validation  # scripts/live_validation.py — 4 new agents on real Foxtale data
make dashboard        # scripts/build_dashboard.py — writes docs/dashboard.html
make ci               # lint + test + backtest
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

- [x] Compound-mode orchestration (Causal + Forecast shared state) — see
      `AnalystAgent.invoke` with multi-mode `modes_triggered`.
- [x] Backtest harness with ±3-day accuracy gate — `scripts/backtest.py`
      (currently passing 5 of 5 against the mock store).
- [x] MCP data adapter (`analyst.data.mcp_adapter.MCPDataStore`) — wire to the
      Q-Comm MCP client at deploy time.
- [ ] Swap the in-house graph runner for `langgraph.graph.StateGraph` in prod.
- [ ] Cloud Run deployment + Pub/Sub event trigger from Fixer OOS events.
- [ ] Full-Shapley attribution (16 coalitions) once ≥ 60d history per
      brand × category × city is available.

## License

Proprietary — Hector.
