# Project: Hector Q-Comm Analyst Agent

This file is the handoff brief for any Claude session picking up this project
(local, web, or spawned subagent). Read it first.

## What this project is

Implementation of the **Analyst Agent** described in the April 19, 2026 Q-Comm
Brain product session. Full transcript is in
[docs/product_session_2026-04-19.txt](docs/product_session_2026-04-19.txt) —
that is the source of truth for product intent.

The agent sits next to Hector's existing Fixer and Finder agents and is the
third "brain": **statistical computation + LLM reasoning**, not LLM-driven gate
logic. It turns MCP/Asgard data into probabilistic estimates in three modes:

| Mode | Question | Tool |
|---|---|---|
| Forecast | Will this SKU go OOS? How much ₹ at risk? | `compute_forecast` |
| Counterfactual | What would have happened if we'd acted? | `compute_counterfactual` |
| Causal | Why did the number move? | `compute_attribution` |

## Architecture (from Turn 5 of the transcript)

```
Entry → Intent Classifier (deterministic) → Data Retrieval (MCP tools)
      → Computation Node (Python statistical tools)
      → LLM Narrator (grounded — never invents numbers)
      → EstimateObject { headline, evidence, estimate, confidence, action, methodology }
```

Four nodes, one shared `AnalystState`. Source files:

- `src/analyst/agent.py` — graph runner
- `src/analyst/schemas.py` — `AnalystState`, `ForecastResult`, `CounterfactualResult`, `AttributionResult`, `EstimateObject`, `Mode`
- `src/analyst/nodes/{classifier,retriever,computer,narrator}.py`
- `src/analyst/tools/{forecast,counterfactual,attribution}.py`
- `src/analyst/data/{mock,quality}.py`
- `src/analyst/cli.py` — `analyst` CLI
- `tests/` — pytest suite (to be added)
- `scripts/backtest.py` — Phase 5 validation (to be added)

## Current state (as of last commit)

**Built and committed:**
- Project scaffold — `pyproject.toml`, `README.md`, `.gitignore`, package layout
- Schemas — all result dataclasses + `AnalystState`
- Mock Asgard data store — synthetic inventory / portal sales / competitor /
  warehouse / ad spend across 4 SKUs × 4 cities × ~76 dark stores × 45 days,
  seeded for reproducibility. Includes a planted OOS event (Face Wash, Mumbai,
  last 5 days) and a planted competitor attack window (Bengaluru Face Wash,
  days 8–10 back) so tools produce meaningful output immediately.
- Data quality grader — grades `high` / `medium` / `low` based on history
  length, store coverage, restock noise. **Hard rule**: every tool consults
  this before emitting a result. No silent low-confidence outputs.
- `compute_forecast` — median demand on available days → per-store
  days-to-OOS → exponential survival model on λ = 1 / mean(days_to_oos),
  city-importance scaled → forward PSL.
- `compute_counterfactual` — three scenarios (`missed_attack_window`,
  `cost_of_delay`, `what_if_osa`).
- `compute_attribution` — partial-attribution decomposition across availability
  / visibility / pricing / competitor, renormalised to the observed
  period-over-period delta. First-order Shapley-style approximation.
- LangGraph-style agent — minimal in-house runner (no `langgraph` dep to keep
  install lightweight; interface matches `StateGraph.invoke` so it can be
  swapped without touching nodes).
- Classifier — deterministic regex-based intent router (forecast / counterfactual
  / causal), per explicit instruction in the brief.
- Narrator — Anthropic Claude Sonnet 4.6 with **template fallback** when no
  `ANTHROPIC_API_KEY` is set, so the agent is runnable offline and in CI.
- Typer CLI — `analyst run`, `analyst status`, `analyst catalog`.

**Not yet done (pick up here):**
1. `scripts/backtest.py` — Phase 5 validation. Pick 5 SKU × city combos with
   full history, hide last 14 days, run `compute_forecast` on day 46, compare
   predicted OOS timing against actual. Pass criterion: predicted OOS within
   ±3 days in ≥ 4 of 5 cases.
2. `scripts/run_demo.py` — end-to-end demo that exercises all three tools and
   prints rich-formatted output.
3. `tests/` — pytest suite covering each tool + the agent end-to-end. Must
   assert: (a) every result carries a confidence grade, (b) data_quality_flags
   is always a list, (c) classifier routes compound queries to multiple modes,
   (d) agent invoke() never raises — errors go into `state.errors`.
4. Local install + smoke test: `pip install -e '.[dev]'`, `pytest`, `analyst
   run "will face wash stock out?" --sku FW100-HEC --city Mumbai`.
5. `docs/ARCHITECTURE.md` — deeper dive than README for the architect/developer
   audience described in Turn 6.
6. Replace mock data store with real MCP tool calls (`get_inventory_snapshot`,
   `get_competitor_visibility`, `get_sales_estimate`, `get_data_status`) — the
   MCP server at `12ce7690-7939-4322-83b5-5173bc48e3c9` already exposes these.

## Critical invariants (do not violate)

From the deployment brief — Turn 6, end:

> The agent never surfaces a ₹ estimate without a confidence label and a
> data_quality_flags list. A confident wrong number is worse than an honest
> uncertain one.

Every tool returns a result with both fields. The LLM narrator is grounded —
it **cannot invent numbers**. If a Claude session is tempted to have the LLM
compute anything, push it back into a Python tool.

The classifier is **deterministic, not LLM**. Explicit in Turn 5.

## Data model notes

From the session Turn 7 — as of 2026-04-19 we have 18 days of Asgard history.
Against tool minima:

- `compute_forecast` needs 14 days → **OK (18)**
- `compute_counterfactual` needs 21 days → **short by 3 days, flag as low confidence**
- `compute_attribution` needs 21 days → **same**

The mock data store uses 45 days by default (well past minima) so demo output
is clean. When swapping to real MCP, the quality grader correctly degrades
confidence when history is short.

Known real-world data issue (per Turn 7): April 16 had a double-scrape. Factor
that out in the real MCP integration path.

## Domain glossary

- **PSL** = Potential Sales Loss. The ₹ figure for revenue lost to OOS.
- **OSA** = On-Shelf Availability (% of dark stores where SKU is available).
- **SOV** = Share of Voice (organic + paid visibility share in a keyword result).
- **DOI** = Days of Inventory at the warehouse.
- **FE / BE** = Front-End / Back-End warehouse stock.
- **Dark store** = small last-mile fulfilment warehouse used by Blinkit/Zepto/Instamart.
- **Asgard** = Hector's scraping pipeline; lands in BigQuery.
- **PartnersBiz** = brand-side portal (gated, requires brand integration).
- **Fixer / Finder** = two existing agents; Analyst is the third.

## Stack

- Python 3.11+
- numpy, pandas, scipy for computation
- pydantic for validation (not heavily used yet)
- typer + rich for CLI
- anthropic SDK for narrator (optional)
- pytest + pytest-cov for tests

**No langgraph dependency** — intentional. The in-house runner is a stand-in
with the same interface. Add langgraph only when deploying to Cloud Run.

## How to continue this project (web session)

1. Clone the repo.
2. `pip install -e '.[dev]'`.
3. Verify current state: `analyst status && analyst run "will face wash stock
   out" --sku FW100-HEC --city Mumbai` — should print a forecast panel.
4. Pick up the "Not yet done" list above in order. Items 1–4 are the immediate
   path to a green CI + a runnable end-to-end demo.
5. When in doubt about *product* intent, re-read Turns 4–6 of
   [docs/product_session_2026-04-19.txt](docs/product_session_2026-04-19.txt).
   When in doubt about *system* architecture, read this file and the README.

## Commit conventions

- Commit messages should be one-line summary + optional body.
- Include `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`
  trailer on Claude-authored commits.
- Push to the remote for this repo after each logical chunk of work lands.

## Contacts

- GitHub account authed locally: `hectoruma`
- Project namespace target: `hectorlive`
