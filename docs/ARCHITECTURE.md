# Hector Analyst Agent — Architecture

**Audience:** technical architect and data-scientist reviewing the system before
production rollout. This document goes deeper than the README.

---

## 1. What this agent is (and is not)

The Analyst Agent is the third brain in the Hector Q-Comm constellation, sitting
alongside Fixer (execution-playbook automation) and Finder (opportunity
discovery). It is a **statistical computation + LLM narration** system — it
**does not** use an LLM to decide what number is true. The LLM's only job is to
take a grounded numerical result and translate it into a brand-readable
paragraph.

Three modes map 1:1 to three tools:

| Mode            | Business question                               | Tool                    |
|-----------------|-------------------------------------------------|-------------------------|
| Forecast        | Will this SKU go OOS? How much ₹ is at risk?    | `compute_forecast`      |
| Counterfactual  | What would have happened if we'd acted?         | `compute_counterfactual`|
| Causal          | Why did the number move?                        | `compute_attribution`   |

Anything that can be reduced to arithmetic lives in Python. Anything that looks
like a judgement call is explicit and traceable.

---

## 2. Graph shape

Four nodes, one shared `AnalystState`, linear composition.

```
  ┌───────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐
  │ classify  │──▶│ retrieve  │──▶│ compute   │──▶│ narrate   │
  └───────────┘   └───────────┘   └───────────┘   └───────────┘
     regex           MCP pulls       3 tools         Claude +
     (det.)                                          template
```

The `AnalystAgent.invoke` runner wraps each node in a try/except that routes
errors into `state.errors` instead of raising. **The agent can never crash a
caller** — every failure is observable in the returned state.

Why a custom runner instead of `langgraph`? Install footprint: we do not need
the full graph machinery for a linear chain. The runner's signature matches
`StateGraph.invoke`, so the swap to langgraph (for branching, human-in-loop, or
persistent checkpointing on Cloud Run) is a one-line change.

### Node responsibilities

| Node      | Responsibility                                      | Determinism |
|-----------|-----------------------------------------------------|-------------|
| classify  | Parse the query, derive `modes_triggered`           | Deterministic (regex) — **not** LLM |
| retrieve  | Pull slices from the DataStore, record data-quality | Deterministic |
| compute   | Dispatch to one or more statistical tools           | Deterministic |
| narrate   | Turn each result into an `EstimateObject`           | LLM with tool-use schema + template fallback |

The classifier is deliberately not an LLM. A mis-routed intent silently changes
the class of estimate a brand receives, so routing must be stable and
auditable.

---

## 3. The three tools

### 3.1 `compute_forecast`

**Inputs:** `sku_id`, `city`, `horizon_days`
**Outputs:** `ForecastResult` (point estimate, range, survival probabilities at
7 and 14 days, forward PSL, stores at risk / total).

**Algorithm**

1. Pull per-store inventory history for the SKU × city.
2. Demand rate per store = median(units consumed on *available* days).
   (Median, not mean, so a handful of outlier days do not warp the estimate.)
3. Current FE stock per store = latest snapshot.
4. Days-to-OOS per store = stock / demand.
5. Aggregate: `mean_days = mean(days_to_oos_per_store)`;
   `[p10, p90]` for the range.
6. Survival rate: `λ = 1 / max(mean_days, 0.5)`, scaled by
   `0.6 + 0.8 × city_importance_weight`. P(OOS by t) = 1 − exp(−λt).
7. Forward PSL = total_demand_rate × expected_oos_days × stores_at_risk_fraction
   × selling_price.

**Why a survival / exponential CDF?** OOS is a right-censored time-to-event
problem. A first-principles survival curve gives us a calibrated probability
and a natural interval, beats an ad-hoc heuristic, and explicitly carries the
censoring information through to the estimate.

**Known limits**

- First-order: ignores restocks and planned inwarding.
- City-importance scaling of λ is a coarse prior — replace with a learned
  multiplier once we have ≥ 60 days of training history.

### 3.2 `compute_counterfactual`

Three scenarios, all answering "what revenue would have been in a world where
X was different":

| Scenario                | What's different                         | Driver                                 |
|-------------------------|------------------------------------------|----------------------------------------|
| `missed_attack_window`  | Brand acted during competitor weakness   | baseline_units × window × (1 + uplift) |
| `cost_of_delay`         | Brand recovered on detectable-OOS day    | healthy-days baseline × window len     |
| `what_if_osa`           | OSA held at `target_osa`                 | per-store OSA gap × available-day demand |

The SOV uplift prior (0.35) during competitor OOS is an empirical placeholder —
calibrate from portal data once we have ≥ 3 completed attack windows per city.

**Delta signs are always from the brand's point of view**:
`delta_rs > 0` means "we left this much on the table / we could have captured
this much more".

### 3.3 `compute_attribution`

Partial-attribution decomposition across four buckets: availability, visibility,
pricing, competitor.

For each driver, we compute the revenue that *would have obtained* if that
driver had held at its comparison-period value while all others moved as
observed. The difference versus actual current-period revenue is the driver's
partial delta.

Response model (intentionally simple for V1):

```
revenue ≈ base × (osa / osa_ref) × (sov / sov_ref) ×
          (price_ref / price) × (1 + 0.4(1 − competitor_osa))
```

The four partial deltas are renormalised so they sum to the observed
period-over-period revenue change. This is a first-order
Shapley-style approximation — exact Shapley would require 2⁴ = 16 coalitions per
window; V1 does 4. When the four drivers are bounded and roughly independent
(which they are, by construction: they are derived from different Asgard
pipelines), the first-order approximation is close enough.

**V2 direction:** true Shapley with Monte-Carlo sampling (16 evaluations × N
samples). Trivial extension — the `_revenue_under` helper already accepts
arbitrary override sets.

---

## 4. Data access layer

```
                ┌──────────────────────────────┐
                │     analyst.data.DataStore   │   Protocol (typed)
                │     ─────────────────────    │
                │  forecast_input()            │
                │  warehouse_for()             │
                │  counterfactual_input()      │
                │  attribution_input()         │
                └──────────────┬───────────────┘
                               │
                ┌──────────────┼──────────────────┐
         MockDataStore              MCPDataStore
        (src/analyst/data/       (src/analyst/data/
           mock.py)                mcp_adapter.py)
       synthetic 45d             binds to Q-Comm
       seeded rng                 MCP server tools
```

Tools do not know which backend they are running against. This is the single
most important architectural property of the system — it is why swapping
BigQuery for a cache, or adding a Snowflake backend, is a matter of adding one
file, not rewriting tools.

### MCP adapter semantics

- **Strictly read-only.** No write tools invoked, no mutation semantics.
- `caller` is injected at construction. In production this binds to the MCP
  client inside a LangGraph MCP node or the Anthropic tool-use loop.
- Response shapes are normalised into MockDataStore's column conventions so
  tool code is identical.
- Empty / None responses flow through without raising — the downstream quality
  grader flags `empty_dataset` and the tool emits a low-confidence result.

### Data-quality grader

Every tool calls `grade_quality` **before** emitting a result. It grades to
`high / medium / low` based on:

- History length vs. tool minimum (14 / 21 / 21 days)
- Store-coverage fraction
- Restock / inwarding noise proxy

It always returns a list of machine-readable flags. The Estimate Object
carries both the grade and the flags so a downstream consumer (Fixer, War
Room dashboard) can filter on them.

---

## 5. The narrator — why the LLM cannot lie

The narrator path is designed so the LLM is physically incapable of inventing
a number.

1. **Structured tool-use.** The narrator calls Anthropic with a strict
   `emit_estimate` tool schema. The model returns a typed object, not
   free-form JSON-in-text. The client never parses model prose.
2. **System prompt discipline.** The prompt explicitly forbids introducing
   numbers that are not in the input object; it must prefer "unknown" over
   estimation.
3. **Template fallback.** When `ANTHROPIC_API_KEY` is absent, or the LLM
   call fails, we fall through to a deterministic template that can only
   interpolate from the computation result. This is the CI / offline path.
4. **Grounding sanity check.** Unit tests assert that the headline contains
   a numerical string that also appears in the computation result.

If product ever asks "can we get the LLM to be more specific about ₹ numbers"
the answer is no — that is where credibility goes to die.

Model: `claude-sonnet-4-6`. Rationale: narrator output is short (a few hundred
tokens), must be cheap, and doesn't require deep reasoning; Sonnet 4.6 hits the
right latency/cost/quality point.

---

## 6. Invariants

The review bar for any PR that touches this repo is: does it preserve these?

1. **Every tool result carries a `confidence` grade and `data_quality_flags` list.**
   Asserted by `test_*_always_has_confidence_and_flags`.
2. **`AnalystAgent.invoke` never raises.** All errors captured in
   `state.errors`. Asserted by `test_invoke_never_raises_on_missing_fields`.
3. **The classifier is deterministic.** Asserted by
   `test_classifier_is_deterministic`.
4. **Narrator output is grounded in numbers from the computation result.**
   Asserted by `test_narrator_grounded_in_numbers`.
5. **Attribution shares sum to 1.0 (± 2 %) and partials sum to total delta.**
   Asserted by `test_attribution_shares_sum_to_one` and
   `test_attribution_deltas_sum_to_total`.
6. **The forecast tool is calibrated.** Asserted by `scripts/backtest.py`
   (Phase 5 gate: ≥ 4 of 5 combos within ±3 days of observed).

---

## 7. Repository map

```
src/analyst/
    agent.py              # linear node runner (LangGraph-compatible signature)
    schemas.py            # AnalystState + every Result dataclass
    cli.py                # `analyst run|status|catalog` (Typer + Rich)
    nodes/
        classifier.py     # deterministic regex router
        retriever.py      # DataStore gate, quality attachment
        computer.py       # dispatch to compute_* tools
        narrator.py       # Anthropic tool-use + template fallback
    tools/
        forecast.py
        counterfactual.py
        attribution.py
    data/
        protocol.py       # DataStore Protocol
        mock.py           # synthetic 45-day store (seed=42)
        mcp_adapter.py    # Q-Comm MCP adapter (production)
        quality.py        # confidence grader + flag emission
tests/                    # pytest suite — 36+ tests, full tool + agent coverage
scripts/
    backtest.py           # Phase 5 calibration gate
    run_demo.py           # end-to-end demo
docs/
    ARCHITECTURE.md       # this file
    product_session_2026-04-19.txt  # source of truth for product intent
```

---

## 8. Deployment path

**V0 (current, local):** `pip install -e '.[dev]'` → `analyst run ...` against
mock store. CI runs `make ci` (ruff + pytest + backtest).

**V1 (production):** Deploy as a Cloud Run service. LangGraph `StateGraph`
replaces the custom runner. `MCPDataStore` is bound to the Q-Comm MCP client.
The agent receives queries from the Fixer / Finder orchestrator or the War
Room UI. Narrator uses the real Anthropic API.

**V2 (learning loop):**
- Backtest runs nightly against the previous 14 days; drift beyond ±3 days
  triggers a model refit.
- Attribution uplift priors (`SOV_UPLIFT_WHEN_COMPETITOR_OOS`) become
  per-category learned constants maintained in a config store.
- Full Shapley attribution when we have ≥ 60 days of portal history per
  (brand × category × city).

---

## 9. What to do when something breaks

| Symptom                                   | Most likely cause                                    | Fix                                                                 |
|-------------------------------------------|------------------------------------------------------|---------------------------------------------------------------------|
| Every result comes back low-confidence    | History too short, or `empty_dataset` on all pulls   | Check MCP `get_data_status`; extend lookback                        |
| Narrator invents a ₹ number               | Should be impossible. File a P0. Check tool-use schema. | Re-verify `emit_estimate` is passed on every LLM call               |
| Forecast off by >3 days                   | Demand regime shift, or restock pattern changed      | Re-run `scripts/backtest.py`; if fails, refit λ scaling             |
| Attribution shares don't sum to 1         | A driver signal is NaN or zero in one period         | Inspect `_signals` output; quality grader should already have flagged |
| Agent raises at the CLI                   | A node threw outside try/except                      | Run `analyst run ... --json`; inspect `errors` array                |

---

Last updated: 2026-04-19
