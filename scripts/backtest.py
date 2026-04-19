"""Phase 5 validation — forecast backtest.

Pick 5 SKU × city combos with full history. Hide the last 14 days. Run
`compute_forecast` at the cut-point. Compare predicted days-to-OOS against the
first *observed* OOS day (or censor at 14 if no OOS seen). Pass criterion:
predicted point estimate within ±3 days of observed in ≥ 4 of 5 cases.

Usage:
    python scripts/backtest.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyst.data.mock import TODAY, get_store  # noqa: E402
from analyst.tools import compute_forecast  # noqa: E402

console = Console()

TOLERANCE_DAYS = 3
HOLDOUT_DAYS = 14
PASS_THRESHOLD = 4  # of 5 combos

COMBOS: list[tuple[str, str]] = [
    ("FW100-HEC", "Mumbai"),
    ("FW100-HEC", "Delhi"),
    ("SH200-HEC", "Bengaluru"),
    ("BL250-HEC", "Pune"),
    ("FW100-MAE", "Mumbai"),
]


@dataclass
class BacktestCase:
    sku_id: str
    city: str
    predicted_days: float
    observed_days: float | None  # None = no OOS observed in hold-out
    within_tolerance: bool
    flags: list[str]


def _mean_days_to_zero(
    inv: pd.DataFrame, sku_id: str, city: str, cut: date, horizon: int
) -> float | None:
    """For each dark store, compute days until cumulative post-cut consumption
    exceeds the cut-day inventory — i.e. no-restock depletion, matching the
    predictor's assumption. Stores that would not deplete within `horizon`
    are censored at `horizon`. Returns mean across stores."""
    future = inv[
        (inv["sku_id"] == sku_id)
        & (inv["city"] == city)
        & (inv["snapshot_date"] > cut)
        & (inv["snapshot_date"] <= cut + timedelta(days=horizon))
    ].copy()
    cut_snapshot = inv[
        (inv["sku_id"] == sku_id)
        & (inv["city"] == city)
        & (inv["snapshot_date"] == cut)
    ].set_index("dark_store_id")["inventory_level"]
    if future.empty or cut_snapshot.empty:
        return None
    future = future.sort_values(["dark_store_id", "snapshot_date"])
    days_list: list[float] = []
    for ds, grp in future.groupby("dark_store_id"):
        start_stock = cut_snapshot.get(ds, 0)
        if start_stock <= 0:
            days_list.append(0.0)
            continue
        cum = 0
        depleted_at: float | None = None
        for i, (_, row) in enumerate(grp.iterrows(), start=1):
            cum += row["units_consumed"]
            if cum >= start_stock:
                depleted_at = float(i)
                break
        days_list.append(depleted_at if depleted_at is not None else float(horizon))
    if not days_list:
        return None
    return sum(days_list) / len(days_list)


def _build_cut_store(full_history: int):
    """Build a store whose 'history' ends 14 days before TODAY."""
    from analyst.data.mock import MockDataStore, _build_ad_spend, _build_competitor
    from analyst.data.mock import _build_inventory, _build_portal_sales, _build_warehouse
    from analyst.data.mock import _date_range
    import numpy as np

    cut_as_of = TODAY - timedelta(days=HOLDOUT_DAYS)
    history = full_history - HOLDOUT_DAYS
    rng = np.random.default_rng(42)  # same seed as default store
    dates = _date_range(history, cut_as_of)
    inventory = _build_inventory(rng, dates, inject_oos_event=True)
    warehouse = _build_warehouse(rng, dates)
    competitor = _build_competitor(rng, dates)
    ad_spend = _build_ad_spend(rng, dates)
    portal_sales = _build_portal_sales(rng, dates, inventory)
    return MockDataStore(
        inventory=inventory,
        portal_sales=portal_sales,
        competitor=competitor,
        warehouse=warehouse,
        ad_spend=ad_spend,
        history_days=history,
        as_of=cut_as_of,
    )


def run_backtest() -> list[BacktestCase]:
    cut_store = _build_cut_store(full_history=45)
    full_store = get_store()
    cut_date = cut_store.as_of
    results: list[BacktestCase] = []
    for sku_id, city in COMBOS:
        r = compute_forecast(sku_id=sku_id, city=city, store=cut_store, horizon_days=HOLDOUT_DAYS)
        predicted = r.days_to_oos_point_estimate
        observed = _mean_days_to_zero(
            full_store.inventory, sku_id, city, cut_date, HOLDOUT_DAYS
        )
        if observed is None:
            within = predicted > HOLDOUT_DAYS
        else:
            within = abs(predicted - observed) <= TOLERANCE_DAYS
        results.append(
            BacktestCase(
                sku_id=sku_id,
                city=city,
                predicted_days=predicted,
                observed_days=observed,
                within_tolerance=within,
                flags=r.data_quality_flags,
            )
        )
    return results


def main() -> int:
    console.print(
        Panel.fit(
            f"Cut point: [cyan]TODAY - {HOLDOUT_DAYS}d[/cyan] "
            f"([cyan]{TODAY - timedelta(days=HOLDOUT_DAYS)}[/cyan])\n"
            f"Tolerance: ±{TOLERANCE_DAYS} days\n"
            f"Pass threshold: ≥{PASS_THRESHOLD} of {len(COMBOS)}",
            title="Forecast Backtest",
            border_style="blue",
        )
    )

    results = run_backtest()

    table = Table(title="Backtest Results")
    table.add_column("SKU")
    table.add_column("City")
    table.add_column("Predicted (d)", justify="right")
    table.add_column("Observed (d)", justify="right")
    table.add_column("Δ (d)", justify="right")
    table.add_column("Within ±3?", justify="center")
    table.add_column("Flags")

    for r in results:
        obs_str = f"{r.observed_days:.1f}" if r.observed_days is not None else "none (censored)"
        delta = (
            f"{r.predicted_days - r.observed_days:+.1f}"
            if r.observed_days is not None
            else "—"
        )
        mark = "[green]✓[/green]" if r.within_tolerance else "[red]✗[/red]"
        table.add_row(
            r.sku_id,
            r.city,
            f"{r.predicted_days:.1f}",
            obs_str,
            delta,
            mark,
            ", ".join(r.flags) or "-",
        )
    console.print(table)

    passed = sum(1 for r in results if r.within_tolerance)
    ok = passed >= PASS_THRESHOLD
    verdict = "[green]PASS[/green]" if ok else "[red]FAIL[/red]"
    console.print(
        Panel.fit(
            f"Passed: [bold]{passed}[/bold] of {len(results)}  →  {verdict}",
            border_style="green" if ok else "red",
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
