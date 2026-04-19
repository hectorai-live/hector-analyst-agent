"""End-to-end demo of the Analyst Agent.

Runs each of the three modes against planted data in the mock store and prints
rich-formatted output. Ideal for product demos / onboarding new engineers.

Usage:
    python scripts/run_demo.py
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyst import run  # noqa: E402
from analyst.data.mock import TODAY  # noqa: E402

console = Console()


def _print_estimate_objects(result: dict) -> None:
    modes = ", ".join(result["modes_triggered"]) or "(none)"
    console.print(f"[dim]modes triggered:[/dim] [yellow]{modes}[/yellow]")
    for est in result["estimate_objects"]:
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_row("[bold]Headline[/bold]", est["headline"])
        table.add_row("[bold]Evidence[/bold]", est["evidence"])
        table.add_row("[bold]Estimate[/bold]", est["estimate"])
        table.add_row("[bold]Confidence[/bold]", est["confidence"])
        table.add_row("[bold]Action[/bold]", est["action"])
        table.add_row("[dim]Methodology[/dim]", f"[dim]{est['methodology']}[/dim]")
        console.print(Panel(table, title=est["source_tool"], border_style="green"))
    if result["errors"]:
        console.print(Panel("\n".join(result["errors"]), title="errors", border_style="red"))


def demo_forecast() -> None:
    console.print(Rule("[bold cyan]1. FORECAST — will face wash stock out in Mumbai?[/bold cyan]"))
    r = run(
        query="will face wash stock out in the next 7 days?",
        brand_id="HEC",
        sku_id="FW100-HEC",
        city="Mumbai",
    )
    _print_estimate_objects(r)


def demo_counterfactual_attack_window() -> None:
    console.print(
        Rule(
            "[bold cyan]2a. COUNTERFACTUAL — missed attack window (Bengaluru)[/bold cyan]"
        )
    )
    r = run(
        query="what would we have made if we had acted on the bengaluru attack window?",
        brand_id="HEC",
        sku_id="FW100-HEC",
        city="Bengaluru",
        scenario="missed_attack_window",
    )
    _print_estimate_objects(r)


def demo_counterfactual_cost_of_delay() -> None:
    console.print(
        Rule("[bold cyan]2b. COUNTERFACTUAL — cost of delay (Mumbai OOS)[/bold cyan]")
    )
    r = run(
        query="what is the cost of delay on the mumbai OOS — what could we have saved?",
        brand_id="HEC",
        sku_id="FW100-HEC",
        city="Mumbai",
        scenario="cost_of_delay",
    )
    _print_estimate_objects(r)


def demo_counterfactual_what_if_osa() -> None:
    console.print(
        Rule("[bold cyan]2c. COUNTERFACTUAL — what-if OSA (Delhi shampoo)[/bold cyan]")
    )
    r = run(
        query="what if OSA had held at 90% in Delhi — we would have made how much more?",
        brand_id="HEC",
        sku_id="SH200-HEC",
        city="Delhi",
        scenario="what_if_osa",
        target_osa=0.9,
    )
    _print_estimate_objects(r)


def demo_attribution() -> None:
    console.print(
        Rule("[bold cyan]3. CAUSAL — why is face wash down in Mumbai?[/bold cyan]")
    )
    # Current window straddles the planted OOS event (last 5 days);
    # comparison window is the 7 healthy days before it.
    start_curr = TODAY - timedelta(days=4)
    end_curr = TODAY
    start_prev = TODAY - timedelta(days=11)
    end_prev = TODAY - timedelta(days=5)
    r = run(
        query="why is face wash revenue down in mumbai last week?",
        brand_id="HEC",
        category="Face Wash",
        city="Mumbai",
        period=f"{start_curr}:{end_curr}",
        comparison_period=f"{start_prev}:{end_prev}",
    )
    _print_estimate_objects(r)


def demo_compound() -> None:
    console.print(
        Rule(
            "[bold cyan]4. COMPOUND — why is it down AND will it get worse?[/bold cyan]"
        )
    )
    start_curr = TODAY - timedelta(days=4)
    end_curr = TODAY
    start_prev = TODAY - timedelta(days=11)
    end_prev = TODAY - timedelta(days=5)
    r = run(
        query=(
            "why is face wash down in mumbai, and will it get worse next 7 days?"
        ),
        brand_id="HEC",
        sku_id="FW100-HEC",
        category="Face Wash",
        city="Mumbai",
        period=f"{start_curr}:{end_curr}",
        comparison_period=f"{start_prev}:{end_prev}",
    )
    _print_estimate_objects(r)


def main() -> int:
    console.print(
        Panel.fit(
            "[bold]Hector Analyst Agent — live demo[/bold]\n"
            f"as_of: [cyan]{TODAY}[/cyan]\n"
            "source: mock Asgard store (deterministic seed=42)",
            border_style="blue",
        )
    )
    demo_forecast()
    demo_counterfactual_attack_window()
    demo_counterfactual_cost_of_delay()
    demo_counterfactual_what_if_osa()
    demo_attribution()
    demo_compound()
    console.print(Rule("[green]demo complete[/green]"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
