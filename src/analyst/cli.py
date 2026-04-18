"""Analyst Agent CLI — `analyst run ...`."""

from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from analyst import run
from analyst.data import get_store
from analyst.data.mock import CATALOG, CITIES

app = typer.Typer(
    add_completion=False,
    help="Hector Q-Comm Analyst Agent — forecasting, counterfactual, causal.",
)
console = Console()


@app.command("run")
def run_cmd(
    query: str = typer.Argument(..., help="Natural-language question for the agent"),
    brand_id: str = typer.Option("HEC", "--brand", "-b"),
    sku_id: str = typer.Option(None, "--sku", help="e.g. FW100-HEC"),
    city: str = typer.Option(None, "--city", help="e.g. Mumbai"),
    category: str = typer.Option(None, "--category", help="e.g. Face Wash"),
    scenario: str = typer.Option(
        None,
        "--scenario",
        help="missed_attack_window | cost_of_delay | what_if_osa",
    ),
    target_osa: float = typer.Option(None, "--target-osa"),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON only"),
) -> None:
    """Run the Analyst Agent against a brand/SKU/city query."""
    result = run(
        query=query,
        brand_id=brand_id,
        sku_id=sku_id,
        city=city,
        category=category,
        scenario=scenario,
        target_osa=target_osa,
    )

    if json_out:
        typer.echo(json.dumps(result, indent=2, default=str))
        return

    modes = ", ".join(result["modes_triggered"]) or "(none)"
    console.print(
        Panel.fit(
            f"[bold]{query}[/bold]\n"
            f"brand=[cyan]{brand_id}[/cyan] "
            f"sku=[cyan]{sku_id}[/cyan] "
            f"city=[cyan]{city}[/cyan]\n"
            f"modes triggered: [yellow]{modes}[/yellow]",
            title="Analyst Agent",
            border_style="blue",
        )
    )

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


@app.command("catalog")
def catalog_cmd() -> None:
    """Show SKUs and cities available in the mock data store."""
    t = Table(title="SKUs")
    t.add_column("sku_id")
    t.add_column("brand")
    t.add_column("category")
    t.add_column("₹ SP")
    for s in CATALOG.values():
        t.add_row(s.sku_id, s.brand_id, s.category, f"{s.selling_price:.0f}")
    console.print(t)

    c = Table(title="Cities")
    c.add_column("city")
    c.add_column("dark stores")
    c.add_column("importance")
    for city in CITIES.values():
        c.add_row(city.city, str(city.n_dark_stores), f"{city.importance_weight:.2f}")
    console.print(c)


@app.command("status")
def status_cmd() -> None:
    """Show data coverage (mirrors get_data_status MCP tool)."""
    store = get_store()
    console.print(
        Panel.fit(
            f"as_of: [cyan]{store.as_of}[/cyan]\n"
            f"history days: [cyan]{store.history_days}[/cyan]\n"
            f"rows (inventory): [cyan]{len(store.inventory):,}[/cyan]\n"
            f"rows (portal_sales): [cyan]{len(store.portal_sales):,}[/cyan]\n"
            f"PSL ready: [green]{'YES' if store.history_days >= 7 else 'NO'}[/green]",
            title="Asgard Data Status (mock)",
            border_style="blue",
        )
    )


if __name__ == "__main__":
    app()
