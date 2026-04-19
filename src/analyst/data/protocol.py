"""Data-store protocol.

Both MockDataStore and MCPDataStore conform to this surface. Tools take a
DataStore instance — the concrete class is injected at the edge (CLI / agent
entrypoint). This is the seam that lets us swap Asgard BigQuery for mock data
without touching any tool code.
"""

from __future__ import annotations

from datetime import date
from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class DataStore(Protocol):
    """The contract that every data backend (mock, MCP, live BigQuery) satisfies."""

    history_days: int
    as_of: date

    def forecast_input(self, sku_id: str, city: str) -> pd.DataFrame: ...

    def warehouse_for(self, sku_id: str) -> pd.DataFrame: ...

    def counterfactual_input(
        self, brand_id: str, sku_id: str, city: str
    ) -> pd.DataFrame: ...

    def attribution_input(
        self, brand_id: str, category: str, city: str
    ) -> pd.DataFrame: ...
