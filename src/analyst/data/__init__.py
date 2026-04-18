"""Data access layer — mock MCP/Asgard retrieval.

In production this module is replaced by the existing MCP tools that read from
BigQuery. The interface surface here is the contract those tools must satisfy.
"""

from analyst.data.mock import MockDataStore, get_store
from analyst.data.quality import grade_quality

__all__ = ["MockDataStore", "get_store", "grade_quality"]
