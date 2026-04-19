"""Data access layer — pluggable backends.

The Protocol in `protocol.py` defines the contract; `mock.MockDataStore` is the
offline/CI implementation; `mcp_adapter.MCPDataStore` is the production path
against the Hector Q-Comm MCP server.
"""

from analyst.data.mcp_adapter import MCPDataStore, build_mcp_store
from analyst.data.mock import MockDataStore, get_store
from analyst.data.protocol import DataStore
from analyst.data.quality import grade_quality

__all__ = [
    "DataStore",
    "MockDataStore",
    "MCPDataStore",
    "build_mcp_store",
    "get_store",
    "grade_quality",
]
