"""A standalone, read-only MCP server for the retail desk. It runs on its own:

    uv run python -m mcp_server [--port 8765]

It imports nothing from `app`: it reads PostgreSQL with its own read-only SQL and applies the shared
pure rules in `retail_domain`. It writes nothing, not even audit rows, which the calling host records.
"""
