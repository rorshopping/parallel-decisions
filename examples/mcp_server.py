#!/usr/bin/env python3
"""MCP server: expose `decide(context, schema)` as a tool for coding agents.

The point is everyday use: an agent (Claude Code, OpenCode, Cursor) can call a local
typed decision instead of asking a frontier model to emit JSON — no API key, no
network, and the answer is a schema-validated value with a confidence.

Requires the MCP SDK, which is deliberately **not** a package dependency:

    .venv/bin/pip install mcp
    .venv/bin/python examples/mcp_server.py            # stdio, one model per process

Then register it. OpenCode (`~/.config/opencode/opencode.json`):

    {"mcp": {"pd": {"type": "local", "command": [".venv/bin/python", "examples/mcp_server.py"]}}}

Claude Code:

    claude mcp add pd -- .venv/bin/python examples/mcp_server.py

Tools exposed:

- `decide(context, schema)` -> values + calibrated probabilities per field
- `validate_schema(schema)` -> token collisions and rename advice, no model needed

The model loads on the first call (~5 s, then cached), so the first tool call is
slower than the rest.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from parallel_decisions import Decider, Schema, SchemaError  # noqa: E402

DECIDER: Decider | None = None


def _decider() -> Decider:
    global DECIDER
    if DECIDER is None:
        DECIDER = Decider(calibration=os.environ.get("PD_CALIBRATION"))
    return DECIDER


def _as_schema(schema: dict) -> Schema:
    """Accept both `{name: spec}` and `{"fields": {name: spec}}`, which is what
    `Schema.to_json()` writes and what agents tend to paste."""
    if isinstance(schema, dict) and isinstance(schema.get("fields"), (dict, list)):
        schema = schema["fields"]
    return Schema(schema)


def decide_tool(context: str, schema: dict) -> str:
    """Answer a schema of typed questions against a context.

    Args:
        context: The text to decide on (ticket, invoice excerpt, email, log).
        schema: {"fields": {name: {"type": "boolean"|"enum"|"multi",
                 "choices": [...] | {choice: description}, "description": "..."}}}.
                 Descriptions and per-choice definitions measurably improve accuracy.

    Returns:
        JSON: per-field value, calibrated probability, and alternatives.
    """
    try:
        result = _decider().decide(context, _as_schema(schema))
    except (SchemaError, ValueError) as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})
    return json.dumps({
        "decisions": result.full_json(),
        "calibrated": result.calibrated,
        "model": result.model,
        "latency_ms": round(result.latency_ms, 1),
    }, default=str)


def validate_schema_tool(schema: dict) -> str:
    """Check a schema for token collisions and indistinguishable answers.

    Args:
        schema: The same schema shape `decide` takes.

    Returns:
        JSON: colliding fields with rename suggestions. No model load.
    """
    from parallel_decisions.lint import lint_schema

    try:
        report = lint_schema(_as_schema(schema), _decider().tokenizer_for_schema)
    except (SchemaError, ValueError) as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})
    return json.dumps(report.to_dict(), default=str)


def main() -> int:
    try:
        from mcp.server.fastmcp import FastMCP
    except ModuleNotFoundError:
        print("this example needs the MCP SDK:  .venv/bin/pip install mcp",
              file=sys.stderr)
        return 2

    server = FastMCP("parallel-decisions")
    server.tool()(decide_tool)
    server.tool()(validate_schema_tool)
    server.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
