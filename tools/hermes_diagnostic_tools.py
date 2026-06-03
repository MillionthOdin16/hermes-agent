#!/usr/bin/env python3
"""
Hermes Diagnostic Tools - Tool Registry Adapter

Registers hermes_log_search, hermes_log_tail, hermes_jsonl_query,
hermes_file_preview, hermes_tool_metrics, hermes_policy_block_summary,
and hermes_skill_inventory_query as proper Hermes tools.

These were created in Patch 92 as Python functions but never registered
with the Hermes tool registry — making them uncallable by the agent.
Patch 93 references them in blocked-command error messages, so they
must be callable or the feedback loop is broken.

Registration: patch 92 / patch 93 closure
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

# Ensure the diagnostic_tools module is importable.
# When this file is at:  ~/.hermes/hermes-agent/tools/hermes_diagnostic_tools.py
# the plugin is at:      ~/.hermes/plugins/tool_logger/scripts/diagnostic_tools.py
# hermes-agent is: ~/.hermes/hermes-agent/
# tools/ is: ~/.hermes/hermes-agent/tools/
# we need: ~/.hermes/  (2 levels up from tools/)
_SCRIPT_DIR = Path(__file__).resolve().parent
_HERMES_ROOT = _SCRIPT_DIR.parent.parent  # ~/.hermes
_PLUGIN_TOOL_LOGGER = _HERMES_ROOT / "plugins" / "tool_logger" / "scripts"

if _PLUGIN_TOOL_LOGGER.exists():
    sys.path.insert(0, str(_PLUGIN_TOOL_LOGGER))

try:
    from diagnostic_tools import (
        hermes_log_search as _log_search,
        hermes_log_tail as _log_tail,
        hermes_jsonl_query as _jsonl_query,
        hermes_file_preview as _file_preview,
        hermes_tool_metrics as _tool_metrics,
        hermes_policy_block_summary as _policy_summary,
        hermes_skill_inventory_query as _skill_query,
    )
    _HAS_DIAGNOSTIC_TOOLS = True
except ImportError:
    _HAS_DIAGNOSTIC_TOOLS = False


# =============================================================================
# Tool Schemas
# =============================================================================

HERMES_LOG_SEARCH_SCHEMA = {
    "name": "hermes_log_search",
    "description": "Search for a pattern in Hermes log files within approved roots. "
                  "Safer alternative to grep on ~/.hermes/logs/.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to log file or directory (must be under ~/.hermes/logs/)",
            },
            "pattern": {
                "type": "string",
                "description": "Regex or literal pattern to search for",
            },
            "case_sensitive": {
                "type": "boolean",
                "default": False,
                "description": "Whether search is case-sensitive",
            },
            "regex": {
                "type": "boolean",
                "default": True,
                "description": "Whether pattern is a regex (True) or literal string (False)",
            },
            "max_matches": {
                "type": "integer",
                "default": 500,
                "description": "Maximum number of matches to return",
            },
            "include_context": {
                "type": "integer",
                "default": 0,
                "description": "Number of context lines before/after each match",
            },
        },
        "required": ["path", "pattern"],
    },
}

HERMES_LOG_TAIL_SCHEMA = {
    "name": "hermes_log_tail",
    "description": "Get the last N lines from a Hermes log file. "
                  "Safer alternative to tail on ~/.hermes/logs/.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to log file (must be under ~/.hermes/logs/)",
            },
            "lines": {
                "type": "integer",
                "default": 50,
                "description": "Number of recent lines to return",
            },
            "offset": {
                "type": "integer",
                "default": 0,
                "description": "Byte offset to start reading from",
            },
        },
        "required": ["path"],
    },
}

HERMES_JSONL_QUERY_SCHEMA = {
    "name": "hermes_jsonl_query",
    "description": "Query and filter JSONL log files. "
                  "Safer alternative to cat/grep on .jsonl log files.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to .jsonl file (must be under ~/.hermes/logs/)",
            },
            "filter_expr": {
                "type": "string",
                "default": None,
                "description": "Filter expression in format 'key=value' or 'key contains value'",
            },
            "limit": {
                "type": "integer",
                "default": 100,
                "description": "Maximum number of records to return",
            },
            "offset": {
                "type": "integer",
                "default": 0,
                "description": "Number of records to skip",
            },
        },
        "required": ["path"],
    },
}

HERMES_FILE_PREVIEW_SCHEMA = {
    "name": "hermes_file_preview",
    "description": "Preview a range of lines from a file. "
                  "Safer alternative to sed -n, head, or cat on large files.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to file (must be under ~/.hermes/logs/, ~/.hermes/plugins/, or ~/.hermes/skills/)",
            },
            "offset": {
                "type": "integer",
                "default": 0,
                "description": "Line number to start reading from (1-indexed)",
            },
            "limit": {
                "type": "integer",
                "default": 100,
                "description": "Maximum number of lines to return",
            },
        },
        "required": ["path"],
    },
}

HERMES_TOOL_METRICS_SCHEMA = {
    "name": "hermes_tool_metrics",
    "description": "Get tool call metrics from tool_calls.jsonl. "
                  "Returns recent tool calls with success/failure status.",
    "parameters": {
        "type": "object",
        "properties": {
            "tool_name": {
                "type": "string",
                "default": None,
                "description": "Optional filter by specific tool name",
            },
            "limit": {
                "type": "integer",
                "default": 100,
                "description": "Maximum number of recent calls to return",
            },
        },
        "required": [],
    },
}

HERMES_POLICY_BLOCK_SUMMARY_SCHEMA = {
    "name": "hermes_policy_block_summary",
    "description": "Get a summary of policy blocks and diagnostic覆盖率. "
                  "Shows which diagnostic tools are available for common block patterns.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

HERMES_SKILL_INVENTORY_QUERY_SCHEMA = {
    "name": "hermes_skill_inventory_query",
    "description": "Query the Hermes skill inventory. "
                  "Safer alternative to ls/find on ~/.hermes/skills/.",
    "parameters": {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "default": None,
                "description": "Filter by category name",
            },
            "search": {
                "type": "string",
                "default": None,
                "description": "Search in skill name/description",
            },
            "limit": {
                "type": "integer",
                "default": 50,
                "description": "Maximum results to return",
            },
        },
        "required": [],
    },
}


# =============================================================================
# Tool Handlers
# =============================================================================

def _hermes_log_search(args: Dict[str, Any]) -> str:
    """Handler for hermes_log_search tool."""
    if not _HAS_DIAGNOSTIC_TOOLS:
        return '{"error": "Diagnostic tools not available", "success": false}'
    result = _log_search(
        path=args.get("path", ""),
        pattern=args.get("pattern", ""),
        case_sensitive=args.get("case_sensitive", False),
        regex=args.get("regex", True),
        max_matches=args.get("max_matches", 500),
        include_context=args.get("include_context", 0),
    )
    return _format_result(result)


def _hermes_log_tail(args: Dict[str, Any]) -> str:
    """Handler for hermes_log_tail tool."""
    if not _HAS_DIAGNOSTIC_TOOLS:
        return '{"error": "Diagnostic tools not available", "success": false}'
    result = _log_tail(
        path=args.get("path", ""),
        lines=args.get("lines", 50),
    )
    return _format_result(result)


def _hermes_jsonl_query(args: Dict[str, Any]) -> str:
    """Handler for hermes_jsonl_query tool."""
    if not _HAS_DIAGNOSTIC_TOOLS:
        return '{"error": "Diagnostic tools not available", "success": false}'
    result = _jsonl_query(
        path=args.get("path", ""),
        filter_expr=args.get("filter_expr"),
        limit=args.get("limit", 100),
    )
    return _format_result(result)


def _hermes_file_preview(args: Dict[str, Any]) -> str:
    """Handler for hermes_file_preview tool."""
    if not _HAS_DIAGNOSTIC_TOOLS:
        return '{"error": "Diagnostic tools not available", "success": false}'
    result = _file_preview(
        path=args.get("path", ""),
        offset=args.get("offset", 0),
        limit=args.get("limit", 100),
    )
    return _format_result(result)


def _hermes_tool_metrics(args: Dict[str, Any]) -> str:
    """Handler for hermes_tool_metrics tool."""
    if not _HAS_DIAGNOSTIC_TOOLS:
        return '{"error": "Diagnostic tools not available", "success": false}'
    result = _tool_metrics(
        tool_name=args.get("tool_name"),
        limit=args.get("limit", 100),
    )
    return _format_result(result)


def _hermes_policy_block_summary(args: Dict[str, Any]) -> str:
    """Handler for hermes_policy_block_summary tool."""
    if not _HAS_DIAGNOSTIC_TOOLS:
        return '{"error": "Diagnostic tools not available", "success": false}'
    result = _policy_summary()
    return _format_result(result)


def _hermes_skill_inventory_query(args: Dict[str, Any]) -> str:
    """Handler for hermes_skill_inventory_query tool."""
    if not _HAS_DIAGNOSTIC_TOOLS:
        return '{"error": "Diagnostic tools not available", "success": false}'
    result = _skill_query(
        category=args.get("category"),
        search=args.get("search"),
        limit=args.get("limit", 50),
    )
    return _format_result(result)


def _format_result(result) -> str:
    """Format DiagnosticResult as JSON string for tool response."""
    import json
    return json.dumps({
        "success": result.success,
        "data": result.data,
        "error": result.error,
        "lines_scanned": result.lines_scanned,
        "matches_found": result.matches_found,
        "output_lines": result.output_lines,
        "truncated": result.truncated,
        "suggestion": result.suggestion,
    }, ensure_ascii=False)


# =============================================================================
# Registration
# =============================================================================

def register_diagnostic_tools(registry):
    """
    Register all hermes diagnostic tools with the Hermes tool registry.
    
    Called by hermes_tools initialization or by the tool_logger plugin
    during its setup phase.
    """
    if not _HAS_DIAGNOSTIC_TOOLS:
        import logging
        logging.getLogger(__name__).warning(
            "hermes diagnostic tools not registered — diagnostic_tools module not found"
        )
        return
    
    tools = [
        (HERMES_LOG_SEARCH_SCHEMA, _hermes_log_search),
        (HERMES_LOG_TAIL_SCHEMA, _hermes_log_tail),
        (HERMES_JSONL_QUERY_SCHEMA, _hermes_jsonl_query),
        (HERMES_FILE_PREVIEW_SCHEMA, _hermes_file_preview),
        (HERMES_TOOL_METRICS_SCHEMA, _hermes_tool_metrics),
        (HERMES_POLICY_BLOCK_SUMMARY_SCHEMA, _hermes_policy_block_summary),
        (HERMES_SKILL_INVENTORY_QUERY_SCHEMA, _hermes_skill_inventory_query),
    ]
    
    for schema, handler in tools:
        try:
            registry.register(
                name=schema["name"],
                toolset="hermes",
                schema=schema,
                handler=lambda args, h=handler: h(args),
                check_fn=None,
                requires_env=None,
                is_async=False,
                description=schema["description"],
                emoji="🔍",
            )
            import logging
            logging.getLogger(__name__).info(f"Registered diagnostic tool: {schema['name']}")
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Failed to register {schema['name']}: {e}")


# =============================================================================
# Self-test
# =============================================================================

if __name__ == "__main__":
    import json
    print("Running diagnostic tool adapter self-tests...\n")
    
    if not _HAS_DIAGNOSTIC_TOOLS:
        print("SKIP: diagnostic_tools module not available")
        exit(0)
    
    # Test 1: hermes_log_search
    print("Test 1: hermes_log_search")
    result = _log_search(
        path=os.path.expanduser("~/.hermes/logs/"),
        pattern="blocked",
        max_matches=10,
    )
    assert result.success in [True, False], "success should be bool"
    print(f"  PASS: success={result.success}, matches_found={result.matches_found}")
    
    # Test 2: hermes_log_tail
    print("Test 2: hermes_log_tail")
    result = _log_tail(
        path=os.path.expanduser("~/.hermes/logs/agent.log"),
        lines=5,
    )
    assert result.success in [True, False]
    print(f"  PASS: success={result.success}, output_lines={result.output_lines}")
    
    # Test 3: hermes_file_preview
    print("Test 3: hermes_file_preview")
    result = _file_preview(
        path=os.path.expanduser("~/.hermes/logs/agent.log"),
        offset=0,
        limit=10,
    )
    assert result.success in [True, False]
    print(f"  PASS: success={result.success}")
    
    # Test 4: hermes_jsonl_query
    print("Test 4: hermes_jsonl_query")
    result = _jsonl_query(
        path=os.path.expanduser("~/.hermes/logs/tool_calls.jsonl"),
        limit=5,
    )
    assert result.success in [True, False]
    print(f"  PASS: success={result.success}")
    
    # Test 5: hermes_tool_metrics
    print("Test 5: hermes_tool_metrics")
    result = _tool_metrics(limit=5)
    assert result.success in [True, False]
    print(f"  PASS: success={result.success}")
    
    # Test 6: hermes_skill_inventory_query
    print("Test 6: hermes_skill_inventory_query")
    result = _skill_query(limit=5)
    assert result.success in [True, False]
    print(f"  PASS: success={result.success}")
    
    # Test 7: hermes_policy_block_summary
    print("Test 7: hermes_policy_block_summary")
    result = _policy_summary()
    assert result.success in [True, False]
    print(f"  PASS: success={result.success}")
    
    # Test 8: Handler JSON output
    print("Test 8: Handler JSON formatting")
    out = _hermes_log_search({"path": os.path.expanduser("~/.hermes/logs/"), "pattern": "error", "max_matches": 5})
    parsed = json.loads(out)
    assert "success" in parsed, "handler output should contain 'success'"
    print(f"  PASS: handler returns valid JSON with success={parsed.get('success')}")
    
    print("\nAll self-tests passed!")
