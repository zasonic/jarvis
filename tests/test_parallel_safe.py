"""Contract tests for the ``parallel_safe`` tool flag.

The planner's direct-exec path may dispatch a batch of independent steps
concurrently, but only for tools that explicitly declare themselves
side-effect-free via ``Tool.parallel_safe``. These tests pin the contract:

- the flag defaults to ``False`` (opt-in, safe-by-default);
- exactly the read-only network/IO builtins opt in;
- anything that writes to the shared DB or has side effects stays sequential.

Asserted against the live registry rather than a hardcoded list, so a new
tool that forgets to set the flag is caught here and a future safe tool can
be added by flipping the override (this test then documents the change).
"""

from __future__ import annotations

from jarvis.tools.base import Tool, ToolContext
from jarvis.tools.registry import BUILTIN_TOOLS
from jarvis.tools.types import ToolExecutionResult


# The set of builtins intended to be safe for concurrent dispatch: read-only,
# network/IO-bound, no shared-DB writes, no dependence on another tool.
EXPECTED_PARALLEL_SAFE = {
    "webSearch",
    "fetchWebPage",
    "getWeather",
    "screenshot",
    "localFiles",
}


def test_base_default_is_not_parallel_safe():
    """A tool that doesn't override the flag is sequential by default."""

    class _Dummy(Tool):
        @property
        def name(self):
            return "dummy"

        @property
        def description(self):
            return "dummy"

        @property
        def inputSchema(self):
            return {"type": "object", "properties": {}}

        def run(self, args, context: ToolContext) -> ToolExecutionResult:
            return ToolExecutionResult(success=True, reply_text="ok")

    assert _Dummy().parallel_safe is False


def test_expected_builtins_are_parallel_safe():
    for name in EXPECTED_PARALLEL_SAFE:
        assert name in BUILTIN_TOOLS, f"{name} missing from registry"
        assert BUILTIN_TOOLS[name].parallel_safe is True, (
            f"{name} should be parallel_safe"
        )


def test_other_builtins_are_not_parallel_safe():
    # Every builtin not in the expected set (nutrition writers, stop,
    # toolSearchTool, refreshMCPTools) must stay sequential.
    for name, tool in BUILTIN_TOOLS.items():
        if name in EXPECTED_PARALLEL_SAFE:
            continue
        assert tool.parallel_safe is False, (
            f"{name} unexpectedly declares parallel_safe=True"
        )


def test_db_writing_tools_are_not_parallel_safe():
    # Explicit guard: tools that mutate the shared SQLite DB must never be
    # batched concurrently, regardless of how the registry evolves.
    for name in ("logMeal", "deleteMeal"):
        assert BUILTIN_TOOLS[name].parallel_safe is False
