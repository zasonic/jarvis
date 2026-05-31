"""listScheduledTasks tool: report the active scheduled/background tasks."""

from __future__ import annotations

from typing import Any, Dict, Optional
from datetime import datetime

from ...base import Tool, ToolContext
from ...types import ToolExecutionResult
from ....scheduling.schedule import KIND_RECURRING


def _format_when(next_run_utc: str, tz_name: Optional[str]) -> str:
    try:
        from ....utils.time_context import format_time_context
        dt = datetime.fromisoformat(next_run_utc)
        return format_time_context(tz_name, now_utc=dt)
    except Exception:
        return next_run_utc


class ListScheduledTasksTool(Tool):
    """List the user's pending scheduled and recurring tasks."""

    @property
    def name(self) -> str:
        return "listScheduledTasks"

    @property
    def description(self) -> str:
        return (
            "List the user's pending scheduled tasks and reminders (their id, "
            "what they do, when they next run, and whether they repeat). Use "
            "when the user asks what's scheduled, what reminders are set, or "
            "before cancelling one."
        )

    @property
    def inputSchema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}}

    def run(
        self, args: Optional[Dict[str, Any]], context: ToolContext
    ) -> ToolExecutionResult:
        try:
            rows = context.db.get_active_scheduled_tasks()
        except Exception as exc:
            return ToolExecutionResult(
                success=False, reply_text=None,
                error_message=f"Could not read scheduled tasks: {exc}",
            )
        if not rows:
            return ToolExecutionResult(
                success=True, reply_text="No scheduled tasks."
            )
        lines = []
        for r in rows:
            recur = " (daily)" if r["kind"] == KIND_RECURRING else ""
            when = _format_when(r["next_run_utc"], r["tz_name"])
            lines.append(f"#{r['id']}: \"{r['prompt']}\" — next {when}{recur}")
        return ToolExecutionResult(
            success=True,
            reply_text=f"Scheduled tasks ({len(rows)}):\n" + "\n".join(lines),
        )
