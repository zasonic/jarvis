"""cancelScheduledTask tool: cancel a pending scheduled/background task by id."""

from __future__ import annotations

from typing import Any, Dict, Optional

from ...base import Tool, ToolContext
from ...types import ToolExecutionResult


class CancelScheduledTaskTool(Tool):
    """Cancel a pending scheduled task by its id."""

    @property
    def name(self) -> str:
        return "cancelScheduledTask"

    @property
    def description(self) -> str:
        return (
            "Cancel a pending scheduled task or reminder by its numeric id. "
            "If the user doesn't know the id, call listScheduledTasks first."
        )

    @property
    def inputSchema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "id": {
                    "type": "integer",
                    "description": "The numeric id of the task to cancel.",
                },
            },
            "required": ["id"],
        }

    def run(
        self, args: Optional[Dict[str, Any]], context: ToolContext
    ) -> ToolExecutionResult:
        args = args or {}
        raw = args.get("id")
        try:
            task_id = int(raw)
        except (TypeError, ValueError):
            return ToolExecutionResult(
                success=False, reply_text=None,
                error_message="A numeric task id is required to cancel a task.",
            )
        try:
            cancelled = context.db.cancel_scheduled_task(task_id)
        except Exception as exc:
            return ToolExecutionResult(
                success=False, reply_text=None,
                error_message=f"Could not cancel task #{task_id}: {exc}",
            )
        if cancelled:
            return ToolExecutionResult(
                success=True, reply_text=f"Cancelled scheduled task #{task_id}."
            )
        return ToolExecutionResult(
            success=False,
            reply_text=None,
            error_message=f"No active scheduled task with id #{task_id}.",
        )
