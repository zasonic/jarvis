"""scheduleTask tool: create a scheduled or background task.

The user asks for something to happen later or on a repeating schedule
("remind me to stretch in 20 minutes", "every morning at 8 tell me the
weather", "research the best laptops and tell me when you're done"). This
tool persists the request; the daemon's scheduler later runs the stored
``prompt`` through the reply engine and speaks the result.

Time understanding is the LLM's job, not the tool's: the model resolves the
utterance into the structured fields below (so the feature works in any
language without hardcoded date parsing). The tool only does deterministic,
timezone-aware arithmetic via ``scheduling.schedule.compute_next_run``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from datetime import datetime, timezone

from ....debug import debug_log
from ...base import Tool, ToolContext
from ...types import ToolExecutionResult
from ....scheduling.schedule import (
    KIND_ONCE,
    KIND_RECURRING,
    DAILY_INTERVAL_SECONDS,
    compute_next_run,
)


def _resolve_tz_name(cfg) -> Optional[str]:
    """Best-effort IANA timezone for the user, or None (callers fall to UTC)."""
    try:
        from ....utils.location import get_location_context_with_timezone
        _, tz_name = get_location_context_with_timezone(
            config_ip=getattr(cfg, "location_ip_address", None),
            auto_detect=getattr(cfg, "location_auto_detect", True),
            resolve_cgnat_public_ip=getattr(cfg, "location_cgnat_resolve_public_ip", True),
            location_cache_minutes=getattr(cfg, "location_cache_minutes", 60),
        )
        return tz_name
    except Exception:
        return None


def _local_when(next_run_utc: datetime, tz_name: Optional[str]) -> str:
    """Render the next firing in the user's local time for the confirmation."""
    try:
        from ....utils.time_context import format_time_context
        return format_time_context(tz_name, now_utc=next_run_utc)
    except Exception:
        return next_run_utc.isoformat()


class ScheduleTaskTool(Tool):
    """Schedule a prompt to run later, once or on a daily recurrence."""

    @property
    def name(self) -> str:
        return "scheduleTask"

    @property
    def description(self) -> str:
        return (
            "Schedule something to happen later or on a repeating daily basis: "
            "reminders ('remind me to stretch in 20 minutes'), recurring "
            "updates ('every morning at 8 tell me the weather'), or a "
            "background job to run now and report back ('research X and tell "
            "me when done', use in_minutes=0). The 'prompt' is what Jarvis "
            "should do or say when it fires. Provide EITHER in_minutes (a "
            "relative delay) OR at_hour (+ optional at_minute) for a "
            "time-of-day. Set recurrence='daily' for something that repeats "
            "every day at at_hour."
        )

    @property
    def inputSchema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": (
                        "What Jarvis should do or say when the task fires, "
                        "phrased as an instruction (e.g. 'tell me today's "
                        "weather', 'remind me to take the pizza out')."
                    ),
                },
                "recurrence": {
                    "type": "string",
                    "enum": [KIND_ONCE, "daily"],
                    "description": "'once' (default) or 'daily' to repeat every day.",
                },
                "in_minutes": {
                    "type": "integer",
                    "description": (
                        "Fire this many minutes from now. Use 0 to run in the "
                        "background immediately. Mutually exclusive with at_hour."
                    ),
                },
                "at_hour": {
                    "type": "integer",
                    "description": "Hour of day to fire, 0-23 (local time).",
                },
                "at_minute": {
                    "type": "integer",
                    "description": "Minute of the hour to fire, 0-59 (defaults to 0).",
                },
            },
            "required": ["prompt"],
        }

    def run(
        self, args: Optional[Dict[str, Any]], context: ToolContext
    ) -> ToolExecutionResult:
        args = args or {}
        prompt = (args.get("prompt") or "").strip()
        if not prompt:
            return ToolExecutionResult(
                success=False, reply_text=None,
                error_message="No task prompt was provided to schedule.",
            )

        recurrence = (args.get("recurrence") or KIND_ONCE).strip().lower()
        recurring = recurrence == "daily"
        in_minutes = args.get("in_minutes")
        at_hour = args.get("at_hour")
        at_minute = args.get("at_minute")

        # Validate the time shape before touching the clock.
        if recurring and at_hour is None:
            return ToolExecutionResult(
                success=False, reply_text=None,
                error_message="A daily task needs a time of day (at_hour).",
            )
        if in_minutes is None and at_hour is None:
            return ToolExecutionResult(
                success=False, reply_text=None,
                error_message=(
                    "Need either a delay (in_minutes) or a time of day "
                    "(at_hour) to schedule a task."
                ),
            )

        tz_name = _resolve_tz_name(context.cfg)
        now_utc = datetime.now(timezone.utc)
        next_run = compute_next_run(
            now_utc=now_utc,
            in_minutes=in_minutes,
            at_hour=at_hour,
            at_minute=at_minute,
            recurring=recurring,
            tz_name=tz_name,
        )
        if next_run is None:
            return ToolExecutionResult(
                success=False, reply_text=None,
                error_message="Could not work out a valid time for that task.",
            )

        kind = KIND_RECURRING if recurring else KIND_ONCE
        interval = DAILY_INTERVAL_SECONDS if recurring else None
        try:
            task_id = context.db.insert_scheduled_task(
                prompt=prompt,
                kind=kind,
                next_run_utc=next_run.isoformat(),
                interval_seconds=interval,
                tz_name=tz_name,
            )
        except Exception as exc:
            debug_log(f"scheduleTask: insert failed — {exc}", "scheduler")
            return ToolExecutionResult(
                success=False, reply_text=None,
                error_message=f"Failed to save the scheduled task: {exc}",
            )

        when = _local_when(next_run, tz_name)
        recur_note = " (repeats daily)" if recurring else ""
        debug_log(
            f"scheduleTask: #{task_id} kind={kind} next={next_run.isoformat()}",
            "scheduler",
        )
        return ToolExecutionResult(
            success=True,
            reply_text=(
                f"Scheduled task #{task_id}: \"{prompt}\". "
                f"Next run: {when}{recur_note}."
            ),
        )
