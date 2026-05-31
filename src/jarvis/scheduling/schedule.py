"""Pure, language-agnostic schedule arithmetic.

The scheduler stores the *what* (a natural-language prompt) and the *when*
(a UTC timestamp). This module owns the *when*: converting the structured
fields a tool resolves from the user's utterance (recurrence, an
hour/minute-of-day, or a relative delay) into a concrete next-run UTC
instant, and advancing a recurring task after it fires.

Why structured fields rather than parsing free text here: Jarvis must
support an arbitrary number of languages, so natural-language time
expressions are understood by the LLM (which fills `at_hour`, `in_minutes`,
etc.) and this module only does deterministic arithmetic on numbers. That
keeps the time logic testable and free of hardcoded language patterns.

All functions are pure and timezone-aware. Daily recurrence anchors on the
user's local wall-clock time (so "every day at 08:00" means 08:00 in their
zone, surviving DST shifts), then converts back to UTC for storage.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

try:  # zoneinfo is stdlib on 3.9+; degrade gracefully to UTC if unavailable
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover - defensive
    ZoneInfo = None  # type: ignore


# Recurrence kinds persisted in the `kind` column.
KIND_ONCE = "once"
KIND_RECURRING = "recurring"

# Seconds in a day — the only recurrence period this MVP exposes ("daily").
# Stored explicitly per row so future periods (weekly, hourly) need no schema
# change, just a different interval_seconds.
DAILY_INTERVAL_SECONDS = 24 * 60 * 60


def _resolve_zone(tz_name: Optional[str]):
    """Return a tzinfo for ``tz_name``, falling back to UTC."""
    if tz_name and ZoneInfo is not None:
        try:
            return ZoneInfo(tz_name)
        except Exception:
            return timezone.utc
    return timezone.utc


def compute_next_run(
    *,
    now_utc: datetime,
    in_minutes: Optional[int] = None,
    at_hour: Optional[int] = None,
    at_minute: Optional[int] = None,
    recurring: bool = False,
    tz_name: Optional[str] = None,
) -> Optional[datetime]:
    """Compute the first firing instant (UTC) for a new task.

    Exactly one of two shapes is expected:

    - **Relative delay** (`in_minutes` set): fire ``in_minutes`` from now.
      ``in_minutes=0`` means "as soon as the scheduler next ticks" — this is
      the "do it in the background now and tell me when done" case.
    - **Time of day** (`at_hour` set, `at_minute` optional, defaults 0): the
      next occurrence of that wall-clock time in ``tz_name``. If that instant
      is already past today, roll to tomorrow.

    Returns a timezone-aware UTC ``datetime``, or ``None`` when the inputs
    don't describe a valid schedule (so callers fail closed rather than
    firing immediately by accident).
    """
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)

    if in_minutes is not None:
        try:
            mins = int(in_minutes)
        except (TypeError, ValueError):
            return None
        if mins < 0:
            return None
        return now_utc + timedelta(minutes=mins)

    if at_hour is not None:
        try:
            hour = int(at_hour)
            minute = int(at_minute) if at_minute is not None else 0
        except (TypeError, ValueError):
            return None
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            return None
        zone = _resolve_zone(tz_name)
        now_local = now_utc.astimezone(zone)
        target_local = now_local.replace(
            hour=hour, minute=minute, second=0, microsecond=0
        )
        if target_local <= now_local:
            target_local = target_local + timedelta(days=1)
        return target_local.astimezone(timezone.utc)

    # Neither shape supplied: not a schedulable request.
    if recurring:
        return None
    return None


def advance_recurring(
    *,
    next_run_utc: datetime,
    interval_seconds: int,
    now_utc: datetime,
) -> datetime:
    """Return the next firing after a recurring task ran.

    Adds whole ``interval_seconds`` steps to ``next_run_utc`` until the result
    is strictly in the future relative to ``now_utc``. Stepping (rather than a
    flat ``now + interval``) keeps a daily task anchored to its original
    wall-clock slot and silently skips missed firings (e.g. the machine was
    asleep) instead of stacking catch-up runs.
    """
    if next_run_utc.tzinfo is None:
        next_run_utc = next_run_utc.replace(tzinfo=timezone.utc)
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    interval = int(interval_seconds)
    if interval <= 0:
        # Degenerate interval — push one minute out so we can't busy-loop.
        return now_utc + timedelta(minutes=1)
    step = timedelta(seconds=interval)
    nxt = next_run_utc
    while nxt <= now_utc:
        nxt = nxt + step
    return nxt
