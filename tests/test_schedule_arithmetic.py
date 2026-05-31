"""Unit tests for the pure schedule arithmetic.

These exercise the real `compute_next_run` / `advance_recurring` functions
with real timezone-aware datetimes (no mocks, no canned data): the only
inputs are concrete instants and the numeric fields a tool would resolve.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from jarvis.scheduling.schedule import (
    DAILY_INTERVAL_SECONDS,
    advance_recurring,
    compute_next_run,
)


_NOON_UTC = datetime(2026, 5, 31, 12, 0, 0, tzinfo=timezone.utc)


class TestComputeNextRun:
    def test_relative_delay_adds_minutes(self):
        got = compute_next_run(now_utc=_NOON_UTC, in_minutes=20)
        assert got == _NOON_UTC + timedelta(minutes=20)

    def test_zero_delay_is_now(self):
        # "do it in the background now" — fires on the next tick.
        assert compute_next_run(now_utc=_NOON_UTC, in_minutes=0) == _NOON_UTC

    def test_negative_delay_rejected(self):
        assert compute_next_run(now_utc=_NOON_UTC, in_minutes=-5) is None

    def test_time_of_day_later_today(self):
        # 18:00 UTC is still ahead of 12:00 UTC today.
        got = compute_next_run(now_utc=_NOON_UTC, at_hour=18, at_minute=30, tz_name="UTC")
        assert got == datetime(2026, 5, 31, 18, 30, tzinfo=timezone.utc)

    def test_time_of_day_already_passed_rolls_to_tomorrow(self):
        # 08:00 UTC is behind 12:00 UTC, so next occurrence is tomorrow.
        got = compute_next_run(now_utc=_NOON_UTC, at_hour=8, tz_name="UTC")
        assert got == datetime(2026, 6, 1, 8, 0, tzinfo=timezone.utc)

    def test_time_of_day_respects_local_timezone(self):
        # 08:00 in London (BST = UTC+1 on this date) is 07:00 UTC.
        # 12:00 UTC = 13:00 local, so 08:00 local already passed → tomorrow.
        got = compute_next_run(now_utc=_NOON_UTC, at_hour=8, tz_name="Europe/London")
        assert got == datetime(2026, 6, 1, 7, 0, tzinfo=timezone.utc)

    def test_unknown_timezone_falls_back_to_utc(self):
        got = compute_next_run(now_utc=_NOON_UTC, at_hour=18, tz_name="Not/AZone")
        assert got == datetime(2026, 5, 31, 18, 0, tzinfo=timezone.utc)

    def test_invalid_hour_rejected(self):
        assert compute_next_run(now_utc=_NOON_UTC, at_hour=25) is None

    def test_no_time_shape_returns_none(self):
        # Neither a delay nor a time-of-day: not schedulable, fail closed.
        assert compute_next_run(now_utc=_NOON_UTC) is None
        assert compute_next_run(now_utc=_NOON_UTC, recurring=True) is None

    def test_naive_now_is_treated_as_utc(self):
        naive = datetime(2026, 5, 31, 12, 0, 0)
        got = compute_next_run(now_utc=naive, in_minutes=10)
        assert got == _NOON_UTC + timedelta(minutes=10)


class TestAdvanceRecurring:
    def test_advances_one_interval_when_just_fired(self):
        nxt = advance_recurring(
            next_run_utc=_NOON_UTC,
            interval_seconds=DAILY_INTERVAL_SECONDS,
            now_utc=_NOON_UTC,
        )
        assert nxt == _NOON_UTC + timedelta(days=1)

    def test_skips_missed_firings_instead_of_stacking(self):
        # Machine was asleep for ~3 days; next run should be the first future
        # slot, not three stacked catch-up runs.
        now = _NOON_UTC + timedelta(days=3, hours=1)
        nxt = advance_recurring(
            next_run_utc=_NOON_UTC,
            interval_seconds=DAILY_INTERVAL_SECONDS,
            now_utc=now,
        )
        assert nxt > now
        assert nxt == _NOON_UTC + timedelta(days=4)

    def test_keeps_original_wall_clock_slot(self):
        # Daily anchored at noon stays at noon after advancing.
        nxt = advance_recurring(
            next_run_utc=_NOON_UTC,
            interval_seconds=DAILY_INTERVAL_SECONDS,
            now_utc=_NOON_UTC + timedelta(minutes=5),
        )
        assert nxt.hour == 12 and nxt.minute == 0

    def test_degenerate_interval_does_not_busy_loop(self):
        nxt = advance_recurring(
            next_run_utc=_NOON_UTC, interval_seconds=0, now_utc=_NOON_UTC,
        )
        assert nxt > _NOON_UTC
