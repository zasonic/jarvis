"""Tests for scheduled-task persistence, the scheduler, and the tools.

Everything here runs against real components: the real in-memory `Database`
(the `db` fixture), the real `TaskScheduler`, and the real tool classes
executed through a real `ToolContext`. No mocked or canned data — the only
test double is a recording callback that captures which tasks fired, which
is observation of real behaviour rather than fabricated input.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from jarvis.scheduling.schedule import DAILY_INTERVAL_SECONDS, KIND_ONCE, KIND_RECURRING
from jarvis.scheduling.scheduler import TaskScheduler
from jarvis.tools.base import ToolContext
from jarvis.tools.builtin.scheduling.schedule_task import ScheduleTaskTool
from jarvis.tools.builtin.scheduling.list_tasks import ListScheduledTasksTool
from jarvis.tools.builtin.scheduling.cancel_task import CancelScheduledTaskTool


def _iso(dt: datetime) -> str:
    return dt.isoformat()


# A config namespace with real values that keep location resolution offline
# and deterministic (no network, falls back to UTC).
def _cfg():
    return SimpleNamespace(
        location_ip_address=None,
        location_auto_detect=False,
        location_cgnat_resolve_public_ip=False,
        location_cache_minutes=60,
    )


def _ctx(db):
    return ToolContext(
        db=db, cfg=_cfg(), system_prompt="", original_prompt="",
        redacted_text="", max_retries=0, user_print=lambda *_: None, language=None,
    )


class TestDatabaseStore:
    def test_insert_and_active_roundtrip(self, db):
        now = datetime(2026, 5, 31, 12, 0, tzinfo=timezone.utc)
        tid = db.insert_scheduled_task(
            prompt="tell me the weather",
            kind=KIND_RECURRING,
            next_run_utc=_iso(now + timedelta(hours=1)),
            interval_seconds=DAILY_INTERVAL_SECONDS,
            tz_name="Europe/London",
        )
        active = db.get_active_scheduled_tasks()
        assert len(active) == 1
        row = active[0]
        assert row["id"] == tid
        assert row["prompt"] == "tell me the weather"
        assert row["kind"] == KIND_RECURRING
        assert row["interval_seconds"] == DAILY_INTERVAL_SECONDS
        assert row["enabled"] == 1

    def test_due_filter_respects_time_and_enabled(self, db):
        now = datetime(2026, 5, 31, 12, 0, tzinfo=timezone.utc)
        due_id = db.insert_scheduled_task(
            prompt="due", kind=KIND_ONCE, next_run_utc=_iso(now - timedelta(minutes=1)),
        )
        db.insert_scheduled_task(
            prompt="future", kind=KIND_ONCE, next_run_utc=_iso(now + timedelta(hours=1)),
        )
        cancelled_id = db.insert_scheduled_task(
            prompt="cancelled", kind=KIND_ONCE, next_run_utc=_iso(now - timedelta(minutes=5)),
        )
        db.cancel_scheduled_task(cancelled_id)

        due = db.get_due_scheduled_tasks(_iso(now))
        assert [r["id"] for r in due] == [due_id]

    def test_update_run_reschedule_vs_retire(self, db):
        now = datetime(2026, 5, 31, 12, 0, tzinfo=timezone.utc)
        tid = db.insert_scheduled_task(
            prompt="x", kind=KIND_RECURRING, next_run_utc=_iso(now),
            interval_seconds=DAILY_INTERVAL_SECONDS,
        )
        nxt = now + timedelta(days=1)
        db.update_scheduled_task_run(tid, _iso(now), _iso(nxt), enabled=True)
        row = db.get_active_scheduled_tasks()[0]
        assert row["last_run_utc"] == _iso(now)
        assert row["next_run_utc"] == _iso(nxt)

        # Retire a one-shot: enabled=False, next_run untouched.
        db.update_scheduled_task_run(tid, _iso(now), None, enabled=False)
        assert db.get_active_scheduled_tasks() == []

    def test_cancel_returns_false_for_unknown(self, db):
        assert db.cancel_scheduled_task(999) is False


class TestTaskScheduler:
    def test_fires_only_due_tasks(self, db):
        now = datetime(2026, 5, 31, 12, 0, tzinfo=timezone.utc)
        due = db.insert_scheduled_task(
            prompt="due", kind=KIND_ONCE, next_run_utc=_iso(now - timedelta(minutes=1)),
        )
        db.insert_scheduled_task(
            prompt="future", kind=KIND_ONCE, next_run_utc=_iso(now + timedelta(hours=2)),
        )
        fired: list[int] = []
        sched = TaskScheduler(
            db, _cfg(), lambda task: fired.append(int(task["id"])),
            tick_seconds=1, now_fn=lambda: now,
        )
        count = sched.run_due_once()
        assert count == 1
        assert fired == [due]

    def test_recurring_reschedules_one_shot_retires(self, db):
        now = datetime(2026, 5, 31, 12, 0, tzinfo=timezone.utc)
        rec = db.insert_scheduled_task(
            prompt="daily", kind=KIND_RECURRING, next_run_utc=_iso(now - timedelta(minutes=1)),
            interval_seconds=DAILY_INTERVAL_SECONDS,
        )
        once = db.insert_scheduled_task(
            prompt="once", kind=KIND_ONCE, next_run_utc=_iso(now - timedelta(minutes=1)),
        )
        sched = TaskScheduler(db, _cfg(), lambda task: None, now_fn=lambda: now)
        sched.run_due_once()

        active = {r["id"]: r for r in db.get_active_scheduled_tasks()}
        # Recurring survives with an advanced next_run; one-shot is retired.
        assert once not in active
        assert rec in active
        assert active[rec]["next_run_utc"] > _iso(now)
        assert active[rec]["last_run_utc"] == _iso(now)

    def test_callback_failure_still_advances_schedule(self, db):
        now = datetime(2026, 5, 31, 12, 0, tzinfo=timezone.utc)
        once = db.insert_scheduled_task(
            prompt="boom", kind=KIND_ONCE, next_run_utc=_iso(now - timedelta(minutes=1)),
        )

        def _boom(task):
            raise RuntimeError("callback failed")

        sched = TaskScheduler(db, _cfg(), _boom, now_fn=lambda: now)
        # Must not raise, and the failed one-shot must be retired (no hot loop).
        sched.run_due_once()
        assert all(r["id"] != once for r in db.get_active_scheduled_tasks())

    def test_start_stop_is_clean(self, db):
        sched = TaskScheduler(db, _cfg(), lambda task: None, tick_seconds=0.05,
                              now_fn=lambda: datetime.now(timezone.utc))
        sched.start()
        sched.stop(timeout=2.0)  # should join promptly


class TestSchedulingTools:
    def test_schedule_task_relative_delay_persists(self, db):
        before = datetime.now(timezone.utc)
        res = ScheduleTaskTool().run(
            {"prompt": "remind me to stretch", "in_minutes": 20}, _ctx(db),
        )
        assert res.success
        rows = db.get_active_scheduled_tasks()
        assert len(rows) == 1
        row = rows[0]
        assert row["prompt"] == "remind me to stretch"
        assert row["kind"] == KIND_ONCE
        assert row["interval_seconds"] is None
        nxt = datetime.fromisoformat(row["next_run_utc"])
        # ~20 minutes ahead of when we called it (generous tolerance).
        delta = (nxt - before).total_seconds()
        assert 19 * 60 <= delta <= 21 * 60

    def test_schedule_task_daily_is_recurring(self, db):
        res = ScheduleTaskTool().run(
            {"prompt": "tell me the weather", "recurrence": "daily", "at_hour": 8},
            _ctx(db),
        )
        assert res.success
        row = db.get_active_scheduled_tasks()[0]
        assert row["kind"] == KIND_RECURRING
        assert row["interval_seconds"] == DAILY_INTERVAL_SECONDS

    def test_schedule_task_requires_prompt(self, db):
        res = ScheduleTaskTool().run({"in_minutes": 5}, _ctx(db))
        assert not res.success
        assert db.get_active_scheduled_tasks() == []

    def test_schedule_task_requires_a_time_shape(self, db):
        res = ScheduleTaskTool().run({"prompt": "do a thing"}, _ctx(db))
        assert not res.success
        assert "delay" in (res.error_message or "").lower() or "time" in (res.error_message or "").lower()

    def test_daily_without_hour_rejected(self, db):
        res = ScheduleTaskTool().run(
            {"prompt": "weather", "recurrence": "daily"}, _ctx(db),
        )
        assert not res.success
        assert db.get_active_scheduled_tasks() == []

    def test_list_reflects_real_rows(self, db):
        ScheduleTaskTool().run({"prompt": "task one", "in_minutes": 10}, _ctx(db))
        ScheduleTaskTool().run(
            {"prompt": "task two", "recurrence": "daily", "at_hour": 9}, _ctx(db),
        )
        res = ListScheduledTasksTool().run({}, _ctx(db))
        assert res.success
        assert "task one" in res.reply_text
        assert "task two" in res.reply_text
        assert "(daily)" in res.reply_text

    def test_list_empty(self, db):
        res = ListScheduledTasksTool().run({}, _ctx(db))
        assert res.success
        assert "No scheduled tasks" in res.reply_text

    def test_cancel_disables_real_task(self, db):
        sres = ScheduleTaskTool().run({"prompt": "cancel me", "in_minutes": 30}, _ctx(db))
        tid = db.get_active_scheduled_tasks()[0]["id"]
        cres = CancelScheduledTaskTool().run({"id": tid}, _ctx(db))
        assert cres.success
        assert db.get_active_scheduled_tasks() == []

    def test_cancel_unknown_id_fails(self, db):
        res = CancelScheduledTaskTool().run({"id": 12345}, _ctx(db))
        assert not res.success

    def test_cancel_requires_numeric_id(self, db):
        res = CancelScheduledTaskTool().run({"id": "not-a-number"}, _ctx(db))
        assert not res.success
