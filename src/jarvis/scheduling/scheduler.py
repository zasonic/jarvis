"""Background scheduler that fires stored prompts through the reply engine.

The daemon owns one :class:`TaskScheduler`. It runs a single background
thread that wakes every ``scheduler_tick_seconds``, asks the DB for due
tasks, and hands each to a ``run_callback`` the daemon supplies (which runs
the reply engine and speaks the result). After a task fires the scheduler
advances recurring tasks to their next slot and retires one-shot tasks.

Design principles:
- **Fail-soft**: a callback that raises must not kill the loop or wedge a
  recurring task into a hot retry loop. We always advance/retire the row.
- **Sequential**: tasks fire one at a time on the scheduler thread, so a
  scheduled announcement never overlaps another. The reply engine and TTS
  coordinate with the live voice path through the existing echo handling.
- **Responsive shutdown**: the loop sleeps on an ``Event`` so ``stop()``
  returns promptly rather than after a full tick.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any, Callable

from ..debug import debug_log
from .schedule import KIND_RECURRING, advance_recurring


class TaskScheduler:
    """Polls the DB for due scheduled tasks and dispatches them."""

    def __init__(
        self,
        db,
        cfg,
        run_callback: Callable[[Any], None],
        *,
        tick_seconds: float | None = None,
        now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self._db = db
        self._cfg = cfg
        self._run_callback = run_callback
        self._tick_seconds = float(
            tick_seconds
            if tick_seconds is not None
            else getattr(cfg, "scheduler_tick_seconds", 30.0)
        )
        self._now_fn = now_fn
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="jarvis-scheduler", daemon=True
        )
        self._thread.start()
        debug_log(
            f"scheduler: started (tick={self._tick_seconds}s)", "scheduler"
        )

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
        self._thread = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.run_due_once()
            except Exception as exc:  # pragma: no cover - defensive
                debug_log(f"scheduler: tick failed — {exc}", "scheduler")
            # Sleep on the stop event so shutdown is immediate.
            self._stop.wait(self._tick_seconds)

    def run_due_once(self) -> int:
        """Fire every task that is due as of now. Returns the count fired.

        Extracted from the loop so it is unit-testable with a fake clock and a
        recording callback, without spinning up the thread.
        """
        now = self._now_fn()
        now_iso = now.isoformat()
        due = self._db.get_due_scheduled_tasks(now_iso)
        fired = 0
        for task in due:
            if self._stop.is_set():
                break
            self._fire(task, now)
            fired += 1
        return fired

    def _fire(self, task, now: datetime) -> None:
        task_id = int(task["id"])
        last_run = now.isoformat()
        # Run the user's callback first; whatever happens, we still advance the
        # schedule so a failing task can't pin the loop.
        try:
            self._run_callback(task)
        except Exception as exc:  # pragma: no cover - defensive
            debug_log(f"scheduler: task {task_id} callback failed — {exc}", "scheduler")
        try:
            if task["kind"] == KIND_RECURRING and task["interval_seconds"]:
                next_run = advance_recurring(
                    next_run_utc=_parse_iso(task["next_run_utc"]),
                    interval_seconds=int(task["interval_seconds"]),
                    now_utc=now,
                )
                self._db.update_scheduled_task_run(
                    task_id, last_run, next_run.isoformat(), enabled=True
                )
            else:
                # One-shot task: retire it.
                self._db.update_scheduled_task_run(
                    task_id, last_run, None, enabled=False
                )
        except Exception as exc:  # pragma: no cover - defensive
            debug_log(f"scheduler: task {task_id} reschedule failed — {exc}", "scheduler")


def _parse_iso(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt
