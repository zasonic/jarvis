# Scheduled & background tasks

## Purpose

Let the user defer or repeat work by voice: reminders ("remind me to
stretch in 20 minutes"), recurring updates ("every morning at 8 tell me
the weather"), and fire-and-forget background jobs ("research the best
laptops and tell me when you're done", scheduled with `in_minutes=0`). A
stored task is a natural-language **prompt** plus a **when**; the daemon's
scheduler runs the prompt through the normal reply engine at its firing
time and speaks the result.

100% local: tasks live in the local SQLite DB, fire on a local thread, and
are spoken by the local TTS engine. No network, no external scheduler.

## Scope

- `src/jarvis/scheduling/schedule.py` — pure schedule arithmetic.
- `src/jarvis/scheduling/scheduler.py` — `TaskScheduler` daemon thread.
- `src/jarvis/tools/builtin/scheduling/` — `scheduleTask`, `listScheduledTasks`,
  `cancelScheduledTask`.
- `scheduled_tasks` table + helpers in `src/jarvis/memory/db.py`.
- Daemon wiring in `src/jarvis/daemon.py`.

## Data model

`scheduled_tasks` row: `id`, `created_utc`, `prompt`, `kind`
(`once` | `recurring`), `next_run_utc` (ISO-8601 UTC), `interval_seconds`
(recurring period; `NULL` for once), `last_run_utc`, `enabled`, `tz_name`
(IANA zone used when scheduling, for display). Created with
`CREATE TABLE IF NOT EXISTS` like every other table — no migration system.

DB helpers (all lock-guarded): `insert_scheduled_task`,
`get_due_scheduled_tasks(now_iso)`, `get_active_scheduled_tasks`,
`update_scheduled_task_run(id, last_run, next_run|None, enabled)`,
`cancel_scheduled_task(id)`.

## Time understanding is the LLM's job

Natural-language time expressions are resolved by the model, not parsed in
code — Jarvis must support arbitrary languages, so there are **no hardcoded
date/time patterns**. The `scheduleTask` schema exposes structured numeric
fields the model fills:

- `prompt` (required) — what to do/say when it fires.
- `recurrence` — `once` (default) or `daily`.
- `in_minutes` — relative delay; `0` = run now in the background.
- `at_hour` (0-23) + optional `at_minute` (0-59) — a wall-clock time.

`compute_next_run` does deterministic, timezone-aware arithmetic on those
numbers: a relative delay adds minutes; a time-of-day yields the next
occurrence in the user's zone (rolling to tomorrow if already past). It
returns `None` (fail closed) for invalid/empty inputs so a malformed
request never fires immediately by accident. Daily recurrence anchors on
the user's local wall-clock slot and survives DST.

`advance_recurring` steps a recurring task forward by whole intervals until
strictly in the future, so missed firings (machine asleep) are skipped
rather than stacked as catch-up runs, and the original slot is preserved.

## Scheduler behaviour

`TaskScheduler` runs one daemon thread that wakes every
`scheduler_tick_seconds`, calls `get_due_scheduled_tasks(now)`, and fires
each due task **sequentially** (no overlap). Per task it invokes the
daemon's `run_callback`, then advances recurring tasks via
`advance_recurring` or retires one-shot tasks (`enabled=0`).

Invariants:
- **Fail-soft**: a callback that raises is caught; the task is still
  advanced/retired so a failing recurring task can't hot-loop.
- **Responsive shutdown**: the loop sleeps on a `threading.Event`, so
  `stop()` returns promptly. The daemon stops the scheduler first on
  teardown so no task fires mid-shutdown.
- `run_due_once()` is the unit-testable core (fake clock + callback);
  the thread is a thin wrapper.

## Daemon integration

When `scheduler_enabled` (default True), the daemon constructs a
`TaskScheduler` after the voice listener and supplies a callback that:

1. Runs `run_reply_engine(db, cfg, tts=None, text=prompt,
   dialogue_memory=<dedicated>, language=None)`. A **dedicated**
   `DialogueMemory` (separate from the live conversation) is used so
   scheduled runs never contaminate the user's hot window.
2. If a reply is produced and TTS is enabled, **pauses the listener**
   (`voice_thread._dictation_active = True`, the same mechanism dictation
   uses) for the announcement so Jarvis doesn't transcribe its own speech,
   speaks the reply, and blocks the scheduler thread until the completion
   callback fires (with a 120s timeout guard), then resumes the listener.

No spoken preamble is added — that would be language-locked; the reply
engine already answers in the user's language.

## Tools

- `scheduleTask` — validates the time shape, resolves the user's timezone
  (best-effort, offline-friendly; falls back to UTC), computes `next_run`,
  persists, and returns a factual confirmation (id + next run, "repeats
  daily" note). Returns `success=False` with a clear message when the
  prompt is empty, no time shape is given, or a daily task lacks `at_hour`.
- `listScheduledTasks` — lists active tasks (id, prompt, next run local,
  recurrence). No args.
- `cancelScheduledTask` — disables a task by numeric `id`.

All three declare `parallel_safe = False` (DB writes / side effects), so the
planner never batches them concurrently.

## Configuration

| Key | Default | Purpose |
|-----|---------|---------|
| `scheduler_enabled` | `True` | Run the scheduler thread. When False, `scheduleTask` still records but nothing fires. |
| `scheduler_tick_seconds` | `30.0` | Poll interval; also the worst-case lateness of an `in_minutes` task. Floored at 1s. |

## Non-goals (current MVP)

- Only `once` and `daily` recurrence (the row stores an arbitrary
  `interval_seconds`, so weekly/hourly need no schema change later).
- No desktop UI for managing tasks yet (voice + DB only).
- A scheduled announcement that collides with the user mid-utterance
  briefly pauses the listener; fine-grained "wait until the user is idle"
  arbitration is future work.
