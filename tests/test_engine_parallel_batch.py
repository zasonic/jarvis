"""Functional test for parallel plan-batch direct-exec.

No mocks and no dummy data: this drives the real ``_execute_parallel_plan_batch``
helper with the real ``localFiles`` builtin tool reading two real files on disk,
a real ``Database``, and a real ``Settings`` config. ``localFiles`` is
parallel-safe and needs no network, so the batch genuinely runs both reads
concurrently and we assert the real file contents come back in plan order.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from jarvis.config import load_settings
from jarvis.reply.engine import _execute_parallel_plan_batch, _maybe_digest_tool_result
from jarvis.tools.registry import generate_tools_json_schema


@pytest.fixture
def real_cfg():
    """A real Settings object loaded from a temp config that disables the
    tool-result digest, so the batch performs zero LLM calls (purely the
    file reads). Restores any prior JARVIS_CONFIG_PATH afterwards."""
    cfg_dir = tempfile.mkdtemp(prefix="jarvis_cfg_")
    cfg_path = os.path.join(cfg_dir, "config.json")
    Path(cfg_path).write_text(json.dumps({"tool_result_digest_enabled": False}))
    prev = os.environ.get("JARVIS_CONFIG_PATH")
    os.environ["JARVIS_CONFIG_PATH"] = cfg_path
    try:
        yield load_settings()
    finally:
        if prev is None:
            os.environ.pop("JARVIS_CONFIG_PATH", None)
        else:
            os.environ["JARVIS_CONFIG_PATH"] = prev


@pytest.fixture
def two_real_files():
    """Create two real files under the user's home directory and yield their
    absolute paths + contents. localFiles only permits paths under
    ``expanduser("~")``, so we place them there and pass absolute paths."""
    home = Path(os.path.expanduser("~")).resolve()
    base = Path(tempfile.mkdtemp(prefix="jarvis_ptest_", dir=str(home)))
    f1 = base / "alpha.txt"
    f2 = base / "beta.txt"
    f1.write_text("ALPHA-CONTENT-12345")
    f2.write_text("BETA-CONTENT-67890")
    rel1 = str(f1.resolve())
    rel2 = str(f2.resolve())
    try:
        yield (rel1, "ALPHA-CONTENT-12345"), (rel2, "BETA-CONTENT-67890")
    finally:
        for f in (f1, f2):
            try:
                f.unlink()
            except OSError:
                pass
        try:
            base.rmdir()
        except OSError:
            pass


def test_independent_reads_run_as_one_parallel_batch(db, real_cfg, two_real_files):
    (rel1, body1), (rel2, body2) = two_real_files
    # Two independent, concrete, parallel-safe plan steps — real file reads.
    plan_tool_steps = [
        f"localFiles operation='read' path='{rel1}'",
        f"localFiles operation='read' path='{rel2}'",
    ]
    action_plan = plan_tool_steps + ["Reply to the user with both files."]
    tools_schema = generate_tools_json_schema(["localFiles"])

    messages: list = []
    recent_sigs: list = []
    history: list = []

    handled = _execute_parallel_plan_batch(
        cfg=real_cfg,
        db=db,
        messages=messages,
        action_plan=action_plan,
        plan_tool_steps=plan_tool_steps,
        start_index=0,
        tools_json_schema=tools_schema,
        allowed_tools=["localFiles", "stop"],
        recent_tool_signatures=recent_sigs,
        invoked_tools_history=history,
        persona_prompt="",
        redacted="read both files",
        language=None,
        maybe_digest=_maybe_digest_tool_result,
    )

    # The batch handled both independent steps.
    assert handled is True

    # Two tool results, appended in PLAN order, each carrying its real file body.
    tool_results = [m for m in messages if m.get("tool_name") == "localFiles"]
    assert len(tool_results) == 2
    assert body1 in tool_results[0]["content"]
    assert body2 in tool_results[1]["content"]
    # Real success flags from the real tool.
    assert all(m.get("tool_failed") is False for m in tool_results)
    # Both calls recorded in history, in order.
    assert [h[0] for h in history] == ["localFiles", "localFiles"]


def test_single_step_is_not_batched(db, real_cfg, two_real_files):
    (rel1, _body1), _ = two_real_files
    # Only one eligible step -> the batch declines (needs >= 2); the engine's
    # sequential path would handle it instead.
    plan_tool_steps = [f"localFiles operation='read' path='{rel1}'"]
    handled = _execute_parallel_plan_batch(
        cfg=real_cfg,
        db=db,
        messages=[],
        action_plan=plan_tool_steps + ["Reply."],
        plan_tool_steps=plan_tool_steps,
        start_index=0,
        tools_json_schema=generate_tools_json_schema(["localFiles"]),
        allowed_tools=["localFiles", "stop"],
        recent_tool_signatures=[],
        invoked_tools_history=[],
        persona_prompt="",
        redacted="read one file",
        language=None,
        maybe_digest=_maybe_digest_tool_result,
    )
    assert handled is False
