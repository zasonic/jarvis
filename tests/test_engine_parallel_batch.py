"""Engine integration tests for parallel plan-batch direct-exec.

When a SMALL-model plan begins with two or more independent, parallel-safe
tool steps (concrete, no placeholder), the engine dispatches them
concurrently in a single turn instead of one-per-turn, then calls the chat
model once for synthesis. These tests assert observable behaviour:

- both independent tools run (without the sequential step-resolver);
- their results land in the message history in PLAN order;
- the chat model is invoked once, for the final synthesis;
- disabling the feature flag restores the sequential single-step path.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


def _assistant_content(text: str):
    return {"message": {"role": "assistant", "content": text}}


# A plan whose first two steps are independent and parallel-safe: a weather
# lookup and a web search. Neither references the other's result, so they can
# run concurrently.
_PARALLEL_PLAN = [
    "getWeather location='London'",
    "webSearch search_query='today top news'",
    "Reply to the user with the combined findings.",
]


def _patches(engine_mod, fake_tool_runner, fake_chat):
    return [
        patch.object(engine_mod, "run_tool_with_retries", side_effect=fake_tool_runner),
        patch.object(engine_mod, "chat_with_messages", side_effect=fake_chat),
        patch.object(
            engine_mod, "select_tools",
            return_value=["getWeather", "webSearch", "stop"],
        ),
        patch.object(
            engine_mod, "extract_search_params_for_memory",
            return_value={"keywords": []},
        ),
        patch.object(engine_mod, "plan_query", return_value=list(_PARALLEL_PLAN)),
    ]


def test_independent_steps_run_in_parallel_batch(mock_config, db, dialogue_memory):
    from jarvis.reply import engine as engine_mod
    from jarvis.tools.types import ToolExecutionResult

    mock_config.ollama_chat_model = "gemma4:e2b"  # SMALL → direct-exec eligible
    mock_config.evaluator_enabled = False
    mock_config.planner_parallel_enabled = True
    mock_config.planner_parallel_max = 4

    invoked: list[str] = []

    def fake_tool_runner(db, cfg, tool_name, tool_args, **kwargs):
        invoked.append(tool_name)
        payload = {
            "getWeather": "London: 14C, light rain.",
            "webSearch": "Headlines: A, B, C.",
        }.get(tool_name, "ok")
        return ToolExecutionResult(success=True, reply_text=payload, error_message=None)

    synthesis_messages: list[list] = []
    chat_calls = [0]

    def fake_chat(*args, **kwargs):
        chat_calls[0] += 1
        msgs = kwargs.get("messages") or (args[2] if len(args) > 2 else [])
        synthesis_messages.append(list(msgs))
        return _assistant_content("It's 14C and rainy in London; top news is A, B, C.")

    # The step-resolver is the SEQUENTIAL path. If the parallel batch handles
    # both steps, the resolver must never be called.
    resolve_mock = patch.object(
        engine_mod, "_resolve_plan_step", return_value=None,
    ).start()
    for p in _patches(engine_mod, fake_tool_runner, fake_chat):
        p.start()
    try:
        engine_mod.run_reply_engine(
            db=db, cfg=mock_config, tts=None,
            text="what's the weather and the top news today?",
            dialogue_memory=dialogue_memory,
        )
    finally:
        patch.stopall()

    # Both independent tools ran.
    assert sorted(invoked) == ["getWeather", "webSearch"], (
        f"both independent tools should run; got {invoked}"
    )
    # The sequential step-resolver was bypassed — the batch handled both steps.
    assert resolve_mock.call_count == 0, (
        "parallel batch should handle both steps without the sequential resolver"
    )
    # Chat model invoked once, for synthesis only.
    assert chat_calls[0] == 1, f"chat model should run once for synthesis; got {chat_calls[0]}"

    # Tool results appear in PLAN order in the synthesis message history.
    final_msgs = synthesis_messages[-1]
    tool_result_order = [m.get("tool_name") for m in final_msgs if m.get("tool_name")]
    assert tool_result_order == ["getWeather", "webSearch"], (
        f"results must be appended in plan order; got {tool_result_order}"
    )


def test_disabling_flag_restores_sequential_path(mock_config, db, dialogue_memory):
    """With planner_parallel_enabled=False the engine must fall back to the
    one-step-per-turn resolver (the parallel batch never runs)."""
    from jarvis.reply import engine as engine_mod
    from jarvis.tools.types import ToolExecutionResult

    mock_config.ollama_chat_model = "gemma4:e2b"
    mock_config.evaluator_enabled = False
    mock_config.planner_parallel_enabled = False

    def fake_tool_runner(db, cfg, tool_name, tool_args, **kwargs):
        return ToolExecutionResult(success=True, reply_text="ok", error_message=None)

    def fake_chat(*args, **kwargs):
        return _assistant_content("done")

    resolved = iter([
        ("getWeather", {"location": "London"}),
        ("webSearch", {"search_query": "today top news"}),
    ])

    def fake_resolve(*args, **kwargs):
        try:
            return next(resolved)
        except StopIteration:
            return None

    with patch.object(engine_mod, "_resolve_plan_step", side_effect=fake_resolve) as resolve_mock:
        for p in _patches(engine_mod, fake_tool_runner, fake_chat):
            p.start()
        try:
            engine_mod.run_reply_engine(
                db=db, cfg=mock_config, tts=None,
                text="what's the weather and the top news today?",
                dialogue_memory=dialogue_memory,
            )
        finally:
            patch.stopall()

    # Sequential resolver drove the steps (called at least once per tool step).
    assert resolve_mock.call_count >= 2, (
        f"sequential resolver should drive steps when parallel disabled; "
        f"got {resolve_mock.call_count} calls"
    )
