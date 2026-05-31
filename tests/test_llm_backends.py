"""Tests for the LLM backend abstraction (Ollama + OpenAI-compatible).

Two layers, both free of mocking libraries:

1. Pure builder/normaliser tests with real dicts — assert the exact request
   shapes and response normalisation per backend.
2. A real round-trip: a throwaway in-process ``http.server`` plays the role of
   a local inference server. We point the client at ``127.0.0.1:<port>`` and
   assert it hits the right path and normalises the real HTTP response. This is
   a genuine socket round-trip, not a stubbed function.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import jarvis.llm as L
from jarvis.memory.embeddings import get_embedding


@pytest.fixture(autouse=True)
def _reset_backend():
    """Keep the process-global backend from leaking across tests."""
    L.configure_llm_backend("ollama", "")
    yield
    L.configure_llm_backend("ollama", "")


class TestUrls:
    def test_ollama_paths(self):
        assert L.chat_url("ollama", "http://h:1") == "http://h:1/api/chat"
        assert L.embeddings_url("ollama", "http://h:1/") == "http://h:1/api/embeddings"
        assert L.models_url("ollama", "http://h:1") == "http://h:1/api/version"

    def test_openai_paths(self):
        assert L.chat_url("openai", "http://h:1") == "http://h:1/v1/chat/completions"
        assert L.embeddings_url("openai", "http://h:1/") == "http://h:1/v1/embeddings"
        assert L.models_url("openai", "http://h:1") == "http://h:1/v1/models"


class TestChatPayload:
    def test_ollama_carries_options_and_think(self):
        p = L.build_chat_payload(
            "ollama", "m", [{"role": "user", "content": "hi"}],
            num_ctx=8192, thinking=True, temperature=0.0,
        )
        assert p["options"]["num_ctx"] == 8192
        assert p["options"]["temperature"] == 0.0
        assert p["think"] is True
        assert "temperature" not in p  # ollama keeps it under options

    def test_openai_omits_options_and_think(self):
        p = L.build_chat_payload(
            "openai", "m", [{"role": "user", "content": "hi"}],
            num_ctx=8192, thinking=True, temperature=0.2,
        )
        assert "options" not in p
        assert "think" not in p
        assert p["temperature"] == 0.2

    def test_tools_passed_through_for_both(self):
        tools = [{"type": "function", "function": {"name": "webSearch"}}]
        for backend in ("ollama", "openai"):
            p = L.build_chat_payload(backend, "m", [], tools=tools)
            assert p["tools"] == tools

    def test_openai_extra_options_go_top_level(self):
        p = L.build_chat_payload(
            "openai", "m", [], extra_options={"max_tokens": 256},
        )
        assert p["max_tokens"] == 256

    def test_ollama_extra_options_merge_into_options(self):
        p = L.build_chat_payload(
            "ollama", "m", [], num_ctx=4096, extra_options={"top_p": 0.9},
        )
        assert p["options"]["top_p"] == 0.9
        assert p["options"]["num_ctx"] == 4096


class TestEmbeddingPayload:
    def test_ollama_uses_prompt(self):
        assert L.build_embeddings_payload("ollama", "m", "txt") == {"model": "m", "prompt": "txt"}

    def test_openai_uses_input(self):
        assert L.build_embeddings_payload("openai", "m", "txt") == {"model": "m", "input": "txt"}


class TestAuthHeaders:
    def test_empty_key_no_header(self):
        assert L.auth_headers("") == {}

    def test_key_sets_bearer(self):
        assert L.auth_headers("sk-x") == {"Authorization": "Bearer sk-x"}


class TestNormaliseChatResponse:
    def test_ollama_shape_passes_through(self):
        raw = {"message": {"content": "hello", "tool_calls": []}}
        assert L.normalise_chat_response(raw) is raw

    def test_openai_shape_lifts_message(self):
        raw = {"choices": [{"message": {"content": "hello"}}]}
        out = L.normalise_chat_response(raw)
        assert out["message"]["content"] == "hello"

    def test_openai_tool_call_arguments_parsed_to_dict(self):
        raw = {
            "choices": [{
                "message": {
                    "content": "",
                    "tool_calls": [{
                        "id": "c1", "type": "function",
                        "function": {"name": "webSearch", "arguments": '{"search_query": "x"}'},
                    }],
                }
            }]
        }
        out = L.normalise_chat_response(raw)
        tc = out["message"]["tool_calls"][0]
        assert tc["function"]["name"] == "webSearch"
        assert tc["function"]["arguments"] == {"search_query": "x"}  # parsed, not a string

    def test_malformed_tool_args_become_empty_dict(self):
        raw = {"choices": [{"message": {"tool_calls": [
            {"function": {"name": "x", "arguments": "not json"}}
        ]}}]}
        out = L.normalise_chat_response(raw)
        assert out["message"]["tool_calls"][0]["function"]["arguments"] == {}

    def test_normalised_output_works_with_extract_text(self):
        raw = {"choices": [{"message": {"content": "hi there"}}]}
        assert L.extract_text_from_response(L.normalise_chat_response(raw)) == "hi there"


class TestParseEmbeddingResponse:
    def test_ollama_shape(self):
        assert L.parse_embedding_response({"embedding": [1.0, 2.0]}) == [1.0, 2.0]

    def test_openai_shape(self):
        assert L.parse_embedding_response({"data": [{"embedding": [3.0]}]}) == [3.0]

    def test_unknown_shape_none(self):
        assert L.parse_embedding_response({"nope": 1}) is None


class TestConfigureBackend:
    def test_default_is_ollama(self):
        assert L.active_backend() == "ollama"

    def test_override_and_unknown_falls_back(self):
        L.configure_llm_backend("openai", "sk-1")
        assert L.active_backend() == "openai"
        assert L.active_api_key() == "sk-1"
        L.configure_llm_backend("bogus", "")
        assert L.active_backend() == "ollama"  # unknown → safe default


# --- Real in-process server round-trip --------------------------------------

class _Recorder:
    paths: list = []


def _make_server(response_for):
    """Start a throwaway HTTP server; response_for(path) returns a JSON dict."""
    _Recorder.paths = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            _Recorder.paths.append(self.path)
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            body = json.dumps(response_for(self.path)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


class TestRealRoundTrip:
    def test_openai_chat_hits_v1_and_normalises(self):
        def respond(path):
            return {"choices": [{"message": {"content": "live answer"}}]}

        server = _make_server(respond)
        try:
            host, port = server.server_address
            base = f"http://{host}:{port}"
            L.configure_llm_backend("openai", "")
            out = L.chat_with_messages(base, "m", [{"role": "user", "content": "hi"}], timeout_sec=5)
        finally:
            server.shutdown()
        assert _Recorder.paths == ["/v1/chat/completions"]
        # Returned dict is Ollama-normalised regardless of backend.
        assert out["message"]["content"] == "live answer"

    def test_ollama_chat_hits_api_chat(self):
        def respond(path):
            return {"message": {"content": "native answer"}}

        server = _make_server(respond)
        try:
            host, port = server.server_address
            base = f"http://{host}:{port}"
            L.configure_llm_backend("ollama", "")
            out = L.chat_with_messages(base, "m", [{"role": "user", "content": "hi"}], timeout_sec=5)
        finally:
            server.shutdown()
        assert _Recorder.paths == ["/api/chat"]
        assert out["message"]["content"] == "native answer"

    def test_openai_embeddings_round_trip(self):
        def respond(path):
            return {"data": [{"embedding": [0.1, 0.2, 0.3]}]}

        server = _make_server(respond)
        try:
            host, port = server.server_address
            base = f"http://{host}:{port}"
            L.configure_llm_backend("openai", "")
            vec = get_embedding("hello", base, "embed-model", timeout_sec=5)
        finally:
            server.shutdown()
        assert _Recorder.paths == ["/v1/embeddings"]
        assert vec == [0.1, 0.2, 0.3]
