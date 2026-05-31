"""Direct LLM interaction utilities without extra features like temporal context.

Backend abstraction
-------------------
Jarvis talks to a *local* inference server. Two wire protocols are supported,
selected by a process-global backend configured once at startup
(``configure_llm_backend``), mirroring how the MCP tool cache is initialised
from settings:

- ``"ollama"`` (default): Ollama's native API — ``/api/chat``,
  ``/api/embeddings``; request carries ``options.num_ctx`` and ``think``.
- ``"openai"``: the OpenAI-compatible API — ``/v1/chat/completions``,
  ``/v1/embeddings`` — exposed by vLLM, llama.cpp ``server``, LM Studio, Jan,
  LocalAI, and by Ollama itself. Stays 100% local; no cloud providers.

The backend-specific request shaping and response **normalisation** live in
this module's pure helpers. Every chat response is normalised back to Ollama's
``{"message": {"content", "tool_calls"}}`` shape, so the reply engine's
tool-call parsing, ``extract_text_from_response``, and every other consumer are
backend-agnostic and untouched.
"""

from __future__ import annotations
from typing import Optional, Any, Dict, List, Callable
import threading
import requests
import json

from .debug import debug_log


class ToolsNotSupportedError(Exception):
    """Raised when the model returns HTTP 400 because native tool calling is not supported."""
    pass


# --- Backend configuration (process-global, set once at startup) ------------

BACKEND_OLLAMA = "ollama"
BACKEND_OPENAI = "openai"

_backend_lock = threading.Lock()
_active_backend = BACKEND_OLLAMA
_active_api_key = ""


def configure_llm_backend(backend: Optional[str], api_key: str = "") -> None:
    """Set the process-wide inference backend. Called once from the daemon.

    Unknown values fall back to Ollama so a misconfiguration can never wedge
    inference. Safe to call again (tests reconfigure between cases).
    """
    global _active_backend, _active_api_key
    b = (backend or BACKEND_OLLAMA).strip().lower()
    if b not in (BACKEND_OLLAMA, BACKEND_OPENAI):
        b = BACKEND_OLLAMA
    with _backend_lock:
        _active_backend = b
        _active_api_key = api_key or ""
    debug_log(f"llm: backend configured = {b}", "llm")


def active_backend() -> str:
    with _backend_lock:
        return _active_backend


def active_api_key() -> str:
    with _backend_lock:
        return _active_api_key


# --- Pure request/response builders (no I/O, no globals) --------------------

def chat_url(backend: str, base_url: str) -> str:
    base = base_url.rstrip("/")
    if backend == BACKEND_OPENAI:
        return f"{base}/v1/chat/completions"
    return f"{base}/api/chat"


def embeddings_url(backend: str, base_url: str) -> str:
    base = base_url.rstrip("/")
    if backend == BACKEND_OPENAI:
        return f"{base}/v1/embeddings"
    return f"{base}/api/embeddings"


def models_url(backend: str, base_url: str) -> str:
    base = base_url.rstrip("/")
    if backend == BACKEND_OPENAI:
        return f"{base}/v1/models"
    return f"{base}/api/version"


def auth_headers(api_key: str) -> Dict[str, str]:
    """Bearer header when an API key is set, else nothing (local servers)."""
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def build_chat_payload(
    backend: str,
    model: str,
    messages: List[Dict[str, Any]],
    *,
    tools: Optional[List[Dict[str, Any]]] = None,
    thinking: bool = False,
    num_ctx: Optional[int] = None,
    temperature: Optional[float] = None,
    stream: bool = False,
    extra_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Shape a chat request for the given backend.

    Ollama carries ``options.num_ctx`` and a top-level ``think`` flag; the
    OpenAI-compatible shape has neither (context length is server-side) and
    uses top-level ``temperature``. Both accept the identical OpenAI-style
    ``tools`` schema Jarvis already generates.
    """
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": stream,
    }
    if backend == BACKEND_OPENAI:
        if temperature is not None:
            payload["temperature"] = temperature
    else:
        options: Dict[str, Any] = {}
        if num_ctx is not None:
            options["num_ctx"] = num_ctx
        if temperature is not None:
            options["temperature"] = temperature
        if extra_options and isinstance(extra_options, dict):
            options.update(extra_options)
        payload["options"] = options
        payload["think"] = thinking
    if backend == BACKEND_OPENAI and extra_options and isinstance(extra_options, dict):
        # OpenAI-compatible servers take these at the top level (e.g. max_tokens).
        payload.update(extra_options)
    if tools and isinstance(tools, list) and len(tools) > 0:
        payload["tools"] = tools
    return payload


def build_embeddings_payload(backend: str, model: str, text: str) -> Dict[str, Any]:
    if backend == BACKEND_OPENAI:
        return {"model": model, "input": text}
    return {"model": model, "prompt": text}


def normalise_chat_response(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce any backend's chat response into Ollama's ``{"message": {...}}``.

    Ollama responses already have ``message``; pass them through. An
    OpenAI-compatible response carries ``choices[0].message`` with
    ``tool_calls[].function.arguments`` as a JSON **string** — we lift that
    message up and parse each tool-call's arguments to a dict so the engine's
    existing Ollama-shaped tool handling works without a special case.
    """
    if not isinstance(raw, dict):
        return raw
    if isinstance(raw.get("message"), dict):
        return raw
    choices = raw.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        msg = choices[0].get("message")
        if isinstance(msg, dict):
            out = dict(msg)
            tcs = out.get("tool_calls")
            if isinstance(tcs, list):
                out["tool_calls"] = [_normalise_tool_call(tc) for tc in tcs]
            return {"message": out}
    return raw


def _normalise_tool_call(tc: Any) -> Any:
    """Parse an OpenAI tool call's stringified ``function.arguments`` to a dict."""
    if not isinstance(tc, dict):
        return tc
    fn = tc.get("function")
    if not isinstance(fn, dict):
        return tc
    args = fn.get("arguments")
    if isinstance(args, str):
        try:
            parsed = json.loads(args) if args.strip() else {}
        except Exception:
            parsed = {}
        new_fn = dict(fn)
        new_fn["arguments"] = parsed
        new_tc = dict(tc)
        new_tc["function"] = new_fn
        return new_tc
    return tc


def parse_embedding_response(raw: Dict[str, Any]) -> Optional[List[float]]:
    """Extract the embedding vector from either backend's response shape."""
    if not isinstance(raw, dict):
        return None
    # Ollama: {"embedding": [...]}
    emb = raw.get("embedding")
    if isinstance(emb, list):
        return emb
    # OpenAI: {"data": [{"embedding": [...]}]}
    data = raw.get("data")
    if isinstance(data, list) and data and isinstance(data[0], dict):
        vec = data[0].get("embedding")
        if isinstance(vec, list):
            return vec
    return None


# --- Public inference functions (use the active backend) --------------------

def call_llm_direct(base_url: str, chat_model: str, system_prompt: str, user_content: str, timeout_sec: float = 10.0, thinking: bool = False, num_ctx: int = 4096, temperature: Optional[float] = None) -> Optional[str]:
    """Direct LLM call without temporal context, location, or other ask_coach features.

    ``num_ctx`` controls Ollama's context window for this call (ignored by the
    OpenAI-compatible backend, where context length is server-side). Default
    4096 is fine for small classification-shaped passes; callers that assemble
    richer prompts (planner with dialogue + memory + tool catalogue) should
    pass a larger value to avoid silent truncation.

    ``temperature`` is forwarded when set. Pass ``0.0`` for classification /
    extraction calls where determinism beats creativity — servers default to
    ~0.8 otherwise, which can flake small models on rule-following tasks.
    """
    backend = active_backend()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    payload = build_chat_payload(
        backend, chat_model, messages,
        thinking=thinking, num_ctx=num_ctx, temperature=temperature, stream=False,
    )
    try:
        with requests.post(
            chat_url(backend, base_url), json=payload,
            headers=auth_headers(active_api_key()), timeout=timeout_sec,
        ) as resp:
            resp.raise_for_status()
            data = resp.json()

        if isinstance(data, dict):
            content = extract_text_from_response(normalise_chat_response(data))
            if isinstance(content, str) and content.strip():
                return content
            debug_log(f"call_llm_direct: empty content from response keys={list(data.keys())}", "llm")
    except requests.exceptions.Timeout:
        debug_log(f"call_llm_direct: timeout after {timeout_sec}s", "llm")
        return None
    except Exception as e:
        debug_log(f"call_llm_direct: request failed — {e}", "llm")
        return None

    return None


def call_llm_streaming(
    base_url: str,
    chat_model: str,
    system_prompt: str,
    user_content: str,
    on_token: Optional[Callable[[str], None]] = None,
    timeout_sec: float = 30.0,
    thinking: bool = False,
) -> Optional[str]:
    """
    Streaming LLM call that invokes on_token callback for each token received.

    Works for both backends: Ollama streams newline-delimited JSON objects with
    ``message.content``; OpenAI-compatible servers stream Server-Sent Events
    (``data: {...}``) with ``choices[0].delta.content``. Both are handled below.

    Returns the complete response text, or None on error.
    """
    backend = active_backend()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    payload = build_chat_payload(
        backend, chat_model, messages,
        thinking=thinking, num_ctx=4096, stream=True,
    )

    # Use ``with`` so the streaming response (and the underlying TCP
    # connection) is released even if iter_lines exits early via an
    # exception or the caller stops consuming.
    try:
        with requests.post(
            chat_url(backend, base_url),
            json=payload,
            headers=auth_headers(active_api_key()),
            timeout=timeout_sec,
            stream=True,
        ) as resp:
            resp.raise_for_status()

            full_response = []
            for line in resp.iter_lines():
                if not line:
                    continue
                token = _extract_stream_token(backend, line)
                if token:
                    full_response.append(token)
                    if on_token:
                        on_token(token)

            result = "".join(full_response)
            return result if result.strip() else None

    except requests.exceptions.Timeout:
        return None
    except Exception:
        return None


def _extract_stream_token(backend: str, line: bytes) -> Optional[str]:
    """Pull the incremental text token from one streamed line, per backend."""
    try:
        if backend == BACKEND_OPENAI:
            # SSE framing: lines look like ``data: {json}`` and a final ``data: [DONE]``.
            text = line.decode("utf-8") if isinstance(line, (bytes, bytearray)) else str(line)
            if text.startswith("data:"):
                text = text[len("data:"):].strip()
            if not text or text == "[DONE]":
                return None
            data = json.loads(text)
            choices = data.get("choices")
            if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if isinstance(content, str) and content:
                    return content
            return None
        # Ollama: each line is a standalone JSON object with message.content.
        data = json.loads(line)
        if "message" in data and isinstance(data["message"], dict):
            content = data["message"].get("content", "")
            if content:
                return content
    except json.JSONDecodeError:
        return None
    except Exception:
        return None
    return None


def extract_text_from_response(data: Dict[str, Any]) -> Optional[str]:
    """Extract text from LLM response - supports multiple response formats."""
    # Preferred: Ollama chat non-stream format
    if "message" in data and isinstance(data["message"], dict):
        content = data["message"].get("content")
        if isinstance(content, str):
            return content

    # Fallback: OpenAI-style format
    if "choices" in data and isinstance(data["choices"], list) and len(data["choices"]) > 0:
        choice = data["choices"][0]
        if isinstance(choice, dict):
            if "message" in choice and isinstance(choice["message"], dict):
                content = choice["message"].get("content")
                if isinstance(content, str):
                    return content
            elif "text" in choice:
                content = choice["text"]
                if isinstance(content, str):
                    return content

    # Another fallback: direct "content" field
    if "content" in data:
        content = data["content"]
        if isinstance(content, str):
            return content

    return None


def chat_with_messages(
    base_url: str,
    chat_model: str,
    messages: List[Dict[str, str]],
    timeout_sec: float = 30.0,
    extra_options: Optional[Dict[str, Any]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    thinking: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Send an arbitrary messages array to the LLM and return a normalised response.

    The returned dict is always Ollama-shaped (``{"message": {...}}``)
    regardless of backend, so callers parse content and tool calls uniformly.

    Args:
        base_url: inference server base URL
        chat_model: model name
        messages: conversation messages
        timeout_sec: request timeout
        extra_options: additional model options (Ollama ``options`` merge, or
            top-level fields like ``max_tokens`` for OpenAI-compatible servers)
        tools: optional list of tools in OpenAI-compatible JSON schema format
        thinking: enable thinking/reasoning mode (Ollama only)

    Returns the normalised JSON response dict on success, or None on error/timeout.
    """
    backend = active_backend()
    # Main agentic chat uses 8192 so the system prompt (tool list + protocol
    # guidance + memory context) doesn't overflow and force truncation — which
    # previously dropped the tool schema on smaller models like gemma4:e2b.
    payload = build_chat_payload(
        backend, chat_model, messages,
        tools=tools, thinking=thinking, num_ctx=8192,
        stream=False, extra_options=extra_options,
    )

    try:
        with requests.post(
            chat_url(backend, base_url), json=payload,
            headers=auth_headers(active_api_key()), timeout=timeout_sec,
        ) as resp:
            resp.raise_for_status()
            data = resp.json()
        if isinstance(data, dict):
            return normalise_chat_response(data)
    except requests.exceptions.Timeout:
        print("  ⏱️ LLM request timed out", flush=True)
        return None
    except requests.exceptions.ConnectionError as e:
        print(f"  ❌ LLM connection error: {e}", flush=True)
        return None
    except requests.exceptions.HTTPError as e:
        # Raise a specific error when the model rejects the tools parameter (HTTP 400).
        # This lets the caller fall back to text-based tool calling automatically.
        if e.response is not None and e.response.status_code == 400 and tools:
            raise ToolsNotSupportedError(
                f"Model {chat_model!r} returned HTTP 400 — native tools API not supported"
            )
        print(f"  ❌ LLM HTTP error: {e}", flush=True)
        return None
    except Exception as e:
        print(f"  ❌ LLM error: {e}", flush=True)
        return None

    return None


def health_check(backend: str, base_url: str, api_key: str = "", timeout_sec: float = 5.0):
    """Check that the inference server is reachable. Returns (ok, detail).

    Ollama: ``GET /api/version`` → version string. OpenAI-compatible:
    ``GET /v1/models`` → number of models. Returns ``(False, None)`` on any
    failure so callers can present a clear "server not reachable" state.
    """
    try:
        resp = requests.get(
            models_url(backend, base_url), headers=auth_headers(api_key), timeout=timeout_sec
        )
        if resp.status_code == 200:
            data = resp.json()
            if backend == BACKEND_OPENAI:
                models = data.get("data") if isinstance(data, dict) else None
                detail = f"{len(models)} models" if isinstance(models, list) else "ok"
                return True, detail
            return True, (data.get("version", "unknown") if isinstance(data, dict) else "unknown")
    except Exception:
        pass
    return False, None
