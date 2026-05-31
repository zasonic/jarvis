# LLM backend abstraction

## Purpose

Jarvis talks to a **local** inference server. Historically that was Ollama's
native API only. This abstraction lets it also drive any local
**OpenAI-compatible** server (vLLM, llama.cpp `server`, LM Studio, Jan,
LocalAI — and Ollama's own `/v1` endpoint) without changing any call site.
It stays 100% local: no cloud providers are offered, by design.

## Scope

`src/jarvis/llm.py` (request shaping, normalisation, the process-global
backend) and `src/jarvis/memory/embeddings.py` (embeddings). Startup wiring
in `daemon.py`; warmup in `listening/intent_judge.py`; health check in
`desktop_app/setup_wizard.py`; settings UI in `desktop_app/settings_window.py`.

## Configuration

| Key | Default | Purpose |
|-----|---------|---------|
| `llm_backend` | `"ollama"` | `"ollama"` (native API) or `"openai"` (OpenAI-compatible API). Unknown values fall back to `"ollama"`. |
| `llm_api_key` | `""` | Optional bearer token; most local servers ignore it. |
| `ollama_base_url` | `http://127.0.0.1:11434` | Endpoint base for **either** backend (name kept for back-compat). |

## Backend selection

A process-global backend is set **once** at daemon startup via
`configure_llm_backend(cfg.llm_backend, cfg.llm_api_key)` (mirroring how the
MCP tool cache is initialised from settings). Inference functions read it
through `active_backend()` / `active_api_key()`, so the ~20 call sites across
the reply engine, planner, router, memory, tools and dictation are untouched.
Tests reconfigure freely; the default is always Ollama.

## Wire differences (confined to llm.py)

| Aspect | `ollama` | `openai` |
|--------|----------|----------|
| Chat path | `/api/chat` | `/v1/chat/completions` |
| Embeddings path | `/api/embeddings` | `/v1/embeddings` |
| Health path | `/api/version` | `/v1/models` |
| Context window | `options.num_ctx` | server-side (omitted) |
| Thinking | top-level `think` | omitted |
| Temperature | under `options` | top-level |
| Embeddings field | `prompt` | `input` |
| Auth | none | `Authorization: Bearer` if key set |

These are produced by **pure** builders (`chat_url`, `embeddings_url`,
`models_url`, `build_chat_payload`, `build_embeddings_payload`,
`auth_headers`) — no I/O, unit-tested with real dicts.

## Response normalisation (the key to zero call-site churn)

Every chat response is coerced back to Ollama's shape
`{"message": {"content", "tool_calls"}}` by `normalise_chat_response` before
returning from `chat_with_messages`. An OpenAI-compatible response's
`choices[0].message` is lifted up, and each
`tool_calls[].function.arguments` JSON **string** is parsed to a dict, so the
reply engine's existing Ollama-shaped tool-call handling and
`extract_text_from_response` work unchanged for both backends. Embeddings are
normalised by `parse_embedding_response` (`embedding` vs `data[0].embedding`).
Streaming handles both Ollama newline-JSON and OpenAI SSE (`data: {...}`)
framing.

## Warmup & health

- `warm_up_ollama_model` is a **no-op for non-Ollama backends** (OpenAI-style
  servers keep models resident and expose no keep-alive warmup), fail-soft.
- `health_check(backend, base_url, api_key)` returns `(ok, detail)`:
  Ollama `/api/version` → version; OpenAI `/v1/models` → model count. The
  setup wizard uses it so a reachable OpenAI-compatible server reads as ready.

## Invariants

- **Ollama default is byte-identical to before** when the new keys are absent:
  same paths, same payloads (`options.num_ctx`, `think`), same parsing.
- **Fail-soft**: unknown `llm_backend` → Ollama; all HTTP errors return
  `None` (or raise `ToolsNotSupportedError` on a 400 with tools, preserved for
  both backends so the text-tool fallback still triggers).
- 100% local: only Ollama and generic OpenAI-compatible **local** servers are
  exposed; no hosted cloud provider is added.

## Tests

`tests/test_llm_backends.py` — pure builders/normalisers asserted with real
dicts, plus a **real round-trip** against an in-process `http.server` standing
in for a local inference server (real socket, no mocking): confirms the client
hits the correct path per backend and normalises real HTTP responses for both
chat and embeddings.
