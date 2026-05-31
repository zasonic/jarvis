from __future__ import annotations
import requests

from ..llm import (
    active_api_key,
    active_backend,
    auth_headers,
    build_embeddings_payload,
    embeddings_url,
    parse_embedding_response,
)


def get_embedding(text: str, base_url: str, model: str, timeout_sec: float = 15.0) -> list[float] | None:
    """Embed ``text`` using the active backend (Ollama or OpenAI-compatible).

    Both wire shapes are handled: Ollama posts ``{"model","prompt"}`` to
    ``/api/embeddings`` and returns ``{"embedding": [...]}``; an
    OpenAI-compatible server posts ``{"model","input"}`` to ``/v1/embeddings``
    and returns ``{"data":[{"embedding":[...]}]}``. Returns None on any failure.
    """
    backend = active_backend()
    try:
        resp = requests.post(
            embeddings_url(backend, base_url),
            json=build_embeddings_payload(backend, model, text),
            headers=auth_headers(active_api_key()),
            timeout=timeout_sec,
        )
        resp.raise_for_status()
        vec = parse_embedding_response(resp.json())
        if isinstance(vec, list):
            return [float(x) for x in vec]
    except Exception:
        return None
    return None
