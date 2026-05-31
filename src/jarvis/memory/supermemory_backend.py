"""Opt-in, cloud-optional memory backend backed by supermemory.

Jarvis is offline-first and 100% local by default. This module is the *only*
place that talks to the optional supermemory service, and it is wired so that
the offline-first promise is preserved:

- ``is_enabled(cfg)`` is the single gate. When it returns False (the default,
  i.e. no API key) every public function returns immediately — nothing is
  imported, no client is built, and no network call is made.
- The ``supermemory`` package is an optional dependency, imported lazily inside
  ``_get_client``. A stock install without the package keeps working; the
  feature simply stays off.
- Every network touch is wrapped fail-open: on any error we ``debug_log`` and
  return empty, so a remote failure is indistinguishable from "no remote
  results" and can never break a turn.

Privacy: only text that has already passed Jarvis's diary scrub (the post-rule
``scrub_deflection_sentences`` pass) and the redaction in
``update_daily_conversation_summary`` is mirrored. Raw transcripts and the hot
dialogue window never leave the device.

See ``supermemory_backend.spec.md`` for the full contract.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, List, Optional

from ..debug import debug_log

# Default container tag (supermemory's per-user isolation namespace) when the
# user has not configured one. Jarvis is single-user per install, so one stable
# tag is sufficient.
DEFAULT_CONTAINER_TAG = "jarvis_default_user"

# Bound every network call so a hung request can never stall a reply. Kept as a
# module constant rather than a config field to keep the surface area small.
_CLIENT_TIMEOUT_SEC = 5.0

# Client cache keyed by (api_key, base_url) so repeated turns reuse one client.
_CLIENT_CACHE: dict[tuple[str, str], Any] = {}
# Remember a missing optional package so the "not installed" notice logs once.
_IMPORT_FAILED = False


def is_enabled(cfg) -> bool:
    """Return True only when the user has explicitly opted in with an API key.

    This is the offline-first gate: when False, no supermemory code path runs,
    nothing is imported, and no network call is made.
    """
    return bool(
        getattr(cfg, "supermemory_enabled", False)
        and getattr(cfg, "supermemory_api_key", "")
    )


def _container_tag(cfg) -> str:
    return getattr(cfg, "supermemory_container_tag", "") or DEFAULT_CONTAINER_TAG


def _get_client(cfg) -> Optional[Any]:
    """Lazily construct and cache a Supermemory client. Never raises.

    Returns None when disabled, when the optional ``supermemory`` package is
    not installed, or when client construction fails — every caller then
    degrades to local-only memory.
    """
    global _IMPORT_FAILED

    if not is_enabled(cfg):
        return None

    api_key = str(getattr(cfg, "supermemory_api_key", "") or "")
    base_url = str(getattr(cfg, "supermemory_base_url", "") or "")
    cache_key = (api_key, base_url)
    cached = _CLIENT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        from supermemory import Supermemory  # optional dependency
    except ImportError:
        if not _IMPORT_FAILED:
            debug_log(
                "supermemory enabled but the 'supermemory' package is not "
                "installed; falling back to local memory (pip install supermemory)",
                "memory",
            )
            _IMPORT_FAILED = True
        return None

    try:
        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        # max_retries=0 + a short timeout keep a transient outage from blocking
        # the offline-first fast path. Guard against SDK versions that don't
        # accept these kwargs.
        try:
            client = Supermemory(
                timeout=_CLIENT_TIMEOUT_SEC, max_retries=0, **kwargs
            )
        except TypeError:
            client = Supermemory(**kwargs)
        _CLIENT_CACHE[cache_key] = client
        return client
    except Exception as e:
        debug_log(f"supermemory client construction failed (non-fatal): {e}", "memory")
        return None


# ── Internal helpers ─────────────────────────────────────────────────────────

def _extract(obj: Any, key: str) -> Any:
    """Read ``key`` from a dict-or-attribute response.

    The SDK may return plain dicts or typed model objects depending on version,
    so we tolerate both shapes.
    """
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _date_prefix(updated_at: Any) -> str:
    """Best-effort ``YYYY-MM-DD`` from an ISO timestamp; today's date as fallback.

    Matching the local diary's ``[YYYY-MM-DD]`` prefix lets remote hits sort and
    digest identically to local ones.
    """
    if isinstance(updated_at, str) and len(updated_at) >= 10:
        candidate = updated_at[:10]
        if candidate[4] == "-" and candidate[7] == "-":
            return candidate
    return datetime.now(timezone.utc).date().isoformat()


def merge_memory_results(
    local: List[str], remote: List[str], max_results: int
) -> List[str]:
    """Merge local + remote diary strings, dedupe, sort newest-first, cap.

    Local entries are the source of truth: added first, so on an exact-text tie
    the local copy is kept and the remote duplicate dropped. Ordering is
    newest-first by the ``[YYYY-MM-DD]`` prefix (Python's stable sort preserves
    the local-before-remote order within a date). Entries with no parseable
    prefix sort last but are still retained.
    """
    merged: List[str] = []
    seen: set[str] = set()
    for text in list(local) + list(remote):
        if not isinstance(text, str) or not text.strip():
            continue
        norm = text.strip()
        if norm in seen:
            continue
        seen.add(norm)
        merged.append(text)

    def _date_key(text: str) -> str:
        if text.startswith("[") and len(text) > 11:
            return text[1:11]
        return ""

    merged.sort(key=_date_key, reverse=True)
    return merged[: max(0, int(max_results))]


# ── Write path ───────────────────────────────────────────────────────────────

def _add(client, *, content: str, container_tag: str, metadata: dict, custom_id: str) -> None:
    """Call ``client.add`` with a stable ``custom_id``, tolerant of SDK versions.

    The ``custom_id`` makes a re-mirror update one logical document instead of
    appending a new one (the daily diary summary is cumulative and rewritten on
    every flush, so without a stable id supermemory would accumulate many
    partial copies). Older/newer SDKs that don't accept ``custom_id`` fall back
    to a plain add.
    """
    try:
        client.add(
            content=content,
            container_tag=container_tag,
            metadata=metadata,
            custom_id=custom_id,
        )
    except TypeError:
        client.add(content=content, container_tag=container_tag, metadata=metadata)


def mirror_diary_summary(
    cfg, summary: str, topics: Optional[str], date_utc: str
) -> None:
    """Best-effort mirror of a daily diary summary to supermemory.

    Only the already-scrubbed summary text is sent. Failures are swallowed; the
    local diary remains the source of truth. A stable per-day ``custom_id`` keeps
    the cumulative summary as a single logical document instead of one entry per
    flush.
    """
    if not is_enabled(cfg) or not summary or not summary.strip():
        return
    client = _get_client(cfg)
    if client is None:
        return
    try:
        metadata: dict[str, Any] = {
            "type": "diary",
            "date": date_utc,
            "source": "jarvis",
        }
        if topics:
            metadata["topics"] = topics
        _add(
            client,
            content=summary,
            container_tag=_container_tag(cfg),
            metadata=metadata,
            custom_id=f"jarvis-diary-{date_utc}",
        )
        debug_log(f"supermemory: mirrored diary summary for {date_utc}", "memory")
    except Exception as e:
        debug_log(f"supermemory diary mirror failed (non-fatal): {e}", "memory")


def mirror_graph_fact(cfg, fact: str, node_name: str) -> None:
    """Best-effort mirror of a single extracted graph fact to supermemory.

    A content-derived ``custom_id`` keeps re-extraction of the same fact (the
    cumulative diary re-extracts facts on every flush) from creating duplicates.
    """
    if not is_enabled(cfg) or not fact or not fact.strip():
        return
    client = _get_client(cfg)
    if client is None:
        return
    try:
        fact_id = hashlib.sha1(fact.strip().encode("utf-8")).hexdigest()[:16]
        _add(
            client,
            content=fact,
            container_tag=_container_tag(cfg),
            metadata={"type": "fact", "node": node_name or "", "source": "jarvis"},
            custom_id=f"jarvis-fact-{fact_id}",
        )
    except Exception as e:
        debug_log(f"supermemory fact mirror failed (non-fatal): {e}", "memory")


# ── Read path ────────────────────────────────────────────────────────────────

def search_memories(cfg, query: str, max_results: int = 10) -> List[str]:
    """Semantic search of supermemory, returned as ``[YYYY-MM-DD] text`` strings.

    The date prefix matches the local diary format so the merged list sorts and
    digests identically. Returns ``[]`` on any failure or when disabled, so the
    result is indistinguishable from "no remote hits".
    """
    if not is_enabled(cfg) or not query or not query.strip():
        return []
    client = _get_client(cfg)
    if client is None:
        return []
    try:
        resp = client.search.memories(
            q=query,
            container_tag=_container_tag(cfg),
            limit=max(1, int(max_results)),
        )
        results = _extract(resp, "results") or []
        out: List[str] = []
        for r in results:
            text = _extract(r, "memory") or _extract(r, "chunk") or ""
            if not isinstance(text, str) or not text.strip():
                continue
            # The documented dict shape uses camelCase ("updatedAt"); a typed
            # model shape would expose snake_case ("updated_at"). Try both.
            updated = _extract(r, "updatedAt") or _extract(r, "updated_at")
            date_str = _date_prefix(updated)
            out.append(f"[{date_str}] {text.strip()}")
        return out[:max_results]
    except Exception as e:
        debug_log(f"supermemory search failed (non-fatal): {e}", "memory")
        return []


def fetch_profile_facts(
    cfg, questions: List[str], max_results: int = 10
) -> List[str]:
    """Fetch user-profile facts (static + dynamic) as graph-style strings.

    Returned strings are prefixed ``[profile]`` so they slot into the same
    "Information the user has shared with you in prior conversations" block as
    local graph hits. Returns ``[]`` on any failure or when disabled.
    """
    if not is_enabled(cfg):
        return []
    client = _get_client(cfg)
    if client is None:
        return []
    try:
        q = "; ".join(s for s in (questions or []) if s) or None
        resp = client.profile(container_tag=_container_tag(cfg), q=q)
        profile = _extract(resp, "profile")
        facts: List[str] = []
        for key in ("static", "dynamic"):
            for fact in (_extract(profile, key) or []):
                if isinstance(fact, str) and fact.strip():
                    facts.append(f"[profile] {fact.strip()}")
        return facts[:max_results]
    except Exception as e:
        debug_log(f"supermemory profile failed (non-fatal): {e}", "memory")
        return []
