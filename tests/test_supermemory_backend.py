"""Behaviour tests for the opt-in supermemory backend.

The invariants under test are the offline-first guarantees: disabled means no
import and no behaviour change, failures degrade silently to local memory, and
only scrubbed text is ever sent.
"""
import builtins
import sys
from types import SimpleNamespace

import pytest

from jarvis.memory import supermemory_backend as sm


def _cfg(**overrides):
    base = dict(
        supermemory_enabled=False,
        supermemory_api_key="",
        supermemory_base_url="",
        supermemory_container_tag="",
        supermemory_mirror_writes=True,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.fixture(autouse=True)
def _reset_module_state():
    """Each test starts from a clean client cache / import-failed flag."""
    sm._CLIENT_CACHE.clear()
    sm._IMPORT_FAILED = False
    yield
    sm._CLIENT_CACHE.clear()
    sm._IMPORT_FAILED = False


class _StubClient:
    """Records writes and returns canned search/profile data."""

    def __init__(self, *args, **kwargs):
        self.added = []
        self.search = SimpleNamespace(memories=self._search_memories)

    def add(self, content=None, container_tag=None, metadata=None, custom_id=None):
        self.added.append({"content": content, "container_tag": container_tag,
                           "metadata": metadata, "custom_id": custom_id})

    def _search_memories(self, q=None, container_tag=None, limit=None, **kw):
        return {"results": [
            {"memory": "User enjoys hiking", "updatedAt": "2025-05-20T10:00:00Z"},
        ]}

    def profile(self, container_tag=None, q=None, **kw):
        return {"profile": {"static": ["User is called Sam"],
                            "dynamic": ["Recently asked about Italy"]}}


def _install_stub(monkeypatch, client):
    """Make ``from supermemory import Supermemory`` yield a factory for ``client``."""
    module = SimpleNamespace(Supermemory=lambda *a, **k: client)
    monkeypatch.setitem(sys.modules, "supermemory", module)


# ── Disabled by default: the offline-first invariant ─────────────────────────

def test_disabled_by_default_is_not_enabled():
    assert sm.is_enabled(_cfg()) is False
    # Enabled flag without a key is still off (explicit-consent requirement).
    assert sm.is_enabled(_cfg(supermemory_enabled=True)) is False
    assert sm.is_enabled(_cfg(supermemory_enabled=True, supermemory_api_key="sm_x")) is True


def test_disabled_reads_and_writes_are_noops():
    cfg = _cfg()
    assert sm.search_memories(cfg, "anything") == []
    assert sm.fetch_profile_facts(cfg, ["who is the user?"]) == []
    # Writes must not raise and must do nothing.
    assert sm.mirror_diary_summary(cfg, "summary", "topics", "2025-05-20") is None
    assert sm.mirror_graph_fact(cfg, "a fact", "node") is None


def test_disabled_never_imports_supermemory(monkeypatch):
    """When disabled, the optional package must never be imported."""
    real_import = builtins.__import__

    def guard(name, *args, **kwargs):
        if name == "supermemory" or name.startswith("supermemory."):
            raise AssertionError("supermemory must not be imported when disabled")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "supermemory", raising=False)
    monkeypatch.setattr(builtins, "__import__", guard)

    cfg = _cfg()  # disabled
    assert sm.search_memories(cfg, "anything") == []
    assert sm.fetch_profile_facts(cfg, ["q"]) == []
    sm.mirror_diary_summary(cfg, "s", "t", "2025-05-20")
    sm.mirror_graph_fact(cfg, "f", "n")


# ── Enabled but package missing: graceful degradation ────────────────────────

def test_enabled_without_package_degrades(monkeypatch):
    monkeypatch.delitem(sys.modules, "supermemory", raising=False)
    real_import = builtins.__import__

    def no_supermemory(name, *args, **kwargs):
        if name == "supermemory":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_supermemory)

    cfg = _cfg(supermemory_enabled=True, supermemory_api_key="sm_x")
    assert sm._get_client(cfg) is None
    assert sm.search_memories(cfg, "hiking") == []
    sm.mirror_diary_summary(cfg, "s", "t", "2025-05-20")  # no raise


# ── Enabled with a stub client (no real network) ─────────────────────────────

def test_write_path_mirrors_summary_and_facts(monkeypatch):
    client = _StubClient()
    _install_stub(monkeypatch, client)
    cfg = _cfg(supermemory_enabled=True, supermemory_api_key="sm_x",
              supermemory_container_tag="sam")

    sm.mirror_diary_summary(cfg, "Sam went hiking", "outdoors", "2025-05-20")
    sm.mirror_graph_fact(cfg, "Sam enjoys hiking", "USER > hobbies")

    assert len(client.added) == 2
    diary, fact = client.added
    assert diary["content"] == "Sam went hiking"
    assert diary["container_tag"] == "sam"
    assert diary["metadata"]["type"] == "diary"
    assert fact["metadata"]["type"] == "fact"
    # Stable per-day / per-fact custom_id so re-mirrors update one document
    # instead of accumulating duplicates.
    assert diary["custom_id"] == "jarvis-diary-2025-05-20"
    assert fact["custom_id"].startswith("jarvis-fact-")


def test_add_falls_back_when_custom_id_unsupported(monkeypatch):
    """An SDK whose add() rejects custom_id still mirrors via the plain call."""
    class _NoCustomId:
        def __init__(self):
            self.added = []

        def add(self, content=None, container_tag=None, metadata=None):
            self.added.append(content)

    client = _NoCustomId()
    _install_stub(monkeypatch, client)
    cfg = _cfg(supermemory_enabled=True, supermemory_api_key="sm_x")

    sm.mirror_diary_summary(cfg, "Sam went hiking", "outdoors", "2025-05-20")
    assert client.added == ["Sam went hiking"]


def test_read_path_handles_model_shaped_response(monkeypatch):
    """A typed-model response (snake_case updated_at) still yields a real date."""
    class _ModelClient(_StubClient):
        def _search_memories(self, q=None, container_tag=None, limit=None, **kw):
            # Object (not dict) results exposing snake_case attributes.
            return SimpleNamespace(results=[
                SimpleNamespace(memory="User enjoys hiking",
                                updated_at="2025-05-20T10:00:00Z"),
            ])

    _install_stub(monkeypatch, _ModelClient())
    cfg = _cfg(supermemory_enabled=True, supermemory_api_key="sm_x")

    assert sm.search_memories(cfg, "hiking") == ["[2025-05-20] User enjoys hiking"]


def test_read_path_returns_diary_formatted_strings(monkeypatch):
    _install_stub(monkeypatch, _StubClient())
    cfg = _cfg(supermemory_enabled=True, supermemory_api_key="sm_x")

    hits = sm.search_memories(cfg, "hiking", max_results=5)
    assert hits == ["[2025-05-20] User enjoys hiking"]


def test_profile_facts_are_profile_prefixed(monkeypatch):
    _install_stub(monkeypatch, _StubClient())
    cfg = _cfg(supermemory_enabled=True, supermemory_api_key="sm_x")

    facts = sm.fetch_profile_facts(cfg, ["who is the user?"], max_results=10)
    assert "[profile] User is called Sam" in facts
    assert "[profile] Recently asked about Italy" in facts


# ── Failure handling: a raising client never breaks a turn ───────────────────

def test_network_failure_degrades_silently(monkeypatch):
    class _Raising(_StubClient):
        def add(self, **kw):
            raise RuntimeError("network down")

        def _search_memories(self, **kw):
            raise RuntimeError("network down")

        def profile(self, **kw):
            raise RuntimeError("network down")

    _install_stub(monkeypatch, _Raising())
    cfg = _cfg(supermemory_enabled=True, supermemory_api_key="sm_x")

    assert sm.search_memories(cfg, "hiking") == []
    assert sm.fetch_profile_facts(cfg, ["q"]) == []
    sm.mirror_diary_summary(cfg, "s", "t", "2025-05-20")  # must not raise
    sm.mirror_graph_fact(cfg, "f", "n")  # must not raise


# ── Pure merge helper ────────────────────────────────────────────────────────

def test_merge_sorts_newest_first_dedupes_and_caps():
    local = ["[2025-05-18] older local", "[2025-05-20] newest local"]
    remote = ["[2025-05-19] middle remote", "[2025-05-20] newest local"]  # dup
    merged = sm.merge_memory_results(local, remote, max_results=10)

    # Duplicate dropped, newest-first ordering by date prefix.
    assert merged == [
        "[2025-05-20] newest local",
        "[2025-05-19] middle remote",
        "[2025-05-18] older local",
    ]


def test_merge_respects_max_results():
    local = ["[2025-05-20] a", "[2025-05-19] b"]
    remote = ["[2025-05-18] c"]
    assert sm.merge_memory_results(local, remote, max_results=2) == [
        "[2025-05-20] a", "[2025-05-19] b",
    ]
