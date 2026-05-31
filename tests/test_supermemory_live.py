"""Live end-to-end smoke test for the supermemory backend.

Skipped by default. The unit tests in ``test_supermemory_backend.py`` cover all
behaviour against a stub; this module proves the integration against the *real*
installed SDK, which is the one thing a fail-open design cannot verify on its
own (a wrong method name or signature would otherwise degrade silently).

Run it by installing the SDK and exporting a key:

    pip install supermemory
    export SUPERMEMORY_API_KEY=sm_...
    # optional, for a self-hosted instance:
    export SUPERMEMORY_BASE_URL=https://my-instance.example.com
    pytest tests/test_supermemory_live.py -v

It uses a throwaway container tag so it never touches real user data, and it
avoids asserting on add->search round-trip timing (supermemory processes writes
asynchronously). It asserts the SDK surface we depend on works: the client
constructs, the startup probe reports connected, and add/search/profile return
the shapes the backend expects.
"""
import os
import time
from types import SimpleNamespace

import pytest

from jarvis.memory import supermemory_backend as sm

_API_KEY = os.environ.get("SUPERMEMORY_API_KEY", "")

pytestmark = pytest.mark.skipif(
    not _API_KEY,
    reason="set SUPERMEMORY_API_KEY to run the live supermemory smoke test",
)


@pytest.fixture
def live_cfg():
    sm._CLIENT_CACHE.clear()
    sm._IMPORT_FAILED = False
    # Throwaway, run-unique container so we never collide with real data.
    tag = f"jarvis-smoketest-{int(time.time())}"
    return SimpleNamespace(
        supermemory_enabled=True,
        supermemory_api_key=_API_KEY,
        supermemory_base_url=os.environ.get("SUPERMEMORY_BASE_URL", ""),
        supermemory_container_tag=tag,
        supermemory_mirror_writes=True,
    )


def test_package_is_installed():
    pytest.importorskip("supermemory")


def test_startup_probe_connects(live_cfg, capsys):
    sm.startup_check(live_cfg)
    out = capsys.readouterr().out
    assert "Supermemory connected" in out, out


def test_write_and_read_surface(live_cfg):
    # Writes must not raise against the real SDK (proves add() signature).
    sm.mirror_diary_summary(
        live_cfg,
        "Smoke test: the user enjoys hiking and prefers tea over coffee.",
        "hobbies, preferences",
        "2026-05-31",
    )
    sm.mirror_graph_fact(live_cfg, "The user prefers tea over coffee.", "USER > preferences")

    # Reads must return the expected types (proves search.memories / profile
    # signatures and our parsing). We don't assert the just-written item is
    # present: supermemory ingests asynchronously, so immediate recall is not
    # guaranteed.
    hits = sm.search_memories(live_cfg, "what does the user like to drink?", max_results=5)
    assert isinstance(hits, list)
    assert all(isinstance(h, str) for h in hits)

    facts = sm.fetch_profile_facts(live_cfg, ["what are the user's preferences?"], max_results=5)
    assert isinstance(facts, list)
    assert all(isinstance(f, str) and f.startswith("[profile]") for f in facts)
