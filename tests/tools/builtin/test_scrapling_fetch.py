"""Tests for the optional Scrapling escalation helper.

The helper shells out to a locally-installed ``scrapling`` CLI only when the
user has opted in via config. These tests pin the behavioural contract the
two callers (``fetchWebPage`` and the ``webSearch`` cascade) rely on:

* it is a no-op when disabled or when the URL fails the SSRF guard,
* it walks the get -> fetch -> stealthy-fetch ladder and stops at the first
  non-empty result,
* it always passes ``--ai-targeted`` (main-content + injection sanitisation),
* a missing binary degrades to ``None`` instead of raising,
* the temp output file is always cleaned up.
"""

import os
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.jarvis.tools.builtin import scrapling_fetch as sf


def _cfg(**overrides):
    base = dict(
        scrapling_fetch_enabled=True,
        scrapling_binary="scrapling",
        scrapling_solve_cloudflare=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class TestGating:
    def test_disabled_is_noop(self):
        with patch.object(sf, "_run_scrapling") as run:
            out = sf.scrapling_fetch("https://example.com", cfg=_cfg(scrapling_fetch_enabled=False))
        assert out is None
        run.assert_not_called()

    def test_missing_cfg_is_noop(self):
        # A config object without the flag must be treated as "off".
        with patch.object(sf, "_run_scrapling") as run:
            out = sf.scrapling_fetch("https://example.com", cfg=SimpleNamespace())
        assert out is None
        run.assert_not_called()

    def test_ssrf_rejected_before_spawn(self):
        with patch.object(sf, "_is_public_url", return_value=False), \
             patch.object(sf, "_run_scrapling") as run:
            out = sf.scrapling_fetch("http://127.0.0.1/admin", cfg=_cfg())
        assert out is None
        run.assert_not_called()

    def test_expired_deadline_is_noop(self):
        past = time.monotonic() - 1.0
        with patch.object(sf, "_is_public_url", return_value=True), \
             patch.object(sf, "_run_scrapling") as run:
            out = sf.scrapling_fetch("https://example.com", cfg=_cfg(), deadline=past)
        assert out is None
        run.assert_not_called()


class TestLadder:
    def test_stops_at_first_nonempty(self):
        with patch.object(sf, "_is_public_url", return_value=True), \
             patch.object(sf, "_run_scrapling", return_value="hello world") as run:
            out = sf.scrapling_fetch("https://example.com", cfg=_cfg())
        assert out == "hello world"
        # First stage succeeded, so only one CLI invocation.
        assert run.call_count == 1
        assert run.call_args_list[0].args[0] == "get"

    def test_escalates_get_then_fetch_then_stealthy(self):
        # get and fetch yield nothing; stealthy-fetch finally returns content.
        with patch.object(sf, "_is_public_url", return_value=True), \
             patch.object(sf, "_run_scrapling", side_effect=[None, None, "deep content"]) as run:
            out = sf.scrapling_fetch("https://example.com", cfg=_cfg())
        assert out == "deep content"
        subcmds = [c.args[0] for c in run.call_args_list]
        assert subcmds == ["get", "fetch", "stealthy-fetch"]

    def test_all_stages_empty_returns_none(self):
        with patch.object(sf, "_is_public_url", return_value=True), \
             patch.object(sf, "_run_scrapling", side_effect=[None, None, None]):
            out = sf.scrapling_fetch("https://example.com", cfg=_cfg())
        assert out is None


class TestCliContract:
    def _capture(self, written_content, solve=False, stage_filter=None):
        """Run scrapling_fetch with a mocked subprocess that writes a file and
        return the list of argv lists actually executed."""
        argvs = []

        def fake_run(argv, **kwargs):
            argvs.append(argv)
            # Command layout: [binary, "extract", subcmd, url, out_path, ...]
            out_path = argv[4]
            subcmd = argv[2]
            if stage_filter is None or subcmd == stage_filter:
                with open(out_path, "w", encoding="utf-8") as fh:
                    fh.write(written_content)
            return Mock(returncode=0, stdout="", stderr="")

        with patch.object(sf, "_is_public_url", return_value=True), \
             patch("subprocess.run", side_effect=fake_run):
            out = sf.scrapling_fetch(
                "https://example.com", cfg=_cfg(scrapling_solve_cloudflare=solve)
            )
        return out, argvs

    def test_ai_targeted_always_present(self):
        out, argvs = self._capture("real content")
        assert out == "real content"
        assert "--ai-targeted" in argvs[0]

    def test_uses_configured_binary(self):
        argvs = []

        def fake_run(argv, **kwargs):
            argvs.append(argv)
            with open(argv[4], "w", encoding="utf-8") as fh:
                fh.write("ok")
            return Mock(returncode=0, stdout="", stderr="")

        with patch.object(sf, "_is_public_url", return_value=True), \
             patch("subprocess.run", side_effect=fake_run):
            sf.scrapling_fetch("https://example.com", cfg=_cfg(scrapling_binary="/opt/scr/scrapling"))
        assert argvs[0][0] == "/opt/scr/scrapling"

    def test_solve_cloudflare_only_on_stealthy_and_when_enabled(self):
        # Force every stage to "fail" so the ladder reaches stealthy-fetch.
        out, argvs = self._capture("x", solve=True, stage_filter="never")
        stealthy = [a for a in argvs if a[2] == "stealthy-fetch"]
        assert stealthy, "ladder should have reached stealthy-fetch"
        assert "--solve-cloudflare" in stealthy[0]
        # get/fetch stages must never carry the cloudflare flag.
        for a in argvs:
            if a[2] in ("get", "fetch"):
                assert "--solve-cloudflare" not in a

    def test_solve_cloudflare_absent_by_default(self):
        out, argvs = self._capture("x", solve=False, stage_filter="never")
        for a in argvs:
            assert "--solve-cloudflare" not in a


class TestRobustness:
    def test_missing_binary_returns_none(self):
        with patch.object(sf, "_is_public_url", return_value=True), \
             patch("subprocess.run", side_effect=FileNotFoundError("scrapling")):
            out = sf.scrapling_fetch("https://example.com", cfg=_cfg())
        assert out is None

    def test_timeout_returns_none(self):
        import subprocess

        with patch.object(sf, "_is_public_url", return_value=True), \
             patch("subprocess.run", side_effect=subprocess.TimeoutExpired("scrapling", 5)):
            out = sf.scrapling_fetch("https://example.com", cfg=_cfg())
        assert out is None

    def test_tempfiles_cleaned_up(self, tmp_path, monkeypatch):
        # Point tempfile at an empty dir we can inspect afterwards.
        monkeypatch.setenv("TMPDIR", str(tmp_path))
        import tempfile
        monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))

        def fake_run(argv, **kwargs):
            with open(argv[4], "w", encoding="utf-8") as fh:
                fh.write("content")
            return Mock(returncode=0, stdout="", stderr="")

        with patch.object(sf, "_is_public_url", return_value=True), \
             patch("subprocess.run", side_effect=fake_run):
            sf.scrapling_fetch("https://example.com", cfg=_cfg())
        leftovers = [p for p in os.listdir(tmp_path) if p.endswith(".md")]
        assert leftovers == []
