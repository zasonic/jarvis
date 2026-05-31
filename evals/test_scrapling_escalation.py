"""
Regression eval: a JavaScript-rendered result rescued by the opt-in Scrapling
escalation.

Before this change the web-search cascade fetched raw HTML only: a result whose
body renders client-side (or sits behind an anti-bot wall) returns nothing the
plain fetch can read, so the cascade drops it and the tool emits the
links-only / honest-block envelope with no usable facts. With
`scrapling_fetch_enabled` set, the cascade escalates the same URLs through a
local Scrapling browser before surrendering, and the rendered content lands in
the untrusted-extract fence like any other source.

This file is behavioural, not judge-driven: it exercises the real
`WebSearchTool.run` against a mocked network and a mocked Scrapling helper, and
asserts the observable outcome flips with the flag. The two tests share an
identical network setup; only the flag differs, so the pair demonstrates the
improvement (no facts before, grounded facts after).

Run: .venv/bin/python -m pytest evals/test_scrapling_escalation.py -v
"""

from unittest.mock import Mock, patch

import pytest

from jarvis.tools.base import ToolContext
from jarvis.tools.builtin.web_search import WebSearchTool


def _make_ctx(scrapling_enabled):
    cfg = Mock()
    cfg.web_search_enabled = True
    cfg.voice_debug = False
    cfg.brave_search_api_key = ""
    cfg.wikipedia_fallback_enabled = False  # isolate the cascade from Wikipedia
    cfg.scrapling_fetch_enabled = scrapling_enabled
    cfg.scrapling_binary = "scrapling"
    cfg.scrapling_solve_cloudflare = False
    ctx = Mock(spec=ToolContext)
    ctx.user_print = Mock()
    ctx.cfg = cfg
    ctx.language = "en"
    return ctx


def _ddg_with_two_results():
    """DDG instant API empty + /lite/ page with two real result links."""
    instant = Mock(status_code=200)
    instant.json.return_value = {}
    instant.raise_for_status = Mock()
    lite = Mock(status_code=200)
    lite.content = (
        b'<html><body>'
        b'<a href="https://quotes.example/app">Quotes to Scrape single-page app</a>'
        b'<a href="https://quotes.example/more">More quotes rendered in JavaScript</a>'
        b'</body></html>'
    )
    return [instant, lite]


@pytest.mark.eval
class TestScraplingEscalationRescuesJsPage:
    """Same blocked JS result: empty before the flag, grounded after it."""

    @patch("jarvis.tools.builtin.scrapling_fetch.scrapling_fetch")
    @patch("jarvis.tools.builtin.web_search._fetch_page_content", return_value=None)
    @patch("jarvis.tools.builtin.web_search.requests.get")
    def test_disabled_yields_no_facts(self, mock_get, _mock_fetch, mock_scrapling):
        mock_get.side_effect = _ddg_with_two_results()

        result = WebSearchTool().run(
            {"search_query": "quotes to scrape javascript"}, _make_ctx(scrapling_enabled=False)
        )

        # No browser escalation happened, and no readable extract was produced.
        mock_scrapling.assert_not_called()
        assert "<<<BEGIN UNTRUSTED WEB EXTRACT>>>" not in result.reply_text

    @patch("jarvis.tools.builtin.scrapling_fetch.scrapling_fetch")
    @patch("jarvis.tools.builtin.web_search._fetch_page_content", return_value=None)
    @patch("jarvis.tools.builtin.web_search.requests.get")
    def test_enabled_rescues_into_fence(self, mock_get, _mock_fetch, mock_scrapling):
        mock_get.side_effect = _ddg_with_two_results()
        # The rendered extract shares query tokens, so it passes the relevance
        # filter and is treated as a genuine answer.
        mock_scrapling.return_value = (
            "Quotes to Scrape: a famous quote rendered in JavaScript by the app."
        )

        result = WebSearchTool().run(
            {"search_query": "quotes to scrape javascript"}, _make_ctx(scrapling_enabled=True)
        )

        mock_scrapling.assert_called()
        assert result.success is True
        # Rendered content must be inside the untrusted fence, like any source.
        assert "<<<BEGIN UNTRUSTED WEB EXTRACT>>>" in result.reply_text
        assert "rendered in JavaScript" in result.reply_text
        # The honest-block envelope must NOT fire — the query was rescued.
        assert "you have failed" not in result.reply_text.lower()
