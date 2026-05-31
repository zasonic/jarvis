"""Tests for fetch web page tool."""

import pytest
from unittest.mock import Mock, patch
import requests

from src.jarvis.tools.builtin.fetch_web_page import FetchWebPageTool
from src.jarvis.tools.base import ToolContext
from src.jarvis.tools.types import ToolExecutionResult


def _make_response_mock(**attrs) -> Mock:
    """Build a Mock that doubles as both the requests response and a context
    manager (the production code uses ``with requests.get(...) as resp`` so
    the connection is released deterministically).
    """
    resp = Mock(**attrs)
    resp.__enter__ = Mock(return_value=resp)
    resp.__exit__ = Mock(return_value=False)
    return resp


class TestFetchWebPageTool:
    """Test fetch web page tool functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tool = FetchWebPageTool()
        self.context = Mock(spec=ToolContext)
        self.context.user_print = Mock()
        # Scrapling escalation is opt-in; keep it OFF for the default-behaviour
        # tests so they never spawn a subprocess or hit the network. The
        # escalation-specific tests below flip it on explicitly.
        self.context.cfg = Mock(scrapling_fetch_enabled=False)

    def test_tool_properties(self):
        """Test tool metadata properties."""
        assert self.tool.name == "fetchWebPage"
        assert "fetch" in self.tool.description.lower()
        assert self.tool.inputSchema["type"] == "object"
        assert "url" in self.tool.inputSchema["required"]

    def test_run_no_args(self):
        """Test fetch web page with no arguments."""
        result = self.tool.run(None, self.context)

        assert isinstance(result, ToolExecutionResult)
        assert result.success is False
        assert "url" in result.reply_text.lower()

    def test_run_empty_url(self):
        """Test fetch web page with empty URL."""
        args = {"url": ""}
        result = self.tool.run(args, self.context)

        assert isinstance(result, ToolExecutionResult)
        assert result.success is False
        assert "url" in result.reply_text.lower()

    @patch('requests.get')
    def test_run_success(self, mock_get):
        """Test successful web page fetch."""
        mock_response = _make_response_mock(
            status_code=200,
            text='<html><head><title>Test</title></head><body><p>Content</p></body></html>',
            content=b'<html><head><title>Test</title></head><body><p>Content</p></body></html>',
            headers={'content-type': 'text/html'},
            raise_for_status=Mock(),
        )
        mock_get.return_value = mock_response

        args = {"url": "https://example.com"}
        result = self.tool.run(args, self.context)

        assert isinstance(result, ToolExecutionResult)
        assert result.success is True
        assert "example.com" in result.reply_text
        self.context.user_print.assert_called()

    @patch('requests.get')
    def test_run_success_without_beautifulsoup(self, mock_get):
        """Test successful web page fetch without BeautifulSoup."""
        mock_response = _make_response_mock(
            status_code=200,
            text='<html><body>Raw content</body></html>',
            content=b'<html><body>Raw content</body></html>',
            headers={'content-type': 'text/html'},
            raise_for_status=Mock(),
        )
        mock_get.return_value = mock_response

        with patch('builtins.__import__', side_effect=ImportError):
            args = {"url": "https://example.com"}
            result = self.tool.run(args, self.context)

        assert isinstance(result, ToolExecutionResult)
        assert result.success is True
        assert "Raw Content" in result.reply_text

    @patch('requests.get')
    def test_run_http_error(self, mock_get):
        """Test fetch web page with HTTP error."""
        mock_response = _make_response_mock(status_code=404)
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Not Found")
        mock_get.return_value = mock_response

        args = {"url": "https://example.com/notfound"}
        result = self.tool.run(args, self.context)

        assert isinstance(result, ToolExecutionResult)
        assert result.success is False
        assert "Failed to fetch page" in result.reply_text

    @patch('requests.get')
    def test_run_request_error(self, mock_get):
        """Test fetch web page with network error."""
        mock_get.side_effect = requests.exceptions.RequestException("Network error")

        args = {"url": "https://example.com"}
        result = self.tool.run(args, self.context)

        assert isinstance(result, ToolExecutionResult)
        assert result.success is False
        assert "Failed to fetch page" in result.reply_text

    def test_run_invalid_url(self):
        """Test fetch web page with invalid URL."""
        args = {"url": "not-a-url"}
        result = self.tool.run(args, self.context)
        assert isinstance(result, ToolExecutionResult)
        assert result.success is False
        assert "failed" in result.reply_text.lower() or "error" in result.reply_text.lower()

    @patch('requests.get')
    def test_run_with_links_extraction(self, mock_get):
        """Test fetch web page including link extraction when include_links=True."""
        html = (
            '<html><head><title>Links Page</title></head>'
            '<body><p>Intro</p>'
            '<a href="/relative">Relative Link</a>'
            '<a href="https://absolute.test/page">Absolute Link</a>'
            '<a href="mailto:test@example.com">Mail</a>'
            '</body></html>'
        )
        mock_response = _make_response_mock(
            status_code=200,
            text=html,
            content=html.encode(),
            raise_for_status=Mock(),
        )
        mock_get.return_value = mock_response

        args = {"url": "https://example.com", "include_links": True}
        result = self.tool.run(args, self.context)
        assert result.success is True
        assert isinstance(result, ToolExecutionResult)
        assert "Links found on page" in result.reply_text
        # relative link should be resolved to absolute
        assert "https://example.com/relative" in result.reply_text
        assert "absolute.test" in result.reply_text


class TestFetchWebPageScraplingEscalation:
    """The opt-in Scrapling fallback for JS-heavy / blocked pages."""

    def setup_method(self):
        self.tool = FetchWebPageTool()
        self.context = Mock(spec=ToolContext)
        self.context.user_print = Mock()
        self.context.cfg = Mock(scrapling_fetch_enabled=True)

    @patch('requests.get')
    @patch('src.jarvis.tools.builtin.fetch_web_page.scrapling_fetch')
    def test_thin_content_triggers_escalation(self, mock_scrapling, mock_get):
        """A near-empty extract (JS shell) escalates and returns Scrapling's text."""
        html = '<html><head><title>App</title></head><body><div id="root"></div></body></html>'
        mock_get.return_value = _make_response_mock(
            status_code=200, text=html, content=html.encode(), raise_for_status=Mock(),
        )
        mock_scrapling.return_value = "# Rendered\n\nReal article body from the SPA."

        result = self.tool.run({"url": "https://spa.example"}, self.context)

        assert result.success is True
        assert "Real article body" in result.reply_text
        mock_scrapling.assert_called_once()
        assert mock_scrapling.call_args.kwargs.get("cfg") is self.context.cfg

    @patch('requests.get')
    @patch('src.jarvis.tools.builtin.fetch_web_page.scrapling_fetch')
    def test_request_failure_triggers_escalation(self, mock_scrapling, mock_get):
        """A transport failure escalates instead of immediately failing."""
        mock_get.side_effect = requests.exceptions.RequestException("blocked")
        mock_scrapling.return_value = "Recovered content via stealth fetch."

        result = self.tool.run({"url": "https://blocked.example"}, self.context)

        assert result.success is True
        assert "Recovered content" in result.reply_text
        mock_scrapling.assert_called_once()

    @patch('requests.get')
    @patch('src.jarvis.tools.builtin.fetch_web_page.scrapling_fetch')
    def test_escalation_declined_keeps_failure(self, mock_scrapling, mock_get):
        """When Scrapling yields nothing, the original failure stands."""
        mock_get.side_effect = requests.exceptions.RequestException("blocked")
        mock_scrapling.return_value = None

        result = self.tool.run({"url": "https://blocked.example"}, self.context)

        assert result.success is False
        assert "Failed to fetch page" in result.reply_text

    @patch('requests.get')
    @patch('src.jarvis.tools.builtin.fetch_web_page.scrapling_fetch')
    def test_rich_content_does_not_escalate(self, mock_scrapling, mock_get):
        """A page that already yields plenty of text must not spawn Scrapling."""
        body = "<p>" + " ".join(f"sentence number {i} here." for i in range(60)) + "</p>"
        html = f'<html><head><title>Rich</title></head><body>{body}</body></html>'
        mock_get.return_value = _make_response_mock(
            status_code=200, text=html, content=html.encode(), raise_for_status=Mock(),
        )

        result = self.tool.run({"url": "https://rich.example"}, self.context)

        assert result.success is True
        mock_scrapling.assert_not_called()
