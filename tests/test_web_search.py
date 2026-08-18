"""
tests/test_web_search.py
========================
Regression and unit tests for actions/web_search.py.

Verifies:
1. `_ddg_search` uses `ddgs.DDGS` (not deprecated `duckduckgo_search`).
2. Correct mapping of DDGS fields:
   - title -> title
   - body  -> snippet
   - href  -> url
3. `_format_ddg` renders formatted search output or handles empty results.
4. `web_search` handles missing queries, player logging, search mode, and compare mode.
5. Error handling returns a graceful failure message if search fails.
"""
import unittest
from unittest.mock import MagicMock, patch

from actions.web_search import (
    _ddg_search,
    _format_ddg,
    _compare,
    web_search,
)


class WebSearchTest(unittest.TestCase):
    def test_ddg_search_imports_ddgs_package(self):
        """ddgs package must be importable and provide DDGS class."""
        import ddgs
        self.assertTrue(hasattr(ddgs, "DDGS"))

    @patch("ddgs.DDGS")
    def test_ddg_search_maps_fields_correctly(self, mock_ddgs_cls):
        mock_instance = MagicMock()
        mock_instance.__enter__.return_value = mock_instance
        mock_instance.text.return_value = [
            {"title": "Sample Title", "body": "Sample Snippet Text", "href": "https://example.com/1"},
            {"title": "Second Title", "body": "Another description", "href": "https://example.com/2"},
        ]
        mock_ddgs_cls.return_value = mock_instance

        results = _ddg_search("test query", max_results=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["title"], "Sample Title")
        self.assertEqual(results[0]["snippet"], "Sample Snippet Text")
        self.assertEqual(results[0]["url"], "https://example.com/1")
        self.assertEqual(results[1]["title"], "Second Title")
        self.assertEqual(results[1]["snippet"], "Another description")
        self.assertEqual(results[1]["url"], "https://example.com/2")
        mock_instance.text.assert_called_once_with("test query", max_results=2)

    def test_format_ddg_empty_results(self):
        formatted = _format_ddg("unknown query", [])
        self.assertEqual(formatted, "No results found for: unknown query")

    def test_format_ddg_with_results(self):
        results = [
            {"title": "First", "snippet": "First snippet", "url": "https://first.com"},
            {"title": "Second", "snippet": "Second snippet", "url": "https://second.com"},
        ]
        formatted = _format_ddg("my query", results)
        self.assertIn("Search results for: my query", formatted)
        self.assertIn("1. First", formatted)
        self.assertIn("   First snippet", formatted)
        self.assertNotIn("https://first.com", formatted)
        self.assertNotIn("https://second.com", formatted)
        self.assertIn("2. Second", formatted)

    def test_web_search_empty_params(self):
        res = web_search({})
        self.assertEqual(res, "Please provide a search query, sir.")

    def test_web_search_logs_to_player(self):
        mock_player = MagicMock()
        with patch("actions.web_search._ddg_search", return_value=[]), \
             patch("actions.web_search._llm_summarize", side_effect=lambda q, r: r):
            res = web_search({"query": "hello world"}, player=mock_player)

        mock_player.write_log.assert_called_once_with("[Search] hello world")
        self.assertIn("No results found for: hello world", res)

    @patch("actions.web_search._ddg_search")
    @patch("actions.web_search._llm_summarize")
    def test_web_search_standard_flow(self, mock_summarize, mock_search):
        mock_search.return_value = [
            {"title": "Result 1", "snippet": "Snippet 1", "url": "https://res1.com"}
        ]
        mock_summarize.return_value = "Sir, here is the answer."

        res = web_search({"query": "test query"})
        self.assertEqual(res, "Sir, here is the answer.")
        mock_search.assert_called_once_with("test query")
        mock_summarize.assert_called_once()

    @patch("actions.web_search._ddg_search")
    def test_web_search_exception_handled(self, mock_search):
        mock_search.side_effect = RuntimeError("Network down")
        res = web_search({"query": "failing query"})
        self.assertTrue(res.startswith("Search failed, sir:"))
        self.assertIn("Network down", res)

    @patch("actions.web_search._ddg_search")
    @patch("actions.web_search._llm_summarize")
    def test_compare_mode(self, mock_summarize, mock_search):
        mock_search.return_value = [{"title": "t", "snippet": "feature snippet", "url": "u"}]
        mock_summarize.side_effect = lambda prompt, raw: f"Summary: {raw}"

        res = web_search({"mode": "compare", "items": ["iPhone", "Pixel"], "aspect": "battery"})
        self.assertIn("Comparison — BATTERY", res)
        self.assertIn("iPhone", res)
        self.assertIn("Pixel", res)


if __name__ == "__main__":
    unittest.main(verbosity=2)
