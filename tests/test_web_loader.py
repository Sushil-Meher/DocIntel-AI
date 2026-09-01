import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.web_loader import load_webpage


def fake_response(html, status_code=200):
    response = Mock()
    response.status_code = status_code
    response.text = html
    response.raise_for_status = Mock()
    return response


class WebLoaderTests(unittest.TestCase):

    def test_valid_page_extracts_useful_text(self):
        html = """
        <html><body>
        <p>Acme Corp builds cloud storage tools for small businesses.</p>
        <p>Our flagship product backs up files automatically every night,
        keeps version history, and restores anything within seconds.</p>
        </body></html>
        """

        with patch("src.web_loader.requests.get", return_value=fake_response(html)):
            document = load_webpage("https://acme.example.com")

        self.assertIn("cloud storage tools", document.text)
        self.assertEqual(document.source, "https://acme.example.com")

    def test_missing_scheme_raises_clear_error(self):
        with patch(
            "src.web_loader.requests.get",
            side_effect=requests.exceptions.MissingSchema("Invalid URL")
        ):
            with self.assertRaises(ValueError):
                load_webpage("example.com")

    def test_http_404_raises_clear_error(self):
        response = fake_response("<html></html>", status_code=404)
        response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=response
        )

        with patch("src.web_loader.requests.get", return_value=response):
            with self.assertRaises(ValueError):
                load_webpage("https://example.com/missing")

    def test_connection_failure_raises_clear_error(self):
        with patch(
            "src.web_loader.requests.get",
            side_effect=requests.exceptions.ConnectionError("boom")
        ):
            with self.assertRaises(ValueError):
                load_webpage("https://unreachable.example.com")

    def test_empty_page_raises_clear_error(self):
        html = "<html><body></body></html>"

        with patch("src.web_loader.requests.get", return_value=fake_response(html)):
            with self.assertRaises(ValueError):
                load_webpage("https://example.com/blank")

    def test_boilerplate_is_stripped_but_content_kept(self):
        html = """
        <html>
        <head><style>body { color: red; }</style></head>
        <body>
        <nav>Home About Contact</nav>
        <script>trackVisit();</script>
        <header>Acme Corp</header>
        <p>Acme Corp builds cloud storage tools for small businesses and
        offers unlimited bandwidth, nightly backups, version history, and
        one-click file restoration on every plan we sell.</p>
        <footer>Copyright 2026 Acme Corp. All rights reserved.</footer>
        </body>
        </html>
        """

        with patch("src.web_loader.requests.get", return_value=fake_response(html)):
            document = load_webpage("https://acme.example.com")

        self.assertIn("cloud storage tools", document.text)
        self.assertNotIn("trackVisit", document.text)
        self.assertNotIn("Copyright 2026", document.text)
        self.assertNotIn("Home About Contact", document.text)


if __name__ == "__main__":
    unittest.main()
