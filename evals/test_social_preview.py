"""Static social-preview checks: no app startup, secrets, models or API calls."""

from collections import Counter
from fnmatch import fnmatchcase
from html.parser import HTMLParser
from pathlib import Path
import struct
import tomllib
import unittest
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "src" / "nom_rag" / "web"
SITE_URL = "https://nom-rag.vercel.app/"
IMAGE_NAME = "nom-rag-social-dark-v2-2026-10-21.png"
IMAGE_URL = SITE_URL + "static/" + IMAGE_NAME


class HeadParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_head = False
        self.meta = {}
        self.counts = Counter()
        self.canonicals = []

    def handle_starttag(self, tag, attrs):
        if tag == "head":
            self.in_head = True
        if not self.in_head:
            return
        attrs = dict(attrs)
        if tag == "meta":
            key = attrs.get("property") or attrs.get("name")
            if key:
                self.counts[key] += 1
                self.meta[key] = attrs.get("content", "")
        elif tag == "link" and attrs.get("rel") == "canonical":
            self.canonicals.append(attrs.get("href"))

    def handle_endtag(self, tag):
        if tag == "head":
            self.in_head = False


def read_head():
    parser = HeadParser()
    parser.feed((WEB / "index.html").read_text(encoding="utf-8"))
    return parser


def test_social_metadata_is_unique_and_in_initial_head():
    head = read_head()
    expected = {
        "og:type": "website",
        "og:locale": "es_MX",
        "og:site_name": "NOM RAG",
        "og:title": "NOM RAG — Consulta NOMs con IA",
        "og:url": SITE_URL,
        "og:image": IMAGE_URL,
        "og:image:type": "image/png",
        "twitter:card": "summary_large_image",
        "twitter:image": IMAGE_URL,
    }
    for key, value in expected.items():
        assert head.meta[key] == value
    assert all(count == 1 for count in head.counts.values())
    assert head.canonicals == [SITE_URL]
    assert head.meta["twitter:title"] == head.meta["og:title"]
    assert head.meta["description"] == head.meta["og:description"]
    assert head.meta["twitter:description"] == head.meta["description"]
    assert 80 <= len(head.meta["description"]) <= 170
    assert head.meta["twitter:image:alt"] == head.meta["og:image:alt"]
    assert "gatito" in head.meta["og:image:alt"]
    for url in (head.meta["og:url"], head.meta["og:image"], head.meta["twitter:image"]):
        assert urlsplit(url).scheme == "https"
        assert urlsplit(url).netloc == "nom-rag.vercel.app"


def test_social_image_matches_declared_dimensions():
    data = (WEB / IMAGE_NAME).read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert data[12:16] == b"IHDR"
    width, height = struct.unpack(">II", data[16:24])
    assert (width, height) == (1732, 908)
    assert abs(width / height - 1.91) < 0.02
    assert len(data) < 5 * 1024 * 1024
    meta = read_head().meta
    assert (int(meta["og:image:width"]), int(meta["og:image:height"])) == (width, height)


def test_social_asset_is_included_in_package_data():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    patterns = config["tool"]["setuptools"]["package-data"]["nom_rag"]
    for filename in ("web/index.html", "web/" + IMAGE_NAME):
        assert any(fnmatchcase(filename, pattern) for pattern in patterns)


if __name__ == "__main__":
    suite = unittest.TestSuite(
        unittest.FunctionTestCase(check)
        for check in (
            test_social_metadata_is_unique_and_in_initial_head,
            test_social_image_matches_declared_dimensions,
            test_social_asset_is_included_in_package_data,
        )
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
