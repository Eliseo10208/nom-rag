"""Download the official DOF HTML for each NOM into data/raw/."""

from pathlib import Path

import httpx

from nom_rag.config import settings
from nom_rag.corpus.sources import SOURCES

UA = "Mozilla/5.0 (compatible; nom-rag/0.1; personal research project)"


def download_all(raw_dir: Path | None = None, force: bool = False) -> list[Path]:
    raw_dir = raw_dir or settings.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    with httpx.Client(headers={"User-Agent": UA}, timeout=60, follow_redirects=True) as client:
        for source in SOURCES.values():
            dest = raw_dir / f"{source.short_id.lower()}.html"
            if dest.exists() and not force:
                saved.append(dest)
                continue
            resp = client.get(source.url)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            saved.append(dest)
    return saved
