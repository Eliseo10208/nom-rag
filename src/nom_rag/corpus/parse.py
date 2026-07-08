"""Parse DOF HTML into citable chunks.

Strategy: extract plain-text lines in document order (one per innermost block
element), then run a state machine over hierarchical clause numbers (5., 5.1,
5.1.1). CSS classes differ across documents and are never relied on. Two
invariants of NOM documents keep false positives out:

- top-level sections are sequential (1, 2, 3, ...), so a stray "1." inside a
  questionnaire table cannot open a new section out of order;
- sub-clauses always share the current section's first component (inside
  section 5 only 5.x is accepted).

Reference guides ("Guía de Referencia I..V"), appendices and TRANSITORIOS are
tracked as their own units; clause numbering inside them is questionnaire
noise and is deliberately ignored.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

from selectolax.parser import HTMLParser

from nom_rag.config import settings
from nom_rag.corpus.sources import SOURCES, NomSource

MIN_CHARS = 200  # units smaller than this merge into a sibling
MAX_CHARS = 4500  # units larger than this split at inner clause boundaries

TOP_RE = re.compile(r"^(\d{1,2})\.\s+(\S.*)")
SUB_RE = re.compile(r"^(\d{1,2}(?:\.\d{1,3})+)\.?\s+(\S.*)")
GUIDE_RE = re.compile(r"^(gu[ií]a de referencia\s+([ivxlc]+))\b\s*(.*)", re.I)
APPENDIX_RE = re.compile(r"^(ap[eé]ndice\s+([A-Z0-9]+))\b\s*(.*)", re.I)
TRANS_RE = re.compile(r"^transitorios?\b", re.I)


@dataclass
class Chunk:
    chunk_id: str
    nom_id: str
    short_id: str
    numeral: str
    section_title: str
    text: str


# ---------------------------------------------------------------- extraction


def _decode(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")


def _fold(text: str) -> str:
    """Lowercase and strip accents, for robust title matching."""
    return "".join(
        ch
        for ch in unicodedata.normalize("NFD", text.lower())
        if not unicodedata.combining(ch)
    )


def _normalize(text: str) -> str:
    # collapse ALL whitespace: Word/DOF exports hard-wrap lines inside blocks
    text = unicodedata.normalize("NFC", text).replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


_BLOCKS = ("p", "div", "table", "li")


def _has_block_descendant(node) -> bool:
    # css_first is self-inclusive in selectolax, so walk direct children instead
    for child in node.iter(include_text=False):
        if child.tag in _BLOCKS or child.css_first("p, div, table, li"):
            return True
    return False


def extract_lines(html: str, container: str | None) -> list[str]:
    """Return normalized text lines, one per innermost block element."""
    if container:
        # nota_detalle pages nest a complete standalone HTML document inside the
        # container div; slice out that inner document and parse it on its own.
        pos = html.find(container.lstrip("#"))
        if pos != -1:
            start = html.find("<!DOCTYPE", pos)
            end = html.upper().find("</HTML>", start)
            if start != -1 and end != -1:
                html = html[start : end + len("</HTML>")]
    tree = HTMLParser(html)
    lines: list[str] = []
    for node in tree.css("p, div, td, th, li"):
        # keep only innermost blocks so parent containers don't duplicate text
        if _has_block_descendant(node):
            continue
        text = _normalize(node.text(deep=True))
        if text:
            lines.append(text)
    return lines


# ------------------------------------------------------------- state machine


@dataclass
class _Unit:
    numeral: str
    section: str  # top-level component ("5") or symbolic ("GR-I", "TRANS")
    section_title: str
    lines: list[str]

    @property
    def text(self) -> str:
        return "\n".join(self.lines).strip()


BODY_START_RE = re.compile(r"^1\.?\s+Objetivo\b", re.I)


def _collect_units(lines: list[str]) -> list[_Unit]:
    # every NOM opens with "1. Objetivo", but it also appears in the table of
    # contents — the real body is the LAST occurrence, so skip everything before
    starts = [i for i, line in enumerate(lines) if BODY_START_RE.match(line)]
    if starts:
        lines = lines[starts[-1] :]

    units: list[_Unit] = []
    current: _Unit | None = None
    top = 0
    top_title = ""
    in_guides = False

    def flush() -> None:
        nonlocal current
        if current is not None and current.text:
            units.append(current)
        current = None

    for line in lines:
        if guide := (GUIDE_RE.match(line) or APPENDIX_RE.match(line)):
            flush()
            in_guides = True
            label = _normalize(guide.group(1))
            ref = guide.group(2).upper()
            prefix = "GR" if label.lower().startswith("gu") else "AP"
            title = _normalize(guide.group(3)) or label
            current = _Unit(f"{prefix}-{ref}", f"{prefix}-{ref}", title, [line])
            continue
        if in_guides:
            if current is not None:
                current.lines.append(line)
            continue
        if TRANS_RE.match(line):
            flush()
            current = _Unit("TRANS", "TRANS", "Transitorios", [line])
            continue
        if current is not None and current.section == "TRANS":
            current.lines.append(line)
            continue

        if (m := TOP_RE.match(line)) and int(m.group(1)) == top + 1:
            flush()
            top += 1
            top_title = m.group(2)
            current = _Unit(str(top), str(top), top_title, [line])
            continue
        if top and (m := SUB_RE.match(line)) and m.group(1).split(".")[0] == str(top):
            numeral = m.group(1)
            # open a new unit only at the x.y level; deeper clauses stay inline
            if numeral.count(".") == 1:
                flush()
                current = _Unit(numeral, str(top), top_title, [line])
                continue
        if current is not None:
            current.lines.append(line)
        # lines before "1. Objetivo" (DOF boilerplate, index) are dropped

    flush()
    return units


# ------------------------------------------------------------ chunk assembly


def _merge_small(units: list[_Unit]) -> list[_Unit]:
    merged: list[_Unit] = []
    pending_intro: _Unit | None = None
    for unit in units:
        if pending_intro is not None:
            if unit.section == pending_intro.section:
                unit.lines = pending_intro.lines + unit.lines
            else:
                merged.append(pending_intro)
            pending_intro = None
        # definitions are self-contained lookup targets: merging several into
        # one chunk dilutes the embedding and sinks definition queries
        # (measured: evals/runs/ baseline vs split-definitions)
        keep_standalone = "definicion" in _fold(unit.section_title)
        if len(unit.text) >= MIN_CHARS or (keep_standalone and unit.numeral != unit.section):
            merged.append(unit)
            continue
        if unit.numeral == unit.section:  # tiny section intro -> merge forward
            pending_intro = unit
        elif merged and merged[-1].section == unit.section:  # merge backward
            merged[-1].lines.extend(unit.lines)
        else:
            merged.append(unit)
    if pending_intro is not None:
        merged.append(pending_intro)
    return merged


def _split_large(unit: _Unit) -> list[list[str]]:
    if len(unit.text) <= MAX_CHARS:
        return [unit.lines]
    parts: list[list[str]] = [[]]
    size = 0
    for line in unit.lines:
        boundary = SUB_RE.match(line) is not None
        if parts[-1] and size + len(line) > MAX_CHARS and (boundary or size > MAX_CHARS):
            parts.append([])
            size = 0
        parts[-1].append(line)
        size += len(line) + 1
    return [p for p in parts if p]


def parse_nom(source: NomSource, html: str) -> list[Chunk]:
    lines = extract_lines(html, source.container)
    units = _merge_small(_collect_units(lines))
    chunks: list[Chunk] = []
    for unit in units:
        parts = _split_large(unit)
        for i, part_lines in enumerate(parts):
            suffix = f"#{i + 1}" if len(parts) > 1 else ""
            header = f"[{source.short_id} §{unit.numeral}] {unit.section_title}"
            body = "\n".join(part_lines).strip()
            chunks.append(
                Chunk(
                    chunk_id=f"{source.short_id}:{unit.numeral}{suffix}",
                    nom_id=source.nom_id,
                    short_id=source.short_id,
                    numeral=unit.numeral,
                    section_title=unit.section_title,
                    text=f"{header}\n{body}",
                )
            )
    return chunks


def build_chunks(raw_dir: Path | None = None, out_path: Path | None = None) -> dict[str, int]:
    """Parse every downloaded NOM and write data/chunks.jsonl. Returns counts."""
    raw_dir = raw_dir or settings.raw_dir
    out_path = out_path or settings.chunks_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    with out_path.open("w", encoding="utf-8") as fh:
        for source in SOURCES.values():
            html = _decode((raw_dir / f"{source.short_id.lower()}.html").read_bytes())
            chunks = parse_nom(source, html)
            counts[source.short_id] = len(chunks)
            for chunk in chunks:
                fh.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
    return counts


def load_chunks(path: Path | None = None) -> list[Chunk]:
    path = path or settings.chunks_path
    with path.open(encoding="utf-8") as fh:
        return [Chunk(**json.loads(line)) for line in fh if line.strip()]
