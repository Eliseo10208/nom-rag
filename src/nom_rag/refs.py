"""Reference resolution: does a chunk cover an official clause reference?

Golden cases and model citations always use OFFICIAL clause numbers
("NOM-035:7.9"), never chunk ids. Chunk boundaries are an implementation
detail of the current chunker (7.9 may live inside the 7.7 chunk today and be
its own chunk tomorrow); resolving references by containment keeps the golden
dataset and the citation checker stable across re-chunking.
"""

from __future__ import annotations

import re


def chunk_covers(short_id: str, numeral: str, text: str, ref: str) -> bool:
    """True if the chunk (short_id, numeral, text) covers the reference
    "NOM-XXX:<clause>". Covers means: same clause, a sub-clause of the cited
    section, or the cited clause number appears at a line start in the text."""
    ref_nom, _, ref_num = ref.partition(":")
    if short_id != ref_nom or not ref_num:
        return False
    base = numeral.split("#")[0]
    if base == ref_num:
        return True
    if base.startswith(f"{ref_num}."):  # section-level ref covered by its clauses
        return True
    # clause folded into a larger chunk: its number opens a line in the text
    return re.search(rf"(?m)^{re.escape(ref_num)}\.?\s", text) is not None


def covered_refs(refs: list[str], chunks: list) -> set[str]:
    """Subset of refs covered by any chunk (objects with short_id, numeral, text)."""
    return {
        ref
        for ref in refs
        if any(chunk_covers(c.short_id, c.numeral, c.text, ref) for c in chunks)
    }
