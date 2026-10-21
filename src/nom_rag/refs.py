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
    # Reference guides, appendices, and transitory provisions are separate
    # units. Numbered list items inside a guide are not official clauses of
    # the norm (e.g. GR-I contains a "4.2.1.6.1" hazard taxonomy entry).
    if base.startswith(("GR-", "AP-")) or base == "TRANS":
        return base == ref_num
    if base == ref_num:
        return True
    if base.startswith(f"{ref_num}."):  # section-level ref covered by its clauses
        return True
    # Only neighboring clauses in the same section can have been folded into
    # this chunk. A conformity table in section 10 can mention "5.1" at the
    # start of a row without actually containing clause 5.1.
    numeric = r"\d{1,2}(?:\.\d{1,3})*"
    if not (re.fullmatch(numeric, base) and re.fullmatch(numeric, ref_num)):
        return False
    if base.split(".")[0] != ref_num.split(".")[0]:
        return False
    return re.search(rf"(?m)^{re.escape(ref_num)}\.?\s", text) is not None


def covered_refs(refs: list[str], chunks: list) -> set[str]:
    """Subset of refs covered by any chunk (objects with short_id, numeral, text)."""
    return {
        ref
        for ref in refs
        if any(chunk_covers(c.short_id, c.numeral, c.text, ref) for c in chunks)
    }
