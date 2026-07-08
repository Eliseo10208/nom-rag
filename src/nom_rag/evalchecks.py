"""Deterministic answer checks — no LLM involved.

Citation accuracy and abstention are verifiable mechanically, so they never
go through the judge: the judge is reserved for what genuinely requires
reading comprehension (groundedness, correctness).
"""

from __future__ import annotations

from dataclasses import dataclass

from nom_rag.pipeline import ABSTENTION_PHRASE, extract_citations
from nom_rag.refs import chunk_covers


@dataclass
class CitationCheck:
    cited: list[str]
    valid: list[str]
    invalid: list[str]

    @property
    def accuracy(self) -> float | None:
        """Fraction of cited clauses that exist in the retrieved context.
        None when the answer cites nothing (nothing to verify)."""
        return len(self.valid) / len(self.cited) if self.cited else None


def check_citations(answer: str, retrieved: list) -> CitationCheck:
    cited = extract_citations(answer)
    valid = [
        ref
        for ref in cited
        if any(chunk_covers(c.short_id, c.numeral, c.text, ref) for c in retrieved)
    ]
    return CitationCheck(cited=cited, valid=valid, invalid=[r for r in cited if r not in valid])


def check_abstention(answer: str) -> bool:
    """True if the answer uses the literal abstention contract from the prompt."""
    return ABSTENTION_PHRASE.lower() in answer.lower()
