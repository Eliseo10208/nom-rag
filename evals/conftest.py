from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

DATASET_DIR = Path(__file__).parent / "dataset"


@dataclass
class Case:
    id: str
    category: str
    question: str
    reference_answer: str
    expected_refs: list[str]
    notes: str = ""
    source: str = "golden"

    @property
    def retrievable(self) -> bool:
        return bool(self.expected_refs)


def load_cases(*names: str) -> list[Case]:
    cases: list[Case] = []
    for name in names or ("golden", "regressions"):
        path = DATASET_DIR / f"{name}.jsonl"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    cases.append(Case(**json.loads(line), source=name))
    return cases


ALL_CASES = load_cases()
RETRIEVABLE_CASES = [c for c in ALL_CASES if c.retrievable]
OUT_OF_SCOPE_CASES = [c for c in ALL_CASES if not c.retrievable]


@pytest.fixture(scope="session")
def retriever():
    from nom_rag.config import settings
    from nom_rag.index.embed import Embedder
    from nom_rag.index.store import Retriever

    if not settings.chroma_dir.exists():
        pytest.skip("Index not built — run: nom-rag download-corpus && nom-rag build-index")
    return Retriever(Embedder(settings.embedding_model))


@dataclass
class RetrievalResults:
    """Retrieval for every retrievable case, computed once per session."""

    by_case: dict[str, list] = field(default_factory=dict)


@pytest.fixture(scope="session")
def retrieval_results(retriever) -> RetrievalResults:
    results = RetrievalResults()
    for case in RETRIEVABLE_CASES:
        results.by_case[case.id] = retriever.query(case.question)
    return results
