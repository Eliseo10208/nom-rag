"""Unit tests of the eval harness itself — fakes only, no network, no index.

If the harness is wrong, every metric it produces is noise; these tests are
what let us trust a red (or green) eval.
"""

from __future__ import annotations

import json

import pytest

from nom_rag.corpus.parse import Chunk, load_chunks
from nom_rag.evalchecks import check_abstention, check_citations
from nom_rag.llm.fake import FakeLLM
from nom_rag.pipeline import ABSTENTION_PHRASE, extract_citations
from nom_rag.refs import chunk_covers, covered_refs

from .conftest import ALL_CASES, RETRIEVABLE_CASES
from .judges.judge import GroundednessVerdict, Judge


def _chunk(short_id="NOM-035", numeral="7.7", text="") -> Chunk:
    return Chunk(
        chunk_id=f"{short_id}:{numeral}",
        nom_id=f"{short_id}-STPS-2018",
        short_id=short_id,
        numeral=numeral,
        section_title="x",
        text=text,
    )


# ------------------------------------------------------------- refs resolution


def test_chunk_covers_exact_and_split():
    assert chunk_covers("NOM-035", "7.9", "", "NOM-035:7.9")
    assert chunk_covers("NOM-035", "7.9#2", "", "NOM-035:7.9")


def test_chunk_covers_clause_folded_into_bigger_chunk():
    text = "7.7 El resultado...\n7.9 La identificación deberá realizarse cada dos años."
    assert chunk_covers("NOM-035", "7.7", text, "NOM-035:7.9")
    assert not chunk_covers("NOM-035", "7.7", text, "NOM-035:7.10")


def test_conformity_table_row_is_not_a_folded_clause():
    table = "10.2 Procedimiento de evaluación\n5.1\nDocumental\nEvidencia de cumplimiento"
    assert not chunk_covers("NOM-035", "10.2#1", table, "NOM-035:5.1")
    assert chunk_covers(
        "NOM-030", "4.6", "4.6 Reportes\n4.8 Conservar documentación dos años",
        "NOM-030:4.8",
    )


def test_chunk_covers_section_level_reference():
    assert chunk_covers("NOM-019", "9.1", "", "NOM-019:9")


def test_chunk_covers_rejects_other_nom():
    assert not chunk_covers("NOM-019", "7.9", "7.9 ...", "NOM-035:7.9")


def test_covered_refs_aggregates():
    chunks = [_chunk(numeral="5.1"), _chunk(numeral="7.7", text="7.9 blah")]
    covered = covered_refs(["NOM-035:5.1", "NOM-035:7.9", "NOM-035:8.1"], chunks)
    assert covered == {"NOM-035:5.1", "NOM-035:7.9"}


# --------------------------------------------------------------- citations


def test_extract_citations_dedup_and_formats():
    answer = "Debe hacerse cada dos años [NOM-035 §7.9]. Ver [NOM-035 § 7.9] y [NOM-019 §9.5]."
    assert extract_citations(answer) == ["NOM-035:7.9", "NOM-019:9.5"]


def test_check_citations_valid_and_invalid():
    retrieved = [_chunk(numeral="7.7", text="7.9 cada dos años")]
    check = check_citations("Cada dos años [NOM-035 §7.9], ver [NOM-035 §5.1].", retrieved)
    assert check.valid == ["NOM-035:7.9"]
    assert check.invalid == ["NOM-035:5.1"]
    assert check.accuracy == 0.5


def test_check_citations_no_citations_is_none():
    assert check_citations("Sin citas.", []).accuracy is None


def test_check_abstention():
    assert check_abstention(f"{ABSTENTION_PHRASE}. Las normas indexadas son...")
    assert not check_abstention("La respuesta es cada dos años [NOM-035 §7.9].")


# ------------------------------------------------------------------- judge


def test_judge_parses_structured_verdicts():
    g = json.dumps(
        {"reasoning": "ok", "unsupported_claims": [], "grounded": True}
    )
    c = json.dumps({"reasoning": "equivalente", "correctness": 5})
    judge = Judge(FakeLLM(responses=[g, c]))
    case = RETRIEVABLE_CASES[0]

    from nom_rag.pipeline import AskResult

    result = AskResult(
        answer="Cada dos años [NOM-035 §7.9].",
        retrieved=[_chunk()],
        citations=["NOM-035:7.9"],
        trace_id="t",
        model="fake-1",
        usage={},
        latency_ms={},
    )
    verdict = judge.evaluate(case, result)
    assert verdict.grounded and verdict.correctness == 5


def test_judge_rejects_malformed_verdict():
    judge = Judge(FakeLLM(responses=["no soy json", "{}"]))
    case = RETRIEVABLE_CASES[0]

    from nom_rag.pipeline import AskResult

    result = AskResult(
        answer="x", retrieved=[], citations=[], trace_id="t",
        model="fake-1", usage={}, latency_ms={},
    )
    with pytest.raises(Exception):
        judge.evaluate(case, result)


def test_groundedness_schema_orders_reasoning_first():
    fields = list(GroundednessVerdict.model_fields)
    assert fields.index("reasoning") < fields.index("grounded")


# ------------------------------------------------- checkpoint / resume logic


def test_aggregate_computes_means_and_ignores_missing():
    from .scoring import aggregate

    per_case = {
        "a": {"grounded": True, "correctness": 5, "citation_accuracy": 1.0},
        "b": {"grounded": False, "correctness": 3, "citation_accuracy": 0.5},
        "c": {"grounded": True, "correctness": 4, "citation_accuracy": None},  # no citations
        "o1": {"abstained": True},
        "o2": {"abstained": False},
    }
    agg = aggregate(per_case)
    assert agg["n_scored"] == 3
    assert agg["groundedness"] == round(2 / 3, 3)
    assert agg["correctness_avg"] == 4.0
    assert agg["citation_accuracy"] == 0.75  # None excluded, not counted as 0
    assert agg["abstention_rate"] == 0.5


def test_aggregate_empty_is_none_not_crash():
    from .scoring import aggregate

    agg = aggregate({})
    assert agg["groundedness"] is None and agg["abstention_rate"] is None


def test_score_generation_skips_resumed_cases(monkeypatch):
    """Resume must not re-spend quota on cases already scored."""
    from . import scoring

    called: list[str] = []

    def fake_score_one(case, retriever, client, judge, k=None):
        called.append(case.id)
        return {"grounded": True, "correctness": 5, "citation_accuracy": 1.0}

    monkeypatch.setattr(scoring, "score_one", fake_score_one)
    monkeypatch.setattr(
        "nom_rag.llm.base.get_client", lambda *a, **kw: FakeLLM()
    )
    monkeypatch.setattr("evals.judges.judge.make_judge", lambda: object())

    cases = RETRIEVABLE_CASES[:3]
    already = {cases[0].id: {"grounded": True, "correctness": 4, "citation_accuracy": 1.0}}
    result = scoring.score_generation(
        None, cases, progress=False, resume_from=already
    )

    assert called == [cases[1].id, cases[2].id], "re-scored an already-measured case"
    assert len(result["per_case"]) == 3


def test_score_generation_raises_quota_with_partial_results(monkeypatch):
    from google.genai import errors as genai_errors  # noqa: F401  (shape reference)

    from . import scoring

    def fake_score_one(case, retriever, client, judge, k=None):
        if case.id == RETRIEVABLE_CASES[1].id:
            raise RuntimeError("429 RESOURCE_EXHAUSTED quota exceeded")
        return {"grounded": True, "correctness": 5, "citation_accuracy": 1.0}

    monkeypatch.setattr(scoring, "score_one", fake_score_one)
    monkeypatch.setattr("nom_rag.llm.base.get_client", lambda *a, **kw: FakeLLM())
    monkeypatch.setattr("evals.judges.judge.make_judge", lambda: object())

    with pytest.raises(scoring.QuotaExhausted) as excinfo:
        scoring.score_generation(None, RETRIEVABLE_CASES[:3], progress=False)

    partial = excinfo.value.args[1]
    assert RETRIEVABLE_CASES[0].id in partial, "partial results must survive the quota wall"


def test_non_quota_errors_are_not_swallowed(monkeypatch):
    from . import scoring

    def boom(case, retriever, client, judge, k=None):
        raise ValueError("bug real, no cuota")

    monkeypatch.setattr(scoring, "score_one", boom)
    monkeypatch.setattr("nom_rag.llm.base.get_client", lambda *a, **kw: FakeLLM())
    monkeypatch.setattr("evals.judges.judge.make_judge", lambda: object())

    with pytest.raises(ValueError):
        scoring.score_generation(None, RETRIEVABLE_CASES[:1], progress=False)


# ------------------------------------------------------------ dataset sanity


def test_case_ids_unique():
    ids = [c.id for c in ALL_CASES]
    assert len(ids) == len(set(ids))


def test_every_expected_ref_exists_in_corpus():
    """Golden refs must resolve against the actual chunked corpus — catches
    typos in refs and chunker regressions that drop clauses."""
    chunks = load_chunks()
    missing = []
    for case in RETRIEVABLE_CASES:
        for ref in case.expected_refs:
            if not any(chunk_covers(c.short_id, c.numeral, c.text, ref) for c in chunks):
                missing.append(f"{case.id}: {ref}")
    assert not missing, f"refs sin chunk que las cubra: {missing}"


def test_out_of_scope_cases_have_no_refs():
    for case in ALL_CASES:
        if case.category == "out_of_scope":
            assert case.expected_refs == []
