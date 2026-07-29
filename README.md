# nom-rag — RAG over Mexican workplace-safety regulations, with an evaluation harness

A question-answering system over four official Mexican safety standards (NOMs), built
around the part most RAG demos skip: **a measurable definition of quality**. Every answer
cites exact clauses, every query leaves a trace, and no prompt or retrieval change lands
without passing an eval suite — pytest gates on retrieval metrics, deterministic citation
checks, and an LLM-as-judge for what genuinely needs reading comprehension.

```
$ nom-rag ask "¿Cada cuánto se evalúa el entorno organizacional?"

┌─────────────────────────── Respuesta ────────────────────────────┐
│ La identificación y análisis de los factores de riesgo           │
│ psicosocial y la evaluación del entorno organizacional deben     │
│ realizarse al menos cada dos años [NOM-035 §7.9].                │
└──────────────────────────────────────────────────────────────────┘
 score │ chunk           │ sección
 0.905 │ NOM-035:7.7 *   │ Identificación y análisis de los factores…
 ...
 modelo=gemini-3.6-flash  tokens=3214/87  latencia=1840ms  trace=93eb1ffe8b
```

## Corpus

Official DOF text (public) of four STPS standards, chosen so cross-document questions
make sense — they legitimately reference each other:

| NOM | Topic |
|---|---|
| NOM-035-STPS-2018 | Psychosocial risk factors |
| NOM-019-STPS-2011 | Safety & hygiene commissions |
| NOM-017-STPS-2024 | Personal protective equipment (2024 revision, in force since Sep 2025) |
| NOM-030-STPS-2009 | Preventive safety & health services |

Parsing works on **text lines + a state machine over hierarchical clause numbers**
(5., 5.1, 5.1.1) rather than CSS classes — the four documents come from three different
DOF export pipelines with inconsistent markup. Two document invariants kill false
positives: top-level sections are sequential, and sub-clauses share the current section's
prefix (a stray "1." inside a questionnaire table can't open a section). Result: 248
citable chunks with `{nom_id, numeral, section_title}` metadata.

## Architecture

```
question ──► e5-small embedding ──► ChromaDB (cosine, k=5) ──► prompt with citation
                                                              contract ──► LLM ──► answer
                                                                            │
   traces/*.jsonl ◄── trace: retrieval scores, tokens, latencies ◄──────────┘
```

- **Embeddings**: `intfloat/multilingual-e5-small`, local and free — retrieval evals run
  offline, in CI, on every push.
- **LLM behind an interface** (`llm/base.py`): Gemini free tier by default
  (`gemini-3.6-flash`); a Claude implementation ships ready
  (`NOMRAG_PROVIDER=anthropic`); a deterministic fake backs harness tests. The pipeline
  and the judge only see the interface — swapping providers is a config change.
- **Citation contract in the prompt**: every claim must cite `[NOM-035 §7.9]`; questions
  outside the corpus must use a literal abstention phrase. Both are machine-checkable.

## The evaluation harness

### Golden dataset

34 hand-curated cases (`evals/dataset/golden.jsonl`), written against the actual chunk
text, across five categories: factual (16), definition (4), multi-hop cross-document (4),
false-premise (5), out-of-scope (5). Expected references use **official clause numbers,
not chunk ids** — a resolver maps them onto whatever the current chunker produced, so
re-chunking never invalidates the dataset. Bugs found in production become cases in
`regressions.jsonl`.

### Three layers of checking

| Layer | What | Cost | When |
|---|---|---|---|
| Retrieval metrics | hit-rate@k, recall@k, MRR vs expected clauses | free, deterministic | every push (CI) |
| Deterministic answer checks | citation accuracy (cited clause exists in retrieved context), abstention on out-of-scope | free, deterministic | with generation evals |
| LLM-as-judge | groundedness (claim-by-claim vs context), correctness 1–5 vs reference answer | ~90 LLM calls | `pytest evals -m llm`, manual CI |

The judge only handles what needs comprehension. Citation validity and abstention are
**parsed and verified mechanically** — no judge drift on the system's core contract.
Judge verdicts are Pydantic-validated structured output with the reasoning field ordered
before the label; a malformed verdict is an error, never a silently wrong metric. The
harness itself has 12 unit tests running on fakes.

### Measured baseline (e5-small, k=5, 29 retrievable cases)

| metric | value | pytest gate |
|---|---|---|
| hit_rate@5 | **0.931** | ≥ 0.90 |
| recall@5 | **0.842** | ≥ 0.80 |
| MRR | **0.722** | ≥ 0.68 |

Generation gates (groundedness ≥ 0.80, correctness ≥ 3.5/5, citation accuracy ≥ 0.90,
abstention ≥ 0.80) run against the live provider: `pytest evals -m llm`.

### A change is only real if the evals say so

The workflow this repo is built to demonstrate:

```
python scripts/run_evals.py --tag before        # snapshot
# ...change the prompt / chunker / embedder...
python scripts/run_evals.py --tag after
python scripts/compare_runs.py before after     # markdown delta + per-case regressions
```

Two real examples from this repo's history (snapshots in `evals/runs/`):

1. **Splitting definition clauses into standalone chunks** — the eval `d02` (a definition
   lookup) retrieved nothing relevant; diagnosis showed four definitions merged into one
   chunk diluting the embedding. Fix: definitions exempted from small-chunk merging.
   Result: `hit_rate 0.897 → 0.931, recall 0.807 → 0.842, zero per-case regressions`.
2. **Excluding bibliography from the index** — hypothesis: bibliography chunks pollute
   retrieval (one ranked #1 for a "violencia laboral" query). Measured effect on the
   golden set: **none** — kept anyway as index hygiene, and documented honestly instead
   of claiming a win.

Two known-weak cases (`f07`, `p04`: commission role/tenure questions whose clauses rank
just outside k=5) are tracked as `xfail`, not hidden. Candidate levers: e5-base, a
reranker, or k=6 — each is one `compare_runs.py` away from being justified with numbers.

### Traces

Every `ask()` appends a JSONL record: retrieval ranks and scores, token usage, per-stage
latency, citations, and the exact answer. Eval snapshots store the trace id of every
case, so any metric traces back to the exact retrieval that produced it.

```
$ nom-rag trace 93eb1ffe8b
```

## Quickstart

```bash
python -m venv ~/.venvs/nom-rag && source ~/.venvs/nom-rag/bin/activate  # or Scripts\activate
pip install -e ".[dev]"

nom-rag download-corpus      # DOF HTML -> data/chunks.jsonl (also versioned in-repo)
nom-rag build-index          # embeds 224 chunks locally (~1 min CPU)

pytest evals -m "not llm"    # harness + retrieval evals: no key needed

# for generation: free key at https://aistudio.google.com/apikey
cp .env.example .env         # put GEMINI_API_KEY there
nom-rag ask "¿Quiénes integran la comisión de seguridad e higiene?"
pytest evals -m llm          # generation + judge evals
```

## Costs

Retrieval evals: $0 (local embeddings). A full generation+judge run is ~90 calls to
Gemini 2.5 Flash — free tier covers ~2 full runs/day; on paid tier ≈ $0.02/run. Switching
`NOMRAG_PROVIDER=anthropic` (Haiku 4.5 generation / Sonnet judge) costs ≈ $0.05/run —
the harness is exactly how you'd justify that upgrade with numbers.

## Honest limitations

- e5-**small** trades recall for CI speed; the two xfail cases document the cost.
- **The judge runs the same model as the generator** (`gemini-3.6-flash`), because it is
  the only tier with free-tier quota — pro and 2.0-flash return 429, 3.5-flash returns
  503. Same-model judging is subject to self-preference bias. This is exactly why the
  system's core contract (citation validity, abstention) is checked **deterministically
  and never by the judge**; the judge only scores groundedness and correctness. On a paid
  key, `NOMRAG_JUDGE_MODEL=gemini-pro-latest` restores an independent, stronger judge.
- The judge is calibrated by rubric design and structured output, not yet meta-evaluated
  against human labels (natural next step: label ~20 answers, measure judge agreement).
- Corpus is 4 NOMs; the STPS catalog has 40+. The chunker's invariants are designed to
  generalize, but each new document format needs a look.
- This is a portfolio system, not legal advice.
