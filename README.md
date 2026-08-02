# nom-rag — RAG over Mexican workplace-safety regulations, with an evaluation harness

[![evals](https://github.com/Eliseo10208/nom-rag/actions/workflows/evals.yml/badge.svg)](https://github.com/Eliseo10208/nom-rag/actions/workflows/evals.yml)

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
- **LLM behind an interface** (`llm/base.py`): Gemini free tier by default — generation on
  `gemini-3.1-flash-lite`, judge on the newer `gemini-3.5-flash-lite`, so verdicts come
  from a **different and not-weaker model than the one being graded**. A Claude
  implementation ships ready (`NOMRAG_PROVIDER=anthropic`); a deterministic fake backs
  harness tests. Pipeline and judge only see the interface — swapping is a config change.
- **Citation contract in the prompt**: every claim must cite `[NOM-035 §7.9]`; questions
  outside the corpus must use a literal abstention phrase. Both are machine-checkable.
- **Follow-up questions** are rewritten into standalone queries before retrieval —
  "¿y si tengo 30 empleados?" embeds to nothing on its own. The rewrite only replaces the
  *retrieval query*; the answer is still generated from the literal question, so a bad
  rewrite degrades retrieval instead of silently changing what was asked. The first turn
  has no history and is never rewritten, so it costs no extra call.

### Interfaces

```bash
nom-rag ask "¿Quiénes integran la comisión?"   # one-shot
nom-rag chat                                   # terminal REPL, model loaded once
uvicorn nom_rag.api:app --port 8017            # web chat at http://localhost:8017
```

The web UI is built around auditing rather than chatting: citations are chips that open
the actual clause text, a side panel shows every retrieved chunk with its similarity score
and whether it was cited, and a banner shows the rewritten query whenever a follow-up was
reformulated. Clicking a citation in an older message restores *that* message's retrieval
— pointing at the wrong evidence would defeat the purpose.

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
| Follow-up rewriting | 8 multi-turn cases: does the rewritten query still retrieve the right clause? | 8 LLM calls | `pytest evals -m llm` |
| LLM-as-judge | groundedness (claim-by-claim vs context), correctness 1–5 vs reference answer | ~90 LLM calls | `pytest evals -m llm`, manual CI |

The judge only handles what needs comprehension. Citation validity and abstention are
**parsed and verified mechanically** — no judge drift on the system's core contract.
Judge verdicts are Pydantic-validated structured output with the reasoning field ordered
before the label; a malformed verdict is an error, never a silently wrong metric. The
harness itself has 12 unit tests running on fakes.

### Measured baseline

Retrieval (e5-small, k=5, 29 retrievable cases) — free, deterministic, gates every push:

| metric | value | pytest gate |
|---|---|---|
| hit_rate@5 | **0.931** | ≥ 0.90 |
| recall@5 | **0.842** | ≥ 0.80 |
| MRR | **0.722** | ≥ 0.68 |

Generation, run `baseline-gen` (34/34 cases, generation `gemini-3.1-flash-lite`, judge
`gemini-3.5-flash-lite`):

| metric | value | pytest gate | what it means |
|---|---|---|---|
| groundedness | **1.00** | ≥ 0.90 | judge found no claim unsupported by the retrieved context |
| correctness | **4.62 / 5** | ≥ 4.0 | vs. expert reference answers |
| citation accuracy | **1.00** ¹ | ≥ 0.90 | every cited clause exists in the context (deterministic) |
| abstention | **0.80** | ≥ 0.75 | 4 of 5 out-of-corpus questions correctly declined |

The one abstention failure is analyzed below — it is a real defect, not rounding.

Follow-up rewriting (8 multi-turn cases): **7/8 = 0.875**, gate ≥ 0.75. The failure is
instructive and *not* a rewriter bug: the user says "fuera de programa" while the norm
says "verificaciones extraordinarias", and e5-small does not bridge that vocabulary gap —
confirmed by checking that the norm's own phrasing retrieves the clause fine. Chat was
added *with* its evals, not before them: adding conversation without covering it would
have created exactly the untested path this repo argues against.

¹ `evals/runs/baseline-gen.json` stores `0.807`, the value measured by the checker *at the
time of the run*; `baseline-gen-rescored.json` holds the `1.00` produced by re-running the
corrected checker over the same answers (story #2 below). Both files are kept — a run
snapshot records what was actually measured, and is never edited after the fact.

### A change is only real if the evals say so

The workflow this repo is built to demonstrate:

```
python scripts/run_evals.py --tag before        # snapshot
# ...change the prompt / chunker / embedder...
python scripts/run_evals.py --tag after
python scripts/compare_runs.py before after     # markdown delta + per-case regressions
```

Four real examples from this repo's history — two changes the evals accepted, two they
rejected (snapshots in `evals/runs/`):

**1. Splitting definition clauses into standalone chunks — accepted.** The eval `d02` (a
definition lookup) retrieved nothing relevant; diagnosis showed four definitions merged
into one chunk, diluting the embedding. Definitions were exempted from small-chunk
merging. Result: `hit_rate 0.897 → 0.931, recall 0.807 → 0.842`, zero per-case
regressions.

**2. The eval found a bug in the eval — accepted.** `citation_accuracy` measured 0.807 and
looked like the system's weak spot. It wasn't. Models cite an inciso —
`[NOM-035 §7.5 a)]` — and the checker parsed that as a clause named `7.5a`, found no such
clause, and marked a *correct* citation invalid. Verified against traces that
`NOM-035:7.5`, `NOM-019:9.13` and `NOM-017:5.9` were all genuinely in the retrieved
context. After normalizing incisos to their parent clause: **0.807 → 1.00**, re-measured
on the *same 34 answers* via `rescore_from_traces.py` at **zero LLM cost**. Had the number
been trusted without auditing, the next hours would have gone into "fixing" a
non-existent problem.

**3. A stronger abstention instruction — rejected.** `o02-alturas` (a question about work
at heights, regulated by a NOM outside the corpus) gets answered instead of declined. The
prompt was hardened to say that a mere mention is not regulation. Measured on the 5
out-of-scope cases: **4/5 before, 4/5 after**. Reverted.

**4. A retrieval relevance floor — rejected.** Top-1 similarity does separate the two
populations on this dataset (in-corpus min **0.859**, out-of-scope max **0.854**), so a
floor was added as a soft hint to the model. Measured: **4/5 again**, and the margin is
only 0.005 across 34 cases — fitting a threshold to that would be fitting noise.
Reverted; the code carries no unmeasured complexity.

Two accepted, two reverted. That ratio is the point: without the harness, changes 3 and 4
both *sound* right and would have shipped.

Two known-weak retrieval cases (`f07`, `p04`: commission role/tenure questions whose
clauses rank just outside k=5) are tracked as `xfail`, not hidden. `f07` is also the
lowest-scoring generation case (correctness 1/5) — a retrieval miss propagating into a
wrong answer, correctly attributed to its actual cause. Candidate levers: e5-base, a
reranker, or k=6 — each is one `compare_runs.py` away from being justified with numbers.

### The most interesting failure: grounded but not responsive

`o02-alturas` asks what the norms require for work at heights — regulated by NOM-009,
which is *not* in the corpus. The system answers anyway, citing `4.2.1.6.1` (anchor
point), `4.2.1.6.2` (scaffolding), and so on. Those numbers are **real**: they come from
NOM-019's Reference Guide I, a taxonomy of hazardous conditions used for accident
investigation, which happens to contain a "Trabajos en Altura" heading.

So the answer is perfectly grounded — a groundedness judge scores it 1.0 — and still
wrong, because a classification annex is not a set of requirements. This is precisely why
abstention is measured **separately and deterministically** rather than folded into a
single quality score: no amount of groundedness catches an answer that is faithful to
retrieved text that should never have been used. Two attempted fixes (README §3 and §4
above) failed to move it; it stands as a documented open defect.

### Traces

Every `ask()` appends a JSONL record: retrieval ranks and scores, token usage, per-stage
latency, citations, and the exact answer. Eval snapshots store the trace id of every
case, so any metric traces back to the exact retrieval that produced it.

```
$ nom-rag trace 93eb1ffe8b
```

Because traces hold the answer *and* the retrieved context, the deterministic metrics of a
past run can be recomputed after fixing a checker — no model calls, no quota:

```
$ python scripts/rescore_from_traces.py baseline-gen
| métrica           | almacenado | recalculado |
| citation_accuracy | 0.807      | 1.0         |
```

That is how story #2 above was measured: the *checker* changed, so previous runs were
re-measured with the new checker instead of comparing a new number against an old one.

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
nom-rag chat                 # interactive; loads the model once (~800ms/answer)
pytest evals -m llm          # generation + judge evals
```

## Costs

Retrieval and harness evals: **$0** — local embeddings, no network, which is why they can
gate every push. A full generation+judge run is ~90 calls (34 answers + 2 judge verdicts
per retrievable case) on the Gemini free tier.

Free-tier quota is the real constraint, and the harness is built around it: generation
scoring **checkpoints after every case**, so hitting a daily wall costs one case of work,
and re-running with the same `--tag` resumes instead of restarting. Deterministic metrics
can also be recomputed on past runs for free (see below), so a fix to a *checker* never
requires re-spending quota on the *model*.

## Honest limitations

- e5-**small** trades recall for CI speed; the two xfail cases document the cost.
- The judge is calibrated by rubric design and structured output, not yet meta-evaluated
  against human labels (natural next step: label ~20 answers, measure judge agreement).
- Model choice is constrained by free-tier quota, not by what would score best: the
  strongest models available on this key (`gemini-pro-latest`, `gemini-2.0-flash`) return
  429 with zero quota, and `gemini-3.6-flash` allows **20 requests/day** against the ~90 a
  full run needs. The lite tier is what actually runs. Both roles are one env var away
  from a stronger model (`NOMRAG_GENERATION_MODEL`, `NOMRAG_JUDGE_MODEL`).
- Corpus is 4 NOMs; the STPS catalog has 40+. The chunker's invariants are designed to
  generalize, but each new document format needs a look.
- This is a portfolio system, not legal advice.
