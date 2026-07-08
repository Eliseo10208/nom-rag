"""Per-query traces: everything needed to reproduce and debug a single ask().

One JSONL record per query, appended to traces/YYYYMMDD.jsonl. Eval runs store
trace ids next to each case, so any metric can be traced back to the exact
retrieval scores, prompt size, token usage and latencies that produced it.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from nom_rag.config import settings


@dataclass
class Trace:
    trace_id: str
    ts: str
    question: str
    provider: str
    model: str
    retrieval: list[dict]  # [{chunk_id, score}] in rank order
    answer: str
    citations: list[str]
    usage: dict = field(default_factory=dict)  # {input_tokens, output_tokens}
    latency_ms: dict = field(default_factory=dict)  # {retrieve, generate, total}


def new_trace_id() -> str:
    return uuid.uuid4().hex[:10]


def write_trace(trace: Trace, traces_dir: Path | None = None) -> Path:
    traces_dir = traces_dir or settings.traces_dir
    traces_dir.mkdir(parents=True, exist_ok=True)
    path = traces_dir / f"{datetime.now(timezone.utc):%Y%m%d}.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(trace), ensure_ascii=False) + "\n")
    return path


def find_trace(trace_id: str, traces_dir: Path | None = None) -> Trace | None:
    traces_dir = traces_dir or settings.traces_dir
    if not traces_dir.exists():
        return None
    for path in sorted(traces_dir.glob("*.jsonl"), reverse=True):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if trace_id in line:
                    record = json.loads(line)
                    if record.get("trace_id") == trace_id:
                        return Trace(**record)
    return None
