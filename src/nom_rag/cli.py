"""nom-rag CLI: download-corpus | build-index | ask | trace"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from nom_rag.config import settings

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


@app.command("download-corpus")
def download_corpus(force: bool = typer.Option(False, help="Re-download even if cached")):
    """Download the official DOF HTML for the 4 NOMs and build data/chunks.jsonl."""
    from nom_rag.corpus.download import download_all
    from nom_rag.corpus.parse import build_chunks

    paths = download_all(force=force)
    console.print(f"[green]Descargados[/green] {len(paths)} documentos en {settings.raw_dir}")
    counts = build_chunks()
    for short_id, n in counts.items():
        console.print(f"  {short_id}: {n} chunks")
    console.print(f"[green]Corpus listo[/green] -> {settings.chunks_path}")


@app.command("build-index")
def build_index_cmd():
    """Embed all chunks and (re)build the ChromaDB index."""
    from nom_rag.corpus.parse import load_chunks
    from nom_rag.index.embed import Embedder
    from nom_rag.index.store import build_index

    chunks = load_chunks()
    console.print(f"Indexando {len(chunks)} chunks con {settings.embedding_model}...")
    n = build_index(chunks, Embedder(settings.embedding_model))
    console.print(f"[green]Índice construido[/green]: {n} chunks en {settings.chroma_dir}")


@app.command()
def ask(
    question: str = typer.Argument(..., help="Pregunta sobre las NOMs indexadas"),
    k: int = typer.Option(None, help="Chunks a recuperar (default: config)"),
    show_context: bool = typer.Option(False, "--show-context", help="Imprime los chunks recuperados"),
):
    """Ask a question; answer includes clause citations and a trace id."""
    from nom_rag.index.embed import Embedder
    from nom_rag.index.store import Retriever
    from nom_rag.llm.base import get_client
    from nom_rag.pipeline import ask as ask_pipeline

    retriever = Retriever(Embedder(settings.embedding_model))
    client = get_client(settings.provider, settings.generation_model)
    result = ask_pipeline(question, retriever, client, k=k)

    console.print(Panel(result.answer, title="Respuesta", border_style="cyan"))

    table = Table(title="Retrieval", show_edge=False)
    table.add_column("score", justify="right")
    table.add_column("chunk")
    table.add_column("sección")
    for chunk in result.retrieved:
        cited = " [green]*[/green]" if f"{chunk.short_id}:{chunk.numeral}" in result.citations else ""
        table.add_row(f"{chunk.score:.3f}", chunk.chunk_id + cited, chunk.section_title[:60])
    console.print(table)

    if show_context:
        for chunk in result.retrieved:
            console.print(Panel(chunk.text[:1200], title=chunk.chunk_id, border_style="dim"))

    usage = result.usage
    console.print(
        f"[dim]modelo={result.model}  tokens={usage['input_tokens']}/{usage['output_tokens']}  "
        f"latencia={result.latency_ms['total']}ms  trace={result.trace_id}[/dim]"
    )


@app.command()
def chat(k: int = typer.Option(None, help="Chunks a recuperar (default: config)")):
    """Interactive session. Loads the embedding model and index once instead of
    paying the ~10s startup on every question.

    Each question is retrieved independently — there is no conversational
    memory. That is deliberate: carrying history would add a code path the eval
    suite does not cover, so the chat would stop reflecting the measured system.
    """
    from nom_rag.index.embed import Embedder
    from nom_rag.index.store import Retriever
    from nom_rag.llm.base import get_client
    from nom_rag.pipeline import ask as ask_pipeline

    console.print("[dim]Cargando modelo e índice...[/dim]")
    retriever = Retriever(Embedder(settings.embedding_model))
    client = get_client(settings.provider, settings.generation_model)

    console.print(
        f"\n[bold cyan]nom-rag[/bold cyan] · {settings.generation_model} · "
        f"NOM-035, NOM-019, NOM-017, NOM-030\n"
        "[dim]Escribe tu pregunta. /contexto muestra los extractos de la última "
        "respuesta, /salir termina.[/dim]\n"
    )

    last: object | None = None
    while True:
        try:
            question = console.input("[bold green]›[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Hasta luego.[/dim]")
            return

        if not question:
            continue
        if question in ("/salir", "/exit", "/quit"):
            console.print("[dim]Hasta luego.[/dim]")
            return
        if question == "/contexto":
            if last is None:
                console.print("[dim]Aún no hay una respuesta.[/dim]\n")
                continue
            for chunk in last.retrieved:  # type: ignore[attr-defined]
                console.print(Panel(chunk.text[:1500], title=chunk.chunk_id, border_style="dim"))
            continue

        try:
            result = ask_pipeline(question, retriever, client, k=k)
        except Exception as exc:  # keep the session alive on quota/network errors
            console.print(f"[red]Error:[/red] {exc}\n")
            continue

        last = result
        console.print(Panel(result.answer, border_style="cyan"))
        cited = ", ".join(result.citations) if result.citations else "sin citas"
        console.print(
            f"[dim]citas: {cited}  ·  {result.latency_ms['total']}ms  ·  "
            f"trace={result.trace_id}[/dim]\n"
        )


@app.command()
def trace(trace_id: str = typer.Argument(..., help="Trace id devuelto por ask")):
    """Pretty-print a stored trace."""
    from nom_rag.tracing import find_trace

    record = find_trace(trace_id)
    if record is None:
        console.print(f"[red]No se encontró el trace {trace_id}[/red]")
        raise typer.Exit(1)

    console.print(Panel(record.question, title=f"trace {record.trace_id} · {record.ts}"))
    table = Table(show_edge=False)
    table.add_column("rank")
    table.add_column("chunk")
    table.add_column("score", justify="right")
    for i, item in enumerate(record.retrieval, 1):
        table.add_row(str(i), item["chunk_id"], f"{item['score']:.3f}")
    console.print(table)
    console.print(Panel(record.answer, title="Respuesta", border_style="cyan"))
    console.print(
        f"[dim]provider={record.provider}  modelo={record.model}  "
        f"tokens={record.usage.get('input_tokens')}/{record.usage.get('output_tokens')}  "
        f"latencias={record.latency_ms}[/dim]"
    )


if __name__ == "__main__":
    app()
