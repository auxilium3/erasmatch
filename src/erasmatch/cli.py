"""Command-line interface.

Every matching command accepts ``--catalog FILE`` (repeatable): the catalogs are loaded into
memory for that run, so you can try the tool without a database.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import typer
import yaml

from erasmatch.config import Settings, get_settings
from erasmatch.domain import LearningAgreementDraft, MatchOption, MatchResult
from erasmatch.errors import ErasmatchError
from erasmatch.service import Services, build_services
from erasmatch.storage.memory import InMemoryRepository

app = typer.Typer(
    help="Find equivalent courses for Erasmus Learning Agreements.", no_args_is_help=True
)

CatalogOpt = typer.Option(
    None, "--catalog", "-c", help="Load catalog file(s) into memory instead of using the DB."
)


def _services(catalogs: list[Path] | None, use_llm: bool = True) -> Services:
    settings: Settings = get_settings()
    if catalogs:
        services = build_services(settings, repo=InMemoryRepository(), use_llm=use_llm)
        services.catalog.import_files(catalogs)
        return services
    return build_services(settings, use_llm=use_llm)


def _fail(exc: Exception) -> None:
    typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


def _codes(option: MatchOption) -> str:
    return " + ".join(c.code for c in option.host_courses)


def _print_match(result: MatchResult, show_topics: bool) -> None:
    h = result.home_course
    typer.secho(f"\n{h.title} [{h.university}:{h.code}] - {h.ects:g} ECTS", bold=True)
    typer.echo(
        f"Host university: {result.host_university}   topic threshold: {result.topic_threshold}\n"
    )
    if not result.options:
        typer.echo("No courses found at the host university.")
        return
    typer.echo(f"{'#':>2}  {'score':>5}  {'cover':>5}  {'ECTS':>5}  {'verdict':<8}  host course(s)")
    for i, o in enumerate(result.options, 1):
        titles = " + ".join(c.title for c in o.host_courses)
        ects = f"{o.ects_total:g}" + ("" if o.ects_ok else "!")
        typer.echo(
            f"{i:>2}  {o.score:>5.2f}  {o.coverage:>5.0%}  {ects:>5}  {o.verdict:<8}  "
            f"{titles} [{_codes(o)}]"
        )
        if o.explanation:
            typer.echo(f"      {o.explanation.summary}")
            if o.explanation.gaps:
                typer.echo(f"      {o.explanation.gaps}")
        elif o.missing_topics:
            shown = o.missing_topics[:3]
            more = len(o.missing_topics) - len(shown)
            suffix = f" (+{more} more)" if more else ""
            typer.echo(f"      missing: {'; '.join(shown)}{suffix}")
    if show_topics:
        best = result.options[0]
        typer.secho(f"\nTopic alignment for option 1 [{_codes(best)}]:", bold=True)
        for a in best.alignments:
            mark = "ok " if a.covered else "-- "
            typer.echo(f"  {mark}{a.similarity:.2f}  {a.home_topic}  ->  {a.host_topic}")
    typer.echo("\n(! = credits below requirement)")


def _agreement_rows(draft: LearningAgreementDraft) -> list[dict[str, str]]:
    rows = []
    for r in draft.rows:
        o = r.option
        rows.append(
            {
                "home_code": r.home_course.code,
                "home_title": r.home_course.title,
                "home_ects": f"{r.home_course.ects:g}",
                "host_codes": _codes(o) if o else "",
                "host_titles": " + ".join(c.title for c in o.host_courses) if o else "",
                "host_ects": f"{o.ects_total:g}" if o else "",
                "coverage": f"{o.coverage:.0%}" if o else "",
                "verdict": o.verdict if o else "",
                "note": r.note or "",
            }
        )
    return rows


def _render_agreement(draft: LearningAgreementDraft, fmt: str) -> str:
    rows = _agreement_rows(draft)
    if fmt == "json":
        return draft.model_dump_json(indent=2)
    if fmt == "csv":
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        return buf.getvalue()
    header = "| Home course | ECTS | Host course(s) | ECTS | Coverage | Verdict | Note |"
    lines = [
        f"## Learning Agreement draft: {draft.home_university} -> {draft.host_university}",
        "",
        header,
        "|---|---:|---|---:|---:|---|---|",
    ]
    for r in rows:
        host = f"{r['host_titles']} ({r['host_codes']})" if r["host_codes"] else "-"
        lines.append(
            f"| {r['home_title']} ({r['home_code']}) | {r['home_ects']} | {host} | "
            f"{r['host_ects'] or '-'} | {r['coverage'] or '-'} | {r['verdict'] or '-'} | "
            f"{r['note']} |"
        )
    lines += [
        "",
        f"Total: {draft.home_ects_total:g} ECTS at home, {draft.host_ects_total:g} ECTS abroad.",
    ]
    if draft.unmatched:
        lines.append(f"Unmatched: {', '.join(draft.unmatched)}")
    return "\n".join(lines)


@app.command()
def migrate() -> None:
    """Create or upgrade the PostgreSQL schema (needs ERASMATCH_DATABASE_URL)."""
    from erasmatch.storage import run_migrations

    settings = get_settings()
    if not settings.database_url.startswith("postgresql"):
        _fail(ValueError("ERASMATCH_DATABASE_URL must point to PostgreSQL"))
    run_migrations(settings.database_url, settings.embedding_dim)
    typer.echo("Database is up to date.")


@app.command("import")
def import_(files: list[Path] = typer.Argument(..., exists=True, readable=True)) -> None:
    """Import catalog files (.yaml/.json) into the database, computing embeddings."""
    try:
        services = _services(None, use_llm=False)
        for path, count in services.catalog.import_files(files).items():
            typer.echo(f"{path}: {count} courses imported with {services.embedder.name}")
    except ErasmatchError as exc:
        _fail(exc)


@app.command()
def courses(university: str, catalog: list[Path] | None = CatalogOpt) -> None:
    """List the courses of a university."""
    services = _services(catalog, use_llm=False)
    for stored in services.repo.list_courses(university):
        c = stored.course
        typer.echo(f"{c.code:<12} {c.ects:>4g} ECTS  {c.title}  ({len(stored.topics)} topics)")


@app.command()
def match(
    home_university: str,
    home_course: str,
    host_university: str,
    top: int = typer.Option(5, "--top", "-k", min=1, max=20),
    explain: bool = typer.Option(False, help="Add natural-language explanations (uses the LLM)."),
    topics: bool = typer.Option(False, help="Show the topic alignment of the best option."),
    as_json: bool = typer.Option(False, "--json", help="Print JSON."),
    catalog: list[Path] | None = CatalogOpt,
) -> None:
    """Rank host courses for one home course."""
    try:
        services = _services(catalog, use_llm=explain)
        result = services.engine.match(
            home_university, home_course, host_university, top_k=top, explain=explain
        )
    except ErasmatchError as exc:
        _fail(exc)
    if as_json:
        typer.echo(result.model_dump_json(indent=2))
    else:
        _print_match(result, show_topics=topics)


@app.command("la")
def learning_agreement(
    home_university: str,
    host_university: str,
    courses_: str = typer.Option(..., "--courses", help="Comma-separated home course codes."),
    fmt: str = typer.Option("markdown", "--format", "-f", help="markdown | csv | json"),
    explain: bool = typer.Option(False, help="Add explanations (uses the LLM)."),
    output: Path | None = typer.Option(None, "--output", "-o"),
    catalog: list[Path] | None = CatalogOpt,
) -> None:
    """Draft a Learning Agreement for several home courses at once."""
    if fmt not in {"markdown", "csv", "json"}:
        _fail(ValueError("--format must be markdown, csv or json"))
    codes = [c.strip() for c in courses_.split(",") if c.strip()]
    try:
        services = _services(catalog, use_llm=explain)
        draft = services.engine.learning_agreement(
            home_university, codes, host_university, explain=explain
        )
    except ErasmatchError as exc:
        _fail(exc)
    text = _render_agreement(draft, fmt)
    if output:
        output.write_text(text, encoding="utf-8")
        typer.echo(f"Written to {output}")
    else:
        typer.echo(text)


@app.command()
def extract(
    file: Path = typer.Argument(..., exists=True, readable=True),
    output: Path | None = typer.Option(None, "--output", "-o", help="Write the course as YAML."),
) -> None:
    """Extract a course from a syllabus PDF/text with the LLM, for review before import."""
    from erasmatch.ingest import file_to_text
    from erasmatch.llm import build_llm
    from erasmatch.llm.extractor import extract_course

    llm = build_llm(get_settings())
    if llm is None:
        _fail(ValueError("Extraction needs an LLM; set ERASMATCH_LLM"))
    text = file_to_text(file.name, file.read_bytes())
    if not text.strip():
        _fail(ValueError("No text found in the file (scanned PDFs need OCR first)."))
    try:
        result = extract_course(text, llm)
    except ErasmatchError as exc:
        _fail(exc)
    rendered = yaml.safe_dump(
        result.course.model_dump(exclude_none=True), allow_unicode=True, sort_keys=False
    )
    for warning in result.warnings:
        typer.secho(f"warning: {warning}", fg=typer.colors.YELLOW, err=True)
    if output:
        output.write_text(rendered, encoding="utf-8")
        typer.echo(f"Written to {output} - review it, then add it to a catalog file.")
    else:
        typer.echo(rendered)


@app.command("eval")
def evaluate_cmd(
    gold: Path = typer.Argument(..., exists=True, readable=True),
    as_json: bool = typer.Option(False, "--json"),
    catalog: list[Path] | None = CatalogOpt,
) -> None:
    """Measure ranking quality (hit@1, hit@3, MRR) against labelled equivalences."""
    from erasmatch.evaluation import GoldSet, evaluate

    try:
        services = _services(catalog, use_llm=False)
        report = evaluate(services.engine, GoldSet.load(gold))
    except ErasmatchError as exc:
        _fail(exc)
    if as_json:
        typer.echo(report.model_dump_json(indent=2))
        return
    typer.echo(f"embedder: {services.embedder.name}   topic threshold: {report.topic_threshold}")
    for p in report.pairs:
        rank = p.rank if p.rank is not None else "-"
        mark = "ok" if p.rank == 1 else "  "
        typer.echo(
            f"  {mark} {p.home:<12} rank {rank!s:>2}  predicted {p.predicted}  "
            f"expected {' | '.join(p.expected)}"
        )
    typer.echo(
        f"\nhit@1 {report.hit_at_1:.2f}   hit@3 {report.hit_at_3:.2f}   MRR {report.mrr:.2f}"
        f"   (n={report.n})"
    )


@app.command()
def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
    reload: bool = False,
) -> None:
    """Run the HTTP API (docs at /docs)."""
    import uvicorn

    uvicorn.run("erasmatch.api:create_app", factory=True, host=host, port=port, reload=reload)


@app.command()
def config() -> None:
    """Print the effective configuration (secrets hidden)."""
    from sqlalchemy.engine import make_url

    data = get_settings().model_dump()
    if data.get("openai_api_key"):
        data["openai_api_key"] = "***"
    if data["database_url"].startswith("postgresql"):
        data["database_url"] = make_url(data["database_url"]).render_as_string(hide_password=True)
    typer.echo(json.dumps(data, indent=2))


if __name__ == "__main__":
    app()
