from __future__ import annotations

import csv
import io
import json

import pytest
from typer.testing import CliRunner

from erasmatch.cli import app
from erasmatch.config import get_settings
from tests.conftest import GOLD, HOME_CATALOG, HOST_CATALOG

runner = CliRunner()
CATALOGS = ["-c", str(HOME_CATALOG), "-c", str(HOST_CATALOG)]


@pytest.fixture(autouse=True)
def offline_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ERASMATCH_EMBEDDER", "hashing")
    monkeypatch.setenv("ERASMATCH_LLM", "none")
    monkeypatch.setenv("ERASMATCH_DATABASE_URL", "memory://")
    get_settings.cache_clear()


def test_match_table() -> None:
    result = runner.invoke(app, ["match", "UNIEX", "WEB", "UEJ", "--topics", *CATALOGS])
    assert result.exit_code == 0, result.output
    assert "Desarrollo de Aplicaciones Web [DAW]" in result.output
    assert "Topic alignment for option 1" in result.output


def test_match_json() -> None:
    result = runner.invoke(app, ["match", "UNIEX", "SO", "UEJ", "--json", *CATALOGS])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["options"][0]["host_courses"][0]["code"] == "SSOO"


def test_match_unknown_course_fails_cleanly() -> None:
    result = runner.invoke(app, ["match", "UNIEX", "NOPE", "UEJ", *CATALOGS])
    assert result.exit_code == 1
    assert "not found" in result.output


def test_learning_agreement_csv() -> None:
    result = runner.invoke(
        app, ["la", "UNIEX", "UEJ", "--courses", "RETI,WEB", "--format", "csv", *CATALOGS]
    )
    assert result.exit_code == 0, result.output
    rows = list(csv.DictReader(io.StringIO(result.output)))
    assert [r["host_codes"] for r in rows] == ["RC + TDR", "DAW"]


def test_learning_agreement_markdown() -> None:
    result = runner.invoke(app, ["la", "UNIEX", "UEJ", "--courses", "WEB,INGSW", *CATALOGS])
    assert result.exit_code == 0, result.output
    assert result.output.startswith("## Learning Agreement draft: UNIEX -> UEJ")
    assert "Total: 12 ECTS at home, 12 ECTS abroad." in result.output


def test_eval() -> None:
    result = runner.invoke(app, ["eval", str(GOLD), *CATALOGS])
    assert result.exit_code == 0, result.output
    assert "hit@1" in result.output
