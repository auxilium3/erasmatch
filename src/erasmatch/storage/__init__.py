"""Storage backends and migrations."""

from __future__ import annotations

from pathlib import Path

from erasmatch.storage.base import Repository

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"


def build_repository(database_url: str) -> Repository:
    if database_url.startswith("memory://"):
        from erasmatch.storage.memory import InMemoryRepository

        return InMemoryRepository()
    if database_url.startswith("postgresql"):
        from erasmatch.storage.postgres import PostgresRepository

        return PostgresRepository(database_url)
    raise ValueError(f"Unsupported ERASMATCH_DATABASE_URL: {database_url!r}")


def run_migrations(database_url: str, embedding_dim: int) -> None:
    """Apply Alembic migrations programmatically (used by `erasmatch migrate`)."""
    from alembic import command
    from alembic.config import Config

    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
    cfg.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
    cfg.attributes["embedding_dim"] = embedding_dim
    command.upgrade(cfg, "head")
