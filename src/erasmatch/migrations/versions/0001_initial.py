"""Initial schema: universities, courses, topics with pgvector embeddings.

Revision ID: 0001
Revises:
Create Date: 2026-11-09
"""

import sqlalchemy as sa
from alembic import context, op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

from erasmatch.config import get_settings

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def _dim() -> int:
    return int(context.config.attributes.get("embedding_dim") or get_settings().embedding_dim)


def upgrade() -> None:
    dim = _dim()
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "universities",
        sa.Column("code", sa.String(40), primary_key=True),
        sa.Column("name", sa.String(300), nullable=False),
        sa.Column("country", sa.String(2), nullable=False),
        sa.Column("language", sa.String(5), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )

    op.create_table(
        "courses",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "university_code",
            sa.String(40),
            sa.ForeignKey("universities.code", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("code", sa.String(40), nullable=False),
        sa.Column("title", sa.String(300), nullable=False),
        sa.Column("ects", sa.Float, nullable=False),
        sa.Column("language", sa.String(5)),
        sa.Column("description", sa.Text, nullable=False, server_default=""),
        sa.Column("declared_topics", postgresql.JSONB, nullable=False, server_default="[]"),
        sa.Column("semester", sa.String(40)),
        sa.Column("level", sa.String(20)),
        sa.Column("url", sa.Text),
        sa.Column("embedding", Vector(dim), nullable=False),
        sa.Column("embedding_model", sa.String(120), nullable=False),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.UniqueConstraint("university_code", "code", name="uq_course_code"),
        sa.CheckConstraint("ects > 0", name="ck_course_ects_positive"),
    )
    op.create_index("ix_courses_university_code", "courses", ["university_code"])

    op.create_table(
        "topics",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column(
            "course_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("courses.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("position", sa.Integer, nullable=False),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("embedding", Vector(dim), nullable=False),
    )
    op.create_index("ix_topics_course_id", "topics", ["course_id"])


def downgrade() -> None:
    op.drop_table("topics")
    op.drop_table("courses")
    op.drop_table("universities")
