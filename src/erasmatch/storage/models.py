"""SQLAlchemy ORM tables. The vector dimension is fixed by the Alembic migration."""

from __future__ import annotations

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class UniversityRow(Base):
    __tablename__ = "universities"

    code: Mapped[str] = mapped_column(String(40), primary_key=True)
    name: Mapped[str] = mapped_column(String(300))
    country: Mapped[str] = mapped_column(String(2))
    language: Mapped[str] = mapped_column(String(5))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class CourseRow(Base):
    __tablename__ = "courses"
    __table_args__ = (UniqueConstraint("university_code", "code", name="uq_course_code"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    university_code: Mapped[str] = mapped_column(
        ForeignKey("universities.code", ondelete="CASCADE"), index=True
    )
    code: Mapped[str] = mapped_column(String(40))
    title: Mapped[str] = mapped_column(String(300))
    ects: Mapped[float] = mapped_column(Float)
    language: Mapped[str | None] = mapped_column(String(5))
    description: Mapped[str] = mapped_column(Text, default="")
    declared_topics: Mapped[list[str]] = mapped_column(JSON().with_variant(JSONB, "postgresql"))
    semester: Mapped[str | None] = mapped_column(String(40))
    level: Mapped[str | None] = mapped_column(String(20))
    url: Mapped[str | None] = mapped_column(Text)
    embedding = mapped_column(Vector())
    embedding_model: Mapped[str] = mapped_column(String(120))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    topics: Mapped[list[TopicRow]] = relationship(
        back_populates="course",
        cascade="all, delete-orphan",
        order_by="TopicRow.position",
        lazy="selectin",
    )


class TopicRow(Base):
    __tablename__ = "topics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    course_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("courses.id", ondelete="CASCADE"), index=True
    )
    position: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    embedding = mapped_column(Vector())

    course: Mapped[CourseRow] = relationship(back_populates="topics")
