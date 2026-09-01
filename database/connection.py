"""
connection.py — SQLite database connection and engine setup.
Uses SQLAlchemy for ORM. Database stored at data/blood_analyzer.db
"""

from __future__ import annotations
import os
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from src.utils import get_logger, ensure_dir, project_path

log = get_logger(__name__)

DB_PATH = project_path("data", "blood_analyzer.db")

# Delay ensure_dir call until get_engine() to avoid import-time side-effects

DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = sessionmaker_factory = None

def get_engine():
    global engine
    if engine is None:
        ensure_dir(DB_PATH.parent)
        engine = create_engine(
            DATABASE_URL,
            connect_args={"check_same_thread": False},
            echo=False,
        )
        log.info("SQLite database engine created: %s", DB_PATH)
    return engine


def get_session_factory():
    global sessionmaker_factory
    if sessionmaker_factory is None:
        sessionmaker_factory = sessionmaker(
            autocommit=False, autoflush=False, bind=get_engine()
        )
    return sessionmaker_factory


def get_db():
    """Dependency for FastAPI — yields a DB session."""
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class Base(DeclarativeBase):
    pass


def init_db():
    """Create all tables and perform lightweight backward-compatible migrations."""
    from database import models  # noqa: F401 — registers models
    eng = get_engine()
    Base.metadata.create_all(bind=eng)
    
    # Lightweight schema migration for SQLite: add missing ML columns if table pre-existed
    with eng.connect() as conn:
        try:
            from sqlalchemy import text
            res = conn.execute(text("PRAGMA table_info(analysis_records)")).fetchall()
            existing_cols = {row[1] for row in res}
            
            new_cols = [
                ("anemia_ml_prediction", "VARCHAR(40)"),
                ("anemia_ml_probability", "FLOAT"),
                ("anemia_ml_model_version", "VARCHAR(40)"),
                ("cbc_inputs_json", "JSON"),
            ]
            for col_name, col_type in new_cols:
                if col_name not in existing_cols:
                    conn.execute(text(f"ALTER TABLE analysis_records ADD COLUMN {col_name} {col_type}"))
                    conn.commit()
                    log.info("Migrated SQLite schema: added column %s", col_name)
        except Exception as me:
            log.warning("Schema migration check warning: %s", me)

    log.info("Database tables initialised.")
