"""
Ingestion environment and path configuration.

INGEST_CLEAN_STORE (default true): remove each system's Chroma persist folder (or Qdrant
collection when ``VECTOR_BACKEND=qdrant``) before re-embedding docs/<system>/.

Chunk size defaults favor Arabic paragraphs; override via INGEST_CHUNK_* in .env.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from core.env_utils import embedding_model, env_bool, env_default_int
from core.paths import PROJECT_ROOT, vector_store_root

# Snapshot after ``load_dotenv()`` for CLI ingest and modules that expect a string constant.
EMBEDDING_MODEL = embedding_model()

# Docs live under repo root; each system is a subfolder (designer, runtime, ...).
DOCS_PATH = PROJECT_ROOT / "docs"
VECTOR_STORE_PATH = vector_store_root()

# Larger chunks preserve Arabic paragraphs and headings; tune via env without code edits.
CHUNK_SIZE = max(200, env_default_int("INGEST_CHUNK_SIZE", 1100))
# Overlap raised to 300 so step-boundary chunks repeat enough text that no
# numbered step is ever cut in half across two consecutive chunks.
CHUNK_OVERLAP = max(0, min(CHUNK_SIZE // 2, env_default_int("INGEST_CHUNK_OVERLAP", 300)))


def ingest_clean_store() -> bool:
    """
    When true (default), delete vector_stores/<system>/ before folder ingest for that system.

    Set INGEST_CLEAN_STORE=false only if you intentionally merge into an existing DB
    (advanced; LangChain/Chroma may still duplicate depending on version).
    """
    return env_bool("INGEST_CLEAN_STORE", True)
