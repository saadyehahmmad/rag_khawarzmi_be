"""
CLI entry for document ingestion: python ingestion/ingest.py [designer runtime ...]

Implementation lives in ingestion.config, ingestion.documents, and ingestion.chroma_ingest.
This file only wires argv, optional KNOWLEDGE_MONOLITH, and prints the next-step hint.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from core.paths import PROJECT_ROOT
from framework.vector_health import SYSTEMS

# config loads .env on import (used by chroma_ingest).
from ingestion.chroma_ingest import get_embeddings, ingest_system, sync_monolith_to_all_systems


def main() -> None:
    embeddings = get_embeddings()
    systems_to_process = sys.argv[1:] if len(sys.argv) > 1 else list(SYSTEMS)

    for system in systems_to_process:
        if system not in SYSTEMS:
            print(f"[ERROR] Unknown system: {system}. Choose from: {list(SYSTEMS)}")
            continue
        ingest_system(system, embeddings)

    monolith_env = os.getenv("KNOWLEDGE_MONOLITH", "").strip()
    if monolith_env:
        mono_path = Path(monolith_env)
        if not mono_path.is_absolute():
            mono_path = (PROJECT_ROOT / mono_path).resolve()
        sync_monolith_to_all_systems(embeddings, mono_path)

    print("\n\nIngestion complete.")
    print("You can now run: uvicorn api.main:app --reload")


if __name__ == "__main__":
    main()
