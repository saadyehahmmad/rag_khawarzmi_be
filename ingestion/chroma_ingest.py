"""
Chroma persistence: idempotent folder rebuilds and monolith upserts.

Design goals (ingestion quality):
- Folder ingest: optional wipe of each system's persist directory before Chroma.from_documents,
  so repeated runs do not accumulate stale or duplicate folder embeddings.
- Monolith ingest: tag chunks with ingest_source=monolith; before each run, delete all prior
  monolith-tagged rows in that collection, then add_documents. Re-running the monolith step
  no longer duplicates the same markdown across runs.
  
Metadata contract (helps debugging and future citation UI):
- ingest_source: "folder" | "monolith"
- source_file: basename of the file
- source_rel: path under docs/<system>/ for folder files; monolith basename for monolith rows
- system: product area name (matches collection_name)
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from agent.vector_health import SYSTEMS
from ingestion.config import DOCS_PATH, EMBEDDING_MODEL, VECTOR_STORE_PATH, ingest_clean_store
from ingestion.documents import build_splitter, detect_language, enrich_with_section_context, load_document




def get_embeddings() -> HuggingFaceEmbeddings:
    """Shared HuggingFace embedding model (same defaults as agent.nodes)."""
    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    print("(First run may download a large model — please wait.)")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def _remove_store_directory(store_path: Path) -> None:
    """Delete a Chroma persist folder so the next from_documents starts empty."""
    if not store_path.is_dir():
        return
    try:
        shutil.rmtree(store_path, ignore_errors=False)
    except OSError as exc:
        # Windows file locks (e.g. API still attached) surface here with a clear message.
        print(f"  [ERROR] Could not remove store directory {store_path}: {exc}")
        raise


def _where_ingest_source(ingest_source: str) -> dict:
    """Chroma where-clause: some stacks accept plain equality, others require $eq."""
    return {"ingest_source": {"$eq": ingest_source}}


def _get_ids_batch(store: Chroma, where: dict, *, limit: int) -> list[str]:
    """Fetch up to `limit` ids matching where; tries get(limit=) then get() without limit."""
    try:
        batch = store.get(where=where, limit=limit)
    except TypeError:
        batch = store.get(where=where)
    return list(batch.get("ids") or [])


def _prune_ingest_source(store: Chroma, ingest_source: str) -> int:
    """
    Delete every vector whose metadata ingest_source matches.

    Batches get + delete(ids=...) so large monoliths are fully cleared even when get() is capped.
    Tries plain equality then $eq where filters so different Chroma stacks still match rows.
    """
    total = 0
    where_variants = (
        {"ingest_source": ingest_source},
        _where_ingest_source(ingest_source),
    )
    for _ in range(50_000):
        ids: list[str] = []
        last_err: Exception | None = None
        for where in where_variants:
            try:
                ids = _get_ids_batch(store, where, limit=2000)
                last_err = None
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                ids = []
        if last_err is not None and not ids:
            if total == 0:
                print(f"  [WARN] Could not query for ingest_source={ingest_source!r}: {last_err}")
            break
        if not ids:
            break
        store.delete(ids=ids)
        total += len(ids)
        if len(ids) < 2000:
            break
    return total


def ingest_system(system: str, embeddings: HuggingFaceEmbeddings) -> None:
    """
    Chunk and embed all files under docs/<system>/ into vector_stores/<system>/.

    When INGEST_CLEAN_STORE is true (default), the target persist directory is removed first
    so this operation is a full rebuild for that system.
    """
    system_docs_path = DOCS_PATH / system
    store_dir = VECTOR_STORE_PATH / system
    store_path = str(store_dir)

    if not system_docs_path.exists():
        print(f"\n[WARN] No docs folder found for system: {system}")
        return

    files = [p for p in system_docs_path.rglob("*") if p.is_file() and not p.name.startswith(".")]
    if not files:
        print(f"\n[WARN] No documents found in docs/{system}/")
        return

    print(f"\n{'=' * 50}\n Processing system: {system.upper()}\n{'=' * 50}")

    all_docs: list[Document] = []
    for file_path in files:
        docs = load_document(file_path)
        rel = file_path.relative_to(system_docs_path).as_posix()
        for doc in docs:
            doc.metadata.update(
                {
                    "system": system,
                    "source_file": file_path.name,
                    "source_rel": rel,
                    "language": detect_language(doc.page_content or ""),
                    "ingest_source": "folder",
                }
            )
        all_docs.extend(docs)

    if not all_docs:
        print(f"  [WARN] No content loaded for {system}")
        return

    # Prepend heading breadcrumbs so every chunk carries its section context.
    all_docs = enrich_with_section_context(all_docs, system)
    print(f"  [CONTEXT] Enriched {len(all_docs)} section(s) with heading breadcrumbs")

    splitter = build_splitter()
    chunks = splitter.split_documents(all_docs)
    print(f"\n  Total chunks created: {len(chunks)}")


    print(f"  Storing in vector store: {store_path}")

    if ingest_clean_store():
        print("  [CLEAN] Removing existing store directory (INGEST_CLEAN_STORE=true).")
        _remove_store_directory(store_dir)

    store_dir.mkdir(parents=True, exist_ok=True)
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=store_path,
        collection_name=system,
    )
    # Write embedding model metadata so vector_health can detect mismatches at startup.
    meta_file = store_dir / ".metadata.json"
    try:
        meta_file.write_text(
            json.dumps(
                {
                    "embedding_model": EMBEDDING_MODEL,
                    "ingest_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "chunks": len(chunks),
                    "system": system,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"  [DONE] {system} ingested successfully — metadata written to {meta_file.name}")
    except OSError as exc:
        print(f"  [WARN] {system} ingested but metadata write failed: {exc}")


def sync_monolith_to_all_systems(embeddings: HuggingFaceEmbeddings, monolith_path: Path) -> None:
    """
    Split one markdown/text file and upsert the same logical chunks into every system collection.

    Upsert = remove prior monolith vectors in each collection, then add the freshly chunked
    monolith. Folder embeddings (ingest_source=folder) are left untouched.
    """
    if not monolith_path.is_file():
        print(f"\n[WARN] KNOWLEDGE_MONOLITH not found: {monolith_path}")
        return

    loader = TextLoader(str(monolith_path), encoding="utf-8")
    docs = loader.load()
    mono_name = monolith_path.name
    for d in docs:
        d.metadata.update(
            {
                "system": "all",
                "source_file": mono_name,
                "source_rel": mono_name,
                "language": detect_language(d.page_content or ""),
                "ingest_source": "monolith",
            }
        )
    # Prepend heading breadcrumbs to monolith sections before splitting.
    docs = enrich_with_section_context(docs, "all")

    splitter = build_splitter()
    base_chunks = splitter.split_documents(docs)
    print(
        f"\n[MONOLITH] {mono_name} -> {len(base_chunks)} chunks per system "
        f"(prune old monolith rows, then add) for {list(SYSTEMS)}"
    )

    for system in SYSTEMS:
        store_path = str(VECTOR_STORE_PATH / system)
        chunks: list[Document] = []
        for c in base_chunks:
            meta = dict(c.metadata)
            meta["system"] = system
            meta["ingest_source"] = "monolith"
            chunks.append(Document(page_content=c.page_content, metadata=meta))

        db = Chroma(
            persist_directory=store_path,
            embedding_function=embeddings,
            collection_name=system,
        )
        removed = _prune_ingest_source(db, "monolith")
        if removed:
            print(f"  [PRUNE] {system}: removed {removed} prior monolith chunk(s)")
        db.add_documents(chunks)
        print(f"  [DONE] monolith upserted -> {system}")
