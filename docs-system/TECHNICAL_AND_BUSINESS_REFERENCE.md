# Al-Khwarizmi RAG — Technical & Business Reference

> **Audience:** engineers, technical leads, product managers, and system administrators.
> Covers every layer of the system: business context, architecture, data flow, configuration, security, and operations.

---

## Table of Contents

1. [Business Overview](#1-business-overview)
2. [System Architecture](#2-system-architecture)
3. [Component Reference](#3-component-reference)
   - 3.1 [Ingestion Pipeline](#31-ingestion-pipeline)
   - 3.2 [LangGraph Agent](#32-langgraph-agent)
   - 3.3 [Retrieval Engine](#33-retrieval-engine)
   - 3.4 [FastAPI Service](#34-fastapi-service)
   - 3.5 [Thread Memory](#35-thread-memory)
   - 3.6 [Governance & Guardrails](#36-governance--guardrails)
   - 3.7 [Observability](#37-observability)
4. [Data Flow — Request Lifecycle](#4-data-flow--request-lifecycle)
5. [API Reference](#5-api-reference)
6. [Configuration Reference](#6-configuration-reference)
7. [Security Model](#7-security-model)
8. [Operations Guide](#8-operations-guide)
9. [Directory Structure](#9-directory-structure)
10. [Dependency Map](#10-dependency-map)
11. [Product Swap Guide](#11-product-swap-guide)

---

## 1. Business Overview

### Purpose

Al-Khwarizmi RAG is the AI assistant backend for the **Al-Khwarizmi statistical survey platform**, used by government agencies and research centres across the Middle East. It answers user questions in **Arabic and English** grounded strictly in the platform's own documentation, eliminating hallucination risks on regulated workflows.

### Business Systems

| System | What it covers |
|--------|----------------|
| **designer** | Survey builder — question types, skip logic, branching, validation, templates |
| **runtime** | Field collection — web/tablet rendering, offline mode, data synchronisation |
| **callcenter** | Phone/email data collection, CATI, agent workflow |
| **admin** | User roles, permissions, project management, organisation setup |

Each system has an isolated Chroma vector store so retrieval is always scoped to the correct product area.

### Key Business Guarantees

- **Grounded answers only** — the LLM is instructed to cite only provided chunks; it refuses to invent policies or APIs.
- **Off-topic rejection** — questions unrelated to the platform are routed to `system=none`, skip retrieval, and return a polite fallback. No survey chunks are returned for a cooking question.
- **Multilingual** — Arabic, English, and code-switched questions are handled with script-ratio detection; Arabic answers use the correct script.
- **Thread continuity** — conversations accumulate up to `MEMORY_MAX_TURNS` prior turns in every prompt, supporting follow-up questions without repeating context.
- **Auditability** — every request gets an `X-Request-ID`; every query is appended to a JSONL log with distance, relevance, fallback flag, and answer preview.

---

## 2. System Architecture

The codebase is split into **three explicit layers**. Swapping the business product
requires changing a **single import line** in `api/deps.py`.

```
┌──────────────────────────────────────────────┐
│         LAYER 3: alkawarzmi/                 │  ← swap this import to change the product
│  prompts · systems · intents · page_map      │
│  fallback · greeting · prescripts · surveys  │
├──────────────────────────────────────────────┤
│         LAYER 2: framework/                  │  ← generic RAG app skeleton
│  graph · nodes · state · survey_store        │
│  parameterised via RAGProfile protocol       │
├──────────────────────────────────────────────┤
│         LAYER 1: core/                       │  ← reusable services (zero business logic)
│  retrieval · memory · governance · NLP       │
│  observability · LLM clients · streaming     │
└──────────────────────────────────────────────┘
```

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         Client (Angular / REST)                          │
│         POST /chat (SSE stream)    POST /chat/sync (JSON)                │
└──────────────────────────────┬───────────────────────────────────────────┘
                               │ HTTP
┌──────────────────────────────▼───────────────────────────────────────────┐
│                         FastAPI  (api/main.py)                           │
│  Middleware: X-Request-ID · CORS · Rate-limit · Governance check         │
│  Routes: /health  /health/ready  /metrics  /llm  /chat  /chat/sync       │
└──────────────────────────────┬───────────────────────────────────────────┘
                               │ invoke / astream
┌──────────────────────────────▼───────────────────────────────────────────┐
│               LangGraph Agent  (framework/graph.py)                      │
│                                                                          │
│  language_detect ──► payload_prescripts ──► rewrite_and_route            │
│                              └── prescripted_answer ──► (stream & done)  │
│                                        │                                 │
│                              retrieval ──► answer                        │
│                                        └──► fallback                     │
│                                                                          │
│  State: AgentState (TypedDict)  ·  Thread memory injected before invoke  │
└──────┬────────────────────────────────────────────────┬──────────────────┘
       │ HuggingFace Embeddings                         │ LLM API (provider-selectable)
       │ intfloat/multilingual-e5-large                 │ Anthropic · OpenAI · Gemma · Ollama
┌──────▼──────────────────────┐              ┌──────────▼──────────────────┐
│  Catalog vector stores      │  4+ systems  │   Chat model                │
│  Chroma (local) or Qdrant   │  designer /  │   rewrite+route (1×)        │
│  vector_stores/<system>/    │  runtime /   │   answer (1×)               │
│                             │  callcenter /│                             │
│  Survey stores (Chroma only)│  admin / …   │                             │
│  vector_stores/surveys/<id>/│              └─────────────────────────────┘
└──────────────┬──────────────┘
               │
┌──────────────▼──────────────┐
│  BM25 (in-proc, rank_bm25)  │  Lexical re-ranking over dense pool (RRF)
└─────────────────────────────┘
```

### LLM call budget per request

| Scenario | LLM calls | Notes |
|----------|-----------|-------|
| On-topic question | **2** | rewrite+route + answer |
| Payload prescript match | **0** | Zero-LLM fast path from client context |
| Off-topic / unsupported script | **1** | rewrite+route only (system=none → skip retrieval) |
| Fallback (empty retrieval) | **1** | rewrite+route only |

---

## 3. Component Reference

### 3.1 Ingestion Pipeline

**Entry point:** `python -m ingestion` (from project root with venv active)

**Files:** `ingestion/config.py`, `ingestion/documents.py`, `ingestion/chroma_ingest.py`, `ingestion/__main__.py`
**Survey ingestion:** `alkawarzmi/ingestion/survey_session.py`

#### What it does

1. Loads `.env` (embedding model, paths, chunk config, vector backend).
2. Builds a `HuggingFaceEmbeddings` singleton (`intfloat/multilingual-e5-large`, CPU).
3. For each system (`designer`, `runtime`, `callcenter`, `admin`):
   - Discovers all files under `docs/<system>/` recursively.
   - Loads each file using the appropriate LangChain loader:
     - `.docx` → `Docx2txtLoader`
     - `.pdf` → `PyPDFLoader`
     - `.txt` / `.md` → `TextLoader` (UTF-8)
     - `.xlsx` → custom openpyxl reader (first sheet, pipe-joined cells)
   - Annotates each document with metadata: `system`, `source_file`, `source_rel`, `language`, `ingest_source=folder`.
   - Splits with `RecursiveCharacterTextSplitter` (Arabic-aware separators).
   - Optionally wipes the existing store directory (`INGEST_CLEAN_STORE=true`, default).
   - Creates or upserts a persist store in the configured `VECTOR_BACKEND` (Chroma or Qdrant).
4. If `KNOWLEDGE_MONOLITH` is set, loads that markdown into all four collections.
5. Writes `.metadata.json` to each `vector_stores/<system>/` directory recording the embedding model name, ingest timestamp, and chunk count. At API startup `framework/vector_health.py` reads this file and logs a `WARNING` when the stored model differs from `EMBEDDING_MODEL`.

#### Survey session ingestion

`alkawarzmi/ingestion/survey_session.py` embeds a live survey's JSON definition into
a per-survey Chroma collection under `vector_stores/surveys/<id>/`. Triggered via
`POST /survey/ingest/<survey_id>` and monitored via `GET /survey/status/<survey_id>`.
Survey stores are always Chroma-only regardless of `VECTOR_BACKEND`.

#### Chunk configuration

| Setting | Default | Effect |
|---------|---------|--------|
| `INGEST_CHUNK_SIZE` | 1100 | Characters per chunk |
| `INGEST_CHUNK_OVERLAP` | 150 | Overlap between adjacent chunks |
| `INGEST_CLEAN_STORE` | true | Wipe store before each system re-ingest |

#### Language metadata

Each chunk is tagged `language=ar` if > 20% of characters are Arabic script, else `language=en`. Used downstream for Arabic normalisation in retrieval.

---

### 3.2 LangGraph Agent

**Files:** `framework/graph.py`, `framework/nodes/`, `framework/state.py`

The pipeline is a **linear StateGraph** with a prescript fast-path and one conditional branch at retrieval:

```
START
  │
  ▼
language_detect_node         — heuristic, no LLM; script gate blocks unsupported writing systems
  │
  ▼
payload_context_node         — zero-LLM prescript from client payload (page_id, survey_id, names)
  │  prescripted_answer set? ──────────────────────────────────────────────────────► DONE (stream)
  │
  ▼
rewrite_and_route_node       — 1 LLM call, typo-normalised query → JSON {rewritten_question, system}
  │
  ▼
retrieval_node               — hybrid dense+BM25+RRF, sets relevance
  │
  ├─── relevance=relevant ──► answer_node    (1 LLM call)
  │
  └─── relevance=irrelevant ► fallback_node  (static message, no LLM)
```

**Profile injection** — `build_graph(profile)` calls `configure_pipeline(profile)` which injects all business-layer dependencies (prompts, systems list, prescripts function, intent detector, fallback function) into `framework/nodes/pipeline.py` module-level variables. Nodes keep their `(state) -> dict` signature unchanged.

#### AgentState fields

| Field | Type | Set by |
|-------|------|--------|
| `question` | `str` | API caller |
| `language` | `"ar"/"en"/"mixed"` | `language_detect_node` |
| `rewritten_question` | `str` | `rewrite_and_route_node` |
| `system` | `str\|None` | `rewrite_and_route_node` — `"none"` for off-topic |
| `ui_system` | `str\|None` | API (from request context hint) |
| `retrieved_chunks` | `list[str]` | `retrieval_node` |
| `retrieved_source_refs` | `list[dict]` | `retrieval_node` |
| `retrieval_best_distance` | `float\|None` | `retrieval_node` — top-1 L2, logged for tuning |
| `relevance` | `"relevant"/"irrelevant"/"unknown"` | `retrieval_node` |
| `answer` | `str` | `answer_node` / `fallback_node` |
| `prescripted_answer` | `str\|None` | `payload_context_node` — non-None skips LLM entirely |
| `image_urls` | `list[str]` | `retrieval_node` — UI image carousel (from screens.json) |
| `thread_id` | `str` | API (resolved before invoke) |
| `conversation_history` | `list[dict]` | Loaded from memory before invoke |
| `request_id` | `str` | API middleware |
| `page_id` | `str\|None` | API (current Angular route from FE) |
| `survey_id` | `str\|None` | API (active survey from FE) |
| `user_name_en/ar` | `str` | API (displayed name for personalised greeting) |
| `is_authenticated` | `bool` | API |
| `system_language` | `str\|None` | API (UI language preference "en"/"ar") |
| `survey_context_missing` | `bool` | `payload_context_node` |
| `survey_ingesting` | `bool` | `payload_context_node` |
| `survey_vector_context_used` | `bool` | `retrieval_node` |
| `survey_index_absent` | `bool` | `retrieval_node` |

#### Off-topic routing

When `rewrite_and_route_node` returns `system="none"`, `retrieval_node` immediately returns `relevance="irrelevant"` with empty chunks — no embedding query, no vector store round-trip, no citations.

---

### 3.3 Retrieval Engine

**File:** `core/retrieval.py`

#### Hybrid retrieval (default)

```
Query (typo-normalised + Arabic-normalised)
  │
  ├─► Dense vector store similarity_search (pool of HYBRID_DENSE_POOL docs)
  │   Backend: Chroma (local) or Qdrant (remote) — set via VECTOR_BACKEND
  │       └── MMR re-ordering (optional, RETRIEVAL_USE_MMR=true)
  │
  ├─► Language filter (RETRIEVAL_LANG_FILTER=true)
  │       └── Drops chunks whose language tag doesn't match query language
  │           when filtered pool ≥ RETRIEVAL_TOP_K (skipped for mixed queries)
  │
  ├─► BM25Okapi over the filtered dense pool (in-process, no index rebuild)
  │       └── Arabic-aware tokenizer: stopword removal, Gulf vocab bridge,
  │           CAMeL multi-lemma expansion, prefix/suffix stripping
  │
  ├─► RRF fusion (RRF_K=60) → top RETRIEVAL_TOP_K docs
  │
  ├─► Optional survey vector store merge (per-survey Chroma, always Chroma)
  │
  └─► Deduplication: near-duplicate chunks (same source_file + first 200 chars)
      are dropped, keeping highest-ranked copy
```

#### Optional Cross-encoder re-ranking

Set `RERANK_CROSS_ENCODER_MODEL=BAAI/bge-reranker-v2-m3` to enable a third-stage re-rank. Improves precision at the cost of ~100–300 ms per request.

#### Relevance gating

`relevance` is set to `"irrelevant"` when:
1. Zero chunks are returned (empty store or all filtered out).
2. `RETRIEVAL_RELEVANCE_MAX_L2` is set and the top-1 dense L2 distance exceeds the threshold.
3. `system="none"` was returned by the router (off-topic).

---

### 3.4 FastAPI Service

**File:** `api/main.py`

#### Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/health` | none | Liveness — always 200 |
| `GET` | `/health/ready` | optional (`RAG_HEALTH_READY_KEY`) | Readiness — 503 if stores missing |
| `GET` | `/metrics` | none | Prometheus exposition format |
| `GET` | `/llm/config` | none | Active LLM provider, model, Ollama URL |
| `GET` | `/llm/ollama/models` | none | Models available from local Ollama daemon |
| `POST` | `/chat` | optional (`RAG_API_KEY`) | SSE token stream |
| `POST` | `/chat/sync` | optional (`RAG_API_KEY`) | Blocking JSON response |
| `POST` | `/survey/ingest/{survey_id}` | optional | Embed a live survey into Chroma |
| `GET` | `/survey/status/{survey_id}` | optional | Ingestion status (`idle`/`ingesting`/`ready`/`error`) |

#### `/chat` — SSE event types

| Event | Payload | When |
|-------|---------|------|
| `node_start` | `{"node": "language_detect"}` … | Before each pipeline step |
| `token` | raw text string | LLM streaming delta (or full fallback/prescript string) |
| `done` | `{"thread_id": "…", "citations": […]}` | Pipeline complete |
| `blocked` | `{"code": "policy_violation", "reasons": […]}` | Governance rejection |
| `error` | `{"message": "…"}` | Unhandled exception |

#### `/chat/sync` — response body

```json
{
  "thread_id": "abc-123",
  "language": "ar",
  "system": "designer",
  "rewritten_question": "How do I add skip logic to a question?",
  "relevance": "relevant",
  "answer": "To add skip logic … [1].\n\n**Sources:** [1]",
  "citations": [
    {
      "ref": 1,
      "source_file": "Al-Khwarizmi Survey Designer.docx",
      "source_rel": "Al-Khwarizmi Survey Designer.docx",
      "system": "designer",
      "ingest_source": "folder"
    }
  ]
}
```

#### Request body

```json
{
  "question": "string (required, min 1 char)",
  "thread_id": "optional — omit to start a new thread"
}
```

#### Authentication

Set `RAG_API_KEY` to require a credential. Accepted via:
- `Authorization: Bearer <key>`
- `X-API-Key: <key>` header
- `?api_key=<key>` query param (SSE-friendly)

`/health`, `/health/ready`, and `/metrics` are always unauthenticated.

---

### 3.5 Thread Memory

**File:** `core/thread_memory.py`

Multi-turn conversation history is stored per `thread_id`. Storage backend is selected automatically:

| Condition | Backend |
|-----------|---------|
| `REDIS_URL` set and reachable | Redis LIST per thread (`rag:thread:<id>:turns`) |
| Otherwise | JSON file per thread (`MEMORY_PATH/threads/<id>.json`) |

**Turn format stored:**
```json
{"role": "user", "content": "…"}
{"role": "assistant", "content": "…"}
```

Up to `MEMORY_MAX_TURNS` (default 10) prior turns are loaded into the prompt prefix. Older turns are trimmed automatically.

**Thread ID rules:**
- Pattern: `^[A-Za-z0-9_.:-]{1,128}$`
- Blank / missing → auto-generated UUID (new thread per request)

---

### 3.6 Governance & Guardrails

**File:** `core/governance.py`

Runs **before** the LangGraph graph on every chat request.

#### Checks (in order)

1. **Empty / length** — rejects blanks and questions > `GOVERNANCE_MAX_QUESTION_CHARS` (default 12 000).
2. **Script gate** (`core/query_script_gate.py`) — questions containing non-Latin / non-Arabic characters return a bilingual prescript immediately, skipping governance and LLM entirely.
3. **Injection heuristics** — 9 English regex patterns + 6 Arabic patterns for common prompt-override phrases.
4. **Substring blocklist** — comma-separated `GOVERNANCE_BLOCK_SUBSTRINGS` env var, plus an optional newline file `GOVERNANCE_BLOCKLIST_FILE`.

On block: returns HTTP 403 (policy violation) or 422 (too long). Logs prefix + SHA-256 hash of the question to `GOVERNANCE_AUDIT_LOG` (never the full text).

Additionally, `core/governance.py` exports `redact_prompt_injection_spans()` which the pipeline uses to sanitise **retrieved chunks and thread history** before they are inserted into LLM prompts.

**Disable:** `GOVERNANCE_ENABLED=false` skips injection + blocklist checks (length checks still apply when `MAX_QUESTION_CHARS > 0`).

#### Rate limiting

`GOVERNANCE_RATE_LIMIT_PER_MINUTE=60` enables a sliding-window per-client rate limit (keyed by `X-Forwarded-For` → `client.host` fallback). Single-process in-memory for dev; use Redis for multi-worker accuracy.

---

### 3.7 Observability

**File:** `core/observability.py`

#### Log format

**Default (text):**
```
10:32:45 INFO [rid=a3f1…  tid=thread-1] framework.nodes: Node: retrieval + relevance (designer)
```

**JSON mode** (`OBSERVABILITY_JSON_STDOUT=true`, for log aggregators):
```json
{"ts":"2026-05-05T10:32:45Z","level":"INFO","logger":"framework.nodes","message":"…","request_id":"a3f1…","thread_id":"thread-1"}
```

#### JSONL metrics log

Set `OBSERVABILITY_METRICS_LOG=./logs/rag_metrics.jsonl` to write one line per completed request containing: `request_id`, `thread_id`, `route`, `duration_ms`, `system`, `relevance`, `fallback`, `question_chars`, `answer_chars`.

#### Per-query JSONL

Every query appends a line to `logs/queries_YYYY-MM-DD.jsonl` containing all state fields useful for offline eval and tuning, including `retrieval_best_distance`.

#### Access log

Every HTTP request appends a JSON record via `log_access_json` to the `rag.access` logger: method, path, status, duration_ms, request_id.

---

## 4. Data Flow — Request Lifecycle

```
Client sends POST /chat {"question": "كيف أضيف منطق التخطي؟", "thread_id": "t1"}
    │
    ├─ Middleware: assign X-Request-ID, bind contextvars, start timer
    │
    ├─ Governance: check length, script gate, injection patterns, blocklist → PASS
    │
    ├─ resolve_thread_id("t1") → validated "t1"
    │
    ├─ load_conversation("t1") → last N turns from Redis / file
    │
    ├─ Build initial AgentState {question, thread_id, conversation_history, request_id,
    │                             page_id, survey_id, user_name_en/ar, system_language, …}
    │
    ├─ LangGraph invoke:
    │     language_detect → language="ar"
    │     payload_context → no prescript match (survey_context_missing=false)
    │     rewrite_and_route (typo-normalised) → {rewritten_question="…", system="designer"}
    │     retrieval → 8 chunks, relevance="relevant", best_l2=0.37
    │     answer → "لإضافة منطق التخطي … [1].\n\n**Sources:** [1]"
    │
    ├─ SSE stream: node_start×4, token×N, done{citations}
    │
    ├─ append_turn("t1", question, answer) → persisted to Redis/file
    │
    ├─ log_rag_summary → rag.rag logger + optional JSONL metrics
    │
    └─ Middleware: log_access_json, set X-Request-ID response header
```

---

## 5. API Reference

### POST /chat

**Request:**
```http
POST /chat HTTP/1.1
Content-Type: application/json
X-API-Key: your-secret        (only when RAG_API_KEY is set)

{
  "question": "What is the CATI module?",
  "thread_id": "session-42"
}
```

**SSE Response stream:**
```
event: node_start
data: {"node":"language_detect"}

event: node_start
data: {"node":"payload_context"}

event: node_start
data: {"node":"rewrite_and_route"}

event: node_start
data: {"node":"retrieval"}

event: node_start
data: {"node":"answer"}

event: token
data: The CATI module allows

event: token
data:  call centre agents to...

event: done
data: {"thread_id":"session-42","citations":[{"ref":1,"source_file":"…"}]}
```

### POST /chat/sync

Same request body; returns a single JSON object (see §3.4).

### GET /health

```json
{"status": "ok", "service": "Al-Khawarzmi AI Assistant", "version": "1.0.0"}
```

### GET /health/ready

Optionally protected by `RAG_HEALTH_READY_KEY`. Returns 200 when all catalog stores are present; 503 otherwise.

**200 OK:**
```json
{
  "status": "ready",
  "vector_stores": {
    "vector_store_root": "/abs/path/vector_stores",
    "current_embedding_model": "intfloat/multilingual-e5-large",
    "collections": {
      "designer": {
        "path": "…/designer", "ready": true,
        "embedding_model": "intfloat/multilingual-e5-large",
        "embedding_model_match": true
      },
      "runtime":    {"ready": true, "embedding_model_match": true},
      "callcenter": {"ready": false, "embedding_model": null, "embedding_model_match": null},
      "admin":      {"ready": false, "embedding_model": null, "embedding_model_match": null}
    }
  }
}
```

`embedding_model_match: false` means the store was ingested with a different model — a `WARNING` is logged at startup. Re-ingest the affected system to fix.

### GET /metrics

Prometheus exposition format. No authentication.

```
# HELP rag_requests_total Total RAG requests processed
# TYPE rag_requests_total counter
rag_requests_total{relevance="relevant",route="ar",system="designer"} 42.0
…
```

### GET /llm/config

Returns non-secret LLM settings (provider, resolved model, Ollama base URL).

```json
{"provider": "anthropic", "model": "claude-haiku-4-5-20251001", "ollama_base_url": null}
```

### GET /llm/ollama/models

Lists models reported by the local Ollama daemon. Works regardless of `LLM_PROVIDER`.

### POST /survey/ingest/{survey_id}

Triggers background embedding of a live survey JSON into a per-survey Chroma collection
under `vector_stores/surveys/<id>/`. Returns `{"status": "started"}` immediately.

### GET /survey/status/{survey_id}

Returns the current ingestion status:

```json
{"survey_id": "123", "status": "ready"}
```

Status values: `idle` | `ingesting` | `ready` | `error`.

---

## 6. Configuration Reference

All settings are read from `.env` at startup via `python-dotenv`. See `.env.example` for the full annotated reference.

### LLM Provider

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `anthropic` | Chat model vendor: `anthropic`, `openai`, `gemma` (Google AI), `ollama` (local) |
| `ANTHROPIC_API_KEY` | — | Required when `LLM_PROVIDER=anthropic` |
| `OPENAI_API_KEY` | — | Required when `LLM_PROVIDER=openai` |
| `GOOGLE_API_KEY` | — | Required when `LLM_PROVIDER=gemma` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama daemon URL (when `LLM_PROVIDER=ollama`) |
| `LLM_MODEL` | provider default | Model ID — e.g. `claude-haiku-4-5-20251001`, `gpt-4o-mini`, `gemma-2-9b-it` |
| `HF_TOKEN` | — | HuggingFace token for private embedding models |

### Embeddings & Vector Stores

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-large` | HuggingFace model for embeddings |
| `VECTOR_STORE_PATH` | `./vector_stores` | Root directory for catalog + survey Chroma stores |
| `VECTOR_BACKEND` | `chroma` | Catalog vector DB: `chroma` (on-disk) or `qdrant` (remote); survey stores are always Chroma |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant HTTP API URL (when `VECTOR_BACKEND=qdrant`) |
| `QDRANT_API_KEY` | — | Qdrant API key |
| `QDRANT_PREFER_GRPC` | `false` | Use Qdrant gRPC transport |

### Paths & Logging

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_PATH` | `./logs` | Directory for query JSONL logs |
| `MEMORY_PATH` | `./memory` | Thread memory root (JSON file backend) |
| `MEMORY_MAX_TURNS` | `10` | Max prior turns in prompt |

### API

| Variable | Default | Description |
|----------|---------|-------------|
| `API_HOST` | `0.0.0.0` | Uvicorn bind host |
| `API_PORT` | `8000` | Uvicorn bind port |
| `ALLOWED_ORIGINS` | `http://localhost:4200,…` | CORS allowed origins |
| `RAG_API_KEY` | _(unset)_ | Bearer/header/query auth for `/chat` and `/chat/sync` |
| `RAG_HEALTH_READY_KEY` | _(unset)_ | Bearer/header auth for `/health/ready`; leave unset for probes |
| `RAG_SYSTEMS` | `designer,runtime,callcenter,admin` | Comma-separated system names; add new systems here without code changes |
| `RAG_REQUIRE_VECTOR_STORES` | `false` | Fail startup if any catalog store missing |
| `APP_VERSION` | _(unset)_ | Shown in `/health` response |

### Retrieval

| Variable | Default | Description |
|----------|---------|-------------|
| `RETRIEVAL_TOP_K` | `8` | Final chunks returned |
| `RETRIEVAL_FETCH_K` | `24` | MMR candidate pool |
| `RETRIEVAL_MMR_LAMBDA` | `0.5` | MMR diversity weight (0=diverse, 1=similar) |
| `RETRIEVAL_USE_MMR` | `true` | Enable MMR diversity in dense pool |
| `RETRIEVAL_HYBRID` | `true` | Enable BM25+RRF hybrid |
| `HYBRID_DENSE_POOL` | `32` | Dense candidates before BM25 |
| `RRF_K` | `60` | RRF rank fusion constant |
| `RETRIEVAL_LANG_FILTER` | `true` | Drop chunks whose language tag doesn't match query language; disabled for mixed queries or when filtered pool < `RETRIEVAL_TOP_K` |
| `RETRIEVAL_RELEVANCE_MAX_L2` | _(unset)_ | L2 distance gate — queries above this threshold fall back; recommended: `1.2`–`1.6` for normalised e5-large |
| `RETRIEVAL_NORMALIZE_AR` | `true` | Normalise Arabic in query before embedding |
| `RERANK_CROSS_ENCODER_MODEL` | _(unset)_ | Enable cross-encoder re-rank (e.g. `BAAI/bge-reranker-v2-m3`) |
| `RERANK_POOL` | `16` | Candidates passed to cross-encoder |
| `RERANK_MAX_CHARS` | `512` | Truncate chunk text before cross-encoder |
| `RETRIEVAL_MIN_RELEVANT_CHUNKS` | `3` | Min chunks to auto-mark retrieval as relevant (skips L2 gate) |
| `QUERY_TYPO_BLOCKLIST` | _(unset)_ | Comma-separated Latin tokens to skip English spell-fix (names, brands) |

### Observability

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Python logging level |
| `OBSERVABILITY_JSON_STDOUT` | `false` | JSON log format for aggregators (Loki, CloudWatch, Datadog) |
| `OBSERVABILITY_METRICS_LOG` | _(unset)_ | Append-only JSONL path for RAG completion events |
| `APP_VERSION` | _(unset)_ | Release label shown in `/health` response |
| `OTLP_ENDPOINT` | _(unset)_ | OpenTelemetry gRPC collector (Jaeger, Grafana Tempo, OTel Collector); requires `opentelemetry-*` packages |

### Governance

| Variable | Default | Description |
|----------|---------|-------------|
| `GOVERNANCE_ENABLED` | `true` | Enable injection + blocklist checks |
| `GOVERNANCE_MAX_QUESTION_CHARS` | `12000` | Max question length |
| `GOVERNANCE_BLOCK_SUBSTRINGS` | _(unset)_ | Comma-separated blocked phrases |
| `GOVERNANCE_BLOCKLIST_FILE` | _(unset)_ | Path to newline-separated blocklist |
| `GOVERNANCE_AUDIT_LOG` | _(unset)_ | JSONL path for blocked attempts |
| `GOVERNANCE_RATE_LIMIT_PER_MINUTE` | `60` | Requests/min per client IP (429 on breach); set `0` to disable for local dev |
| `SAFETY_STRICT` | `false` | When `true`, substitute fallback message if output safety flags are raised |

### Redis (optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | _(unset)_ | e.g. `redis://localhost:6379/0` |
| `REDIS_KEY_PREFIX` | `rag:thread:` | Key prefix for thread lists |
| `REDIS_MEMORY_TTL` | `0` | Thread key TTL in seconds (0=no expiry) |

### LLM retries

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MAX_TOKENS` | `4096` | Max tokens for final answer |
| `LLM_MAX_RETRIES` | `3` | Max retry attempts on transient LLM errors |
| `LLM_RETRY_BASE_SEC` | `0.6` | Exponential backoff base (doubles each attempt) |

### Ingestion

| Variable | Default | Description |
|----------|---------|-------------|
| `INGEST_CLEAN_STORE` | `true` | Wipe `vector_stores/<system>/` before each folder re-ingest |
| `INGEST_CHUNK_SIZE` | `1100` | Characters per chunk |
| `INGEST_CHUNK_OVERLAP` | `150` | Overlap between adjacent chunks |
| `KNOWLEDGE_MONOLITH` | _(unset)_ | Path to a single `.md` ingested into all system collections |

---

## 7. Security Model

### OWASP A01 — Broken Access Control
- `RAG_API_KEY` guards chat endpoints; `/health` is always open for load balancers.
- Thread IDs are validated against `^[A-Za-z0-9_.:-]{1,128}$` — no path traversal possible in file backend.

### OWASP A03 — Injection
- Governance layer blocks prompt-injection patterns in 9 English + 6 Arabic regexes before the LLM is reached.
- Script gate (`core/query_script_gate.py`) rejects non-Latin/non-Arabic input before governance runs — no LLM cost.
- `core/governance.py` exports `redact_prompt_injection_spans()` which sanitises retrieved chunks and thread history before they enter LLM prompts.
- LLM prompts use explicit governance blocks instructing the model to ignore override attempts.
- `GOVERNANCE_BLOCKLIST_FILE` allows operators to add site-specific blocked phrases.

### OWASP A05 — Security Misconfiguration
- CORS origins are explicit (no wildcard `*` by default).
- `RAG_REQUIRE_VECTOR_STORES=true` prevents a misconfigured server silently returning empty answers.

### OWASP A06 — Vulnerable Components
- All dependencies are pinned with minimum versions in `requirements.txt`; run `pip list --outdated` regularly.

### OWASP A09 — Security Logging
- Governance audit log records: timestamp, blocked reason codes, thread_id, request_id, question prefix (≤50 chars), SHA-256 hash of the full question. The full question body is never written to disk in the audit log.
- Access log covers every request with method, path, status, duration.

### Secrets
- API keys are in `.env` (excluded from VCS via `.gitignore`).
- `.env.example` uses placeholder values only.

---

## 8. Operations Guide

### Starting the server

```powershell
# From project root with venv active
.\.venv\Scripts\python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Use `--workers 4` (without `--reload`) in production. Set `REDIS_URL` for shared thread memory across workers.

### Ingesting documents

```powershell
# All systems
python -m ingestion

# Specific systems only
python -m ingestion designer runtime

# Re-ingest after adding docs (INGEST_CLEAN_STORE=true is default — store wiped then rebuilt)
python -m ingestion designer
```

### Adding documents

Drop files into `docs/<system>/` — any subfolder depth is scanned. Supported formats: `.docx`, `.pdf`, `.txt`, `.md`, `.xlsx`. Re-run `python -m ingestion <system>` after adding files.

### Switching LLM provider

Set `LLM_PROVIDER` in `.env` and the matching API key. No code changes required.

```dotenv
# Anthropic (default)
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-…
LLM_MODEL=claude-haiku-4-5-20251001

# OpenAI
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-…
LLM_MODEL=gpt-4o-mini

# Local Ollama
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=gemma2:9b
```

### Switching vector store backend

```dotenv
# Chroma (default, local on-disk)
VECTOR_BACKEND=chroma

# Qdrant (remote)
VECTOR_BACKEND=qdrant
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your-key
```

Re-run ingestion after changing backend. Survey stores always remain Chroma regardless of this setting.

### Health checks

```bash
curl http://localhost:8000/health
curl http://localhost:8000/health/ready    # 503 if any catalog store empty
curl http://localhost:8000/llm/config      # Active provider + model
```

### Log locations

| Log | Path | Format |
|-----|------|--------|
| Query log | `logs/queries_YYYY-MM-DD.jsonl` | JSONL, one line per query |
| RAG metrics | `OBSERVABILITY_METRICS_LOG` | JSONL, one line per request |
| Governance audit | `GOVERNANCE_AUDIT_LOG` | JSONL, blocked attempts only |
| Thread memory | `memory/threads/<id>.json` | JSON array of turns |

### Production checklist

- [ ] `ANTHROPIC_API_KEY` (or provider key) set to real key
- [ ] `RAG_API_KEY` set to a long random secret
- [ ] `REDIS_URL` set (multi-worker deployments)
- [ ] `ALLOWED_ORIGINS` restricted to production domains
- [ ] `RAG_REQUIRE_VECTOR_STORES=true`
- [ ] `GOVERNANCE_RATE_LIMIT_PER_MINUTE=60` (with Redis for multi-worker accuracy)
- [ ] `GOVERNANCE_AUDIT_LOG=./logs/governance_audit.jsonl`
- [ ] `OBSERVABILITY_METRICS_LOG=./logs/rag_metrics.jsonl`
- [ ] `OBSERVABILITY_JSON_STDOUT=true` (if shipping to a log aggregator)
- [ ] `APP_VERSION=1.0.0` set to current release

---

## 9. Directory Structure

```
RAG_BE/
├── .env                         # Active secrets (git-ignored)
├── .env.example                 # Full config documentation
├── requirements.txt
│
├── core/                        # Layer 1 — reusable services, zero business logic
│   ├── retrieval.py             # Hybrid dense+BM25+RRF+optional CrossEncoder
│   ├── text_ar.py               # Arabic script helpers, normalisation, CAMeL lemmatizer
│   ├── thread_memory.py         # Redis / file thread memory
│   ├── governance.py            # Input guardrails, rate limit, audit log, redact spans
│   ├── output_safety.py         # Post-generation safety check
│   ├── observability.py         # Logging setup, access/metrics helpers, OTel
│   ├── llm_helpers.py           # LLM + embeddings singletons (provider-selectable)
│   ├── vector_stores.py         # Catalog vector stores: Chroma or Qdrant
│   ├── env_utils.py             # env_bool(), EMBEDDING_MODEL, provider helpers
│   ├── paths.py                 # PROJECT_ROOT, vector_store_root()
│   ├── greeting_intent.py       # Heuristic is_greeting()
│   ├── closing_intent.py        # Heuristic is_closing()
│   ├── client_locale.py         # say(), say_prompt(), ui_reply_language()
│   ├── session_store.py         # In-memory/Redis ingestion status store
│   ├── query_script_gate.py     # Block non-Latin/non-Arabic scripts (no LLM)
│   ├── query_typo_normalize.py  # English spell-fix before rewrite step
│   ├── ollama_models.py         # List models from local Ollama daemon
│   └── nodes/
│       ├── config.py            # Retrieval tuning constants (env-driven)
│       ├── relevance.py         # Score-based relevance decision
│       ├── rewrite_parse.py     # JSON parser for route LLM output
│       └── query_log.py         # JSONL query log appender
│
├── framework/                   # Layer 2 — generic RAG app skeleton
│   ├── profile.py               # RAGProfile dataclass + Protocol interfaces (THE CONTRACT)
│   ├── graph.py                 # LangGraph StateGraph builder
│   ├── state.py                 # AgentState TypedDict
│   ├── vector_health.py         # Catalog store readiness checks
│   ├── survey_store.py          # Survey-scoped Chroma client
│   └── nodes/
│       ├── pipeline.py          # All node callables + configure_pipeline()
│       ├── answer_prompt.py     # Prompt assembly + configure_answer_prompt()
│       ├── intent.py            # is_greeting() + is_platform_overview() delegation
│       ├── fallback_text.py     # Delegates to profile.fallback
│       ├── retrieval_step.py    # Hybrid retrieval node + survey merge
│       └── streaming.py        # SSE token streaming
│
├── alkawarzmi/                  # Layer 3 — Al-Khawarzmi business logic
│   ├── profile.py               # PROFILE instance — THE SINGLE SWAP POINT
│   ├── prompts.py               # AlKhawarzmiPrompts adapter
│   ├── prompt_templates.py      # Raw LLM prompt strings
│   ├── intents.py               # AlKhawarzmiIntentDetector adapter
│   ├── fallback.py              # AlKhawarzamiFallback adapter
│   ├── prescripts.py            # AlKhawarzmiPrescripts adapter
│   ├── payload_context.py       # Zero-LLM prescript logic (page_id, survey_id, names)
│   ├── survey_retrieval.py      # Survey vector context hooks
│   ├── image_selection.py       # Screen image URL mapping
│   ├── greeting_reply.py        # GREETING_MESSAGE_EN/AR constants
│   ├── closing_reply.py         # Closing message constants
│   ├── designer/
│   │   ├── page_map.py          # Angular route → system/context mapping
│   │   ├── prescripts.py        # Designer-specific payload prescripts
│   │   ├── survey_listing_intent.py
│   │   └── survey_layout_retrieval.py
│   └── ingestion/
│       └── survey_session.py    # Embed live survey JSON into per-survey Chroma
│
├── api/                         # Thin HTTP layer
│   ├── deps.py                  # Graph singleton + auth + ChatRequest model
│   │                            # ← from alkawarzmi.profile import PROFILE  (swap point)
│   ├── main.py                  # FastAPI app, lifespan, CORS, middleware
│   └── routers/
│       ├── chat.py              # POST /chat (SSE) + POST /chat/sync
│       ├── health.py            # GET /health + GET /health/ready + GET /metrics
│       ├── llm.py               # GET /llm/config + GET /llm/ollama/models
│       └── survey.py            # POST /survey/ingest + GET /survey/status
│
├── ingestion/                   # Standalone CLI: python -m ingestion [systems…]
│   ├── __main__.py
│   ├── ingest.py
│   ├── chroma_ingest.py
│   ├── documents.py
│   └── config.py
│
├── docs/
│   ├── designer/                # Source documents for designer system
│   ├── runtime/
│   ├── callcenter/
│   └── admin/
│
├── vector_stores/
│   ├── designer/                # Chroma/Qdrant catalog stores
│   ├── runtime/
│   ├── callcenter/
│   ├── admin/
│   └── surveys/                 # Per-survey Chroma session stores
│       └── <survey_id>/
│
├── memory/
│   └── threads/                 # JSON thread history files
│
├── logs/
│   └── queries_YYYY-MM-DD.jsonl
│
├── eval/
│   ├── golden.json
│   └── run_eval.py
│
└── tests/
    └── test_queries.py
```

---

## 10. Dependency Map

```
api/main.py
  ├── fastapi, uvicorn, sse-starlette
  ├── api.deps
  │     ├── alkawarzmi.profile      → PROFILE (single swap point)
  │     └── framework.graph
  │           └── framework.nodes   → core.retrieval (rank_bm25)
  │                                 → alkawarzmi.prompt_templates (via configure_pipeline)
  │                                 → core.llm_helpers → langchain_anthropic / langchain_openai
  │                                                       / langchain_google_genai / langchain_openai[ollama]
  │                                 → core.vector_stores → langchain_chroma / qdrant_client
  │                                 → langchain_huggingface (HuggingFaceEmbeddings)
  │                                     → sentence-transformers
  ├── core.governance     → core.text_ar
  ├── core.observability
  ├── core.thread_memory  → redis (optional)
  └── framework.vector_health → core.paths

ingestion/
  ├── langchain_chroma (or qdrant_client)
  ├── langchain_community (loaders: Docx2txt, PyPDF, TextLoader)
  ├── langchain_text_splitters
  ├── docx2txt, pypdf, openpyxl
  └── core.paths, core.llm_helpers, framework.vector_health

alkawarzmi.ingestion.survey_session
  ├── framework.survey_store → langchain_chroma
  └── core.llm_helpers → HuggingFaceEmbeddings
```

---

## 11. Product Swap Guide

The entire Al-Khawarzmi business layer lives in `alkawarzmi/`. To deploy the same
RAG engine for a different product:

1. Create a new package (e.g. `myproduct/`) mirroring `alkawarzmi/`:
   - `myproduct/prompts.py` — implement `PromptProvider` protocol
   - `myproduct/intents.py` — implement `IntentDetector` protocol
   - `myproduct/fallback.py` — implement `FallbackProvider` protocol
   - `myproduct/prescripts.py` — implement `PrescriptProvider` protocol
   - `myproduct/profile.py` — assemble `PROFILE = RAGProfile(...)`

2. Change **one line** in `api/deps.py`:
   ```python
   # Before
   from alkawarzmi.profile import PROFILE as _ACTIVE_PROFILE
   # After
   from myproduct.profile import PROFILE as _ACTIVE_PROFILE
   ```

3. Ingest your product's docs into `docs/` subdirectories, update `RAG_SYSTEMS` in `.env`, and run `python -m ingestion`.

Everything in `core/` and `framework/` is unchanged. `api/` is unchanged.
