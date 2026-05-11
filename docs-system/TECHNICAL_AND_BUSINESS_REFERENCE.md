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

```
┌────────────────────────────────────────────────────────────────────────┐
│                           Client (Angular / REST)                      │
│          POST /chat  (SSE stream)   POST /chat/sync  (JSON)            │
└──────────────────────────────┬─────────────────────────────────────────┘
                               │ HTTP
┌──────────────────────────────▼─────────────────────────────────────────┐
│                         FastAPI  (api/main.py)                         │
│  Middleware: X-Request-ID · CORS · Rate-limit · Governance check       │
│  Routes: /health  /health/ready  /chat  /chat/sync                     │
└──────────────────────────────┬─────────────────────────────────────────┘
                               │ invoke / astream
┌──────────────────────────────▼─────────────────────────────────────────┐
│                      LangGraph Agent  (agent/)                         │
│                                                                         │
│  language_detect ──► rewrite_and_route ──► retrieval ──► answer        │
│                                                       └──► fallback     │
│                                                                         │
│  State: AgentState (TypedDict)  ·  Thread memory injected before invoke│
└──────┬────────────────────────────────────────────────────┬────────────┘
       │ HuggingFace Embeddings                             │ Anthropic API
       │ intfloat/multilingual-e5-large                     │ claude-haiku / sonnet
┌──────▼──────────┐                              ┌──────────▼─────────────┐
│  Chroma (local) │  4 collections               │   Claude LLM           │
│  vector_stores/ │  designer / runtime /         │   rewrite+route (1×)   │
│  admin          │  callcenter / admin           │   answer (1×)          │
└─────────────────┘                              └────────────────────────┘
       │
┌──────▼──────────┐
│  BM25 (in-proc) │  Lexical re-ranking over dense candidate pool (RRF)
└─────────────────┘
```

### LLM call budget per request

| Scenario | LLM calls | Notes |
|----------|-----------|-------|
| On-topic question | **2** | rewrite+route + answer |
| Off-topic question | **1** | rewrite+route only (system=none → skip retrieval) |
| Fallback (empty retrieval) | **1** | rewrite+route only |

---

## 3. Component Reference

### 3.1 Ingestion Pipeline

**Entry point:** `python -m ingestion` (from project root with venv active)

**Files:** `ingestion/config.py`, `ingestion/documents.py`, `ingestion/chroma_ingest.py`, `ingestion/__main__.py`

#### What it does

1. Loads `.env` (embedding model, paths, chunk config).
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
   - Creates a `Chroma.from_documents` persist store.
4. If `KNOWLEDGE_MONOLITH` is set, loads that markdown into all four collections, de-duplicating previous monolith vectors before re-inserting.
5. Writes `.metadata.json` to each `vector_stores/<system>/` directory recording the embedding model name, ingest timestamp, and chunk count. At API startup `vector_health.py` reads this file and logs a `WARNING` when the stored model differs from the current `EMBEDDING_MODEL` — preventing silent quality regressions after a model change.

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

**File:** `agent/graph.py`, `agent/nodes.py`, `agent/state.py`

The pipeline is a **linear StateGraph** with one conditional branch:

```
START
  │
  ▼
language_detect_node        — heuristic, no LLM
  │
  ▼
rewrite_and_route_node      — 1 LLM call, returns JSON {rewritten_question, system}
  │
  ▼
retrieval_node              — hybrid dense+BM25+RRF, sets relevance
  │
  ├─── relevance=relevant ──► answer_node    (1 LLM call)
  │
  └─── relevance=irrelevant ► fallback_node  (static message, no LLM)
```

#### AgentState fields

| Field | Type | Set by |
|-------|------|--------|
| `question` | `str` | API caller |
| `language` | `"ar"/"en"/"mixed"` | `language_detect_node` |
| `rewritten_question` | `str` | `rewrite_and_route_node` |
| `system` | `str\|None` | `rewrite_and_route_node` — `"none"` for off-topic |
| `retrieved_chunks` | `list[str]` | `retrieval_node` |
| `retrieved_source_refs` | `list[dict]` | `retrieval_node` |
| `retrieval_best_distance` | `float\|None` | `retrieval_node` — top-1 L2, logged for tuning |
| `relevance` | `"relevant"/"irrelevant"/"unknown"` | `retrieval_node` |
| `answer` | `str` | `answer_node` / `fallback_node` |
| `thread_id` | `str` | API (resolved before invoke) |
| `conversation_history` | `list[dict]` | Loaded from memory before invoke |
| `request_id` | `str` | API middleware |

#### Off-topic routing

When `rewrite_and_route_node` returns `system="none"`, `retrieval_node` immediately returns `relevance="irrelevant"` with empty chunks — no embedding query, no Chroma round-trip, no citations.

---

### 3.3 Retrieval Engine

**File:** `agent/retrieval.py`

#### Hybrid retrieval (default)

```
Query
  │
  ├─► Dense Chroma similarity_search (pool of HYBRID_DENSE_POOL=32 docs)
  │       └── MMR re-ordering (optional, RETRIEVAL_USE_MMR=true)
  │
  ├─► Language filter (RETRIEVAL_LANG_FILTER=true)
  │       └── Drops chunks whose language tag doesn't match query language
  │           when filtered pool ≥ RETRIEVAL_TOP_K (skipped for mixed queries)
  │
  ├─► BM25Okapi over the filtered dense pool (in-process, no index rebuild)
  │       └── Arabic-aware tokenizer: strips diacritics + light morphological
  │           stemming (prefix/suffix stripping via stem_arabic_token)
  │
  ├─► RRF fusion (RRF_K=60) → top RETRIEVAL_TOP_K docs
  │
  └─► Deduplication: near-duplicate chunks (same source_file + first 200 chars)
      are dropped, keeping highest-ranked copy
```

#### Optional Cross-encoder re-ranking

Set `RERANK_CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2` to enable a third-stage re-rank. This improves precision at the cost of ~100–300 ms load per request and requires `sentence-transformers` (already in `requirements.txt`).

#### Relevance gating

`relevance` is set to `"irrelevant"` when:
1. Zero chunks are returned (empty store or all filtered out).
2. `RETRIEVAL_RELEVANCE_MAX_L2` is set and the top-1 dense L2 distance exceeds the threshold.
3. `system="none"` was returned by the router (off-topic).

Without `RETRIEVAL_RELEVANCE_MAX_L2`, cases 1 and 3 are the only hard gates. The LLM grounding prompt (case 2 fallback) handles borderline content at the answer layer.

---

### 3.4 FastAPI Service

**File:** `api/main.py`

#### Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/health` | none | Liveness — always 200 |
| `GET` | `/health/ready` | optional (`RAG_HEALTH_READY_KEY`) | Readiness — 503 if stores missing |
| `GET` | `/metrics` | none | Prometheus exposition format metrics |
| `POST` | `/chat` | optional (`RAG_API_KEY`) | SSE token stream |
| `POST` | `/chat/sync` | optional (`RAG_API_KEY`) | Blocking JSON response |

#### `/chat` — SSE event types

| Event | Payload | When |
|-------|---------|------|
| `node_start` | `{"node": "language_detect"}` … | Before each pipeline step |
| `token` | raw text string | LLM streaming delta (or full fallback string) |
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

`/health` and `/health/ready` are always unauthenticated.

---

### 3.5 Thread Memory

**File:** `agent/thread_memory.py`

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

**File:** `agent/governance.py`

Runs **before** the LangGraph graph on every chat request.

#### Checks (in order)

1. **Empty / length** — rejects blanks and questions > `GOVERNANCE_MAX_QUESTION_CHARS` (default 12 000).
2. **Injection heuristics** — 9 English regex patterns + 6 Arabic patterns for common prompt-override phrases.
3. **Substring blocklist** — comma-separated `GOVERNANCE_BLOCK_SUBSTRINGS` env var, plus an optional newline file `GOVERNANCE_BLOCKLIST_FILE`.

On block: returns HTTP 403 (policy violation) or 422 (too long). Logs prefix + SHA-256 hash of the question to `GOVERNANCE_AUDIT_LOG` (never the full text).

**Disable:** `GOVERNANCE_ENABLED=false` skips injection + blocklist checks (length checks still apply when `MAX_QUESTION_CHARS > 0`).

#### Rate limiting

`GOVERNANCE_RATE_LIMIT_PER_MINUTE=60` enables a sliding-window per-client rate limit (keyed by `X-Forwarded-For` → `client.host` fallback). Single-process in-memory for dev; use Redis in production for multi-worker accuracy.

---

### 3.7 Observability

**File:** `agent/observability.py`

#### Log format

**Default (text):**
```
10:32:45 INFO [rid=a3f1…  tid=thread-1] agent.nodes: Node: retrieval + relevance (designer)
```

**JSON mode** (`OBSERVABILITY_JSON_STDOUT=true`, for log aggregators):
```json
{"ts":"2026-05-05T10:32:45Z","level":"INFO","logger":"agent.nodes","message":"…","request_id":"a3f1…","thread_id":"thread-1"}
```

#### JSONL metrics log

Set `OBSERVABILITY_METRICS_LOG=./logs/rag_metrics.jsonl` to write one line per completed request containing: `request_id`, `thread_id`, `route`, `duration_ms`, `system`, `relevance`, `fallback`, `question_chars`, `answer_chars`.

#### Per-query JSONL

Every query (answer or fallback) appends a line to `logs/queries_YYYY-MM-DD.jsonl` containing all state fields useful for offline eval and tuning, including `retrieval_best_distance`.

#### Access log

Every HTTP request appends a JSON record via `log_access_json` to the `rag.access` logger: method, path, status, duration_ms, request_id.

---

## 4. Data Flow — Request Lifecycle

```
Client sends POST /chat {"question": "كيف أضيف منطق التخطي؟", "thread_id": "t1"}
    │
    ├─ Middleware: assign X-Request-ID, bind contextvars, start timer
    │
    ├─ Governance: check length, injection patterns, blocklist → PASS
    │
    ├─ resolve_thread_id("t1") → validated "t1"
    │
    ├─ load_conversation("t1") → last N turns from Redis / file
    │
    ├─ Build initial AgentState {question, thread_id, conversation_history, request_id}
    │
    ├─ LangGraph invoke:
    │     language_detect → language="ar"
    │     rewrite_and_route → {rewritten_question="…", system="designer"}
    │     retrieval → 8 chunks, relevance="relevant", best_l2=0.37
    │     answer → "لإضافة منطق التخطي … [1].\n\n**Sources:** [1]"
    │
    ├─ SSE stream: node_start×3, token×N, done{citations}
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
{"status": "ok", "service": "Al-Khwarizmi AI Assistant"}
```

### GET /health/ready

Optionally protected by `RAG_HEALTH_READY_KEY` (Bearer or `X-API-Key` header). Leave unset for load-balancer probes.

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
      "runtime":  {"path": "…/runtime", "ready": true, "embedding_model_match": true},
      "callcenter": {"path": "…/callcenter", "ready": false, "embedding_model": null, "embedding_model_match": null},
      "admin":    {"path": "…/admin", "ready": false, "embedding_model": null, "embedding_model_match": null}
    }
  }
}
```

`embedding_model_match: false` means the store was ingested with a different embedding model — a `WARNING` is also logged at startup. Re-ingest the affected system to fix.

**503 Not Ready** (same body, HTTP 503) — use for Kubernetes readiness probe.

### GET /metrics

Returns Prometheus exposition format text. No authentication. Returns 503 when `prometheus-client` is not installed.

```
# HELP rag_requests_total Total RAG requests processed
# TYPE rag_requests_total counter
rag_requests_total{relevance="relevant",route="ar",system="designer"} 42.0
# HELP rag_request_duration_seconds End-to-end RAG request duration in seconds
# TYPE rag_request_duration_seconds histogram
rag_request_duration_seconds_bucket{le="0.5",route="en"} 3.0
…
```

---

## 6. Configuration Reference

All settings are read from `.env` at startup via `python-dotenv`.

### Required

| Variable | Example | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | `sk-ant-…` | Anthropic API credential |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-large` | HuggingFace model for embeddings |
| `LLM_MODEL` | `claude-haiku-4-5-20251001` | Anthropic model ID |
| `VECTOR_STORE_PATH` | `./vector_stores` | Root directory for Chroma stores |
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
| `RAG_REQUIRE_VECTOR_STORES` | `false` | Fail startup if any store missing |
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
| `RETRIEVAL_LANG_FILTER` | `true` | Drop chunks whose language tag doesn't match query language (Arabic ↔ English); disabled for mixed queries or when filtered pool < `RETRIEVAL_TOP_K` |
| `RETRIEVAL_RELEVANCE_MAX_L2` | _(unset)_ | L2 distance gate — queries above this threshold return a fallback instead of the LLM; recommended start: `1.2` for e5-normalised embeddings |
| `RETRIEVAL_NORMALIZE_AR` | `true` | Normalise Arabic in query before embedding |
| `RERANK_CROSS_ENCODER_MODEL` | _(unset)_ | Enable cross-encoder re-rank (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) |
| `RERANK_POOL` | `16` | Candidates passed to cross-encoder |
| `RERANK_MAX_CHARS` | `512` | Truncate chunk text before cross-encoder |

### Observability

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Python logging level |
| `OBSERVABILITY_JSON_STDOUT` | `false` | JSON log format for aggregators (Loki, CloudWatch, Datadog) |
| `OBSERVABILITY_METRICS_LOG` | _(unset)_ | Append-only JSONL path for RAG completion events |
| `APP_VERSION` | _(unset)_ | Release label shown in `/health` response |

### Governance

| Variable | Default | Description |
|----------|---------|-------------|
| `GOVERNANCE_ENABLED` | `true` | Enable injection + blocklist checks |
| `GOVERNANCE_MAX_QUESTION_CHARS` | `12000` | Max question length |
| `GOVERNANCE_BLOCK_SUBSTRINGS` | _(unset)_ | Comma-separated blocked phrases |
| `GOVERNANCE_BLOCKLIST_FILE` | _(unset)_ | Path to newline-separated blocklist |
| `GOVERNANCE_AUDIT_LOG` | _(unset)_ | JSONL path for blocked attempts |
| `GOVERNANCE_RATE_LIMIT_PER_MINUTE` | `60` | Requests/min per client IP (429 on breach); set `0` to disable for local dev. Use Redis (`REDIS_URL`) for multi-worker accuracy. |

### Redis (optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | _(unset)_ | e.g. `redis://localhost:6379/0` |
| `REDIS_KEY_PREFIX` | `rag:thread:` | Key prefix for thread lists |
| `REDIS_MEMORY_TTL` | `0` | Thread key TTL in seconds (0=no expiry) |

### LLM retries

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MAX_RETRIES` | `3` | Max retry attempts on transient LLM errors |
| `LLM_RETRY_BASE_SEC` | `0.6` | Exponential backoff base (doubles each attempt) |

### Ingestion

| Variable | Default | Description |
|----------|---------|-------------|
| `INGEST_CLEAN_STORE` | `true` | Wipe `vector_stores/<system>/` before each folder re-ingest (idempotent rebuilds) |
| `INGEST_CHUNK_SIZE` | `1100` | Characters per chunk |
| `INGEST_CHUNK_OVERLAP` | `150` | Overlap between adjacent chunks |
| `KNOWLEDGE_MONOLITH` | _(unset)_ | Path to a single `.md` file ingested into all system collections simultaneously |

---

## 7. Security Model

### OWASP A01 — Broken Access Control
- `RAG_API_KEY` guards chat endpoints; `/health` is always open for load balancers.
- Thread IDs are validated against `^[A-Za-z0-9_.:-]{1,128}$` — no path traversal possible in file backend.

### OWASP A03 — Injection
- Governance layer blocks prompt-injection patterns in 9 English + 6 Arabic regexes before the LLM is reached.
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
.\venv\Scripts\python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
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

### Health checks

```bash
curl http://localhost:8000/health
curl http://localhost:8000/health/ready    # 503 if any store empty
```

### Log locations

| Log | Path | Format |
|-----|------|--------|
| Query log | `logs/queries_YYYY-MM-DD.jsonl` | JSONL, one line per query |
| RAG metrics | `OBSERVABILITY_METRICS_LOG` | JSONL, one line per request |
| Governance audit | `GOVERNANCE_AUDIT_LOG` | JSONL, blocked attempts only |
| Thread memory | `memory/threads/<id>.json` | JSON array of turns |

### Production checklist

- [ ] `ANTHROPIC_API_KEY` set to real key
- [ ] `RAG_API_KEY` set to a long random secret
- [ ] `REDIS_URL` set (multi-worker deployments)
- [ ] `ALLOWED_ORIGINS` restricted to production domains
- [ ] `RAG_REQUIRE_VECTOR_STORES=true`
- [ ] `GOVERNANCE_RATE_LIMIT_PER_MINUTE=60` (with Redis)
- [ ] `GOVERNANCE_AUDIT_LOG=./logs/governance_audit.jsonl`
- [ ] `OBSERVABILITY_METRICS_LOG=./logs/rag_metrics.jsonl`
- [ ] `OBSERVABILITY_JSON_STDOUT=true` (if shipping to a log aggregator)
- [ ] `APP_VERSION=1.0.0` set to current release

---

## 9. Directory Structure

```
RAG/
├── .env                       # Active secrets (git-ignored)
├── .env.example               # Full config documentation
├── requirements.txt
│
├── api/
│   └── main.py                # FastAPI app, lifespan, routes, SSE
│
├── agent/
│   ├── graph.py               # LangGraph StateGraph builder
│   ├── nodes.py               # All node functions + PRE_ANSWER_PIPELINE
│   ├── retrieval.py           # Hybrid dense+BM25+RRF+optional CrossEncoder
│   ├── prompts.py             # All LLM prompt templates
│   ├── state.py               # AgentState TypedDict
│   ├── governance.py          # Input guardrails, rate limit, audit log
│   ├── observability.py       # Logging setup, access/metrics helpers
│   ├── thread_memory.py       # Redis / file thread memory
│   ├── paths.py               # PROJECT_ROOT, vector_store_root()
│   ├── text_ar.py             # Arabic script helpers, normalisation
│   └── vector_health.py       # Chroma readiness checks
│
├── ingestion/
│   ├── __main__.py            # CLI entry: python -m ingestion
│   ├── ingest.py              # main() wiring
│   ├── chroma_ingest.py       # Chroma persistence logic
│   ├── documents.py           # File loaders, splitter, language detection
│   └── config.py              # Env vars for ingestion
│
├── docs/
│   ├── designer/              # Source documents for designer system
│   ├── runtime/               # Source documents for runtime system
│   ├── callcenter/            # Source documents for callcenter system
│   └── admin/                 # Source documents for admin system
│
├── vector_stores/
│   ├── designer/              # Chroma persist dir (chroma.sqlite3)
│   ├── runtime/
│   ├── callcenter/
│   └── admin/
│
├── memory/
│   └── threads/               # JSON thread history files
│
├── logs/
│   └── queries_YYYY-MM-DD.jsonl
│
├── eval/
│   ├── golden.json            # Ground-truth eval cases
│   └── run_eval.py            # Offline eval runner
│
└── tests/
    └── test_queries.py        # Pytest smoke tests
```

---

## 10. Dependency Map

```
api/main.py
  ├── fastapi, uvicorn, sse-starlette
  ├── agent.graph           → agent.nodes → agent.retrieval (rank_bm25)
  │                                       → agent.prompts
  │                                       → langchain_anthropic
  │                                       → langchain_chroma (chromadb)
  │                                       → langchain_community (HuggingFaceEmbeddings)
  │                                           → sentence-transformers
  ├── agent.governance      → agent.text_ar
  ├── agent.observability
  ├── agent.thread_memory   → redis (optional)
  └── agent.vector_health   → agent.paths

ingestion/
  ├── langchain_chroma
  ├── langchain_community (loaders: Docx2txt, PyPDF, TextLoader)
  ├── langchain_text_splitters
  ├── docx2txt, pypdf, openpyxl
  └── agent.paths, agent.vector_health
```
