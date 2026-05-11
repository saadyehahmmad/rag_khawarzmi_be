# Al-Khawarzmi RAG System — Senior Engineer Deep-Dive

> **Audience:** Senior AI / ML engineers  
> **Format:** Technical walkthrough of every non-obvious decision  
> **Goal:** Explain what the system does, why each layer was built the way it was, and what separates it from a tutorial RAG

---

## 0. The Problem Statement

Al-Khawarzmi is a statistical survey platform with four distinct sub-products — **Designer**, **Admin** (Field Management), **Call Center**, and **Runtime** — each with its own Arabic/English manual. Users ask support questions in:

- Modern Standard Arabic (MSA)
- Gulf dialect (Khaleeji) — structurally different from MSA
- English
- Code-switched mixed language ("كيف أعمل skip logic في designer؟")

A naive single-collection RAG fails because:
1. Gulf dialect words are different *lexemes*, not just inflections — a model trained on MSA won't match them
2. Questions about Designer and Admin frequently overlap in vocabulary
3. A single dense retriever over a merged 4-system corpus produces irrelevant cross-system citations

---

## 1. System Architecture (30-second map)

```
OFFLINE ──────────────────────────────────────────────────────────────────
  docs/<system>/*.md/pdf/docx
         │
         ▼ ingestion/
  Section enrichment → Chunking → multilingual-e5-large → Chroma
  One persist directory per system: vector_stores/{designer,admin,callcenter,runtime}/

ONLINE ───────────────────────────────────────────────────────────────────
  Browser POST /chat (SSE)
         │
         ▼ FastAPI + sse-starlette
  Governance gate (injection heuristics, rate limit, blocklist)
         │
         ▼ LangGraph StateGraph
  [language_detect] → [rewrite_and_route] → [retrieval]
                                                 │
                           ┌─────── relevant ────┘──── irrelevant ───┐
                           ▼                                         ▼
                     [answer_node]                           [fallback_node]
                     Claude stream                           static bilingual
                           │
                     SSE token deltas → Browser
```

---

## 2. Ingestion Pipeline — What Makes It Non-Trivial

### 2.1 Section context enrichment

**Problem:** Chunked text loses its heading context. A 1,100-character chunk about "skip logic" has no idea it came from `Designer > Builder > Rules > Skip Logic`. When the LLM cites it, the user gets `[1]` with no breadcrumb.

**Solution:** Before splitting, `enrich_with_section_context()` scans each document for Markdown heading levels (`#`, `##`, `###`) and prepends the full heading path as metadata:

```
section_path = "Survey Builder > Rules Engine > Skip Logic"
system_label = "Survey Designer"
```

At retrieval time, `format_numbered_context()` reconstructs the header on every chunk even after splitting:

```
[Survey Designer | Survey Builder > Rules Engine > Skip Logic]

To add skip logic to a question…
```

The LLM now sees the context envelope on every chunk, not just the first one in a section.

---

### 2.2 Per-system isolated stores

Four Chroma persist directories, one per sub-product. This is a deliberate architecture decision over a single merged collection because:

| Merged collection | Per-system stores |
|-------------------|-------------------|
| Cross-system noise in top-K | Queries isolated to the routed system |
| Hard to rebuild one system | `python -m ingestion designer` rebuilds only that store |
| One model mismatch kills everything | `.metadata.json` per store tracks embedding model used |

The `.metadata.json` file written at ingest time contains:
```json
{
  "embedding_model": "intfloat/multilingual-e5-large",
  "ingest_time": "2026-05-10T14:32:00Z",
  "chunks": 847,
  "system": "designer"
}
```

At API startup, `vector_health.py` reads this and warns when the model used during ingest differs from the current `EMBEDDING_MODEL` env var — a mismatch that would silently produce wrong distances.

---

### 2.3 Monolith upsert without duplicates

Optional `KNOWLEDGE_MONOLITH` allows one shared document (e.g. company-wide policy) to be ingested into all four collections. The upsert is surgical:

1. Tag all monolith chunks with `ingest_source=monolith`
2. Before each re-run: batch-delete all `ingest_source=monolith` rows in each collection
3. Add fresh chunks

Folder chunks (`ingest_source=folder`) are never touched. This avoids the classic "re-run duplicates everything" bug that plagues most ingestion scripts.

---

## 3. Embedding Model — Why `multilingual-e5-large`

| Model | Arabic quality | Gulf dialect | Size | Cost |
|-------|---------------|--------------|------|------|
| `text-embedding-ada-002` | Moderate | Poor | API | Per token |
| `paraphrase-multilingual-mpnet-base` | Good | Limited | 278M | Free |
| **`intfloat/multilingual-e5-large`** | **Excellent** | **Good** | **560M** | **Free** |
| `Arabic-BERT` | Excellent (MSA only) | No | 135M | Free |

Key constraint: **`normalize_embeddings=True` is not optional.** Without L2 normalisation, dot-product ≠ cosine similarity, breaking both Chroma similarity search and the semantic flow-image matching which relies on `dot(q_vec, flow_vec) == cosine_sim`.

The same model singleton is used for:
1. Indexing chunks during ingestion
2. Querying at retrieval time
3. Semantic image matching against `screens.json` flow descriptions

Three use cases, one model load, zero extra memory.

---

## 4. The Arabic NLP Stack — Three Layers

This is the part most RAG systems skip entirely.

### Layer 1 — Script ratio language detection (no LLM)

```python
arabic_ratio = count(chars in U+0600–U+06FF) / len(text)

if ratio >= 0.32 → "ar"
if ratio >= 0.06 → "mixed"
else             → "en"
```

Zero latency, zero cost, deterministic. Used to:
- Select fallback message language (Arabic or English)
- Decide whether to apply Arabic normalisation before retrieval
- Bias BM25 tokenisation toward Arabic or English paths

---

### Layer 2 — Gulf dialect vocabulary bridge

CAMeL Tools handles *morphological inflections* within a dialect — different word forms of the same root. It **cannot** bridge *vocabulary* gaps between dialects because Gulf and MSA use different lexemes:

| Gulf (user input) | MSA (corpus) | Meaning |
|-------------------|--------------|---------|
| وين | أين | where |
| أبغى | أريد | I want |
| أقدر | أستطيع | I can |
| أسوي | أفعل | I do/make |
| مو | ليس | not |
| الحين | الآن | now |

The `_GULF_VOCAB_BRIDGE` dictionary (35 entries) maps Gulf lexemes to MSA **before** CAMeL lemmatisation. This is applied during BM25 tokenisation so the Gulf question matches MSA corpus chunks.

Without this bridge, a question like `"أبغى أسوي استمارة جديدة"` ("I want to create a new survey") would score near zero against corpus chunks written in MSA, even though the semantic content is identical.

---

### Layer 3 — CAMeL Tools morphological lemmatisation

Arabic is morphologically rich — a single root can produce hundreds of inflected forms. BM25 is a term-frequency model; without lemmatisation it misses inflections.

```
يضيف → يضيف (adds)    — in user question
أضيف → أضيف (add)     — in corpus chunk
```

These are the same root (`ض-ي-ف`) but different surface forms. BM25 would not match them.

CAMeL provides two analyzers:
- **`calima-glf-01`** — Gulf Arabic morphology database
- **`calima-msa-r13`** — MSA morphology database (fallback)

`camel_get_lemmas(token)` runs both analyzers and returns up to 3 unique lemmas, deduplicating across databases. All lemmas become separate BM25 tokens — **soft query expansion** without LLM involvement.

**Graceful degradation:** if CAMeL is not installed or data is missing, the function returns `[token]` unchanged. BM25 still works, just without lemmatisation.

**Stopword removal:** 30 Arabic function words (`في`, `من`, `على`, `هو`, `التي`, …) are stripped before BM25 scoring to prevent high-frequency particles from dominating term weights.

---

## 5. Retrieval — The Hybrid Pipeline

### 5.1 Why pure dense retrieval fails here

Dense retrieval finds semantically similar text. It fails when:
- The user mentions an exact UI label (`"زر إضافة سؤال"` — the "Add Question" button)
- A product-specific code or feature name (`"calima-glf-01"`, `"skip logic"`)
- Arabic terms that happen to have low cosine similarity to the corpus phrasing despite being semantically equivalent

BM25 finds exact term overlap. It fails when synonyms are used (`"استمارة"` vs `"نموذج"` for "form").

The hybrid approach gets both.

---

### 5.2 Why BM25 over the dense pool (not the full corpus)

A common implementation runs BM25 over the entire corpus. This is slow at query time because BM25 must score every chunk.

The architecture here runs BM25 **only over the dense candidate pool** (default: top-32 chunks from Chroma). The dense retriever acts as a semantic pre-filter, and BM25 reorders that pool by lexical precision. This keeps BM25 scoring fast regardless of corpus size.

```
Dense pool (k=32 from Chroma similarity_search_with_score)
         │
         ├── BM25 score each of the 32 docs
         │
         └── RRF fusion → top-8 → [optional CrossEncoder] → deduplicate → return
```

---

### 5.3 Reciprocal Rank Fusion — the math

RRF is parameter-free and does not require score normalisation across two different scoring functions (L2 distance vs BM25 score are not comparable units):

```
RRF(doc) = 1 / (60 + rank_dense) + 1 / (60 + rank_bm25)
```

`k=60` from the original Cormack et al. 2009 paper. It smooths the advantage of being ranked #1 vs #2. Raising `k` flattens the curve; lowering it amplifies top-rank advantage.

Result: a document that ranks 2nd in dense and 2nd in BM25 outscores a document that ranks 1st in dense but 30th in BM25. Consistent relevance beats single-axis dominance.

---

### 5.4 Optional CrossEncoder reranking

When `RERANK_CROSS_ENCODER_MODEL=cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` is set:

- `cross-encoder/mmarco-*` was trained on MS MARCO with multilingual queries — strong Arabic + English support
- The cross-encoder is fed `(query, chunk)` pairs and produces a single relevance score
- Applied only to the top-24 RRF results to keep latency bounded
- Adds ~100-200ms but significantly improves precision for ambiguous queries

Not enabled by default because the performance gain requires the model to be pre-downloaded locally.

---

### 5.5 Relevance gate — removing the LLM grader

An earlier version used a separate Claude call to grade `relevant`/`irrelevant`. This was removed because:

1. **Slow** — an extra round trip after retrieval before the answer
2. **Expensive** — a third LLM call when 2 are enough
3. **Unreliable** — the grader was biased toward "relevant" even for weak retrieval results

Replacement: two complementary heuristics.

**Heuristic 1 — chunk count:**
```python
if len(chunks) >= RETRIEVAL_MIN_RELEVANT_CHUNKS:  # default: 3
    return "relevant"
```
If the hybrid pipeline returned ≥ 3 chunks, the corpus clearly has signal. BM25 is hard to fool — it requires actual term overlap.

**Heuristic 2 — top-1 L2 distance (optional):**
```python
if best_l2_distance <= RETRIEVAL_RELEVANCE_MAX_L2:  # default: 1.6
    return "relevant"
```
Only applied when chunk count is below the threshold. `0 = identical, 1.6 = far, 2.0 = unrelated` for normalised E5 embeddings.

Net effect: 5 LLM calls → 2 LLM calls per question. 60% reduction in API cost and latency.

---

## 6. LangGraph — Why a Graph Over a Chain

### 6.1 What a chain cannot do

LangChain LCEL (`prompt | llm | parser`) is a linear pipeline. It cannot:

1. **Branch conditionally** — `answer` vs `fallback` based on retrieval relevance
2. **Share mutable state** across all steps — each LCEL step only sees the previous step's output
3. **Emit node lifecycle events** — no hook to fire `node_start` SSE events as each step begins

### 6.2 The single source of truth pattern

```python
PRE_ANSWER_PIPELINE: tuple[tuple[str, NodeFn], ...] = (
    ("language_detect",   language_detect_node),
    ("rewrite_and_route", rewrite_and_route_node),
    ("retrieval",         retrieval_node),
)
```

This tuple is iterated by **both**:
- `graph.py` — to build the LangGraph `StateGraph` edges
- `api/main.py` `_sse_stream` — to execute nodes sequentially while emitting SSE events

One definition, two consumers. Adding a new node means adding one line here. The graph and streaming stay in sync by construction.

### 6.3 State as a TypedDict

```python
class AgentState(TypedDict):
    question: str
    language: Literal["ar", "en", "mixed"]
    rewritten_question: str
    system: Optional[str]
    retrieved_chunks: list[str]
    retrieved_source_refs: list[dict]
    retrieval_best_distance: float | None
    image_urls: list[str]
    relevance: Literal["relevant", "irrelevant", "unknown"]
    answer: str
    thread_id: str
    conversation_history: list[dict]
    request_id: str
```

Every node receives the full state and returns a **partial update dict**. LangGraph merges the partial. No node knows which step it is in the pipeline — they are pure functions of state.

`TypedDict` over Pydantic: zero runtime overhead (it's just a type hint). Pydantic would add `.model_dump()` / `.model_validate()` at every node boundary — unnecessary here.

---

## 7. The Rewrite + Route Node — One LLM Call for Two Tasks

### 7.1 Why combine rewrite and route

Query rewriting (resolve pronouns, clarify follow-ups) and routing (which sub-product) require the **same context**: the current question + conversation history.

Separating them would fire two LLM calls with identical input. The combined approach uses one prompt that returns structured JSON:

```json
{"rewritten_question": "How do I add skip logic to a question in the Survey Designer?", "system": "designer"}
```

### 7.2 Defensive JSON parsing

Claude almost always returns valid JSON, but the parser (`_parse_rewrite_route_json`) is written for the worst case:
- Strip markdown code fences (` ```json ... ``` `)
- Find outermost `{` and `}` braces (tolerates leading/trailing text)
- Validate `system` is one of the five valid values
- Fall back to `("original_question", "designer")` on any parse failure

This function has never caused a 500 error in production because it degrades gracefully.

### 7.3 Follow-up resolution

The rewrite prompt has 9 explicitly labelled cases:
- Language switch (`"بالعربي"` → rewrite the previous topic in Arabic)
- Step reference (`"explain step 2"` → expand with the full workflow from history)
- Pronoun reference (`"how do I do that?"` → substitute the actual subject)
- Elaboration, example, negation, next step, implicit subject, repetition

Without this, multi-turn conversations break — `"please in english"` would be embedded as a retrieval query and return zero relevant chunks.

---

## 8. Semantic Image Matching

`docs/screens.json` maps UI flows to screenshot filenames, with bilingual descriptions:

```json
{
  "id": "skip_logic",
  "system": "designer",
  "desc": {
    "en": "Add skip logic to a question",
    "ar": "إضافة منطق التخطي للسؤال"
  },
  "steps": ["step1_en.png", "step2_en.png"]
}
```

At startup, all flow descriptions (en + ar concatenated) are embedded in one batch call using the same model singleton — no extra memory, no extra model load.

At query time:
```python
sim = dot(embed(rewritten_question), flow_embedding)
# dot-product == cosine similarity because embeddings are L2-normalised
if sim >= 0.65:
    return flow["steps"]  # attach screenshots to response
```

The threshold `0.65` means the question must be about the same topic as the flow, not just in the same system. A question about "user roles" will not return screenshots for "skip logic" even if both are in Designer.

**Critical detail:** only the LLM-rewritten question is embedded, not the raw user input. For a follow-up like `"what was step 2?"`, the rewritten question is `"Step 2 of adding skip logic in the Survey Designer"` — which matches the flow embedding. The raw input would match nothing.

---

## 9. SSE Streaming Architecture

### 9.1 Why SSE over WebSocket

| | SSE | WebSocket |
|-|-----|-----------|
| Direction | Server → Client | Bidirectional |
| Protocol overhead | HTTP/1.1 | Upgraded TCP |
| Firewall/proxy | Rarely blocked | Sometimes blocked |
| POST body support | Via `fetch()` trick | Native |
| Per-request lifecycle | Natural (one response per request) | Manual session management |

For token streaming, SSE is the right choice. The channel is inherently unidirectional — the server sends tokens, the client never needs to send data mid-stream.

### 9.2 Why `fetch()` instead of `EventSource`

The browser's native `EventSource` API only supports GET requests. The chat endpoint requires POST to send `question` and `thread_id` in the body. The Angular service uses `fetch()` with `response.body.getReader()` and manually parses the `event: \ndata: \n\n` SSE format.

### 9.3 The event protocol

```
event: node_start   data: {"node": "language_detect"}
event: node_start   data: {"node": "rewrite_and_route"}
event: node_start   data: {"node": "retrieval"}
event: node_start   data: {"node": "answer"}
event: token        data: The survey
event: token        data:  designer allows...
...
event: done         data: {"thread_id": "abc", "images": ["/images/step1.png"]}
```

The client progressively renders each token and only knows the `thread_id` (for multi-turn) and `images` when the stream ends. The `node_start` events drive a progress indicator showing which pipeline stage is running.

---

## 10. Thread Memory — Two-Tier Strategy

```python
if REDIS_URL:
    # Redis list rag:thread:<id>:turns — shared across workers, survives restarts
    redis_client.lpush(key, json.dumps(turn))
    redis_client.ltrim(key, 0, MEMORY_MAX_TURNS - 1)
else:
    # JSON file memory/<id>.json — zero infrastructure for local dev
    Path(f"memory/threads/{thread_id}.json").write_text(json.dumps(turns))
```

`MEMORY_MAX_TURNS=10` caps conversation history to bound prompt size. Oldest turns are dropped first — recency is more useful than history for support conversations.

The conversation history is loaded **before** the graph is invoked and injected into state. Nodes read it but never mutate it — history is immutable within a single request.

---

## 11. Governance and Output Safety

### 11.1 Pre-LLM governance (agent/governance.py)

Applied before the graph runs, so blocked requests never consume LLM tokens:

1. **Length check** — `GOVERNANCE_MAX_QUESTION_CHARS=12000` → 422
2. **Injection heuristics** — regex patterns for `ignore previous instructions`, `system:`, `<|im_start|>` → 403
3. **Blocklist** — optional newline-separated file, loaded once at startup → 403
4. **Rate limit** — per-IP counter with Redis (multi-worker) or in-memory dict (single process) → 429

The `sanitized_question` (control chars stripped, whitespace normalised) is what gets passed to the LLM — never the raw input.

### 11.2 Post-LLM output safety (agent/output_safety.py)

Four rule-based checks run after every `answer_node` call:

| Check | What it catches |
|-------|----------------|
| **Hallucination self-disclosure** | LLM phrases like `"i made up"`, `"i cannot verify"` |
| **Cross-system leakage** | LLM gives authoritative instructions for the wrong sub-product |
| **Citation overflow** | `[n]` where `n > len(retrieved_chunks)` — LLM invented a citation index |
| **Empty answer** | Blank LLM response |

Default mode: monitoring-only (warnings logged). `SAFETY_STRICT=true`: fallback message substituted on any flag. This gives operators a tunable lever without code changes.

---

## 12. Observability Stack

### 12.1 Correlation IDs

Every HTTP request gets an `X-Request-ID` (UUID4, or passthrough from upstream). It is:
- Set as a `contextvars.ContextVar` before the request handler runs
- Automatically injected into every `logging.LogRecord` via a custom `Filter`
- Returned in response headers so clients can correlate UI errors with backend logs

`thread_id` is similarly bound via `thread_scope(tid)` context manager, so all log lines within a conversation are filterable in one query.

### 12.2 Query JSONL

Every completed question appends one JSON line to `logs/queries_YYYY-MM-DD.jsonl`:

```json
{
  "system": "designer",
  "relevance": "relevant",
  "retrieval_best_distance": 0.34,
  "chunks_count": 7,
  "coverage_gap": false,
  "fallback": false,
  "answer_preview": "To add skip logic…"
}
```

`coverage_gap=true` is auto-detected when the answer contains phrases like `"I don't have enough information"`. Mining these entries identifies documentation topics that need to be written.

### 12.3 OpenTelemetry (optional)

When `OTLP_ENDPOINT` is set, `otel_init()` creates an OTLP gRPC exporter. Each pipeline node gets a span:

```
rag.graph.invoke
  ├── rag.node.language_detect   (microseconds)
  ├── rag.node.rewrite_and_route (LLM call — ~800ms)
  └── rag.node.retrieval         (Chroma + BM25 — ~50ms)
```

`start_span()` falls back to a `nullcontext` no-op when OTel is not installed — callers never need to guard. Zero performance cost when disabled.

---

## 13. What Makes This System Stand Out (Summary)

| Decision | Standard RAG | This System |
|----------|-------------|-------------|
| Corpus architecture | Single merged collection | Per-system isolated Chroma stores |
| Retrieval | Dense only | Dense + BM25 + RRF + optional CrossEncoder |
| Arabic handling | Raw embedding | Gulf bridge + CAMeL lemmatisation + stopword removal + script-ratio language detection |
| Relevance gate | LLM grader (extra call) | Chunk count + L2 distance heuristics |
| LLM calls per request | 4–5 | **2** (rewrite+route, answer) |
| Chunk metadata | file path only | section breadcrumb + language + system + image refs |
| Follow-up questions | Broken (raw input embedded) | Resolved via 9 rewrite cases before embedding |
| Image attachment | None | Semantic cosine similarity against flow embeddings |
| Safety | Input only | Input governance + post-LLM output check |
| Memory | None / always full history | Capped at 10 turns; Redis or JSON with zero config switch |
| Observability | `print()` | Correlation IDs, query JSONL, Prometheus, OTel spans |
| Streaming | Full response wait | Token-level SSE with node_start lifecycle events |

---

## 14. Numbers That Matter

| Metric | Value |
|--------|-------|
| LLM calls per question | 2 (was 5) |
| Embedding model size | 560M params, local, no API cost |
| Dense pool size | 32 → BM25 reranks → top 8 |
| Chunk size | 1,100 chars with 150 char overlap |
| Gulf vocabulary bridge entries | 35 lexeme mappings |
| Arabic stopwords removed | 30 high-frequency function words |
| Conversation memory cap | 10 turns |
| Golden evaluation cases | 52 (routing, dialect, fallback, cross-system) |
| Output safety checks | 4 post-LLM rule-based checks |
| OTel span cost when disabled | 0 (nullcontext) |

---

*The system is a production RAG, not a tutorial. Every layer has a specific failure mode it was built to prevent.*
