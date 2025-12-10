# AI Fraud Intelligence System

This project implements an end-to-end **Fraud Intelligence & Analytics System** combining:

- Retrieval-Augmented Generation (RAG) over regulatory PDFs (Bhatla, EBA/ECB 2024)
- SQL-driven fraud analytics over transaction data
- Multilingual support (English/Indonesian)
- Enterprise-grade guardrails, safety, rewriting, and scoring
- Streamlit UI components for chat, trace viewer, and charts
- Clean, modular backend orchestrator

The system is designed for **accuracy**, **coverage**, **readability**, **robust exception handling**, and **performance.**

---

# 1. Architecture Overview

```
User → Orchestrator → (Intent Classifier)
          ├── Analytics Pipeline → SQL → Supabase → Summary + Chart
          └── RAG Pipeline → Retriever → Reranker → LLM → Scoring → Insight
```

### Main Components

| Module                        | Responsibility                                                                                                           |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `src/orchestrator.py`       | Central router: sanitizes input, detects language, rewrites questions, routes analytics vs RAG, scoring, fallback logic. |
| `src/rag/`                  | RAG pipeline (retrieval, reranking, prompt building, insight layer).                                                     |
| `src/analytics/`            | Natural-language-to-SQL analytics engine with summaries + charts.                                                        |
| `src/llm/`                  | Unified LLM wrapper (cost guardrail, retry, schema validation).                                                          |
| `src/db/supabase_client.py` | Supabase client + PostgreSQL direct connection.                                                                          |
| `src/embeddings/`           | OpenAI embedding model wrapper.                                                                                          |
| `ui/components/`            | Streamlit chat window, charts, citation viewer.                                                                          |
| `ingest_docs.py`            | PDF ingestion to Supabase (page-based embedding).                                                                        |
| `ingest_tabular.py`         | Fraud CSV ingestion to PostgreSQL.                                                                                       |

---

# 2. RAG Pipeline

## 2.1 Retrieval

`retriever_direct.py`

- Uses Supabase RPC `match_documents` for vector similarity.
- Optionally filters by document source.
- Returns top-K contextual rows (page, content, similarity).

## 2.2 Ranking

`reranker_hybrid.py` implements:

- Deduplication
- Embedding similarity
- BM25 keyword match
- Coherence scoring
- Document-aware boosts
- Optional LLM cross-encoder reranking
- Hybrid score fusion

## 2.3 Prompting

`rag_chain.py` assembles:

- Context (max 12k chars)
- RAG system prompt
- Multilingual answer instruction
- Fallback answer

Special case: **Merchant Fraud Inference Mode** loads all merchant-related pages for high-accuracy reasoning.

## 2.4 Insight Layer

`insight_layer.py`

- Generates 3–5 sentence grounded insights (EN/ID)
- No hallucinations; based only on answer + context

## 2.5 Scoring

`scoring.py`

- Lightweight answer grade (length-based + keyword overlap)
- Used for low-confidence fallback

---

# 3. Analytics Pipeline

## 3.1 Intent Detection

`classify_analytics_intent()` distinguishes:

- `timeseries`
- `merchant_rank`
- `category_rank`
- `generic`

## 3.2 NL → SQL

`nl_to_sql()`

- Hard-coded templates for merchant/category ranking
- Strict daily/monthly time-series templates
- Controlled LLM SQL generation for generic questions
- Safety guardrails: SELECT-only, no DML

## 3.3 SQL Execution

Via `DB.sql()` using PostgreSQL (Supabase Pooler).

## 3.4 Summary Builders

### Timeseries

- Trend detection
- Min/max
- Volatility
- Confidence score

### Rankings

- Top entity
- Contribution
- Fraud rate

## 3.5 Optional LLM Refinement

Shortens and polishes summaries without altering numbers.

## 3.6 UI Output

Analytics returns:

- Summary text
- Data points
- Chart data
- Confidence

---

# 4. Orchestrator

`orchestrator.py` is the heart of the system.

### Responsibilities

1. Sanitize query input
2. Detect user language
3. Detect intent (analytics vs RAG)
4. Safe rewrite + domain checks
5. Route to analytics or RAG
6. Score the answer
7. Fallback if low confidence
8. Generate insights
9. Translate output if Indonesian

### Fallback behavior

If `final_score < 0.12`, reply with a strict domain fallback message.

---

# 5. LLM Client

`llm_client.py` wraps OpenAI with:

- Retry logic
- Cost guardrails per session
- Automatic downgrade to cheaper model
- Local model fallback (optional)
- JSON-block extraction
- Pydantic schema validation support

---

# 6. Ingestion Pipelines

## 6.1 PDF Ingestion

`ingest_docs.py`

- Loads each PDF
- Extracts **pages**, not chunks
- Computes embeddings in batches
- Inserts into Supabase `documents` + `document_embeddings`

Benefit: page-level context is ideal for regulatory/RAG datasets.

## 6.2 Fraud CSV Ingestion

`ingest_tabular.py`

- Reads large CSV in 50k-row batches
- Normalizes timestamps, boolean fields
- Ensures table structure exists
- Inserts via PostgreSQL connection

---

# 7. UI Components (Streamlit)

### Chat Window

`chat_window.py`

- Styled bubbles for user + agent
- Safe HTML escaping

### Charts

`charts.py`

- Auto-detects date labels → line chart
- Else → bar chart (ranking)

### Trace Viewer

`trace_viewer.py`

- Shows page-level citations with expandable blocks

---

# 8. Safety & Guardrails

`safety/guardrails.py`

- Query length restrictions
- Prompt-injection detection
- Domain verification (fraud only)
- Language detection (EN/ID)
- Sanitization & noise filtering

If invalid → returns structured error messages.

---

# 9. Testing

`tests/test_runner.py`

- Runs 6 core queries that validate:
  - Analytics routing
  - RAG routing
  - Structural correctness
  - Absence of errors

---

# 10. Environment Variables

Create `.env` with:

```
OPENAI_API_KEY=...
DEFAULT_MODEL=gpt-4o-mini
SUPABASE_URL=...
SUPABASE_ANON_KEY=...
SUPABASE_SERVICE_ROLE_KEY=...
SUPABASE_DB_URL=...
FRAUD_CSV_PATH=data/raw/fraudTrain.csv
ANALYTICS_USE_LLM_SQL=1
ANALYTICS_USE_LLM_SUMMARY=0
```

---

# 11. Running the System

### Start Streamlit UI

```
streamlit run app.py
```

### Run Ingestion

```
python ingest_docs.py
python ingest_tabular.py
```

### Run Tests

```
pytest -q
```

or

```
python tests/test_queries.py
```

---

# 12. Key Design Strengths

### ✔ Accuracy

- Hybrid reranking
- Merchant inference mode
- Strict grounding (context-only)
- Scoring + fallback

### ✔ Coverage

- Dual pipeline: RAG + SQL analytics
- Multilingual EN/ID
- Handles conceptual, regulatory, numerical, and pattern-based questions

### ✔ Readability

- Clean modular structure
- Documented modules
- Consistent naming

### ✔ Exception Handling

- Try/except with detailed logs
- Safe fallbacks for SQL, RAG, ingestion, LLM
- Input guardrails

### ✔ Performance

- Page-level embeddings (52 total)
- Hybrid scoring avoids expensive cross-encoders unless needed
- Batched ingestion
- Cost-aware LLM usage

---

# 13. Conclusion

This codebase is built for **production clarity**, **enterprise-grade reliability**, and **extensibility**. With clear routing, strict constraints, and modular components.
