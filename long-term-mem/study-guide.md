# AllInLog Study Guide — Deep Architecture & Concepts Reference

**Purpose**: Complete reference for understanding the AllInLog project offline.
Covers every module, data flow, design decision, and research concept.

**Created**: 2026-02-13 | **Author**: Project working notes

---

## Table of Contents

1. [Project Overview & Research Framing](#1-project-overview--research-framing)
2. [System Architecture (Big Picture)](#2-system-architecture-big-picture)
3. [Data Pipeline: From Raw Logs to Sessions](#3-data-pipeline-from-raw-logs-to-sessions)
4. [The Screener: Linformer-Based Anomaly Detection](#4-the-screener-linformer-based-anomaly-detection)
5. [Evidence Store & Log Normalization](#5-evidence-store--log-normalization)
6. [Retrieval: BM25 and Mixed Retrieval](#6-retrieval-bm25-and-mixed-retrieval)
7. [Prompt Building & Trace Schema](#7-prompt-building--trace-schema)
8. [LLM Client: Ollama / OpenAI Integration](#8-llm-client-ollama--openai-integration)
9. [Signature Normalization: The Consolidation Layer](#9-signature-normalization-the-consolidation-layer)
10. [Verification: 8-Check Faithfulness System](#10-verification-8-check-faithfulness-system)
11. [Signature Discovery & Pattern Cards](#11-signature-discovery--pattern-cards)
12. [Complete End-to-End Pipeline Flow](#12-complete-end-to-end-pipeline-flow)
13. [Dataset-Specific Differences: BGL vs HDFS](#13-dataset-specific-differences-bgl-vs-hdfs)
14. [Design Decisions & Lessons Learned](#14-design-decisions--lessons-learned)
15. [Key Files & Module Map](#15-key-files--module-map)
16. [Glossary](#16-glossary)

---

## 1. Project Overview & Research Framing

### What This Project Does

AllInLog is a **Screener–Reasoner** framework for **explainable log-based anomaly detection**. It combines:

- A **lightweight deep learning screener** (Linformer) that detects anomalies at near-perfect accuracy (F1 ≈ 0.999 BGL, ≈ 0.997 HDFS)
- An **LLM-based reasoner** (Llama 3.1:8b) that generates structured, evidence-grounded explanations for *why* each detected anomaly is anomalous

The key innovation is **NOT** the detection — that's already solved. The research focus is:

> Can we produce **traceable, verifiable, structured explanations** for log anomalies using RAG + LLM, without compromising detection performance?

### Research Questions

1. **RQ1**: Can we produce traceable, verifiable explanations while maintaining detection performance?
2. **RQ2**: Does RAG + structured output improve faithfulness and reduce hallucinations?
3. **RQ3**: How do different gating strategies affect quality vs cost under fixed LLM budgets?

### Why "Screener–Reasoner"?

Traditional anomaly detection says "this is anomalous" but never explains why. The screener handles the high-volume filtering (71K+ sessions in seconds), and the reasoner (expensive LLM) only processes the detected anomalies (~3-9% of sessions). This is a **cost-aware** architecture:

```
71,221 test sessions → Screener → 6,295 anomalies (8.8%) → LLM Reasoner → Explanations
```

The LLM never does detection. It only explains. It gets evidence from RAG (training data corpus), not from its parametric knowledge.

### Datasets

| Property | BGL | HDFS |
|----------|-----|------|
| Source | Blue Gene/L supercomputer | Hadoop cluster |
| Total log lines | 4,747,963 | 11,175,629 |
| Session definition | Sliding window (10 lines, step 10) | Group by block_id |
| Total sessions | 474,796 | 575,061 |
| Test sessions | 71,221 | 86,260 |
| Test anomaly rate | 8.22% | 2.93% |
| Anomaly type | **Lexical** (error keywords) | **Structural** (incomplete operations) |

---

## 2. System Architecture (Big Picture)

### The Pipeline Diagram

```
┌──────────────────────────────────────────────────────────┐
│                    TRAINING PHASE                         │
│                                                          │
│  Raw Logs ──→ DataLoader ──→ Train Sessions              │
│                                    │                     │
│                    ┌───────────────┼───────────────┐     │
│                    ▼               ▼               ▼     │
│              EvidenceStore   SignatureGenerator  Screener │
│              (normalized     (pattern cards)    (trained │
│               text corpus)                      model)   │
│                    │               │                     │
│                    ▼               ▼                     │
│              BM25 Index    patterns/*.json               │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│                   INFERENCE PHASE                         │
│                                                          │
│  Test Sessions ──→ Screener ──→ Filter Anomalies         │
│                                      │                   │
│                    For each anomaly:  │                   │
│                    ┌─────────────────┘                    │
│                    ▼                                     │
│              Retriever (BM25)                            │
│              ├── 4 anomaly exemplars                     │
│              └── 1 normal contrast                       │
│                    │                                     │
│                    ▼                                     │
│              PromptBuilder                               │
│              ├── System prompt (rules, schema)           │
│              └── User prompt (session + evidence)        │
│                    │                                     │
│                    ▼                                     │
│              LLM Client (Ollama → Llama 3.1:8b)         │
│              └── JSON structured output                  │
│                    │                                     │
│                    ▼                                     │
│              Normalizer                                  │
│              └── COMPONENT__ERROR_TYPE canonical form    │
│                    │                                     │
│                    ▼                                     │
│              Verifier (8 checks)                         │
│              └── PASS / FAIL                             │
│                    │                                     │
│                    ▼                                     │
│              Results (JSONL + metrics)                   │
└──────────────────────────────────────────────────────────┘
```

### Key Principle: No Data Leakage

The evidence store contains **only training data**. Test sessions are never put into the evidence corpus. The LLM explains by comparing the test session against similar/contrasting training sessions — it never sees test labels, and the evidence it references is always from a separate data partition.

---

## 3. Data Pipeline: From Raw Logs to Sessions

### File: `src/data_loader.py` (~406 lines)

### The `Session` Object

Every module operates on `Session` objects — the universal data unit:

```python
@dataclass
class Session:
    session_id: str           # "BGL_00042150" or "HDFS_blk_-123456789"
    split: str                # "train", "val", or "test"
    label: int                # 0=normal, 1=anomaly
    lines: List[str]          # Original log lines (raw text)
    metadata: Dict            # Dataset-specific info
```

**Critical**: The `label` field is used for evaluation and evidence store partitioning, but is **never** given to the LLM. The LLM never knows the ground truth.

### BGL Session Construction

BGL creates sessions via **sliding windows**:

```
Line 1:  - 1117838570 R02-M1-N0-C:J12-U11 ... (normal)
Line 2:  - 1117838573 R02-M1-N0-C:J12-U11 ... (normal)
...
Line 10: 1117838612 R02-M1-N0-C:J12-U11 ... (anomaly — no leading "-")

Window → Session with 10 lines
Label = 1 if ANY line lacks the "-" prefix
```

- `window_size = 10`, `step_size = 10` (non-overlapping)
- A session is anomalous if **any** line in the window is anomalous
- The label indicator character (`-` for normal) is stripped from the content after labeling
- This produces ~474K sessions from 4.7M lines

### HDFS Session Construction

HDFS creates sessions by **block ID grouping**:

```
081109 203615 148 INFO dfs.DataNode$PacketResponder: blk_-1608999687 ...
081109 203615 148 INFO dfs.DataNode$PacketResponder: blk_-1608999687 ...
→ All lines with "blk_-1608999687" become one session
```

- Block IDs extracted via regex: `blk_-?\d+`
- Labels come from external CSV file (`anomaly_label_HDFS.csv`)
- String labels like `"Anomaly"` converted to int `1`
- This produces ~575K sessions from 11.1M lines
- Average 19.4 lines/session (longer than BGL's 10)

### Train/Val/Test Split

Both datasets use the same stratified split:
1. Separate sessions by label (normal vs anomaly)
2. Shuffle each group independently (with seed=42)
3. Split each group by ratio: 70% train, 15% val, 15% test
4. Recombine and shuffle the final splits

This ensures anomaly ratio is preserved across splits.

---

## 4. The Screener: Linformer-Based Anomaly Detection

### File: `src/screener.py` (~610 lines)

### What is the Screener?

The screener is a binary classifier that decides: **is this session anomalous?**

It's based on Linformer — a variant of Transformer with **O(n)** attention complexity instead of O(n²). This allows processing long sequences (BGL: up to 2,549 tokens, HDFS: up to 15,166 tokens) without the quadratic memory cost of standard attention.

### Model Architecture (`AllLinLog`)

```
Input tokens → EmbeddingLayer → LinformerTransformerEncoder → Mean Pool → Linear(128,2) → Softmax
```

#### EmbeddingLayer
- **Token embedding**: Maps each BPE token to a 128-dim vector (vocab_size = 100,264 — GPT-4 cl100k_base)
- **Segment embedding**: Maps each line index to a 128-dim vector. If a session has 10 lines, tokens from line 0 get segment_id=0, line 1 gets segment_id=1, etc. This encodes positional structure **across lines**
- **Position embedding**: Standard positional encoding within the full sequence
- All three are **summed** (not concatenated):
  ```
  embedding = token_emb + segment_emb + position_emb
  ```

#### LinformerEncoderLayer (×1)
- **Linformer attention**: Projects K and V matrices from n×d to k×d before computing attention, where k=32 is constant regardless of sequence length n. This is why it's O(n) instead of O(n²).
  - `one_kv_head=True`: Single K/V head shared across attention heads
  - `share_kv=True`: K and V share the same projection matrix
  - 4 attention heads
- **FFN**: Linear(128→128) with GELU activation
- **LayerNorm** after both attention and FFN (Post-LN)
- **Dropout**: 0.5 (during training)

#### Classifier
- **Mean pooling** over the entire sequence (not CLS token) → single 128-dim vector
- **Linear(128, 2)** → logits for [normal, anomaly]

**Key design decision**: When batching, sequences are padded to the **max length within the batch**, not to `max_seq_len`. This prevents mean pooling from being diluted by pad tokens.

#### Total Parameters
- BGL: ~13.4M parameters
- HDFS: ~15.5M parameters (larger due to more segment IDs)

### Tokenization

Uses **tiktoken** with GPT-4's `cl100k_base` encoding (100,264 tokens). Each session is tokenized as:

```
<BOS> line_0_tokens <EOS> <BOS> line_1_tokens <EOS> ... <BOS> line_n_tokens <EOS>
```

Where `<BOS>` = `<|startoftext|>` and `<EOS>` = `<|endoftext|>`. This preserves line boundaries.

### ScreenerOutput

```python
@dataclass
class ScreenerOutput:
    session_id: str
    pred: int           # 0=normal, 1=anomaly
    logits: List[float] # [logit_normal, logit_anomaly]
    prob: List[float]   # [p_normal, p_anomaly] (softmax)
    margin: float       # abs(p_anomaly - p_normal)
```

- **`is_anomaly`**: `pred == 1`
- **`anomaly_prob`**: `prob[1]` — probability of being anomalous
- **`confidence`**: `prob[pred]` — probability of the predicted class
- **`margin`**: Lower margin = less confident prediction (useful for budgeted gating — explain only the most uncertain anomalies)

---

## 5. Evidence Store & Log Normalization

### Files: `src/evidence_store.py` (~274 lines), `src/normalizer.py` (~623 lines)

### Normalization: Why and How

Raw log lines contain variable-length numbers, IPs, paths, and timestamps that dilute BM25 matching. Normalization replaces these with placeholders:

```
Before: 1117838573 2005.06.03 R02-M1-N0-C:J12-U11 RAS APP FATAL ciod: LOGIN chdir(/p/gb1/dave): No such file
After:  <NUM> <TIMESTAMP> <NODE> RAS APP FATAL ciod: LOGIN chdir(<PATH>): No such file
```

#### LogNormalizer (Base — 20 regex patterns)

Applied in priority order (first match wins):

| Priority | Pattern | Placeholder | Example |
|----------|---------|-------------|---------|
| 1 | IPv4 addresses | `<IPV4>` | `192.168.1.1` → `<IPV4>` |
| 2 | IPv6 addresses | `<IPV6>` | |
| 3 | MAC addresses | `<MAC>` | |
| 4 | UUIDs | `<UUID>` | |
| 5 | Hex values (0x... or 8+ hex) | `<HEX>` | `0xdeadbeef` → `<HEX>` |
| 6 | HDFS block IDs | `<BLOCK_ID>` | |
| 7 | Unix paths | `<PATH>` | `/home/user/file.txt` → `<PATH>` |
| 8 | URLs | `<URL>` | |
| 9 | Email | `<EMAIL>` | |
| 10-12 | Timestamps/dates | `<TIMESTAMP>`, `<TIME>`, `<DATE>` | |
| 13 | Memory addresses | `<MEMADDR>` | |
| 14 | Port numbers | `<PORT>` | |
| 15-16 | PID/TID | `<PID>`, `<TID>` | |
| 17-18 | Numbers (6+ digits, then 2-5) | `<NUM>` | |

#### BGLNormalizer (adds 5 patterns)

| Pattern | Placeholder | Example |
|---------|-------------|---------|
| Node IDs | `<NODE>` | `R02-M1-N0-C:J12-U11` → `<NODE>` |
| Core IDs | `core.<CORE>` | `core.14` → `core.<CORE>` |
| 8-char hex | `<HEX8>` | `0000abcd` → `<HEX8>` |
| DDR memory | `DDR(<MEMLOC>)` | |
| Torus coordinates | `(<COORD>)` | `(0,1,0,1,0)` → `(<COORD>)` |

#### HDFSNormalizer (adds 5 patterns + structural summary)

| Pattern | Placeholder |
|---------|-------------|
| Block IDs | `<BLOCK>` |
| DataNode addresses | `<DATANODE>` |
| Hadoop paths | `<HDFS_PATH>`, `<TMP_PATH>` |
| Replication/size info | `replicas=<NUM>`, `size=<SIZE>` |

**Structural Summary** (HDFS only): Counts operations via regex and generates discriminative tags:

```python
def structural_summary(self, session) -> str:
    # Counts: receiving_block, received_block, allocateblock,
    #         addstoredblock, packetresponder, exception/error/failed,
    #         writeblock, delete/invalidate
    
    # Then generates tags:
    # INCOMPLETE_PIPELINE: receives > received
    # EXCESS_REPLICATION: addstoredblock > 3
    # HAS_EXCEPTION: any exception/error
    # WRITE_FAILURE: writes with no success
    # BLOCK_DELETION: block invalidated
    # MISSING_ACKNOWLEDGMENT: no packetresponder
    # ORPHAN_BLOCK: allocate but no receive
```

This structural summary is appended to the normalized text. This is critical for HDFS because HDFS anomalies are **structural**, not lexical (see Section 13).

### Evidence Store

The evidence store converts training sessions into a searchable corpus for BM25 retrieval:

```python
class EvidenceDoc:
    evidence_id: str        # "E_BGL_00001234"
    session_id: str         # Source session ID
    text: str               # Normalized text (via LogNormalizer)
    evidence_type: str      # "session", "signature", or "profile"
    metadata: Dict          # label, dataset, param_stats, etc.
```

**Build process** (`build_from_sessions`):
1. Take all **train** sessions (no test/val — data leakage prevention)
2. Normalize each session's lines via the dataset-specific normalizer
3. If dataset provides a structural summary, append it
4. Create one `EvidenceDoc` per session
5. Store as JSON for reuse

**Typical sizes**:
- BGL: 332,356 documents (305K normal + 27K anomaly)
- HDFS: 402,542 documents (390K normal + 11.7K anomaly)

**Memory optimization**: Since BGL has 305K normal docs but only 27K anomaly, the pipeline **samples** down to 20K normal docs at runtime (with `MAX_NORMAL_EVIDENCE = 20,000`) to avoid BM25 index OOM:

```
BGL:  332,356 → 47,315 (27,315 anomaly + 20,000 normal)
HDFS: 402,542 → 31,786 (11,786 anomaly + 20,000 normal)
```

---

## 6. Retrieval: BM25 and Mixed Retrieval

### File: `src/retriever.py` (~520 lines)

### BM25 Retrieval

BM25 (Best Match 25) is a term-frequency-based retrieval algorithm. It ranks documents by how well their terms match the query, with:
- **TF saturation**: Repeated terms give diminishing returns (controlled by `k1=1.5`)
- **Document length normalization**: Short documents aren't penalized (controlled by `b=0.75`)

```
score(q, d) = Σ IDF(qi) × [tf(qi,d) × (k1+1)] / [tf(qi,d) + k1 × (1-b+b×|d|/avgdl)]
```

The index operates on **normalized text** — the query session is also normalized before retrieval. This ensures BGL node IDs, HDFS block IDs, etc. don't cause spurious matches.

Uses the `rank_bm25.BM25Okapi` library with whitespace tokenization.

### Why Mixed Retrieval?

Standard top-k retrieval tends to return only anomaly exemplars (because anomalies have distinctive terms). But explanations also need **contrast evidence** — examples of what *normal* looks like — to support "contrast" claims.

**Mixed retrieval** splits the results:

```python
def retrieve_for_session_mixed(session, top_k_anomaly=4, top_k_normal=1):
    # 1. Run BM25 on full corpus
    # 2. Separate results by metadata["label"]
    # 3. Take top 4 anomaly docs + top 1 normal doc
    # 4. Return combined (5 evidence docs total)
```

This enables the LLM to produce three types of claims:
- **observation**: "E0 (the query) shows error X at lines L3-L5"
- **pattern_match**: "E1 (anomaly exemplar) shows the same error pattern"
- **contrast**: "E4 (normal session) lacks this error, confirming it's anomaly-specific"

### Evidence ID Mapping

Evidence is given simple IDs for prompt brevity:

```
E0 = query session (the anomaly being explained)
E1 = first retrieved evidence (anomaly exemplar, BM25 rank 1)
E2 = second retrieved evidence (anomaly exemplar, BM25 rank 2)
E3 = third retrieved evidence (anomaly exemplar, BM25 rank 3)
E4 = fourth retrieved evidence (often normal contrast)
```

The mapping `{E0: session_id, E1: evidence_id_1, ...}` is stored for traceability.

---

## 7. Prompt Building & Trace Schema

### File: `src/prompt_builder.py` (~585 lines)

### System Prompt Structure

The system prompt tells the LLM who it is and what rules to follow. It's ~500 tokens and includes:

1. **Role**: "You are an expert log analyst producing forensic, evidence-grounded explanations"
2. **Task**: Explain WHY a log session is anomalous based on evidence
3. **References**: Use line-level references (E0-L1, E1-L3) — never cite evidence without specifying which line
4. **Three claim types**:
   - `observation`: What you see in E0 (the query session)
   - `pattern_match`: How E0 matches anomaly exemplars (E1-E3)
   - `contrast`: How E0 differs from normal sessions (E4)
5. **Signature naming**: `COMPONENT__ERROR_TYPE` format, no severity (FATAL/WARN/etc.), exactly one `__` separator
6. **Dataset-specific examples**:
   - BGL: `KERNEL__DATA_TLB_ERROR`, `APP__CIOD_STREAM_ERROR`, components: `KERNEL, APP, MMCS, LINKCARD`
   - HDFS: `DATANODE__BLOCK_VERIFICATION_FAILED`, `NAMENODE__REPLICATION_INCOMPLETE`, components: `DATANODE, NAMENODE, FSDATASET, BLOCKSCANNER`
7. **Seven critical rules** (quantify errors, never exceed line ranges, correct span format, etc.)

### User Prompt Structure

Each user prompt has a fixed template:

```
=== ANOMALOUS SESSION {session_id} ===
Anomaly probability: {anomaly_prob:.4f}
Decision margin: {margin:.4f}

--- Query Session (E0) ---
E0-L1: <log line 1>
E0-L2: <log line 2>
...

=== EVIDENCE ===
--- E1 [session | anomaly | score=3.45] ---
E1-L1: <evidence line 1>
E1-L2: <evidence line 2>
...

--- E4 [session | normal | score=1.23] ---
E4-L1: <evidence line 1>
...

=== INSTRUCTIONS ===
Produce a JSON explanation following the schema...
```

Key formatting details:
- Every line gets a line number (E0-L1, E1-L1, etc.)
- Evidence is capped at `max_evidence_items=5` pieces
- Each evidence text is truncated at `max_chars_per_evidence=500` characters
- The query session is capped at `max_log_lines=20` lines
- Evidence metadata (type, label, BM25 score) is shown in the header

### Trace Schema (The JSON Output)

The LLM must produce this exact JSON structure:

```json
{
  "prediction": "anomaly",
  "summary": "Brief forensic summary...",
  "signature": {
    "name": "KERNEL__DATA_TLB_ERROR",
    "matched_evidence_ids": ["E1", "E2"]
  },
  "claims": [
    {
      "type": "observation",
      "claim": "E0 shows a data TLB error interrupt at L3, indicating...",
      "evidence_ids": ["E0"],
      "evidence_spans": ["E0-L3"],
      "confidence": "high"
    },
    {
      "type": "pattern_match",
      "claim": "E1 exhibits the same data TLB error pattern at L5-L7...",
      "evidence_ids": ["E0", "E1"],
      "evidence_spans": ["E0-L3", "E1-L5"],
      "confidence": "high"
    },
    {
      "type": "contrast",
      "claim": "E4 (normal) contains no TLB-related errors, confirming...",
      "evidence_ids": ["E0", "E4"],
      "evidence_spans": ["E4-L1"],
      "confidence": "medium"
    }
  ]
}
```

### Data Classes

```python
class Claim:
    type: str                       # "observation", "pattern_match", "contrast"
    claim: str                      # The claim text
    evidence_ids: List[str]         # ["E0", "E1"]
    evidence_spans: List[str]       # ["E0-L3", "E1-L5"] — line-level refs
    confidence: Optional[str]       # "high", "medium", "low"

class Signature:
    name: str                       # "KERNEL__DATA_TLB_ERROR"
    matched_evidence_ids: List[str] # ["E1", "E2"]

class TraceExplanation:
    prediction: str                 # "anomaly" or "normal"
    summary: str                    # Brief forensic summary
    signature: Optional[Signature]  # Anomaly signature card
    claims: List[Claim]             # Evidence-grounded claims
    insufficient_evidence: bool     # True if evidence not sufficient
    raw_response: str               # Original LLM text

class ExplanationResult:
    session_id: str
    session: Session
    screener_output: ScreenerOutput
    evidence_hits: List[RetrievalHit]
    explanation: TraceExplanation
    evidence_id_mapping: Dict[str, str]  # E0 → session_id, E1 → evidence_id
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    created_at: str
```

---

## 8. LLM Client: Ollama / OpenAI Integration

### File: `src/llm_client.py` (~351 lines)

### Architecture

Uses OpenAI-compatible `/v1/chat/completions` API, which Ollama also supports. This means the same client works for both local (Ollama) and cloud (OpenAI) LLMs.

```python
client = LLMClient(
    provider="ollama",           # or "openai"
    model="llama3.1:8b",
    temperature=0.1,             # Low temperature for consistent outputs
    max_tokens=1024,             # Completion budget
    timeout=120                  # Seconds per call
)
```

### How `generate_json` Works

```python
parsed_dict, llm_response = client.generate_json(prompt, system_prompt)
```

1. Sends `system_prompt` + `prompt` to `/v1/chat/completions` with `json_mode=True`
2. Receives raw text response
3. Strips markdown code blocks (```json ... ```) if present
4. Finds JSON boundaries `{...}` in the response
5. Parses with `json.loads`
6. Returns both the parsed dict and the raw `LLMResponse` metadata (tokens, latency, etc.)

### Current Production Config

- **Provider**: Ollama (local, free)
- **Model**: `llama3.1:8b` (Meta's Llama 3.1, 8 billion parameters)
- **Temperature**: 0.1 (near-deterministic — want consistent signatures)
- **Max tokens**: 1024 (sufficient for structured JSON output)
- **Timeout**: 120s (HDFS sessions with 20+ lines can take >60s)

### Performance Characteristics

| Metric | BGL | HDFS |
|--------|-----|------|
| Avg tokens/session | ~3,479 | ~3,330 |
| Avg latency | ~9.8s | ~10.2s |
| P95 latency | ~11.2s | ~12.3s |
| Parse success rate | 100% | 100% |

---

## 9. Signature Normalization: The Consolidation Layer

### File: `src/normalizer.py` (signature parts)

### The Problem

An 8B parameter LLM generates signature names with high variance:
- `KERNEL__FATAL__DATA_TLB_ERROR` (severity leaked into name)
- `KERNEL__FATAL_data_TLB_ERROR` (mixed case)
- `KERNEL__FATAL__data TLB error interrupt` (verbose literal from log text)
- `RAS_APP_FATAL__CIOD_STREAM_ERROR` (RAS_ prefix from log format)
- `DATANODE__INFO__BLOCK_WRITE_FAILURE` (severity as middle segment)

Without normalization, BGL generates **71 unique signatures** from 500 sessions. With normalization: **27 unique signatures**. That's a 62% reduction.

### The Solution: Post-Hoc Normalization

A deterministic, rule-based normalizer runs **after** the LLM, transforming any variant into a canonical form:

```
KERNEL__FATAL__DATA_TLB_ERROR  →  KERNEL__DATA_TLB_ERROR
RAS_APP_FATAL__CIOD_STREAM_ERROR  →  APP__CIOD_STREAM_ERROR
APP__CREATING_NODE_MAP  →  APP__NODE_MAP_ERROR
```

### BGL Normalizer: 7-Step Pipeline

```
Input: "RAS_KERNEL_FATAL__data TLB error interrupt"

Step 1: Split on "__", uppercase, strip pure severity segments
        → prefix="RAS_KERNEL_FATAL", segments=["RAS_KERNEL_FATAL", "DATA_TLB_ERROR_INTERRUPT"]

Step 2: Clean component prefix
        → Strip "RAS_" → "KERNEL_FATAL"
        → Strip severity suffix → "KERNEL"

Step 3: Rejoin error type segments
        → error_type = "DATA_TLB_ERROR_INTERRUPT"

Step 4: Normalize whitespace and special characters
        → error_type = "DATA_TLB_ERROR_INTERRUPT"

Step 5: Verbose pattern collapsing
        → (no match — this step handles "kernel terminated for reason 1001" etc.)

Step 6: Strip severity prefix from error type
        → "FATAL_DATA_TLB_ERROR" → "DATA_TLB_ERROR" (removed "FATAL_")
        → Exception: FATAL_ERROR, FATAL_MESSAGE keep their prefix

Step 7: Canonical error type + component lookup
        → _ERROR_TYPE_CANONICAL: "DATA_TLB_ERROR_INTERRUPT" → "DATA_TLB_ERROR"
        → _COMPONENT_MAP: "DATA_TLB_ERROR" → "KERNEL"

Output: "KERNEL__DATA_TLB_ERROR"
```

### Key Canonical Mappings (BGL)

| Input | Canonical | Rationale |
|-------|-----------|-----------|
| `DATA_TLB_ERROR_INTERRUPT` | `DATA_TLB_ERROR` | Suffix variant |
| `TLB_ERROR` | `DATA_TLB_ERROR` | Too-short variant |
| `CIOD_SOCKET_ERROR` | `CIOD_STREAM_ERROR` | Same root cause |
| `CIOD_UNEXPECTED_EOF` | `CIOD_STREAM_ERROR` | Same root cause |
| `CREATING_NODE_MAP` | `NODE_MAP_ERROR` | Verbose literal from log |
| `LINK_SEVERED` | `LOAD_MESSAGE_ERROR` | Verbose literal from log |
| `TERMINATION` | `KERNEL_TERMINATED` | Too-short variant |

### Key Design Decision

**Why normalizer instead of prompt engineering?**

We tested both approaches (2026-02-13):
- **Prompt hints** (+45 tokens/session): 13 → 14 unique sigs = noise, no benefit
- **Stricter prompt rules** (+189 tokens/session): 13 → 15 unique sigs = made things *worse*
- **Normalizer mappings** (0 tokens): 29 → 27 unique sigs = actual improvement

**Conclusion**: For 8B models, the LLM can correctly *read* log content but cannot reliably *follow* abstract naming conventions. Post-hoc normalization is deterministic, zero-cost at inference, and trivially verifiable.

---

## 10. Verification: 8-Check Faithfulness System

### File: `src/verifier.py` (~684 lines)

### Why Verify?

LLMs can hallucinate — claim things aren't in the evidence, cite wrong line numbers, or produce structurally invalid output. The verifier checks every explanation for faithfulness.

### The 8 Checks

| # | Check | Type | What It Does |
|---|-------|------|-------------|
| 1 | Structure | FAIL | `prediction`, `summary`, `claims` fields must exist |
| 2 | Evidence IDs | FAIL | All referenced IDs (E0, E1...) must exist in the mapping |
| 3 | Evidence Coverage | FAIL | ≥80% of claims must cite at least one evidence ID |
| 4 | Keyword Match | WARN | Words from claims should appear in referenced evidence |
| 5 | Empty Claims | WARN | Claims must be ≥10 characters |
| 6 | Evidence Spans | FAIL | `E0-L3` format valid, line numbers within range |
| 7 | Signature | WARN | Signature exists, not "UNKNOWN", contains "__" |
| 8 | Span Keywords | WARN | Claims reference specific lines that contain relevant keywords |

### Pass/Fail Logic

- An explanation **PASSES** if it has **zero FAIL** issues
- **WARNINGs** are recorded but don't cause failure
- Current pass rate: **100% on both BGL and HDFS** (500-session tests)

### Span Parsing

The verifier robustly handles multiple LLM output formats for evidence spans:

```python
# Standard:          "E0-L12"               → (E0, 12)
# Range:             "E0-L6 to E0-L10"      → (E0, 6)
# Comma:             "E0-L6, E0-L8"         → (E0, 6)
# Slash:             "E5-L3/E5-L4"          → (E5, 3)
# Hyphen:            "E0-L7-E0-L12"         → (E0, 7)
```

---

## 11. Signature Discovery & Pattern Cards

### Files: `src/signature_generator.py` (~381 lines), `patterns/*.json`, `notebooks/05_signature_audit.ipynb`

### How Pattern Cards Are Created

Pattern cards are **mined from training data** (not hand-crafted). The process:

1. Load all training anomaly sessions
2. Normalize and extract keywords
3. Cluster by keyword fingerprint (sorted unique keywords)
4. For each cluster, create a pattern card:

```json
{
  "name": "KERNEL__DATA_TLB_ERROR",
  "description": "Data TLB error interrupt (11052 sessions)",
  "keywords": ["data", "error", "fatal", "interrupt", "tlb"],
  "frequency": 11052,
  "fingerprint": "data+error+fatal+interrupt+tlb",
  "discriminative_score": 0.75,
  "discriminative_level": "high",
  "anomaly_rate": 0.87,
  "normal_rate": 0.12
}
```

### Coverage

- **BGL**: 34 patterns covering 99.8% of 27,315 training anomalies
- **HDFS**: 26 patterns covering 99.8% of 11,786 training anomalies

### How Pattern Cards Enter the Pipeline

Pattern cards don't go directly to the LLM. They're injected into the **evidence store** as documents:

```
EvidenceDoc(
    evidence_id = "E_SIG_pattern_001",
    text = "ERROR SIGNATURE: KERNEL__DATA_TLB_ERROR\nDescription: ...\nKey Indicators: ...",
    evidence_type = "signature",
    metadata = {"label": 1, "signature_name": "KERNEL__DATA_TLB_ERROR", ...}
)
```

When BM25 retrieves evidence for a query session, pattern cards can appear alongside normal session documents — providing the LLM with a pre-mined description of the error pattern.

### Discriminative Scores

Discriminative score = `anomaly_rate - normal_rate` (keyword containment in anomaly vs normal sessions).

**BGL**: All 34 patterns are "high" discriminative (+0.48 to +0.88). Keywords like `tlb`, `ciod`, `parity` appear almost exclusively in anomaly sessions. BGL anomalies are **lexical anomalies**.

**HDFS**: All 26 patterns are "low" discriminative (≈ +0.0001). Keywords like `exception`, `writeblock`, `error` appear in both anomaly AND normal sessions at ~100%. HDFS anomalies are **structural anomalies** — the keywords are the same, but the *sequence* and *absence* of operations is different.

---

## 12. Complete End-to-End Pipeline Flow

### What Happens For One Anomaly Session

Let's trace a single BGL session through the entire pipeline:

#### Step 1: Data Loading
```
Raw log → BGLDataLoader → Session(session_id="BGL_00042150", lines=[...10 lines...], label=1)
```

#### Step 2: Screening
```
Session → Screener.screen_session() → ScreenerOutput(pred=1, anomaly_prob=0.9987, margin=0.9974)
```
The screener tokenizes the session with GPT-4 BPE, passes through Linformer, and outputs a binary prediction.

#### Step 3: Evidence Retrieval
```
Session → Retriever.retrieve_for_session_mixed() → [E1(anomaly), E2(anomaly), E3(anomaly), E4(anomaly), E5(normal)]
```
The session is normalized, then BM25 ranks all documents in the evidence store by relevance. The top 4 anomaly docs and top 1 normal doc are returned.

#### Step 4: Prompt Building
```
Session + ScreenerOutput + Evidence → PromptBuilder.build_prompt() → (system_prompt, user_prompt)
```
The system prompt (~500 tokens) sets rules. The user prompt (~2,900 tokens) contains the session text with line numbers, evidence with line numbers, anomaly probability, and the JSON schema.

#### Step 5: LLM Generation
```
(system_prompt, user_prompt) → LLMClient.generate_json() → (parsed_dict, LLMResponse)
```
Ollama's Llama 3.1:8b generates a JSON response (~500 tokens) in ~10 seconds.

#### Step 6: Parse & Normalize
```
parsed_dict → TraceExplanation.from_dict() → explanation
explanation.signature.name → Normalizer.normalize_signature() → "KERNEL__DATA_TLB_ERROR"
```
The JSON is parsed into structured objects. The signature name is normalized to canonical form.

#### Step 7: Verification
```
explanation → Verifier.verify() → VerificationResult(passed=True, total_checks=8)
```
All 8 checks run. If any FAIL check triggers, `passed=False`.

#### Step 8: Save
```
ExplanationResult → .to_dict() → JSON → append to JSONL file
```
Each result is written to disk immediately (crash-resilient).

### Full Pipeline Runtime

| Dataset | Sessions | Time | Rate |
|---------|----------|------|------|
| BGL 500 | 500 anomalies | 91.5 min | 0.09/s |
| HDFS 500 | 500 anomalies | 100.9 min | 0.08/s |
| BGL full | ~6,295 anomalies | ~19 hours | 0.09/s |
| HDFS full | ~2,527 anomalies | ~8 hours | 0.08/s |

The bottleneck is LLM inference (~10s per session). Everything else (screening, retrieval, verification) takes milliseconds.

---

## 13. Dataset-Specific Differences: BGL vs HDFS

### Anomaly Types

This is the most important conceptual distinction in the project:

#### BGL: Lexical Anomalies

BGL anomalies contain **distinctive error keywords** that don't appear in normal sessions:
- `tlb`, `interrupt`, `ciod`, `parity`, `machine check`
- Discriminative scores +0.48 to +0.88
- The **presence** of these words is the signal
- BM25 retrieval works excellently (keyword match = anomaly match)
- Normalizer handles 25+ canonical mappings because BGL has diverse error vocabulary

#### HDFS: Structural Anomalies

HDFS anomalies contain the **same keywords** as normal sessions:
- `exception`, `writeblock`, `error`, `replicate` appear in ~100% of both anomaly and normal sessions
- Discriminative scores ≈ +0.0001
- The **absence or incompleteness of operations** is the signal
- Example: A normal session has `receiving block → received block → packet response → add stored block`. An anomaly might have `receiving block` but never `received block` (pipeline was interrupted)
- HDFSNormalizer's `structural_summary()` is crucial — it counts operations and flags structural anomalies like `INCOMPLETE_PIPELINE`, `MISSING_ACKNOWLEDGMENT`

### Component & Signature Conventions

| Property | BGL | HDFS |
|----------|-----|------|
| Components | `KERNEL, APP, MMCS, LINKCARD` | `DATANODE, NAMENODE, FSDATASET, BLOCKSCANNER` |
| Default component | `KERNEL` | `DATANODE` |
| Unique sigs (500-session) | 27 | 8 |
| Dominant signature | `KERNEL__DATA_TLB_ERROR` (43%) | `DATANODE__BLOCK_VERIFICATION_FAILED` (64%) |

### Model Configuration

| Property | BGL | HDFS |
|----------|-----|------|
| Max sequence length | 2,549 tokens | 15,166 tokens |
| Segment vocab size | 10 (10 lines/session) | 298 (up to 298 lines/session) |
| Model parameters | 13.4M | 15.5M |
| Test anomalies | 5,854 (8.2%) | 2,526 (2.9%) |

---

## 14. Design Decisions & Lessons Learned

### Decision 1: Post-Hoc Normalization > Prompt Engineering (for 8B models)

**Tested**:
- Signature hints in prompt: +45 tokens, no benefit
- Bad→good naming examples in prompt: +189 tokens, made things worse
- Normalizer canonical mappings: 0 tokens, measurable improvement

**Lesson**: 8B models can read log content correctly but cannot follow abstract naming rules. The normalizer is deterministic and zero-cost.

### Decision 2: Mixed Retrieval (4 anomaly + 1 normal)

**Why**: Without normal contrast evidence, the LLM can only say "this is anomalous because it has errors." With a normal example, it can say "this has errors at L3-L5 that are absent in the normal session E4, confirming the anomaly."

**Result**: Enables "contrast" claim type, which is the third pillar of forensic explanation.

### Decision 3: Signatures as Open-Set Labels

Signatures are NOT a fixed taxonomy. The LLM can create new categories. This means:
- We don't want to force every anomaly into a predefined class (label collapse)
- The normalizer consolidates naming variants, not concepts
- If the LLM discovers a genuinely new error pattern, it should be allowed to name it

### Decision 4: Verification with Soft Constraints

The verifier uses 8 checks but only 3 can cause FAIL (structure, evidence IDs, evidence coverage). The rest are WARNINGs. This is because:
- We want to detect hallucinations (FAILs) but not penalize stylistic differences (WARNINGs)
- **keyword_match_ratio=0.0** — we don't fail on keyword mismatches because log normalization replaces many keywords with placeholders

### Decision 5: Unified Framework with Dataset Subclasses

~83% of code is shared between BGL and HDFS. Dataset-specific logic is concentrated in:
- `BGLNormalizer` / `HDFSNormalizer` (normalize patterns + signature normalization)
- `SIGNATURE_EXAMPLES` in prompt builder (component names, example signatures)
- `Screener.CONFIGS` (model dimensions)

### Decision 6: Train-Only Evidence Store

Evidence comes exclusively from training data. This prevents information leakage:
- The LLM never sees the test session's label
- The retrieved evidence might be wrong label (that's fine — it's what the model would encounter in production)
- The evidence is always from a separate data partition

---

## 15. Key Files & Module Map

### Source Code (`src/`)

| File | Lines | Purpose | Key Classes |
|------|-------|---------|-------------|
| `data_loader.py` | 406 | Load raw logs → Session objects | `Session`, `BGLDataLoader`, `HDFSDataLoader` |
| `screener.py` | 610 | Linformer anomaly detection | `AllLinLog`, `Screener`, `ScreenerOutput` |
| `evidence_store.py` | 274 | Normalized training corpus | `EvidenceStore`, `EvidenceDoc` |
| `retriever.py` | 520 | BM25 retrieval | `BM25Retriever`, `Retriever`, `RetrievalHit` |
| `prompt_builder.py` | 585 | LLM prompt assembly | `PromptBuilder`, `TraceExplanation`, `Claim`, `Signature` |
| `llm_client.py` | 351 | Ollama/OpenAI API client | `LLMClient`, `LLMResponse` |
| `normalizer.py` | 623 | Text normalization + signature normalization | `LogNormalizer`, `BGLNormalizer`, `HDFSNormalizer` |
| `verifier.py` | 684 | 8-check faithfulness verification | `Verifier`, `VerificationResult` |
| `signature_generator.py` | 381 | Mine patterns from training | `SignatureGenerator`, `ErrorSignature` |

### Pipeline & Notebooks

| File | Purpose |
|------|---------|
| `pipelines/explain_all.py` | Standalone end-to-end pipeline class |
| `notebooks/06_full_run.ipynb` | Primary run notebook (BGL & HDFS) |
| `notebooks/05_signature_audit.ipynb` | Signature discovery & pattern analysis |
| `notebooks/01-04_*.ipynb` | Development & testing notebooks |

### Data & Config

| File | Purpose |
|------|---------|
| `configs/config.yaml` | All model/pipeline parameters |
| `patterns/bgl_patterns.json` | 34 BGL pattern cards with discriminative scores |
| `patterns/hdfs_patterns.json` | 26 HDFS pattern cards with discriminative scores |
| `logs/BGL.log` | Reconstructed BGL log file (4.7M lines) |
| `logs/HDFS.log` | Reconstructed HDFS log file (11.1M lines) |
| `best_model/*.pth` | Pre-trained Linformer screener weights |

### Results

| File Pattern | Content |
|-------------|---------|
| `results/explanations_BGL_*.jsonl` | One JSON record per explained session |
| `results/explanations_BGL_*.metrics.json` | Aggregate metrics for the run |
| `results/evidence_store_BGL.json` | Serialized evidence store (reusable) |

### Documentation

| File | Purpose |
|------|---------|
| `long-term-mem/*.md` | Daily progress logs (read latest for context) |
| `long-term-mem/discriminative-analysis.md` | BGL vs HDFS anomaly type comparison |
| `long-term-mem/signature-lifecycle.md` | How signatures flow through the pipeline |
| `project-description/*.md` | Research framing & coding blueprint |

### Module Dependency Graph

```
                    data_loader.py
                   ╱      |      ╲
                  ╱       |       ╲
          normalizer.py   |    screener.py
              |           |        |
        evidence_store.py |        |
              |           |        |
          retriever.py    |        |
              |           |        |
        prompt_builder.py ←────────┘
              |
          llm_client.py     (standalone)
              |
          verifier.py
              |
     pipelines/explain_all.py  (imports ALL)
```

---

## 16. Glossary

| Term | Definition |
|------|-----------|
| **AllLinLog** | The Linformer-based anomaly detection model (screener) |
| **Screener** | The lightweight model that detects anomalies (fast, runs on all sessions) |
| **Reasoner** | The LLM that explains anomalies (slow, runs only on detected anomalies) |
| **Session** | A group of log lines forming one analysis unit (sliding window for BGL, block_id for HDFS) |
| **Evidence Store** | Normalized training sessions used as the RAG corpus for retrieval |
| **BM25** | Best Match 25 — term-frequency-based retrieval algorithm with TF saturation and length normalization |
| **Mixed Retrieval** | Retrieving both anomaly exemplars (4) and normal contrast (1) for each query |
| **Trace Explanation** | The structured JSON output with prediction, summary, signature, and evidence-grounded claims |
| **Claim** | A single evidence-grounded statement in an explanation. Three types: observation, pattern_match, contrast |
| **Evidence Span** | Line-level reference like `E0-L3` pointing to line 3 of evidence E0 |
| **Signature** | A canonical anomaly type label in `COMPONENT__ERROR_TYPE` format |
| **Normalizer** | Dual role: (1) normalize log text for BM25 matching, (2) normalize LLM signature names to canonical form |
| **Linformer** | Linear self-attention variant with O(n) complexity, projecting K/V to fixed dimension k |
| **GPT-4 BPE** | The tokenizer used (cl100k_base, 100,264 tokens) — used for the screener, NOT for LLM calls |
| **Evidence ID Mapping** | E0→query session, E1-EN→retrieved evidence documents |
| **Verification** | 8-check system ensuring LLM output is faithful to the provided evidence |
| **Pattern Card** | Pre-mined error signature with name, keywords, frequency, discriminative score |
| **Discriminative Score** | Keyword containment rate difference between anomaly and normal sessions |
| **Lexical Anomaly** | Anomaly detectable by presence of specific keywords (BGL) |
| **Structural Anomaly** | Anomaly detectable by absence/incompleteness of operations (HDFS) |
| **Open-Set Label** | Signature naming paradigm where the LLM can create new categories (not fixed taxonomy) |
| **JSONL** | JSON Lines format — one JSON object per line, used for incremental crash-resilient saving |

---

*End of study guide. Read `long-term-mem/` daily logs for session-by-session development history.*
