# Signature Lifecycle in the Pipeline

Reference document explaining how signatures flow through each stage of the AllInLog pipeline.

---

## 1. Pattern Discovery → Pattern Cards

Signatures originate in `patterns/bgl_patterns.json` and `patterns/hdfs_patterns.json`, mined from training data by `notebooks/05_signature_audit.ipynb`. Each card has:

```json
{
  "name": "KERNEL__DATA_TLB_ERROR",
  "description": "Data TLB error interrupt (11052 sessions)",
  "keywords": ["data", "error", "fatal", "interrupt", "tlb"],
  "frequency": 11052,
  "fingerprint": "data+error+fatal+interrupt+tlb"
}
```

- BGL: 34 pattern cards
- HDFS: 26 pattern cards
- Coverage: 99.8% of training anomalies for both datasets

---

## 2. Evidence Store — Signature Cards as Documents

Pattern cards are injected into the `EvidenceStore` as `EvidenceDoc` objects with `evidence_type="signature"`. Their text is a human-readable card:

```
ERROR SIGNATURE: KERNEL__DATA_TLB_ERROR
Description: Data TLB error interrupt (11052 sessions)
Key Indicators: data, error, fatal, interrupt, tlb
Frequency: 11052 occurrences
Fingerprint: data+error+fatal+interrupt+tlb
```

They live alongside session documents (anomaly/normal) in the store.

**Key code** (`notebooks/06_full_run.ipynb`, cell 5):
```python
doc = EvidenceDoc(
    evidence_id=f"E_SIG_{pid}",
    session_id=pid,
    text=sig_text,
    evidence_type="signature",
    metadata={"label": 1, "dataset": DATASET,
              "signature_name": info['name'],
              "frequency": info['frequency'],
              "keywords": info['keywords']}
)
evidence_store.documents.append(doc)
```

---

## 3. Retrieval — BM25 Index Enrichment

Signature cards are tokenized into the BM25 index alongside all other docs. **However**, in `retrieve_for_session_mixed()`, they're **excluded** from the anomaly/normal split (they have no `label=0` or `label=1`):

```python
if label == 1:   # Anomaly
    anomaly_scored.append((idx, score))
elif label == 0: # Normal
    normal_scored.append((idx, score))
# Signature cards (no label) are skipped
```

Their primary value is **vocabulary enrichment** — the keywords in card text help BM25 scoring, but cards themselves don't appear in retrieval hits used by the prompt.

---

## 4. Prompt Building — Guiding the LLM

Two mechanisms teach the LLM about signatures:

### a) `SIGNATURE_EXAMPLES` dict (`src/prompt_builder.py`)

Provides dataset-specific example names and component lists:

```python
SIGNATURE_EXAMPLES = {
    "BGL": {
        "examples": [
            "KERNEL__DATA_TLB_ERROR",
            "KERNEL__DATA_STORAGE_INTERRUPT",
            "APP__CIOD_STREAM_ERROR",
            "KERNEL__LUSTRE_MOUNT_FAILED",
            "KERNEL__KERNEL_TERMINATED",
        ],
        "components": "KERNEL, APP, MMCS, LINKCARD",
    },
    "HDFS": {
        "examples": [
            "DATANODE__BLOCK_VERIFICATION_FAILED",
            "NAMENODE__REPLICATION_INCOMPLETE",
            "DATANODE__WRITE_PIPELINE_FAILURE",
            "DATANODE__BLOCK_WRITE_FAILURE",
            "DATANODE__BLOCK_RECEIVE_FAILURE",
        ],
        "components": "DATANODE, NAMENODE, FSDATASET, BLOCKSCANNER",
    },
}
```

### b) System Prompt

Has a full `=== SIGNATURE NAMING ===` section instructing:
- Format: `COMPONENT__ERROR_TYPE` (double underscore, NO severity)
- 3 example signatures from the dataset
- "You MUST create YOUR OWN signature based on what you see in [E0]"

### c) JSON Schema

The `TRACE_SCHEMA` makes `signature` a **required** output field:

```python
"signature": {
    "type": "object",
    "required": ["name", "matched_evidence_ids"],
    "properties": {
        "name": {"type": "string",
                 "description": "Anomaly signature name: COMPONENT__ERROR_TYPE"},
        "matched_evidence_ids": {"type": "array", "items": {"type": "string"}}
    }
}
```

---

## 5. LLM Output

The LLM produces:

```json
{
  "signature": {
    "name": "KERNEL__FATAL__DATA_TLB_ERROR",
    "matched_evidence_ids": ["E1", "E2"]
  }
}
```

- `name`: The LLM's best-effort signature name (often includes severity despite instructions)
- `matched_evidence_ids`: Which retrieved evidence the LLM considered relevant to that signature

Parsed via `TraceExplanation.from_dict()` → `Signature.from_dict()`.

---

## 6. Normalization — Post-LLM Cleanup

`normalize_signature()` fires immediately after LLM response parsing. It fixes the LLM's inconsistencies.

### BGLNormalizer

| Pattern | Example | Result |
|---------|---------|--------|
| Strip severity | `KERNEL__FATAL__DATA_TLB_ERROR` | `KERNEL__DATA_TLB_ERROR` |
| Strip RAS_ prefix | `RAS_APP_FATAL__CIOD_STREAM_ERROR` | `APP__CIOD_STREAM_ERROR` |
| Collapse verbose | `kernel terminated for reason 1001` | `KERNEL_TERMINATED` |
| Canonical mapping | `TLB_ERROR` | `DATA_TLB_ERROR` |
| Component fix | `KERNEL__CIOD_STREAM_ERROR` | `APP__CIOD_STREAM_ERROR` |

### HDFSNormalizer

| Pattern | Example | Result |
|---------|---------|--------|
| Severity as prefix suffix | `DATANODE_INFO__BLOCK_WRITE_FAILURE` | `DATANODE__BLOCK_WRITE_FAILURE` |
| Severity as middle segment | `DATANODE__INFO__BLOCK_WRITE_FAILURE` | `DATANODE__BLOCK_WRITE_FAILURE` |
| Canonical mapping | `BLOCK_RECEIVING_FAILED` | `BLOCK_RECEIVE_FAILURE` |
| Component fix | `FSDATASET__BLOCK_WRITE_FAILURE` | `DATANODE__BLOCK_WRITE_FAILURE` |

**Code** (`pipelines/explain_all.py` and `notebooks/06_full_run.ipynb`):
```python
if explanation.signature and explanation.signature.name:
    explanation.signature.name = normalizer.normalize_signature(
        explanation.signature.name
    )
```

---

## 7. Verification — Soft Checks

`Verifier._check_signature()` runs three checks, all producing **warnings** (not failures):

| Check | Result if failed |
|-------|-----------------|
| Signature is `None` | WARNING |
| Name is empty or `"UNKNOWN"` | WARNING |
| No `__` in name | WARNING (wrong format) |

Signature issues alone **won't fail** verification — they're advisory. This is intentional since the LLM is *creating* signatures, not matching against a fixed set.

---

## Data Flow Diagram

```
patterns/*.json ──→ EvidenceStore (signature docs) ──→ BM25 Index (vocabulary)
                                                            │
                    SIGNATURE_EXAMPLES ──→ System Prompt     │
                                               │            │
test session ──→ Screener ──→ Retriever ───────┼────────────┘
                                 │             │
                            evidence hits   prompt
                                 └──→ PromptBuilder ──→ LLM
                                                        │
                                                  JSON { signature }
                                                        │
                                              normalize_signature()
                                                        │
                                              Verifier (soft check)
                                                        │
                                              metrics.signature_counts
```

---

## Key Insight

Signatures serve a dual purpose:

1. **Retrieval enrichment** — pattern card keywords in the BM25 index improve scoring for relevant anomaly sessions
2. **Output standardization** — prompt examples + post-LLM normalization ensure consistent naming across thousands of LLM calls

The verification is intentionally soft since the LLM is *creating* signatures from what it observes, not matching against a fixed closed set.
