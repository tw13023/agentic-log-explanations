# Discriminative Analysis: BGL vs HDFS Anomaly Patterns

## Key Finding

The two datasets exhibit fundamentally different anomaly characteristics, yet are handled by the same unified framework.

---

## BGL = Lexical Anomalies

- **All 34 patterns are "high" discriminative** (scores +0.48 to +0.88)
- Keywords like `tlb`, `ciod`, `parity`, `storage`, `interrupt` are strongly anomaly-specific
- Anomaly rate ≈ 100% for all patterns — these keywords appear in nearly every anomaly session
- Normal rate ranges 12–34% — keywords occasionally appear in normal sessions but far less frequently
- **Implication**: BGL anomalies can be identified by the *presence* of specific error keywords

### BGL Discriminative Score Distribution

| Score Range | Count | Example Patterns |
|-------------|-------|-----------------|
| +0.80 to +0.89 | 7 | FATAL_ERROR, CIOD_ERROR, DATA_STORAGE_INTERRUPT |
| +0.70 to +0.80 | 14 | DATA_TLB_ERROR, CIOD_SOCKET_ERROR, HARDWARE_ERROR |
| +0.60 to +0.70 | 10 | PROGRAM_INTERRUPT, DATA_INTERRUPT, CIOD_SIGNAL_DELIVERED |
| +0.40 to +0.60 | 3 | HARDWARE_ERROR (rare subtypes) |

---

## HDFS = Structural Anomalies

- **All 26 patterns are "low" discriminative** (scores ≈ +0.0001)
- Keywords like `exception`, `writeblock`, `error`, `replicate` appear in **both** anomaly and normal sessions at nearly identical rates (~100%)
- **Implication**: HDFS anomalies cannot be identified by keyword presence alone — they are characterized by the *absence* of expected normal operations (e.g., missing `received block`, incomplete replication pipeline)

### Why HDFS Keywords Don't Discriminate

| Keyword | Anomaly Rate | Normal Rate | Discriminative Score |
|---------|-------------|-------------|---------------------|
| writeblock | 100.0% | 100.0% | +0.0002 |
| exception | 100.0% | 100.0% | +0.0002 |
| replicate | 100.0% | 100.0% | +0.0001 |
| error | 100.0% | 100.0% | +0.0001 |

Normal HDFS sessions also contain these keywords because block operations naturally generate exceptions and errors during routine replication. The anomaly signal is **structural**: certain operations start but never complete.

---

## Unified Framework Advantage

AllInLog handles both anomaly types through the same pipeline:

1. **Screener** (Linformer): Detects anomalies regardless of whether the signal is lexical (BGL) or structural (HDFS)
2. **Retriever** (BM25): Retrieves relevant evidence — for BGL, keyword overlap naturally finds similar errors; for HDFS, structural similarity emerges from operation co-occurrence patterns
3. **LLM Explainer**: Generates forensic explanations grounded in evidence — adapts its reasoning to the anomaly type visible in the logs
4. **Normalizer**: Consolidates LLM-generated signature names into canonical forms — handles both keyword-rich BGL names and operation-based HDFS names

The system requires **zero dataset-specific tuning** beyond the normalizer subclass (~17% of code). The same prompt structure, retrieval strategy, and verification pipeline work for both fundamentally different anomaly types.

---

## Paper Framing Suggestion

> "We evaluate AllInLog on two datasets with contrasting anomaly characteristics: BGL, where anomalies are marked by highly discriminative error keywords (mean discriminative score = 0.75), and HDFS, where anomaly-specific keywords are indistinguishable from normal operations (mean discriminative score ≈ 0). This contrast demonstrates the framework's ability to explain anomalies regardless of whether the underlying signal is lexical or structural."

---

## Data Source

Discriminative scores computed from training data using keyword containment rates:
- `discriminative_score = anomaly_rate - normal_rate`
- Stored in `patterns/bgl_patterns.json` and `patterns/hdfs_patterns.json`
- Computed on 2026-02-13
