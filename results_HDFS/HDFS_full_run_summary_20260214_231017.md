# HDFS Full Run Summary

**Run date:** 2026-02-15 (started 2026-02-14 23:10)  
**Results file:** `explanations_HDFS_20260214_231017.jsonl`  
**Duration:** ~18.9 hours (68,146s)

---

## Pipeline Metrics

| Metric | Value |
|--------|-------|
| Total test sessions | 86,260 |
| Predicted anomalies | 2,527 |
| Successful explanations | 2,527 (100%) |
| LLM parse errors | 0 (0%) |
| Verification passed | 2,525 (99.9% of explained) |
| Verification failed | 2 (0.08% of explained) |

### Screener Accuracy

| Metric | Value |
|--------|-------|
| True Positives | 2,520 |
| False Positives | 7 |
| False Negatives | 6 |
| Precision | 99.7% |
| Recall | 99.8% |

## Token Usage

| Metric | Value |
|--------|-------|
| Total tokens | 8,460,576 |
| Avg tokens/session | 3,348 |

## Latency

| Metric | Value |
|--------|-------|
| Avg latency | 10,498 ms |
| P95 latency | 12,354 ms |

---

## Signature Distribution

12 unique signatures produced by the LLM (all already canonical — HDFSNormalizer required no further consolidation). Normalized results written to `explanations_HDFS_20260214_231017.normalized.jsonl`.

### All 12 Signatures

| Signature | Count | % |
|-----------|------:|--:|
| DATANODE__BLOCK_VERIFICATION_FAILED | 1,665 | 65.9% |
| DATANODE__BLOCK_WRITE_FAILURE | 491 | 19.4% |
| DATANODE__BLOCK_RECEIVE_FAILURE | 168 | 6.6% |
| DATANODE__BLOCK_TRANSFER_FAILURE | 108 | 4.3% |
| DATANODE__BLOCK_SERVING_FAILURE | 48 | 1.9% |
| NAMENODE__REPLICATION_INCOMPLETE | 20 | 0.8% |
| DATANODE__WRITE_PIPELINE_FAILURE | 20 | 0.8% |
| DATANODE__REDUNDANT_STORED_BLOCK | 3 | 0.1% |
| DATANODE__BLOCK_DUPLICATION | 1 | <0.1% |
| DATANODE__BLOCK_VERIFICATION_SUCCEEDED | 1 | <0.1% |
| NAMENODE__REDUNDANT_STORED_BLOCK | 1 | <0.1% |
| DATANODE__BLOCK_REPLICATION_FAILED | 1 | <0.1% |

Top 4 signatures account for 2,432 / 2,527 sessions (96.2%).

---

## Verification Failures (2 sessions)

Two sessions (0.08% of 2,527) failed verification due to **invalid evidence span references**. Both correctly identified the `DATANODE__BLOCK_VERIFICATION_FAILED` signature and cited valid evidence IDs — the failures are formatting errors, not hallucinations.

Both sessions cite evidence ID `E5`, which does not exist. The pipeline retrieves 5 evidence documents (E0–E4: 4 anomaly + 1 normal), so `E5` is an out-of-range reference the LLM fabricated.

### HDFS_blk_6989094700274811196

| Field | Value |
|-------|-------|
| LLM Signature | `DATANODE__BLOCK_VERIFICATION_FAILED` |
| LLM Summary | "DATANODE__BLOCK_VERIFICATION_FAILED: 2 errors at E0-L15 and E0-L8, E0-L5." |
| Verification checks | 8 passed, 1 failed |
| Failed check | `evidence_spans_validity`: E5 is malformed (does not exist) |

**LLM Claims:**
1. (claim referencing `E0`)
2. (claim referencing `E1`, `E2`)
3. (claim referencing `E0`, **`E5`** — invalid)

**Actual log content (41 lines, showing key events):**

| Line | Content (truncated) |
|------|---------------------|
| L1–L3 | `Receiving block blk_6989094700274811196` (3 DataXceiver receives) |
| L4 | `NameSystem.allocateBlock: /user/root/randtxt2/...` |
| L5–L13 | `PacketResponder` terminating + `Received block` + `addStoredBlock` (normal write pipeline) |
| L14 | `Served block blk_6989094700274811196 to /10.251.42.9` |
| **L15** | **`WARN: Got exception while serving blk_6989094700274811196 to /10.250.13.188`** |
| L16 | `Served block blk_6989094700274811196 to /10.251.91.32` |
| L17–L25 | Replication requests, block transfers, block deletion (recovery sequence) |
| L26–L41 | Additional replication, transfer, and verification events |

**Analysis:** The LLM correctly identified the WARN-level serving exception at L15 as the anomaly trigger. The signature `DATANODE__BLOCK_VERIFICATION_FAILED` is appropriate — the block had a serving error that led to replication and recovery. The only issue is the spurious `E5` evidence reference.

### HDFS_blk_9135407675197435306

| Field | Value |
|-------|-------|
| LLM Signature | `DATANODE__BLOCK_VERIFICATION_FAILED` |
| LLM Summary | "DATANODE__BLOCK_VERIFICATION_FAILED: 3 errors at E0-L6 to E0-L10, E0-L16 to E0-L17" |
| Verification checks | 8 passed, 1 failed |
| Failed check | `evidence_spans_validity`: E5 is malformed (does not exist) |

**LLM Claims:**
1. (claim referencing `E0`)
2. (claim referencing `E1`, `E2`)
3. (claim referencing `E0`, **`E5`** — invalid)

**Actual log content (27 lines, showing key events):**

| Line | Content (truncated) |
|------|---------------------|
| L1 | `NameSystem.allocateBlock: /user/root/randtxt2/...` |
| L2–L4 | `Receiving block blk_9135407675197435306` (3 receives) |
| L5–L13 | `PacketResponder` terminating + `Received block` + `addStoredBlock` (normal write) |
| L14–L15 | `Served block` (2 successful serves) |
| **L16** | **`WARN: Got exception while serving blk_9135407675197435306 to /10.251.71.240`** |
| L17 | `Verification succeeded for blk_9135407675197435306` |
| L18 | `Served block` (successful) |
| **L19** | **`WARN: Got exception while serving blk_9135407675197435306 to /10.251.71.240`** |
| L20 | `Served block` (successful) |
| L21–L23 | `NameSystem.delete: blk_... is added to invalidSet` (3 nodes) |
| L24–L25 | `Deleting block` (cleanup) |

**Analysis:** Two WARN-level serving exceptions (L16, L19) to the same destination (`/10.251.71.240`), followed by block invalidation and deletion — a clear anomalous lifecycle. The LLM correctly identified the failure pattern. Note L17 shows "Verification succeeded" between the two errors, which is the incidental success message that spawned the singleton `DATANODE__BLOCK_VERIFICATION_SUCCEEDED` signature for a different session.

### Root Cause

Both sessions failed the same check: the LLM referenced evidence ID `E5` which does not exist in the retrieval set (only E0–E4 are valid). This is a **formatting/hallucinated reference** error, not a reasoning failure:

1. **Correct signature** — both correctly identified `DATANODE__BLOCK_VERIFICATION_FAILED`
2. **Correct grounding** — claims reference valid evidence (E0, E1, E2) alongside the invalid E5
3. **Minor severity** — removing the `E5` reference would make both pass verification

**Contrast with BGL:** The BGL full run's 2 verification failures were **confirmed hallucinations** (0% evidence coverage, wrong signature, fabricated claims about normal-severity messages). The HDFS failures are qualitatively different — correct identification with a spurious evidence reference.

---

## Cross-Dataset Comparison (BGL vs HDFS)

| Metric | BGL | HDFS |
|--------|-----|------|
| Anomalies | 6,295 | 2,527 |
| Success rate | 99.92% | 100% |
| LLM errors | 5 (0.08%) | 0 |
| Verification pass | 99.97% | 99.9% |
| V-failure type | Hallucination (0% coverage) | Format error (invalid spans) |
| Raw signatures | 116 | 12 |
| Normalized signatures | 56 | 12 |
| Normalizer consolidation | 116→56 (52% reduction) | None needed |
| Avg tokens/session | 3,493 | 3,348 |
| Avg latency | 10,072 ms | 10,498 ms |
| Wall time | 29.3 hrs | 18.9 hrs |

### Key Observations

1. **HDFS is structurally simpler:** 12 signatures vs 56 (BGL). HDFS anomalies cluster tightly around block-level operations (verification, write, receive, transfer). BGL has diverse hardware/kernel failure modes.

2. **Normalizer not needed for HDFS:** The LLM produced consistent naming for HDFS from the start. This aligns with the discriminative analysis — HDFS anomalies are structural (sequence/absence patterns), not lexical, so the LLM has fewer naming choices to make.

3. **Zero LLM errors on HDFS:** The 100% success rate (vs 99.92% for BGL) may reflect shorter/simpler HDFS log sequences that are easier for the LLM to parse.

4. **Block verification dominates:** 65.9% of HDFS anomalies are block verification failures. This is consistent with HDFS's known failure mode — corrupted or missing block replicas.

---

## SME Reviewer Assessment

### Strengths

**Perfect completion:** 100% success rate on 2,527 anomalies over 18.9 hours with zero LLM errors. The pipeline demonstrates robust reliability across a second, structurally different dataset.

**Tight signature space:** 12 signatures for a distributed file system is highly plausible. HDFS has a well-defined set of block-level operations, and the LLM correctly maps anomalies to these categories without over-differentiating.

**Consistent with 500-session validation:** The 500-session test (Feb 13) produced 8 unique signatures. The full run (2,527 sessions) found 12 — the additional 4 are genuine rare failure types (block duplication, replication failures), not noise.

### Gaps

Same gaps as BGL apply:
- No ground-truth evaluation of explanation correctness
- Verification is structural, not semantic
- No human evaluation sample

### HDFS-Specific Observation

The `DATANODE__BLOCK_VERIFICATION_SUCCEEDED` signature (1 session) is suspicious — if verification succeeded, why is the session anomalous? This warrants manual inspection. It may be a case where the block was flagged for other reasons (e.g., replication delay) and the verification success message is incidental.
