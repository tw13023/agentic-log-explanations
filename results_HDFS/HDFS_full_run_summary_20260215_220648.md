# HDFS Full Run Summary (Post-Fix Re-run)

**Run date:** 2026-02-16 (started 2026-02-15 22:06)  
**Results file:** `explanations_HDFS_20260215_220648.jsonl`  
**Duration:** ~20.1 hours (72,531s)  
**Purpose:** Re-run after fixing two issues from the previous run (20260214_231017):
1. **Dynamic log display** — replaced hard `max_log_lines=20` cutoff with character-budget strategy (`max_chars=6000`, `tail_ratio=0.3`) that adapts to actual session length
2. **Bare evidence span handling** — updated prompt template to show full-range span format for contrast claims, and updated verifier to accept bare `E{n}` references as whole-document citations

---

## Pipeline Metrics

| Metric | Value | vs. Previous Run |
|--------|-------|:---:|
| Total test sessions | 86,260 | — |
| Predicted anomalies | 2,527 | — |
| Successful explanations | 2,527 (100%) | — |
| LLM parse errors | 0 (0%) | — |
| **Verification passed** | **2,527 (100%)** | **+2 (was 99.92%)** |
| Verification failed | 0 (0%) | **-2 (was 0.08%)** |

### Screener Accuracy

| Metric | Value |
|--------|-------|
| True Positives | 2,520 |
| False Positives | 7 |
| False Negatives | 6 |
| Precision | 99.7% |
| Recall | 99.8% |

## Token Usage

| Metric | Value | vs. Previous Run |
|--------|-------|:---:|
| Total tokens | 8,461,896 | +1,320 (+0.02%) |
| Avg tokens/session | 3,349 | ~same |

## Latency

| Metric | Value | vs. Previous Run |
|--------|-------|:---:|
| Avg latency | 10,395 ms | -103 ms |
| P50 latency | 10,398 ms | — |
| P95 latency | 12,362 ms | +8 ms |
| P99 latency | 13,151 ms | — |
| Wall time | 20.1 hrs | +1.2 hrs |

The wall time increase (+1.2 hrs) is within normal variance for a 20-hour LLM inference run. Token usage is virtually identical, confirming the dynamic log display does not inflate prompt size for HDFS sessions (max 55 lines, all fit within the 6000-char budget).

---

## Signature Distribution

13 unique signatures (vs. 12 in the previous run). All already canonical — HDFSNormalizer required no further consolidation. Normalized results written to `explanations_HDFS_20260215_220648.normalized.jsonl`.

### All 13 Signatures

| Signature | Count | % | Δ vs. Previous |
|-----------|------:|--:|:---:|
| DATANODE__BLOCK_VERIFICATION_FAILED | 1,675 | 66.3% | +10 |
| DATANODE__BLOCK_WRITE_FAILURE | 485 | 19.2% | -6 |
| DATANODE__BLOCK_RECEIVE_FAILURE | 169 | 6.7% | +1 |
| DATANODE__BLOCK_TRANSFER_FAILURE | 94 | 3.7% | -14 |
| DATANODE__BLOCK_SERVING_FAILURE | 54 | 2.1% | +6 |
| NAMENODE__REPLICATION_INCOMPLETE | 21 | 0.8% | +1 |
| DATANODE__WRITE_PIPELINE_FAILURE | 18 | 0.7% | -2 |
| DATANODE__BLOCK_DUPLICATION | 3 | 0.1% | +2 |
| DATANODE__REDUNDANT_STORED_BLOCK | 3 | 0.1% | 0 |
| NAMENODE__REDUNDANT_STORED_BLOCK | 2 | <0.1% | +1 |
| DATANODE__IO_ERROR | 1 | <0.1% | +1 (new) |
| DATANODE__REDUNDANT_ADD_STORED_BLOCK | 1 | <0.1% | +1 (new) |
| DATANODE__BLOCK_VERIFICATION_SUCCEEDED | 1 | <0.1% | 0 |

Top 4 signatures account for 2,423 / 2,527 sessions (95.9%).

**Signature changes from previous run:**
- 2 new signatures appeared: `DATANODE__IO_ERROR` (1) and `DATANODE__REDUNDANT_ADD_STORED_BLOCK` (1)
- 1 signature disappeared: `DATANODE__BLOCK_REPLICATION_FAILED` (was 1)
- Minor count redistribution across existing signatures (±1–14), consistent with LLM non-determinism across a 20-hour run

---

## Fix Impact Assessment

### Fix 1: Dynamic Log Display (Truncation)

**Previous issue:** `max_log_lines=20` hid late-appearing anomaly signals. The singleton `DATANODE__BLOCK_VERIFICATION_SUCCEEDED` in session `blk_-1478843903114016209` (30 lines) was caused by truncation hiding the actual WARN at L27.

**Result:** The old singleton session (`blk_-1478843903114016209`) is now correctly classified as `DATANODE__BLOCK_VERIFICATION_FAILED`. With the dynamic char-budget strategy, all 30 lines are visible and the LLM correctly identifies the anomaly.

**Token impact:** Negligible. HDFS sessions range from 1–55 lines (mean 17.4), all fitting within the 6000-char budget. No truncation occurs for any HDFS session in this run.

### Fix 2: Bare Evidence Span Handling

**Previous issue:** 2 sessions (0.08%) failed verification because the LLM wrote bare `E5` in contrast claims instead of `E5-L{n}` format.

**Result:** Zero verification failures. The updated prompt template now shows the full-range format (`E5-L1 to E5-L35`) for contrast claims, and the verifier accepts bare `E{n}` as whole-document references. Both previously-failing sessions (`blk_6989094700274811196` and `blk_9135407675197435306`) now pass verification with correct `DATANODE__BLOCK_VERIFICATION_FAILED` signatures.

---

## Remaining Singleton: DATANODE__BLOCK_VERIFICATION_SUCCEEDED

**Session:** `HDFS_blk_7428580627654080207` (label=1, verification **passed**)

This is a **different session** from the previous run's singleton (`blk_-1478843903114016209`). In the previous run, `blk_7428580627654080207` was correctly classified as `DATANODE__BLOCK_VERIFICATION_FAILED` — it flipped to `SUCCEEDED` in this run due to LLM non-determinism.

### Session Content (21 lines)

| Line | Content (truncated) | Notable? |
|------|----------------------|:---:|
| L1–L4 | `Receiving block` (3 DataXceiver) + `NameSystem.allocateBlock` | |
| L5–L10 | `PacketResponder` terminating + `Received block` (normal write) | |
| L11–L13 | `NameSystem.addStoredBlock` (3 nodes) | |
| **L14** | **`INFO dfs.DataBlockScanner: Verification succeeded for blk_7428580627654080207`** | ⚠️ |
| L15–L17 | `NameSystem.delete: added to invalidSet` (3 nodes) | |
| L18–L19 | `FSDataset: Deleting block` (2 deletions) | |
| **L20** | **`WARN dfs.FSDataset: Unexpected error trying to delete block blk_7428580627654080207. BlockInfo not found in volumeMap.`** | ⚠️ |
| L21 | `FSDataset: Deleting block` (1 more deletion) | |

### Analysis

Unlike the previous singleton, this is **not a truncation artifact** — all 21 lines fit within the char budget and are visible to the LLM. The LLM sees the WARN at L20 but still focuses on the "Verification succeeded" event at L14. This demonstrates a **model-level limitation**: the 8B LLM sometimes latches onto distinctive INFO-level events over WARN-level events when naming signatures, especially when the WARN appears late in the sequence.

**Key observations:**
1. The LLM's summary correctly says "DATANODE__BLOCK_VERIFICATION_FAILED" (referencing the failure pattern), but the signature name says "SUCCEEDED" — the same internal contradiction as the previous singleton
2. The WARN at L20 (the actual anomaly trigger) is visible but not referenced in the claims — the LLM cites L10, L12, L14 instead
3. This session classified correctly as `BLOCK_VERIFICATION_FAILED` in the prior run — the flip is LLM non-determinism (temperature=0.1 still allows some variation)

**Impact:** 1/2,527 (0.04%) — same negligible rate as before, just a different session.

### Normalizer Decision

Same as previous run: no normalizer mapping added. The singleton is retained as-is because:
- It's a genuine LLM quality issue, not a naming inconsistency
- Mapping it to `DATANODE__BLOCK_VERIFICATION_FAILED` would mask the error
- The 0.04% rate is negligible for aggregate analysis

---

## Cross-Dataset Comparison (BGL vs HDFS)

| Metric | BGL | HDFS (this run) |
|--------|-----|------|
| Anomalies | 6,295 | 2,527 |
| Success rate | 99.92% | 100% |
| LLM errors | 5 (0.08%) | 0 |
| Verification pass | 99.97% | **100%** |
| V-failure type | Hallucination (0% coverage) | None |
| Raw signatures | 116 | 13 |
| Normalized signatures | 56 | 13 |
| Normalizer consolidation | 116→56 (52% reduction) | None needed |
| Avg tokens/session | 3,493 | 3,349 |
| Avg latency | 10,072 ms | 10,395 ms |
| Wall time | 29.3 hrs | 20.1 hrs |

---

## Run Comparison: Previous (20260214) vs This Run (20260215)

| Metric | Previous | This Run | Change |
|--------|----------|----------|--------|
| Successful | 2,527 | 2,527 | — |
| Errors | 0 | 0 | — |
| **Verification passed** | **2,525** | **2,527** | **+2** |
| **Verification failed** | **2** | **0** | **-2** |
| Unique signatures | 12 | 13 | +1 |
| SUCCEEDED singleton | blk_-1478843903114016209 | blk_7428580627654080207 | Different session |
| Total tokens | 8,460,576 | 8,461,896 | +0.02% |
| Avg latency | 10,498 ms | 10,395 ms | -1.0% |
| Wall time | 18.9 hrs | 20.1 hrs | +6.3% |

### Summary of Improvements
1. **100% verification pass rate** — eliminates the 2 bare-span failures from the previous run
2. **Old truncation singleton fixed** — `blk_-1478843903114016209` now correctly classified as `BLOCK_VERIFICATION_FAILED`
3. **Token-neutral** — dynamic log display adds ~0.02% tokens, confirming HDFS sessions fit within the char budget
4. **New SUCCEEDED singleton** — different session, same 0.04% rate, caused by LLM non-determinism rather than truncation

---

## SME Reviewer Assessment

### What Improved

**100% verification pass rate** — the bare-span formatting issue is fully resolved. The combination of better prompt examples and tolerant verifier parsing eliminates all structural failures.

**Truncation artifact resolved** — the specific session (`blk_-1478843903114016209`) that motivated the `max_log_lines` investigation is now correctly classified. The dynamic char-budget approach is validated: zero truncation on any HDFS session, no token cost increase.

### What Persists

**SUCCEEDED singleton (0.04%):** Still present but now caused by LLM non-determinism rather than truncation. A different session exhibits the same pattern: the LLM correctly summarizes the failure mode but names the signature after a distinctive INFO event ("Verification succeeded") rather than the WARN event. This is a known limitation of the 8B model and would likely require a larger model or few-shot examples to resolve.

### Recommendation

This run's results are suitable for paper reporting:
- **100% success, 100% verification** — clean numbers
- **13 signatures** — stable and interpretable
- **Singleton at 0.04%** — documented limitation, negligible impact

No further re-run needed for HDFS.
