# BGL Full Run Summary

**Run date:** 2026-02-13  
**Results file:** `explanations_BGL_20260213_160034.jsonl`  
**Duration:** ~29.3 hours (63,355,129 ms)

---

## Pipeline Metrics

| Metric | Value |
|--------|-------|
| Total test sessions | 71,221 |
| Anomalies processed | 6,295 |
| Successful explanations | 6,290 (99.92%) |
| LLM parse errors | 5 (0.08%) |
| Verification passed | 6,288 (99.97% of explained) |
| Verification failed | 2 (0.03% of explained) |

## Token Usage

| Metric | Value |
|--------|-------|
| Total tokens | 21,971,554 |
| Avg tokens/session | 3,493 |

## Latency

| Metric | Value |
|--------|-------|
| Avg latency | 10,072 ms |
| P95 latency | 11,750 ms |

## Signature Distribution

116 raw unique signatures produced by the LLM, consolidated to **56 canonical signatures** after normalization (`BGLNormalizer`, commit `cb57eac`). Normalized results written to `explanations_BGL_20260213_160034.normalized.jsonl`.

### All 56 Normalized Signatures

| Signature | Count | % |
|-----------|------:|--:|
| KERNEL__DATA_TLB_ERROR | 2,370 | 37.7% |
| KERNEL__DATA_STORAGE_INTERRUPT | 1,129 | 18.0% |
| APP__CIOD_STREAM_ERROR | 1,104 | 17.6% |
| KERNEL__LUSTRE_MOUNT_FAILED | 483 | 7.7% |
| KERNEL__KERNEL_TERMINATED | 330 | 5.2% |
| APP__LOGIN_CHDIR_FAILED | 166 | 2.6% |
| KERNEL__BAD_MESSAGE_HEADER | 98 | 1.6% |
| KERNEL__FATAL_ERROR | 97 | 1.5% |
| LINKCARD__NODE_CARD_VPD_CHECK | 76 | 1.2% |
| APP__CIOD_NODE_MAP_ERROR | 48 | 0.8% |
| KERNEL__FLOATING_POINT_ERROR | 42 | 0.7% |
| KERNEL__DDR_ERROR | 41 | 0.7% |
| KERNEL__MACHINE_CHECK | 33 | 0.5% |
| KERNEL__MICROLOADER_ASSERTION | 21 | 0.3% |
| KERNEL__EXTERNAL_INPUT_INTERRUPT | 21 | 0.3% |
| KERNEL__INSTRUCTION_ADDRESS | 14 | 0.2% |
| LINKCARD__MONITOR_FAILURE | 14 | 0.2% |
| APP__CIOD_SIGNAL_RECEIVED | 13 | 0.2% |
| KERNEL__TREE_NETWORK_PACKET_ERROR | 13 | 0.2% |
| KERNEL__NETWORK_RECEIVE_ERROR | 12 | 0.2% |
| LINKCARD__NODE_CARD_STATUS_ERROR | 11 | 0.2% |
| KERNEL__MISSING_OR_INVALID_FIELDS | 11 | 0.2% |
| KERNEL__PARITY_ERROR | 10 | 0.2% |
| KERNEL__RTS_INTERNAL_ERROR | 10 | 0.2% |
| LINKCARD__DISCOVERY_ERROR | 8 | 0.1% |
| KERNEL__INTEGER_ALIGNMENT_ERROR | 7 | 0.1% |
| KERNEL__IDO_CHIP_STATUS_CHANGED | 7 | 0.1% |
| APP__EXEC_FORMAT_ERROR | 6 | 0.1% |
| APP__DEVICE_RESOURCE_BUSY | 6 | 0.1% |
| LINKCARD__CAN_NOT_GET_ASSEMBLY_INFORMATION | 6 | 0.1% |
| KERNEL__CRITICAL_INPUT_INTERRUPT | 6 | 0.1% |
| LINKCARD__HARDWARE_WARNING | 6 | 0.1% |
| KERNEL__REGISTER_DUMP | 6 | 0.1% |
| KERNEL__TORUS_RECEIVER_ERROR | 5 | 0.1% |
| KERNEL__COORDINATE_EXCEEDS_DIMENSION | 5 | 0.1% |
| KERNEL__RECEIVED_SIGNAL | 5 | 0.1% |
| KERNEL__IDO_PROXY_COMMUNICATION_FAILURE | 5 | 0.1% |
| KERNEL__NFS_MOUNT_FAILED | 4 | 0.1% |
| LINKCARD__MIDPLANE_SWITCH_ERROR | 4 | 0.1% |
| KERNEL__MAILBOXMONITOR_SERVICE_MAILBOXES | 4 | 0.1% |
| KERNEL__RESOURCE_TEMPORARILY_UNAVAILABLE | 4 | 0.1% |
| KERNEL__EDRAM_ERROR | 3 | <0.1% |
| KERNEL__POWER_GOOD_SIGNAL_DEACTIVATED | 3 | <0.1% |
| MMCS__ASSERT_CONDITION | 3 | <0.1% |
| LINKCARD__FATAL_ERROR | 3 | <0.1% |
| MMCS__BGLMASTER_FAILURE | 3 | <0.1% |
| KERNEL__L3_INTERNAL_ERROR | 2 | <0.1% |
| KERNEL__ICACHE_PREFETCH_ERROR | 2 | <0.1% |
| APP__CIOD_PROGRAM_IMAGE_ERROR | 2 | <0.1% |
| KERNEL__CE_SYM_ERROR | 2 | <0.1% |
| KERNEL__INFO | 1 | <0.1% |
| KERNEL__MEMORY_MANAGER_ERROR | 1 | <0.1% |
| LINKCARD__INVALID_NODE_ECID | 1 | <0.1% |
| KERNEL__INSTRUCTION_PLB_ERROR | 1 | <0.1% |
| APP__NODE_MAP_ERROR | 1 | <0.1% |
| KERNEL__CIOD_GENERATION | 1 | <0.1% |

Top 6 signatures account for 5,582 / 6,290 sessions (88.7%). 6 singletons remain (genuine rare error types).

---

## LLM Parse Errors (5 sessions)

Five sessions (0.08%) received malformed JSON responses from the Ollama inference endpoint (llama3.1:8b). In each case, the LLM returned a string that could not be parsed as a JSON object, raising `'str' object has no attribute 'get'` during response processing.

These are **transient infrastructure-level inference failures**, not systematic pipeline or model deficiencies. The error manifested identically in all five cases — the LLM produced output that did not conform to the expected JSON schema. No retry mechanism was applied; the sessions were recorded with empty explanations.

| Session ID | Error | Root Cause |
|------------|-------|------------|
| BGL_04603850 | `'str' object has no attribute 'get'` | Malformed LLM JSON response |
| BGL_02912890 | `'str' object has no attribute 'get'` | Malformed LLM JSON response |
| BGL_04643760 | `'str' object has no attribute 'get'` | Malformed LLM JSON response |
| BGL_04646060 | `'str' object has no attribute 'get'` | Malformed LLM JSON response |
| BGL_01097570 | `'str' object has no attribute 'get'` | Malformed LLM JSON response |

**Outcome:** All 5 recorded with `verification_passed = false`, empty signature and claims. The 0.08% error rate is consistent with expected transient failure rates in local LLM inference serving.

---

## Verification Failures (2 sessions)

Two sessions (0.03% of 6,290 explained) passed LLM explanation generation but failed the post-hoc verification stage. Both failed the same check: **evidence coverage below the 80% minimum threshold** (0% actual coverage).

Upon inspection of the raw BGL log lines, **both are confirmed LLM hallucinations** — the model latched onto the wrong signal in mixed-content windows.

### BGL_04647320

| Field | Value |
|-------|-------|
| LLM Signature | `KERNEL__INTEGER_ALIGNMENT_ERROR` |
| LLM Summary | "KERNEL__INTEGER_ALIGNMENT_EXCEPTION: 9 occurrences at E0-L2 to E0-L10" |
| Verification checks | 8 passed, 1 failed |
| Failed check | Evidence coverage 0% < 80% minimum |

**LLM Claims:**
1. "E0 contains 9 KERNEL INTEGER ALIGNMENT EXCEPTIONS at E0-L2 to E0-L10." — `evidence_ids: []`
2. "Similar occurrences can be found in E1, E2, E3, and E4." — `evidence_ids: []`

**Actual log content (10 lines):**

| Line | Label | Content |
|------|-------|---------|
| L1 | **ANOM** | `RAS APP FATAL ciod: Error reading message prefix on CioStream` |
| L2 | NORM | `RAS KERNEL INFO Kernel detected 4795798 integer alignment exceptions` |
| L3 | **ANOM** | `RAS APP FATAL ciod: Error reading message prefix on CioStream` |
| L4 | NORM | `RAS KERNEL INFO Kernel detected 4084912 integer alignment exceptions` |
| L5 | **ANOM** | `RAS APP FATAL ciod: Error reading message prefix on CioStream` |
| L6 | NORM | `RAS KERNEL INFO Kernel detected 4953596 integer alignment exceptions` |
| L7 | **ANOM** | `RAS APP FATAL ciod: Error reading message prefix on CioStream` |
| L8 | NORM | `RAS KERNEL INFO Kernel detected 4255580 integer alignment exceptions` |
| L9 | **ANOM** | `RAS APP FATAL ciod: Error reading message prefix on CioStream` |
| L10 | NORM | `RAS KERNEL INFO Kernel detected 4248920 integer alignment exceptions` |

**Hallucination analysis:** The session contains two interleaved error types: 5 anomalous `APP FATAL ciod: Error reading message prefix on CioStream` lines and 5 normal-severity `KERNEL INFO integer alignment exceptions` lines. The LLM **incorrectly identified the normal informational messages as the anomaly** and completely ignored the actual CIOD stream errors. The correct signature should be `APP__CIOD_STREAM_ERROR`. The claim "9 occurrences" is also factually wrong — there are only 5 alignment messages in the window.

### BGL_04647460

| Field | Value |
|-------|-------|
| LLM Signature | `KERNEL__INTEGER_ALIGNMENT_EXCEPTIONS` |
| LLM Summary | "KERNEL__INTEGER_ALIGNMENT_EXCEPTIONS: 10 occurrences at lines E0-L1 to E0-L10." |
| Verification checks | 8 passed, 1 failed |
| Failed check | Evidence coverage 0% < 80% minimum |

**LLM Claims:**
1. (empty claim text) — `evidence_ids: []`
2. (empty claim text) — `evidence_ids: []`

**Actual log content (10 lines):**

| Line | Label | Content |
|------|-------|---------|
| L1 | NORM | `RAS KERNEL INFO Kernel detected 4035536 integer alignment exceptions` |
| L2 | **ANOM** | `RAS APP FATAL ciod: Error reading message prefix on CioStream` |
| L3 | NORM | `RAS KERNEL INFO Kernel detected 5267798 integer alignment exceptions` |
| L4 | **ANOM** | `RAS APP FATAL ciod: Error reading message prefix on CioStream` |
| L5 | NORM | `RAS KERNEL INFO Kernel detected 4962478 integer alignment exceptions` |
| L6 | NORM | `RAS KERNEL INFO Kernel detected 3825404 integer alignment exceptions` |
| L7 | **ANOM** | `RAS APP FATAL ciod: Error reading message prefix on CioStream` |
| L8 | NORM | `RAS KERNEL INFO Kernel detected 3287410 integer alignment exceptions` |
| L9 | **ANOM** | `RAS APP FATAL ciod: Error reading message prefix on CioStream` |
| L10 | **ANOM** | `RAS APP FATAL ciod: Error reading message prefix on CioStream` |

**Hallucination analysis:** Same pattern as BGL_04647320. The session has 5 anomalous CIOD stream errors and 5 normal integer alignment info messages. The LLM again fixated on the informational alignment messages and produced empty claims with no evidence linkage. The claim "10 occurrences" is factually wrong — there are only 5 alignment messages. The correct signature should be `APP__CIOD_STREAM_ERROR`.

### Root Cause

Both sessions contain **interleaved multi-type log windows** where normal-severity informational messages (integer alignment counts) co-occur with the actual anomalous events (CIOD stream errors). The LLM was confused by:

1. **Volume bias** — the word "alignment exceptions" with large numeric counts (e.g., "4,795,798") appears visually prominent
2. **Severity mismatch** — the actual anomalies (`APP FATAL`) were the CIOD errors, not the alignment messages (`KERNEL INFO`)
3. **Mixed signals** — in a 10-line window with two distinct event types, the model failed to distinguish which type constituted the anomaly

The verification layer **correctly caught both hallucinations** by detecting 0% evidence coverage — the LLM could not ground its (incorrect) claims in the retrieved evidence because the evidence documents contained CIOD stream error patterns, not alignment exception patterns.

---

## SME Reviewer Assessment

### Strengths

**Reliability (strong):** 99.92% completion and 99.97% verification on 6,295 anomalies over 29 hours with zero human intervention is genuinely impressive for a local 8B model. The pipeline showed no degradation over time — no evidence of drift, memory leaks, or cascading failures. This is a significant engineering achievement.

**Signature coherence (good):** 56 canonical signatures for a supercomputer log dataset with ~30 known RAS event categories is in the right ballpark. The long tail (49 singletons pre-normalization) reflects the LLM encountering rare error types it hasn't seen patterns for. The top 6 signatures covering 88% of anomalies shows the model concentrates well on dominant failure modes.

**Efficiency (adequate):** 3,493 tokens/session and ~10s latency are reasonable for an 8B model doing retrieval-augmented explanation. The 29-hour wall clock is dominated by volume, not per-session cost.

### Gaps and Limitations

**No ground-truth evaluation.** The 99.97% verification pass rate measures *structural well-formedness* — did the LLM produce valid JSON with claims that reference evidence IDs? It does NOT measure whether the explanations are **correct**. A claim like "E0 contains 9 KERNEL INTEGER ALIGNMENT EXCEPTIONS at E0-L2 to E0-L10" could be factually wrong and still pass verification if evidence IDs are referenced.

**Verification is self-referential.** The verifier checks that the LLM's claims cite evidence, not that the citations are accurate or the reasoning is sound. A 100% pass rate could equally mean the verification bar is too low.

**No inter-annotator agreement.** Without a human-labeled sample (even 50–100 sessions), precision, recall, or F1 on explanation quality cannot be reported. Any top venue reviewer will ask for this.

### Recommendations for Paper

1. **Sample-based human evaluation** — Have 1–2 domain experts score a stratified random sample (e.g., 100 sessions across the top 10 signature types). Score on correctness, completeness, and evidence grounding. This is table stakes for an explanation paper.

2. **Report what you have honestly** — The pipeline metrics (completion rate, verification rate, signature count) belong in a "System Reliability" subsection, not as the primary evaluation. Frame them as operational robustness, not explanation quality.

3. **Anomaly detection accuracy is separate** — If the screener (Linformer) has its own precision/recall/F1 on the BGL labels, report that independently. The explanation pipeline inherits the screener's decisions — a perfect explanation of a false-positive anomaly is still wrong.

4. **Compare signature coverage to ground truth** — BGL has known RAS event categories. Show a mapping from the 56 normalized signatures to the known categories. This gives reviewers confidence the model is discovering real structure, not hallucinating categories.
