# Human Eval Issues Log

Issues discovered during human evaluation of GPT-5.1 results.
Reviewed against evidence store (normalized) content, not just E0 log lines.

---

## Issue HE-001 — LLM mis-describes Normal session content in contrast claim

**Discovered at:** HDFS IDX=5, session `HDFS_blk_-963887220294303910`
**Signature:** NAMENODE__EXCESS_REPLICATION_INVALIDATION
**Evaluated on:** 2026-03-04

### What happened

Claim 3 (contrast) states:
> "E5, labeled normal, shows only routine 'Receiving block' and 'NameSystem.allocateBlock' INFO lines
> without any 'invalidSet' or 'Deleting block' entries in its span E5-L1 to E5-L30."

E5 = `HDFS_blk_-1549752419809595077`

Actual E5 content (from evidence_store_HDFS.json):
- L20: `FSDataset: Deleting block`
- L21: `WARN DataNode$DataXceiver: Got exception while serving`
- L24-L26: `FSNamesystem: NameSystem.delete: is added to invalidSet`
- L27-L29: `FSDataset: Deleting block` (3 more)
- L30: `STRUCTURAL: EXCESS_REPLICATION HAS_EXCEPTION`

E5 **does** contain invalidSet and Deleting lines — LLM's description is factually incorrect.

### Is E5 correctly a Normal session?

Yes. `anomaly_label_HDFS.csv` labels `blk_-1549752419809595077` as **Normal** (label=0).
The evidence store also has `metadata.label = 0`.

The STRUCTURAL tag `EXCESS_REPLICATION` is a log pattern, not an anomaly label. In the HDFS
dataset, excess replication cleanup (invalidate surplus replicas) is **normal HDFS behavior** and
is labeled Normal. This is not a screener false negative.

### Root cause of the LLM error

The LLM appears to have hallucinated a description of E5's content. Even though E5 is
legitimately a Normal session, it has invalidSet/Deleting lines as part of its normal excess
replication cleanup, and those lines are present in E5-L20 to L29. The LLM's contrast claim
incorrectly states these lines do not exist.

This is a **cross-evidence claim hallucination**: the LLM correctly identified E5 as a Normal
reference but then fabricated a description of what E5 contains rather than checking the actual
line content.

### Impact on scores

| Dimension       | Initial | Revised | Reason |
|-----------------|---------|---------|--------|
| Correctness     | 5       | 3       | Claim 3 factually wrong about E5 content |
| Completeness    | 5       | 5       | E0 anomaly analysis complete |
| Evid. Grounding | 5       | 3       | Claim 3 cites E5-L1 to L30 but describes content that contradicts those lines |
| Actionable      | Y       | Y       | E0 core analysis still actionable |

### Key finding for paper

The automated verifier (9-check system) **cannot detect this class of error** because it only
validates that cited line numbers are in range, not that the LLM's description of those lines
is accurate. Human evaluation is necessary to catch cross-evidence claim hallucinations.

### Action items

- [ ] Consider adding a verifier check: for contrast claims citing external sessions (E1-E5),
  verify that key terms in the claim description (e.g., "no invalidSet") are consistent with
  the actual token distribution of those cited lines.
- [ ] Track frequency of this error type across the full 200-session human eval.
- [ ] If frequency > 5%, document in paper as a known LLM failure mode.

---
