# Agentic Log-based Anomaly Detection（Screener–Reasoner）研究大綱與架構圖

> 目標：在 **維持 AllLinLog 高偵測效能** 的前提下，提供 **evidence-grounded（有證據鏈）** 的 **可追溯解釋**，並透過 gating 進一步做 **成本可控（cost-aware）** 的進階實驗。

---

## 一、研究動機與定位（Introduction Arguments）

### Problem Context
Modern large-scale computing systems generate massive volumes of runtime logs that serve as the primary diagnostic record for system health and failure events. Automated log-based anomaly detection has emerged as a critical component of AIOps pipelines, enabling operators to surface abnormal behavior without manual inspection of millions of log lines. Recent neural approaches, including log-parsing-free and self-attention-based methods, have pushed detection accuracy to near-ceiling levels on established benchmarks: AllLinLog, a linear self-attention screener, achieves F1 ≈ 0.999 on BGL and F1 ≈ 0.997 on HDFS, leaving little room for further improvement in binary classification performance.

### The Explainability Gap
Despite these advances, state-of-the-art anomaly detectors remain fundamentally opaque: they identify *that* an anomaly occurred, but provide no account of *why* it occurred, *which* log evidence supports the decision, or *how* the incident relates to previously observed failure patterns. In operational practice, this opacity forces site reliability engineers to re-examine raw logs manually after every alert, negating much of the productivity gain promised by automated detection. Furthermore, without a traceable justification, anomaly alerts are difficult to audit, challenge in incident post-mortems, or feed into downstream root-cause analysis pipelines. As detection accuracy approaches saturation, the central bottleneck shifts from *Can the system detect anomalies?* to *Can the system explain them in a trustworthy and actionable way?*

### Limitations of Existing Explanation Approaches
Prior work on log explanation either relies on template-based summarization—which lacks semantic depth and fails on novel log patterns—or applies general-purpose LLMs without grounding outputs in concrete log evidence, making explanations susceptible to hallucination and unverifiable in production settings. Neither line of work provides a mechanism to (i) anchor every explanatory claim to specific, retrievable log evidence, (ii) enforce a structured, machine-verifiable output format, or (iii) operate within a controllable inference cost envelope.

### Our Approach and Positioning
This paper proposes the **Screener–Reasoner** framework, a two-stage hybrid pipeline that decouples anomaly *detection* from anomaly *explanation*. The Screener (AllLinLog) performs efficient, high-accuracy session-level classification across all incoming log sessions. The Reasoner, an LLM augmented with retrieval from a training-set evidence store, is invoked selectively on flagged anomalies to produce structured, evidence-grounded explanations in which every claim cites verifiable evidence spans. A confidence-based gating mechanism further controls when the Reasoner is engaged, enabling cost-quality trade-off analysis under realistic operational budgets.

The goal of this work is therefore **not** to improve detection F1—a metric already near saturation—but to establish that anomaly detection systems can be augmented to produce traceable, verifiable explanations at scale, and to quantify the conditions under which such explanations are faithful, cost-effective, and practically deployable.

---

## 二、研究問題（Research Questions）

- **RQ1（Traceable Explanation Quality）**: To what extent can the Screener–Reasoner framework produce traceable, evidence-grounded explanations for anomalous log sessions, as measured by structured output compliance, evidence citation coverage, and human-evaluated explanation quality?

- **RQ2（RAG Contribution to Explanation Faithfulness）**: How does retrieval-augmented evidence grounding affect the faithfulness and groundedness of LLM-generated anomaly explanations compared to a no-retrieval baseline, and to what degree does structured output enforcement reduce unsupported claims?

- **RQ3（Cost-Quality Trade-off under Confidence-Based Gating）**: Under a fixed LLM inference budget, how do confidence-based gating strategies affect the trade-off between explanation coverage, per-session quality, and operational cost, as measured by trigger rate, token consumption, and explanation quality metrics?

---

## 三、整體架構：Screener–Reasoner（Hybrid Agentic Explanation）
### 3.1 Screener（AllLinLog）
- 對所有 log sessions 做快速偵測  
- 輸出：normal/anomaly logits（與預測 label）

### 3.2 Reasoner（LLM 用於解釋，不做偵測）
- 僅對被選中的異常 sessions 啟動
- 目標：產出 **可追溯解釋**（每個 claim 都引用 evidence id）

### 3.3 Gating（成本/延遲控制器）
- 目的：決定哪些 anomalies 值得花 LLM 成本做解釋
- 實驗設計：
  - **Baseline：Explain-All**（所有 anomalous sessions 都解釋）
  - **進階：Budgeted Explain（margin gating）**（在預算內優先解釋低信心 anomalies）

---

## 四、RAG（Evidence Store 的建立與使用）
### 4.1 Evidence Corpus 的來源（可重現、可過審）
- 使用 **train split 的所有 sessions（normal + anomaly）** 建立 evidence store  
  - **不解釋 normal**，但 **保留 normal 作為對照 evidence**，提升 anomaly 解釋說服力
  - 避免只用 anomaly evidence 造成偏差（bias）

### 4.2 Retrieval（Top-k）
- 對每個要解釋的 anomaly session，取回 top-k 相似 evidence（k=5 作為起始）
- 用途：作為 LLM 解釋時的「可引用證據」

---

## 五、Trace Schema（可追溯解釋的輸出格式）
- LLM 必須輸出結構化格式（例如 JSON）：
  - `prediction`（可引用 Screener 結果）
  - `claims`（每個 claim 必須附 `evidence_ids`）
  - `optional: insufficient_evidence`

---

## 六、Baseline 與進階實驗設計
### 6.1 Baseline：Explain-All + RAG + Trace Schema
- 觸發：predicted anomaly → 一律解釋
- 目的：先做出穩定的可追溯解釋能力與品質評估

### 6.2 進階：Budgeted Explain（Cost-aware agentic explanation）
- 固定 LLM 預算（最多解釋前 K% anomalies 或每天最多 M 次）
- gating：依 logits margin（低信心優先）選擇要解釋的 anomalies
- 目的：展示「成本–品質」trade-off

---

## 七、評估指標（不只 F1）

### 7.1 Detection Performance（addresses RQ1 baseline）
- F1 / Precision / Recall（Screener, reported on BGL and HDFS test sets）

### 7.2 Explanation Quality（addresses RQ1 & RQ2）
- **Structured output compliance rate**: percentage of sessions with valid JSON output matching the trace schema
- **Verification pass rate**: percentage of explanations passing all automated verifier checks (evidence span validity, claim grounding, signature consistency)
- **Evidence citation coverage**: fraction of claims that cite at least one valid evidence ID
- **Human evaluation score**: Likert-scale ratings for correctness, faithfulness, and usefulness (sampled subset)
- **RAG ablation — faithfulness delta**: difference in verification pass rate and citation coverage between RAG-on and no-retrieval conditions (addresses RQ2)

### 7.3 Cost & Latency（addresses RQ3）
- Trigger rate: fraction of sessions that receive LLM explanation under a given gating budget
- Token consumption: total and average tokens per explained session
- Avg / P95 latency per session
- Cost-quality Pareto curve: explanation quality vs. token budget across gating thresholds
- Unique signature yield: number of distinct anomaly signatures discovered at each budget level

---

## 八、論文貢獻點（Contributions）

1. **Screener–Reasoner framework** — We propose a hybrid pipeline that decouples anomaly *detection* (AllLinLog linear self-attention screener) from anomaly *explanation* (LLM reasoner), preserving near-perfect detection performance (BGL F1 ≈ 0.999, HDFS F1 ≈ 0.997) while enabling post-hoc traceable explanations. *(answers RQ1)*

2. **Evidence-grounded explanation with automated verification** — We introduce a RAG-backed trace schema that requires every LLM-generated claim to cite specific evidence IDs, and an automated multi-check verifier that enforces grounding at scale. Ablation experiments quantify the faithfulness gain of retrieval augmentation over a no-retrieval baseline. *(answers RQ2)*

3. **Cost-aware gating and cost-quality trade-off analysis** — We evaluate a confidence-based gating strategy (logit-margin gating) against an explain-all baseline, producing a cost-quality Pareto analysis that guides practitioners in balancing operational LLM cost against explanation coverage and quality. *(answers RQ3)*

4. **Large-scale empirical evaluation** — We report full-dataset explanation results on BGL (6,295 anomaly sessions) and HDFS (full test set), including token cost, latency distribution, and a taxonomy of 360+ automatically discovered anomaly signatures.

---

## 九、架構圖
> 圖檔另附（下載連結如下）。

![Screener–Reasoner Architecture](A_flowchart_diagram_illustrates_a_hybrid_AI_system.png)



---
