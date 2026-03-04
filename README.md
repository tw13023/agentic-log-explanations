
# AllLinLog — Explainable Log-Based Anomaly Detection with Linear Self-Attention

This repository implements **AllLinLog**, a framework for log-based anomaly detection using linear self-attention (Linformer), extended with an agentic **Screener–Reasoner** pipeline that produces evidence-grounded, traceable explanations for detected anomalies.

The AllLinLog screener achieves near-perfect detection (BGL F1 ≈ 0.999, HDFS F1 ≈ 0.997). The research focus of this project is producing **structured, verifiable explanations** via RAG-augmented LLM reasoning — not improving detection accuracy.

---

## Architecture

```
Raw Logs → DataLoader (Session objects)
                ↓
  ┌─────────────┴──────────────┐
  │  Train split               │  Test split
  │     ↓                      │     ↓
  │  EvidenceStore             │  Screener (AllLinLog)
  │  (normalized docs)         │     ↓
  │     ↓                      │  Predicted anomalies
  │  SignatureGenerator         │     ↓
  │  (error pattern cards)     │  Retriever (BM25)
  │     ↓                      │  ← queries evidence store
  │  BM25 Index                │     ↓
  └────────────────────────────│  PromptBuilder
                               │  (session + evidence → prompt)
                               │     ↓
                               │  LLMClient (Ollama / OpenAI)
                               │     ↓
                               │  TraceExplanation (JSON)
                               │     ↓
                               │  Verifier (rule-based faithfulness)
                               │     ↓
                               │  Results (JSONL + metrics JSON)
```

**Key design principles:**
- **No data leakage** — evidence store uses train split only; labels are metadata, never shown to the LLM.
- **Forensic scope** — the system produces pattern-matching and contrast observations, not root cause analysis.
- **Mixed retrieval** — 4 anomaly + 1 normal evidence per query, enabling observation, pattern-match, and contrast claims.
- **Config-driven** — `configs/config.yaml` centralizes all parameters.

---

## Project Structure

```
├── configs/
│   └── config.yaml                  # Central configuration
├── src/                             # Core modules
│   ├── data_loader.py               # Session loaders (BGL sliding-window, HDFS block-id)
│   ├── normalizer.py                # Log normalization for RAG (IPs, hex, paths → placeholders)
│   ├── screener.py                  # AllLinLog model + GPT-4 BPE tokenizer wrapper
│   ├── evidence_store.py            # RAG corpus builder from training sessions
│   ├── signature_generator.py       # Error signature cards (pattern clustering)
│   ├── retriever.py                 # BM25 evidence retrieval with mixed-mode support
│   ├── prompt_builder.py            # Prompt assembly + TraceExplanation schema
│   ├── llm_client.py                # Unified LLM client (Ollama / OpenAI compatible)
│   └── verifier.py                  # 8-check faithfulness verification
├── pipelines/
│   └── explain_all.py               # End-to-end Explain-All pipeline (CLI + API)
├── notebooks/
│   ├── 01_pipeline_test.ipynb       # Initial component testing
│   ├── 02_pipeline_walkthrough.ipynb # Step-by-step interactive walkthrough
│   ├── 03_pipeline_complete.ipynb   # Complete BGL pipeline run
│   └── 04_pipeline_HDFS.ipynb       # Complete HDFS pipeline run
├── allinlog_BGL_inMem_GPT4BPE.ipynb # BGL model training notebook
├── allinlog_HDFS_inMEM_GPT4BPE.ipynb # HDFS model training notebook
├── BGL_screener.ipynb               # BGL screener inference testing
├── HDFS_screener.ipynb              # HDFS screener inference testing
├── best_model/                      # Pretrained BGL model
├── best_model_HDFS/                 # Pretrained HDFS model
├── logs/                            # Log datasets (compressed; see below)
├── results/                         # BGL pipeline outputs (JSONL + metrics)
├── results_HDFS/                    # HDFS pipeline outputs
└── long-term-mem/                   # Development journal / decision log
```

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/tw13023/agentic-log-explanations.git
cd agentic-log-explanations
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

**Key dependencies:** PyTorch (CUDA 12.8), Linformer, tiktoken, rank-bm25, scikit-learn, pandas, requests, PyYAML.

### 3. Reconstruct large log files

The original logs are too large for GitHub. They are compressed and (for HDFS) split into chunks.

**BGL:**
```bash
cd logs
gunzip BGL.log.gz
```

**HDFS:**
```bash
cat logs/HDFS_part_*.gz | gunzip > logs/HDFS.log
```

### 4. Run the explanation pipeline

**Option A — Jupyter Notebook (recommended for exploration):**

| Dataset | Notebook |
|---------|----------|
| BGL     | `notebooks/03_pipeline_complete.ipynb` |
| HDFS    | `notebooks/04_pipeline_HDFS.ipynb` |

**Option B — Command line:**

```bash
python pipelines/explain_all.py --dataset BGL --max-sessions 100 --llm-model llama3.1:8b
```

### 5. Train models from scratch (optional)

- `allinlog_BGL_inMem_GPT4BPE.ipynb` — BGL model training
- `allinlog_HDFS_inMEM_GPT4BPE.ipynb` — HDFS model training

Pretrained models are included in `best_model/` and `best_model_HDFS/`.

---

## Datasets

| Dataset | Source | Session Strategy | Log Lines | Test Sessions | Test Anomaly Rate |
|---------|--------|------------------|-----------|---------------|-------------------|
| **BGL** | Blue Gene/L supercomputer | Sliding window (w=10, s=10) | ~4.7M | ~71K | ~8.2% |
| **HDFS** | Hadoop Distributed File System | Group by `block_id` | Large | ~86K | ~3.4% |

---

## Core Modules

| Module | Purpose |
|--------|---------|
| **DataLoader** | Loads raw logs into `Session` objects. BGL uses sliding windows; HDFS groups by block ID. Stratified train/val/test split. |
| **Normalizer** | Replaces dynamic values (IPs, hex, paths, timestamps, block IDs) with placeholders for pattern-level RAG matching. Dataset-specific patterns. |
| **Screener** | AllLinLog model wrapper — Linformer encoder with GPT-4 BPE tokenization. Returns predictions, probabilities, and confidence margins. |
| **EvidenceStore** | Builds RAG corpus from train-split sessions and signature cards. Serializable to JSON. |
| **SignatureGenerator** | Clusters training anomalies into named error patterns (9 predefined BGL patterns). Generates signature cards injected into the evidence store. |
| **Retriever** | BM25-based retrieval with mixed mode (anomaly + normal exemplars). Supports batch processing. |
| **PromptBuilder** | Assembles LLM prompts with session content, retrieved evidence, and dataset-specific instructions. Defines the `TraceExplanation` JSON schema. |
| **LLMClient** | Unified client for Ollama (local) and OpenAI. Tracks token usage, latency, and cost. |
| **Verifier** | 8-check faithfulness verification: structure, evidence ID validity, coverage (≥80%), keyword matching, span validity, signature format, and more. |

---

## Explanation Schema

Each anomaly explanation is a structured JSON trace:

```json
{
  "prediction": "anomaly",
  "summary": "Memory parity error detected on node R00-M0-N0...",
  "signature": "RAS_KERNEL_FATAL__DATA_STORAGE_INTERRUPT",
  "claims": [
    {
      "type": "observation",
      "text": "Line 8 shows a data storage interrupt...",
      "evidence_spans": ["E0-L8"]
    },
    {
      "type": "pattern_match",
      "text": "This matches the memory parity pattern in E1...",
      "evidence_spans": ["E1-L3", "E1-L7"]
    },
    {
      "type": "contrast",
      "text": "Normal session E5 shows no parity errors...",
      "evidence_spans": ["E5-L1"]
    }
  ]
}
```

**Conventions:**
- `[E0]` = the query session under analysis (test set); `[E1]`–`[E5]` = retrieved evidence (train set).
- Signature format: `COMPONENT_SEVERITY__ERROR_TYPE` (double underscore), dataset-specific.
- Three claim types: `observation` (from E0), `pattern_match` (matches exemplars), `contrast` (differs from normal).

---

## Results

### Human Evaluation — GPT-5.1 (2026-03-04)

Manual evaluation of 100 sampled sessions per dataset on four dimensions: Correctness, Completeness, Evidence Grounding (Likert 1–5), and Actionable (Y/N). Stratified by signature to ensure coverage. Evaluated in `notebooks/09_human_evaluation.ipynb`.

**HDFS (100/100 complete):**

| Dimension | Mean | Std | Distribution |
|-----------|------|-----|--------------|
| Correctness | 4.99 | 0.10 | 4:1, 5:99 |
| Completeness | 4.99 | 0.10 | 4:1, 5:99 |
| Evidence Grounding | 4.04 | 0.45 | 3:8, 4:80, 5:12 |
| Actionable | 100% | — | Y:100 |

Evidence grounding scores were slightly lower than correctness and completeness. The gap arises because each reference evidence document carries a compact outcome label (e.g., `exceptions=0`, `NORMAL_FLOW`) but does not enumerate which log operations occurred. The model occasionally treated a "normal outcome" label as proof that certain operations were absent from the reference session, whereas those operations were actually present — they simply completed without error. This led to contrast claims that were directionally correct but overstated at the event level.

**BGL (0/100 — in progress):** Evaluation begins 2026-03-05.

---

### BGL Full Pipeline Run (2026-02-02)

| Metric | Value |
|--------|-------|
| Test sessions | 71,221 |
| Predicted anomalies | 5,849 (5,844 TP, 5 FP) |
| Explanations generated | 5,849 / 5,849 (100%) |
| JSON parse success | 100% |
| Verification pass rate | 99.7% (5,830 / 5,849) |
| Avg tokens / explanation | ~2,654 |
| Avg latency / explanation | ~5.8s |
| Total wall time | ~9.5 hours |
| LLM | Llama 3.1:8b (Ollama, local) |

After implementing mixed retrieval (4 anomaly + 1 normal), all 19 verification failures were resolved → **100% pass rate**.

---

## Configuration

All settings are centralized in `configs/config.yaml`:

- **Dataset paths** and model paths
- **Model hyperparameters** (vocab_size, embedding_dim, layers, heads, Linformer k)
- **RAG settings** (retriever type, top_k)
- **LLM settings** (provider, model, temperature, timeout)
- **Gating mode** — `explain_all` (implemented) or `budgeted` (future: margin-based top-k%)
- **Output settings** (results directory, save format)

---

## LLM Requirements

The explanation pipeline requires a running LLM server. Default: **Ollama** with Llama 3.1:8b.

```bash
# Install Ollama (https://ollama.ai)
ollama pull llama3.1:8b
ollama serve
```

OpenAI is also supported — set `llm.provider: "openai"` in config and provide an API key in a `.env` file.

---


## License

See repository for license information.

---

## Citation

If you use this work, please cite the xxxxxx paper / repository.

