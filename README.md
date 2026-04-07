
# Explainable Log-Based Anomaly Detection with Linear Self-Attention

This repository implements **Screener-Reasoner**, a framework for log-based anomaly detection using linear self-attention (Linformer) producing evidence-grounded, traceable explanations for detected anomalies.

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
│   ├── gating.py                    # Gating logic (explain_all / top_k modes)
│   ├── config_loader.py             # YAML config loader and validation
│   └── verifier.py                  # 8-check faithfulness verification
├── pipelines/
│   ├── explain_all.py               # End-to-end Explain-All pipeline (CLI + API)
│   └── auto_evaluator.py            # Automated evaluation of explanation outputs
├── notebooks/
│   ├── 01_pipeline_test.ipynb       # Initial component testing
│   ├── 02_pipeline_walkthrough.ipynb # Step-by-step interactive walkthrough
│   ├── 03_pipeline_BGL.ipynb        # Complete BGL pipeline run
│   ├── 04_pipeline_HDFS.ipynb       # Complete HDFS pipeline run
│   ├── 05_signature_audit.ipynb     # Signature card review and audit
│   ├── 06_full_run.ipynb            # Full dataset pipeline execution
│   ├── 07_gating_analysis.ipynb     # Gating mode analysis
│   ├── 08_end_to_end_gating.ipynb   # End-to-end gating simulation
│   ├── 09_human_evaluation.ipynb    # Human evaluation of explanations
│   ├── 10_gpt51_explanation_audit.ipynb  # GPT-5.1 explanation audit
│   ├── 11_rq2_rag_ablation.ipynb    # RQ2: RAG ablation study
│   ├── 12_rq3_cost_quality_gating.ipynb  # RQ3: Cost-quality gating analysis
│   ├── 13_rq1_explanation_quality.ipynb  # RQ1: Explanation quality metrics
│   ├── 14_BGL_screener.ipynb        # BGL screener inference testing
│   └── 15_HDFS_screener.ipynb       # HDFS screener inference testing
├── allinlog_BGL_inMem_GPT4BPE.ipynb # BGL model training notebook
├── allinlog_HDFS_inMEM_GPT4BPE.ipynb # HDFS model training notebook
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

### 2. Create a virtual environment

**Python 3.12 is required.**

```bash
python3.12 -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

> The `--extra-index-url` flag is required for PyTorch CUDA 12.8 builds (`torch==2.7.1+cu128`). Without it, pip cannot find this version on PyPI.

**Key dependencies:** PyTorch (CUDA 12.8), Linformer, tiktoken, rank-bm25, scikit-learn, pandas, requests, python-dotenv, PyYAML.

### 3. Prepare log files

**BGL:**

`BGL.log` is not included in this repository (too large for GitHub and not redistributable). Download it from the [LogHub dataset collection](https://github.com/logpai/loghub) and place it at `logs/BGL.log`.

**HDFS:**

The HDFS log is split into compressed chunks and included in the repository. Reconstruct it with:

```bash
cat logs/HDFS_part_*.gz | gunzip > logs/HDFS.log
```

### 4. Run the explanation pipeline

**Option A — Jupyter Notebook (recommended for exploration):**

| Dataset | Notebook |
|---------|----------|
| BGL     | `notebooks/03_pipeline_BGL.ipynb` |
| HDFS    | `notebooks/04_pipeline_HDFS.ipynb` |

**Option B — Command line:**

```bash
# Run with default config (configs/config.yaml)
python pipelines/explain_all.py --dataset BGL

# Limit to a subset of sessions for a quick test
python pipelines/explain_all.py --dataset BGL --max-sessions 100

# Use a custom config file
python pipelines/explain_all.py --dataset HDFS --config path/to/config.yaml
```

> LLM provider and model are configured in `configs/config.yaml` (`llm.provider`, `llm.model`).

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
| **Gating** | Selects which predicted anomalies to explain. Mode `explain_all`: all anomalies; mode `top_k`: budget-constrained by screener uncertainty score. |
| **ConfigLoader** | Loads and validates `configs/config.yaml`; centralizes all pipeline parameters. |
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

Manual evaluation of 50 sampled sessions per dataset on four dimensions: Correctness, Completeness, Evidence Grounding (Likert 1–5), and Actionable (Y/N). Stratified by signature to ensure coverage. Evaluated in `notebooks/09_human_evaluation.ipynb`.

**HDFS (50/50 complete):**

| Dimension | Mean | Std |
|-----------|------|-----|
| Correctness | 4.92 | 0.27 |
| Completeness | 4.96 | 0.20 |
| Evidence Grounding | 4.64 | 0.52 |
| Actionable | 100% | — |



**BGL (50/50 complete):**

| Dimension | Mean | Std |
|-----------|------|-----|
| Correctness | 4.94 | 0.24 |
| Completeness | 5.00 | 0.00 |
| Evidence Grounding | 4.80 | 0.40 |
| Actionable | 100% | — |


**Overall (100 sessions):**

| Dataset | n | Correctness | Completeness | Evid. Grounding | Actionable |
|---------|:---:|:---:|:---:|:---:|:---:|
| BGL | 50 | 4.94 ± 0.24 | 5.00 ± 0.00 | 4.80 ± 0.40 | 100% |
| HDFS | 50 | 4.92 ± 0.27 | 4.96 ± 0.20 | 4.64 ± 0.52 | 100% |
| Overall | 100 | 4.93 ± 0.25 | 4.98 ± 0.14 | 4.72 ± 0.49 | 100% |

---

### BGL Full Pipeline Run (2026-03-13)

| Metric | Value |
|--------|-------|
| Test sessions | 71,221 |
| Predicted anomalies | 5,850 (5,840 TP, 10 FP) |
| Explanations generated | 5,850 / 5,850 (100%) |
| JSON parse success | 100% |
| Verification pass rate | 100% (5,850 / 5,850) |
| Avg tokens / explanation | ~6,496 |
| Avg latency / explanation | ~7.6s |
| Total wall time | ~12.4 hours |
| LLM | GPT-5.1 (OpenAI) |

---

## Configuration

All settings are centralized in `configs/config.yaml`:

- **Dataset paths** and model paths
- **Model hyperparameters** (vocab_size, embedding_dim, layers, heads, Linformer k)
- **RAG settings** (retriever type, top_k)
- **LLM settings** (provider, model, temperature, timeout)
- **Gating mode** — `explain_all` (all anomalies explained) or `top_k` (budget-constrained by screener uncertainty score)
- **Output settings** (results directory, save format)

---

## LLM Requirements

The explanation pipeline requires access to an LLM. Default: **OpenAI GPT-5.1** (set `llm.provider: "openai"` in config and provide an API key in a `.env` file).

```bash
# .env file
OPENAI_API_KEY=your-key-here
```

**Ollama (local alternative):** Set `llm.provider: "ollama"` and `llm.model: "llama3.1:8b"` in config.

```bash
# Install Ollama (https://ollama.ai)
ollama pull llama3.1:8b
ollama serve
```

---


## License

See repository for license information.

---

## Citation

If you use this work, please cite:

> Evidence-Grounded Explanations for Log-Based Anomaly Detection: A Screener–Reasoner Framework with Automated Evidence Verification

