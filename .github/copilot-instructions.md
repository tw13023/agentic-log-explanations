# Copilot Instructions for agentic-log-explanations

## Project Overview
- This repository implements **AllLinLog**, a framework for log-based anomaly detection using linear self-attention (Linformer).
- Main datasets: **BGL** and **HDFS**. Large logs are reconstructed from compressed chunks (see `/logs`).

## Key Components
- `src/`: Core modules for data loading, normalization, evidence storage, LLM client, prompt building, retrieval, screening, signature generation, and verification.
- `pipelines/`: Contains scripts for end-to-end log explanation and anomaly detection workflows.
- `notebooks/`: Jupyter notebooks for pipeline testing, walkthroughs, and complete runs.
- `results/` and `results_HDFS/`: Output metrics, explanations, and JSONL files.
- `configs/config.yaml`: Configuration for pipeline and model parameters.

## Developer Workflows
- **Reconstruct logs**: Use shell commands from README to decompress and merge log files.
- **Run notebooks**: Use Jupyter for interactive pipeline execution. Key notebooks: `allinlog_BGL_inMem_GPT4BPE.ipynb`, `allinlog_HDFS_inMEM_GPT4BPE.ipynb`, and those in `/notebooks`.
- **Model files**: Pretrained models are stored in `/best_model/` and `/best_model_HDFS/`.
- **Config-driven**: Pipelines and scripts read from `configs/config.yaml`.

## Patterns & Conventions
- **Data flow**: Logs → Data loader → Normalizer → Evidence store → LLM client → Prompt builder → Retriever → Screener → Signature generator → Verifier → Results.
- **Modular design**: Each step is a separate module in `src/`.
- **Metrics and explanations**: Output files are JSONL and metrics JSON, organized by dataset and timestamp.
- **Naming**: File and folder names encode dataset, pipeline, and run time for traceability.

## Integration & Dependencies
- **PyTorch** (torch), **Linformer**, **tiktoken**, **LLM API** (via `llm_client.py`), **scikit-learn**, **pandas**, **numpy**.
- **External logs**: Must be reconstructed before use.
- **Config YAML**: Central for parameter tuning and pipeline selection.

## Examples
- To run BGL pipeline: Open `allinlog_BGL_inMem_GPT4BPE.ipynb` and follow cell instructions.
- To run HDFS pipeline: Open `allinlog_HDFS_inMEM_GPT4BPE.ipynb`.
- To add a new dataset: Add log chunks to `/logs`, update config, and create a new pipeline script/notebook.

## References
- See `README.md` for setup, requirements, and log reconstruction commands.
- See `src/` for module patterns and cross-component communication.
- See `pipelines/explain_all.py` for main pipeline script.

---
For questions or unclear conventions, review the README or reference the config and pipeline scripts. Suggest improvements or ask for clarification if needed.
