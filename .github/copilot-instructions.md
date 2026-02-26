# Copilot Instructions for agentic-log-explanations

## Language Preferences
- **繁體中文 (Traditional Chinese)** for general discussions, explanations, and daily logs.
- **Keep English** for:
  - Code snippets and variable names
  - Technical terms (e.g., screener, retriever, BM25, gating, margin, coverage)
  - File paths and command outputs
  - Function/class names
  - Mathematical formulas and notation
- **Context restoration**: At the start of each new session, read the most recent daily log in `long-term-mem/` to understand current progress.

## Code Style Rules
- **No emojis** in code, comments, or print/log messages. Use plain text markers like `[OK]`, `[WARN]`, `[RESUME]` instead.
- **Grayscale plots only** — no color. Differentiate lines via linestyle, marker, and gray level. Save at 600 dpi.

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

## Obsidian Integration
- **Vault (WSL path)**: `/mnt/c/Users/dave/Obsidian-repo/My-knowledge-base`
- **Vault (Windows path)**: `C:\Users\dave\Obsidian-repo\My-knowledge-base`
- **CLI wrapper**: `obs` (in `~/bin/`, routes to `Obsidian.com` via PowerShell). Requires Obsidian to be running.
- **Vault-scoped wrapper**: `obsv` (cd's into vault, then calls `obs`)
- **Thesis helper**: `obs-thesis-create <name>` creates notes in the thesis folder
- **Thesis notes**: `10-Projects/thesis_agentic_log_analysis/` — chapters 00-overview through 07-conclusion
- **Templates**: `thesis-chapter` and `thesis-experiment` in `Templates/` folder (Templater auto-triggers on file creation)
- **Agent content markers**: `<!-- agent-generated -->` sections can be updated by the agent; `<!-- user-section -->` must NOT be modified
- **VS Code tasks**: 11 Obsidian tasks in `.vscode/tasks.json` (run via Ctrl+Shift+P → "Tasks: Run Task")
- **Full guide**: `30-Resources/Obsidian CLI + VS Code Copilot Agent Guide.md` in the vault

---
For questions or unclear conventions, review the README or reference the config and pipeline scripts. Suggest improvements or ask for clarification if needed.
