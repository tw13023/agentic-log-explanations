# Long-Term Memory

This folder stores daily progress logs for the Agentic Log Anomaly Explanation project.

## 🌐 Communication Preferences

**Language**: 繁體中文 (Traditional Chinese) for general discussions, keep English for:
- Code snippets and variable names
- Technical terms (e.g., screener, retriever, BM25)
- File paths and command outputs
- Function/class names

**Context Restoration**: At the start of each new session, read the most recent daily log to understand current progress.

## Purpose

- Track development progress day by day
- Maintain context across sessions
- Document decisions, fixes, and learnings

## Format

Each file is named `YYYY-MM-DD.md` and contains:
- Accomplishments
- Bug fixes
- Technical details
- Next steps
- Session notes

## Index

| Date | Summary |
|------|---------|
| [2026-02-14](2026-02-14.md) | BGL full run post-analysis: normalizer consolidation (116→56 sigs), hallucination investigation (2 confirmed), SME review, normalized JSONL; HDFS full run prepared |
| [2026-02-13](2026-02-13.md) | Phase 1+2 prompt engineering tested & reverted (no benefit for 8B), 7 new BGL normalizer mappings, discriminative scores as paper metadata, 500-session BGL (29→27 sigs) + HDFS (8 sigs) validation |
| [2026-02-12](2026-02-12.md) | BGL+HDFS signature normalizers, prompt alignment, pipeline wiring (71→24 BGL, 26→10 HDFS sigs) |
| [2026-02-10](2026-02-10.md) | Data-driven signature discovery (26 HDFS + 34 BGL patterns, 99.8% coverage) |
| [2026-02-04](2026-02-04.md) | HDFS pipeline fix + dataset-specific prompts |
| [2026-02-03](2026-02-03.md) | Phase 2: Mixed retrieval implementation |
| [2026-02-01](2026-02-01.md) | Phase 1 verification system + 100% pass rate |
| [2026-01-30](2026-01-30.md) | Phase 1 complete, end-to-end pipeline verified |
