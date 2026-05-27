"""
RQ2: RAG Ablation Study
========================
Reproduces the following thesis results (Section 4.5):
  - Table 13: Baseline metrics (RAG-on) for all anomaly sessions
  - Table 14: RAG-on vs no-RAG ablation results

Inputs  (inputs/rq2/):
  bgl_rq2_results.json              -- BGL phase0/phase2 ablation results
  hdfs_rq2_results.json             -- HDFS phase0/phase2 ablation results
  semantic_faithfulness_results.json -- LLM-as-Judge faithfulness scores

Outputs (results/rq2/):
  table13_rag_baseline.txt
  table14_rag_ablation.txt
  rq2_results.json

Usage:
  python pipelines/rq2_rag_ablation.py
"""

import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR    = PROJECT_ROOT / 'inputs' / 'rq2'
OUT_DIR      = PROJECT_ROOT / 'results' / 'rq2'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _write_txt(path: Path, text: str) -> None:
    with open(path, 'w') as f:
        f.write(text)


def _sep(width: int = 72) -> str:
    return '-' * width


def _double_sep(width: int = 72) -> str:
    return '=' * width


# ---------------------------------------------------------------------------
# Table 13: Baseline metrics (RAG-on) for all anomaly sessions
# ---------------------------------------------------------------------------

def build_table13(bgl: dict, hdfs: dict) -> str:
    bp0 = bgl['phase0']
    hp0 = hdfs['phase0']

    bgl_n  = bp0['n_sessions']
    hdfs_n = hp0['n_sessions']

    rows = [
        (
            'Context Precision@4',
            f"{bp0['context_precision_at_4_mean']:.3f} "
            f"(+/- {bp0['context_precision_at_4_std']:.3f})",
            f"{hp0['context_precision_at_4_mean']:.3f} "
            f"(+/- {hp0['context_precision_at_4_std']:.3f})",
        ),
        (
            'Evidence Utilization',
            f"{bp0['evidence_utilization']:.3f}",
            f"{hp0['evidence_utilization']:.3f}",
        ),
    ]

    w_m, w_b, w_h = 24, 24, 24
    header = (f"{'Metric':<{w_m}}  "
              f"{f'BGL (n = {bgl_n:,})':>{w_b}}  "
              f"{f'HDFS (n = {hdfs_n:,})':>{w_h}}")

    lines = [
        'Table 13: Baseline metrics (RAG-on) for all anomaly sessions',
        _double_sep(),
        header,
        _sep(),
    ]
    for metric, bv, hv in rows:
        lines.append(f"{metric:<{w_m}}  {bv:>{w_b}}  {hv:>{w_h}}")
    lines.append(_double_sep())
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Table 14: RAG-on vs no-RAG ablation results
# ---------------------------------------------------------------------------

def build_table14(bgl: dict, hdfs: dict, sem: dict) -> str:
    bp2 = bgl['phase2']
    hp2 = hdfs['phase2']
    bs  = sem['BGL']
    hs  = sem['HDFS']

    # Phantom citations: reported as count in phase2, but Table 14 shows
    # the number of sessions (= n_no_rag for HDFS=245, BGL=388 → but thesis
    # shows BGL 338).  Use no_rag_phantom_citations / 3 claims = sessions
    # with phantom = no_rag_phantom_citations (already per-session count
    # if every session has exactly 1 phantom claim).
    # From notebook: BGL 388 sessions → 388 phantom claims; but thesis
    # shows 338 → that is the no_rag_e0_only_claims / something.
    # Actually from notebook Cell 23: claims_with_phantom_eids = 388 (BGL),
    # 245 (HDFS); but thesis Table 14 shows 338 and 245.
    # BGL: 388 sessions * 3 claims = 1164 total; 388 phantom/1164 = 33.3%
    # 338 = 388 - 50? Or it's the raw phantom count from a prior run.
    # The semantic_faithfulness_results n=388 (BGL), n=245 (HDFS) matches
    # the no_rag sample sizes.  338 for BGL appears to be the phantom count
    # from an earlier notebook run stored in a different field.
    # Check: no_rag_e0_only_claims BGL=776 (= 388*2), HDFS=490 (= 245*2).
    # Best match: use no_rag_phantom_citations directly from the JSON which
    # stores the value that was computed — BGL=388, but thesis=338.
    # The thesis may be from an earlier run (n_no_rag was smaller then).
    # Use the values from the JSON as the definitive source of truth.
    bgl_phantom  = bp2['no_rag_phantom_citations']
    hdfs_phantom = hp2['no_rag_phantom_citations']

    rows = [
        (
            'Faithfulness',
            f"{bs['rag_on']['mean']:.3f}",
            f"{bs['no_rag']['mean']:.3f}",
            f"{hs['rag_on']['mean']:.3f}",
            f"{hs['no_rag']['mean']:.3f}",
        ),
        (
            'Grounding Breadth',
            f"{bp2['rag_on_grounding_breadth']:.3f}",
            f"{bp2['no_rag_grounding_breadth']:.3f}",
            f"{hp2['rag_on_grounding_breadth']:.3f}",
            f"{hp2['no_rag_grounding_breadth']:.3f}",
        ),
        (
            'Claims per session',
            f"{bp2['rag_on_claims_per_session']:.1f}",
            f"{bp2['no_rag_claims_per_session']:.1f}",
            f"{hp2['rag_on_claims_per_session']:.1f}",
            f"{hp2['no_rag_claims_per_session']:.1f}",
        ),
        (
            'Phantom citations',
            '0',
            str(bgl_phantom),
            '0',
            str(hdfs_phantom),
        ),
    ]

    w_m, w_c = 22, 10
    header = (f"{'Metric':<{w_m}}  "
              f"{'BGL':>{w_c}}  {'BGL':>{w_c}}  "
              f"{'HDFS':>{w_c}}  {'HDFS':>{w_c}}")
    sub    = (f"{'': <{w_m}}  "
              f"{'RAG-on':>{w_c}}  {'no-RAG':>{w_c}}  "
              f"{'RAG-on':>{w_c}}  {'no-RAG':>{w_c}}")

    lines = [
        'Table 14: RAG-on vs no-RAG ablation results',
        _double_sep(),
        header,
        sub,
        _sep(),
    ]
    for metric, br, bn, hr, hn in rows:
        lines.append(
            f"{metric:<{w_m}}  "
            f"{br:>{w_c}}  {bn:>{w_c}}  "
            f"{hr:>{w_c}}  {hn:>{w_c}}"
        )
    lines.append(_double_sep())

    # Footnote
    lines.append(
        f"\nNote: Faithfulness = semantic faithfulness (LLM-as-Judge, "
        f"{sem['judge_model']}, n_BGL={bs['rag_on']['n']}, "
        f"n_HDFS={hs['rag_on']['n']})."
    )
    lines.append(
        "      Grounding Breadth = fraction of claims citing external "
        "evidence (E1-E4)."
    )
    lines.append(
        "      Phantom citations = claims citing evidence IDs absent from "
        "the no-RAG prompt."
    )
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# rq2_results.json
# ---------------------------------------------------------------------------

def build_rq2_results(bgl: dict, hdfs: dict, sem: dict) -> dict:
    return {
        'rq':    'RQ2',
        'title': 'RAG Ablation Study',
        'bgl_run':  bgl['timestamp'],
        'hdfs_run': hdfs['timestamp'],
        'table13': {
            'BGL': {
                'n_sessions':               bgl['phase0']['n_sessions'],
                'context_precision_at_4':   round(bgl['phase0']['context_precision_at_4_mean'], 4),
                'context_precision_at_4_std': round(bgl['phase0']['context_precision_at_4_std'], 4),
                'evidence_utilization':     round(bgl['phase0']['evidence_utilization'], 4),
            },
            'HDFS': {
                'n_sessions':               hdfs['phase0']['n_sessions'],
                'context_precision_at_4':   round(hdfs['phase0']['context_precision_at_4_mean'], 4),
                'context_precision_at_4_std': round(hdfs['phase0']['context_precision_at_4_std'], 4),
                'evidence_utilization':     round(hdfs['phase0']['evidence_utilization'], 4),
            },
        },
        'table14': {
            'judge_model': sem['judge_model'],
            'BGL': {
                'rag_on_faithfulness':   round(sem['BGL']['rag_on']['mean'], 4),
                'no_rag_faithfulness':   round(sem['BGL']['no_rag']['mean'], 4),
                'rag_on_grounding_breadth': round(bgl['phase2']['rag_on_grounding_breadth'], 4),
                'no_rag_grounding_breadth': round(bgl['phase2']['no_rag_grounding_breadth'], 4),
                'rag_on_claims_per_session': bgl['phase2']['rag_on_claims_per_session'],
                'no_rag_claims_per_session': bgl['phase2']['no_rag_claims_per_session'],
                'no_rag_phantom_citations':  bgl['phase2']['no_rag_phantom_citations'],
                'n_ablation': sem['BGL']['rag_on']['n'],
            },
            'HDFS': {
                'rag_on_faithfulness':   round(sem['HDFS']['rag_on']['mean'], 4),
                'no_rag_faithfulness':   round(sem['HDFS']['no_rag']['mean'], 4),
                'rag_on_grounding_breadth': round(hdfs['phase2']['rag_on_grounding_breadth'], 4),
                'no_rag_grounding_breadth': round(hdfs['phase2']['no_rag_grounding_breadth'], 4),
                'rag_on_claims_per_session': hdfs['phase2']['rag_on_claims_per_session'],
                'no_rag_claims_per_session': hdfs['phase2']['no_rag_claims_per_session'],
                'no_rag_phantom_citations':  hdfs['phase2']['no_rag_phantom_citations'],
                'n_ablation': sem['HDFS']['rag_on']['n'],
            },
        },
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    missing = [p for p in [
        INPUT_DIR / 'bgl_rq2_results.json',
        INPUT_DIR / 'hdfs_rq2_results.json',
        INPUT_DIR / 'semantic_faithfulness_results.json',
    ] if not p.exists()]
    if missing:
        print('[ERROR] Missing input files:')
        for p in missing:
            print(f'  {p}')
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    bgl = _load_json(INPUT_DIR / 'bgl_rq2_results.json')
    hdfs = _load_json(INPUT_DIR / 'hdfs_rq2_results.json')
    sem  = _load_json(INPUT_DIR / 'semantic_faithfulness_results.json')

    # Table 13
    t13 = build_table13(bgl, hdfs)
    print('\n' + t13)
    _write_txt(OUT_DIR / 'table13_rag_baseline.txt', t13 + '\n')
    print('[OK] Saved table13_rag_baseline.txt')

    # Table 14
    t14 = build_table14(bgl, hdfs, sem)
    print('\n' + t14)
    _write_txt(OUT_DIR / 'table14_rag_ablation.txt', t14 + '\n')
    print('[OK] Saved table14_rag_ablation.txt')

    # JSON summary
    results = build_rq2_results(bgl, hdfs, sem)
    out_json = OUT_DIR / 'rq2_results.json'
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print('[OK] Saved rq2_results.json')

    print('\n[OK] All outputs saved to results/rq2/')


if __name__ == '__main__':
    main()
