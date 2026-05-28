"""
RQ1: Traceable Explanation Quality
===================================
Reproduces the following thesis results (Section 4.4):
  - Table 10: Reasoner pipeline results on BGL and HDFS
  - Table 11: Human evaluation results (5-point scale, n=50 per dataset)
  - Table 12: Top 5 frequent anomaly signatures per dataset
  - Figure 14: Rank-frequency distribution of anomaly signatures (log scale)

Inputs  (inputs/rq1/):
  bgl_metrics.json        -- BGL full-run pipeline metrics (2026-03-13)
  hdfs_metrics.json       -- HDFS full-run pipeline metrics (2026-03-11)
  human_eval_ratings.json -- 100 human-rated explanations (50 BGL + 50 HDFS)

Outputs (results/rq1/):
  table10_pipeline_summary.txt
  table11_human_eval.txt
  table12_top_signatures.txt
  figure14_signature_zipf.png
  rq1_results.json

Usage:
  python pipelines/rq1_explanation_quality.py
"""

import json
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR    = PROJECT_ROOT / 'inputs' / 'rq1'
OUT_DIR      = PROJECT_ROOT / 'results' / 'rq1'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict:
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def _write_txt(path: Path, text: str) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)


def _separator(width: int = 68) -> str:
    return '-' * width


def _double_sep(width: int = 68) -> str:
    return '=' * width


# ---------------------------------------------------------------------------
# Table 10: Reasoner pipeline results on BGL and HDFS
# ---------------------------------------------------------------------------

def build_table10(bgl: dict, hdfs: dict) -> str:
    def n_sigs(m):
        return len(m['signatures'])

    rows = [
        ('Anomalies Explained',
         f"{bgl['counts']['total_anomalies']:,}",
         f"{hdfs['counts']['total_anomalies']:,}"),
        ('Verification pass rate',
         f"{bgl['verification']['pass_rate']:.0%}",
         f"{hdfs['verification']['pass_rate']:.0%}"),
        ('Unique signatures',
         str(n_sigs(bgl)),
         str(n_sigs(hdfs))),
        ('Avg tokens/session',
         f"{round(bgl['tokens']['avg']):,}",
         f"{round(hdfs['tokens']['avg']):,}"),
        ('Mean latency (seconds)',
         f"{bgl['latency']['avg_ms'] / 1000:.1f}",
         f"{hdfs['latency']['avg_ms'] / 1000:.1f}"),
        ('P95 latency (seconds)',
         f"{bgl['latency']['p95_ms'] / 1000:.1f}",
         f"{hdfs['latency']['p95_ms'] / 1000:.1f}"),
    ]

    w_metric, w_bgl, w_hdfs = 28, 10, 10
    header = f"{'Metric':<{w_metric}}  {'BGL':>{w_bgl}}  {'HDFS':>{w_hdfs}}"
    lines = [
        'Table 10: Reasoner pipeline results on BGL and HDFS',
        _double_sep(),
        header,
        _separator(),
    ]
    for metric, bv, hv in rows:
        lines.append(f"{metric:<{w_metric}}  {bv:>{w_bgl}}  {hv:>{w_hdfs}}")
    lines.append(_double_sep())
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Table 11: Human evaluation results
# ---------------------------------------------------------------------------

def build_table11(human_eval: dict) -> str:
    ratings = human_eval['ratings']

    bgl_rows  = [v for v in ratings.values() if v['dataset'] == 'BGL']
    hdfs_rows = [v for v in ratings.values() if v['dataset'] == 'HDFS']

    def stats(rows, dim):
        vals = [r[dim] for r in rows]
        return statistics.mean(vals), statistics.stdev(vals)

    dims = [
        ('Correctness',       'correctness'),
        ('Completeness',      'completeness'),
        ('Evidence grounding','evidence_grounding'),
    ]

    bgl_act  = sum(1 for r in bgl_rows  if r['actionable'] == 'Y') / len(bgl_rows)  * 100
    hdfs_act = sum(1 for r in hdfs_rows if r['actionable'] == 'Y') / len(hdfs_rows) * 100

    n_bgl, n_hdfs = len(bgl_rows), len(hdfs_rows)

    w_dim, w_col = 22, 22
    header = (f"{'Dimension':<{w_dim}}  "
              f"{'BGL (mean +/- std)':>{w_col}}  "
              f"{'HDFS (mean +/- std)':>{w_col}}")
    lines = [
        f'Table 11: Human evaluation results (5-point scale, '
        f'n = {n_bgl} per dataset)',
        _double_sep(),
        header,
        _separator(),
    ]
    for label, key in dims:
        bm, bs = stats(bgl_rows, key)
        hm, hs = stats(hdfs_rows, key)
        bv = f"{bm:.2f} +/- {bs:.2f}"
        hv = f"{hm:.2f} +/- {hs:.2f}"
        lines.append(f"{label:<{w_dim}}  {bv:>{w_col}}  {hv:>{w_col}}")
    lines.append(_separator())
    lines.append(f"{'Actionable (Y)':<{w_dim}}  "
                 f"{f'{bgl_act:.0f}%':>{w_col}}  "
                 f"{f'{hdfs_act:.0f}%':>{w_col}}")
    lines.append(_double_sep())
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Table 12: Top 5 frequent anomaly signatures per dataset
# ---------------------------------------------------------------------------

def build_table12(bgl: dict, hdfs: dict) -> str:
    bgl_top5  = sorted(bgl['signatures'].items(),  key=lambda x: -x[1])[:5]
    hdfs_top5 = sorted(hdfs['signatures'].items(), key=lambda x: -x[1])[:5]

    w_rank, w_sig, w_cnt = 4, 42, 8
    header = (f"{'Rank':<{w_rank}}  "
              f"{'BGL Signature':<{w_sig}}  {'Count':>{w_cnt}}  "
              f"{'HDFS Signature':<{w_sig}}  {'Count':>{w_cnt}}")
    lines = [
        'Table 12: Top 5 frequent anomaly signatures per dataset',
        _double_sep(len(header)),
        header,
        _separator(len(header)),
    ]
    for i, ((bsig, bcnt), (hsig, hcnt)) in enumerate(zip(bgl_top5, hdfs_top5), 1):
        lines.append(
            f"{i:<{w_rank}}  "
            f"{bsig:<{w_sig}}  {bcnt:>{w_cnt},}  "
            f"{hsig:<{w_sig}}  {hcnt:>{w_cnt},}"
        )
    lines.append(_double_sep(len(header)))
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Figure 14: Signature rank-frequency distribution (Zipf, log scale)
# ---------------------------------------------------------------------------

def build_figure14(bgl: dict, hdfs: dict, out_path: Path) -> None:
    plt.rcParams.update({
        'font.size':        11,
        'axes.titlesize':   12,
        'axes.labelsize':   11,
        'xtick.labelsize':  10,
        'ytick.labelsize':  10,
        'figure.dpi':       150,
        'axes.grid':        True,
        'grid.linestyle':   '--',
        'grid.alpha':       0.5,
    })

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    datasets = [
        ('BGL',  bgl['signatures']),
        ('HDFS', hdfs['signatures']),
    ]

    for ax, (ds_name, sigs) in zip(axes, datasets):
        counts   = sorted(sigs.values(), reverse=True)
        ranks    = np.arange(1, len(counts) + 1)
        n_sigs   = len(counts)
        n_sing   = sum(1 for c in counts if c == 1)
        sing_pct = n_sing / n_sigs * 100

        ax.loglog(ranks, counts, 'o-', color='0.3', markersize=4, linewidth=1.2)

        # Annotate top 3
        for k in range(min(3, len(counts))):
            ax.annotate(
                f"{counts[k]:,}",
                xy=(ranks[k], counts[k]),
                xytext=(2, 4),
                textcoords='offset points',
                fontsize=9,
                color='0.15',
            )

        ax.set_xlabel('Rank')
        ax.set_ylabel('Frequency')
        ax.set_title(
            f"{ds_name}: {n_sigs} signatures\n"
            f"(singletons: {n_sing}, {sing_pct:.0f}%)"
        )

    fig.suptitle('Signature Rank-Frequency Distribution (Zipf)', fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=600, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# rq1_results.json
# ---------------------------------------------------------------------------

def build_rq1_results(bgl: dict, hdfs: dict, human_eval: dict) -> dict:
    ratings   = human_eval['ratings']
    bgl_rows  = [v for v in ratings.values() if v['dataset'] == 'BGL']
    hdfs_rows = [v for v in ratings.values() if v['dataset'] == 'HDFS']

    def he_stats(rows):
        dims = ['correctness', 'completeness', 'evidence_grounding']
        out = {}
        for d in dims:
            vals = [r[d] for r in rows]
            out[d] = {
                'mean': round(statistics.mean(vals), 4),
                'std':  round(statistics.stdev(vals), 4),
                'n':    len(vals),
            }
        out['actionable_pct'] = round(
            sum(1 for r in rows if r['actionable'] == 'Y') / len(rows) * 100, 1
        )
        return out

    def pipeline_summary(m):
        sigs = m['signatures']
        top5 = sorted(sigs.items(), key=lambda x: -x[1])[:5]
        return {
            'anomalies_explained':   m['counts']['total_anomalies'],
            'verification_pass_rate': m['verification']['pass_rate'],
            'unique_signatures':     len(sigs),
            'singleton_rate':        round(
                sum(1 for c in sigs.values() if c == 1) / len(sigs), 4
            ),
            'avg_tokens':            round(m['tokens']['avg'], 1),
            'mean_latency_s':        round(m['latency']['avg_ms'] / 1000, 1),
            'p95_latency_s':         round(m['latency']['p95_ms'] / 1000, 1),
            'top5_signatures':       top5,
        }

    return {
        'rq':    'RQ1',
        'title': 'Traceable Explanation Quality',
        'model': 'GPT-5.1',
        'bgl_run':  '2026-03-13',
        'hdfs_run': '2026-03-11',
        'pipeline': {
            'BGL':  pipeline_summary(bgl),
            'HDFS': pipeline_summary(hdfs),
        },
        'human_eval': {
            'BGL':  he_stats(bgl_rows),
            'HDFS': he_stats(hdfs_rows),
        },
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    # Validate inputs
    missing = [p for p in [
        INPUT_DIR / 'bgl_metrics.json',
        INPUT_DIR / 'hdfs_metrics.json',
        INPUT_DIR / 'human_eval_ratings.json',
    ] if not p.exists()]
    if missing:
        print('[ERROR] Missing input files:')
        for p in missing:
            print(f'  {p}')
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    bgl        = _load_json(INPUT_DIR / 'bgl_metrics.json')
    hdfs       = _load_json(INPUT_DIR / 'hdfs_metrics.json')
    human_eval = _load_json(INPUT_DIR / 'human_eval_ratings.json')

    # Table 10
    t10 = build_table10(bgl, hdfs)
    print('\n' + t10)
    _write_txt(OUT_DIR / 'table10_pipeline_summary.txt', t10 + '\n')
    print(f'[OK] Saved table10_pipeline_summary.txt')

    # Table 11
    t11 = build_table11(human_eval)
    print('\n' + t11)
    _write_txt(OUT_DIR / 'table11_human_eval.txt', t11 + '\n')
    print(f'[OK] Saved table11_human_eval.txt')

    # Table 12
    t12 = build_table12(bgl, hdfs)
    print('\n' + t12)
    _write_txt(OUT_DIR / 'table12_top_signatures.txt', t12 + '\n')
    print(f'[OK] Saved table12_top_signatures.txt')

    # Figure 14
    fig_path = OUT_DIR / 'figure14_signature_zipf.png'
    build_figure14(bgl, hdfs, fig_path)
    print(f'[OK] Saved figure14_signature_zipf.png')

    # rq1_results.json
    results = build_rq1_results(bgl, hdfs, human_eval)
    out_json = OUT_DIR / 'rq1_results.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f'[OK] Saved rq1_results.json')

    print(f'\n[OK] All outputs saved to results/rq1/')


if __name__ == '__main__':
    main()
