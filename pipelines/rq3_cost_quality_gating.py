"""
RQ3: Cost-Quality Trade-off Under Confidence-Based Gating
==========================================================
Reproduces the following thesis results (Section 4.6):
  - Table 15: Signature coverage (%) by gating strategy and budget (B)
  - Figure 15: Coverage-Cost Pareto Curves by Gating Strategy

Inputs  (inputs/rq3/):
  rq3_gating_simulation_results.json -- pre-computed gating simulation results

Outputs (results/rq3/):
  table15_signature_coverage.txt
  figure15_pareto_curves.png
  rq3_results.json

Usage:
  python pipelines/rq3_cost_quality_gating.py
"""

import json
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
INPUT_DIR    = PROJECT_ROOT / 'inputs' / 'rq3'
OUT_DIR      = PROJECT_ROOT / 'results' / 'rq3'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict:
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def _write_txt(path: Path, text: str) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)


def _sep(width: int = 72) -> str:
    return '-' * width


def _double_sep(width: int = 72) -> str:
    return '=' * width


# ---------------------------------------------------------------------------
# Table 15: Signature coverage (%) by gating strategy and budget (B)
# ---------------------------------------------------------------------------

def build_table15(sim_results: list) -> str:
    table_strategies = ['Random', 'Uncertainty', 'Novelty', 'Combined']
    strategy_labels  = {
        'Random':     'Random',
        'Uncertainty': 'Uncertainty',
        'Novelty':    'Novelty',
        'Combined':   'Combined (a=0.5)',
    }
    target_budgets = [0.1, 0.2, 0.5]

    # Build lookup: (strategy, budget, dataset) -> coverage
    lookup = {}
    for r in sim_results:
        lookup[(r['strategy'], r['budget'], r['dataset'])] = r['coverage']

    w_s, w_c = 18, 10
    # Header row
    header1 = (f"{'Strategy':<{w_s}}  "
               f"{'BGL':^{w_c}}  {'BGL':^{w_c}}  {'BGL':^{w_c}}  "
               f"{'HDFS':^{w_c}}  {'HDFS':^{w_c}}  {'HDFS':^{w_c}}")
    header2 = (f"{'': <{w_s}}  "
               f"{'B = 0.1':^{w_c}}  {'B = 0.2':^{w_c}}  {'B = 0.5':^{w_c}}  "
               f"{'B = 0.1':^{w_c}}  {'B = 0.2':^{w_c}}  {'B = 0.5':^{w_c}}")

    lines = [
        'Table 15: Signature coverage (%) by gating strategy and budget (B)',
        _double_sep(),
        header1,
        header2,
        _sep(),
    ]

    for strat in table_strategies:
        label = strategy_labels[strat]
        vals = []
        for ds in ['BGL', 'HDFS']:
            for b in target_budgets:
                cov = lookup.get((strat, b, ds), float('nan'))
                vals.append(round(cov * 100, 1))
        row = (f"{label:<{w_s}}  "
               f"{vals[0]:^{w_c}.1f}  {vals[1]:^{w_c}.1f}  {vals[2]:^{w_c}.1f}  "
               f"{vals[3]:^{w_c}.1f}  {vals[4]:^{w_c}.1f}  {vals[5]:^{w_c}.1f}")
        lines.append(row)

    lines.append(_double_sep())
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Figure 15: Coverage-Cost Pareto Curves by Gating Strategy
# ---------------------------------------------------------------------------

STRATEGY_STYLES = {
    'Uncertainty': {'linestyle': '-',  'marker': 'o', 'color': 'black',  'linewidth': 1.8},
    'Novelty':     {'linestyle': '--', 'marker': 's', 'color': 'black',  'linewidth': 1.2},
    'Combined':    {'linestyle': '-.',  'marker': '+', 'color': 'black',  'linewidth': 1.2},
    'Random':      {'linestyle': ':',  'marker': '*', 'color': '0.55',   'linewidth': 1.0},
    'Upper Bound': {'linestyle': '-',  'marker': None,'color': '0.75',   'linewidth': 1.0},
}

STRATEGY_LEGEND = {
    'Uncertainty': 'Uncertainty',
    'Novelty':     'Novelty',
    'Combined':    'Combined',
    'Random':      'Random',
    'Upper Bound': 'Upper Bound',
}

PLOT_ORDER = ['Uncertainty', 'Novelty', 'Combined', 'Random']


def _plot_dataset(ax, sim_results: list, dataset: str, all_strategies: list) -> None:
    budgets = sorted(set(r['budget'] for r in sim_results))

    for strat in all_strategies:
        rows = sorted(
            [r for r in sim_results if r['strategy'] == strat and r['dataset'] == dataset],
            key=lambda r: r['budget']
        )
        if not rows:
            continue
        xs   = [r['budget'] for r in rows]
        ys   = [r['coverage'] * 100 for r in rows]
        stds = [r.get('coverage_std', float('nan')) or 0.0 for r in rows]
        stds = [0.0 if (s != s) else s * 100 for s in stds]  # NaN -> 0

        style = STRATEGY_STYLES.get(strat, {})
        ax.plot(xs, ys,
                label=STRATEGY_LEGEND.get(strat, strat),
                linestyle=style.get('linestyle', '-'),
                marker=style.get('marker'),
                color=style.get('color', 'black'),
                linewidth=style.get('linewidth', 1.2),
                markersize=5)

        # Error band (only for Random which has std)
        has_std = any(s > 0 for s in stds)
        if has_std:
            lower = [y - s for y, s in zip(ys, stds)]
            upper = [y + s for y, s in zip(ys, stds)]
            ax.fill_between(xs, lower, upper,
                            alpha=0.15, color=style.get('color', 'black'))

    ax.set_title(dataset, fontsize=10)
    ax.set_xlabel('Budget Ratio B', fontsize=9)
    ax.set_ylabel('Signature Coverage', fontsize=9)
    ax.set_xlim(0.0, 1.05)
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda y, _: f'{int(y)}%'))
    ax.grid(True, linestyle=':', linewidth=0.5, color='0.8')
    ax.legend(fontsize=8, loc='lower right')


def build_figure15(sim_results: list, out_path: Path) -> None:
    plot_strategies = PLOT_ORDER + ['Upper Bound']

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle('Coverage-Cost Pareto Curves by Gating Strategy', fontsize=11)

    for ax, ds in zip(axes, ['BGL', 'HDFS']):
        _plot_dataset(ax, sim_results, ds, plot_strategies)

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=600, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# rq3_results.json
# ---------------------------------------------------------------------------

def build_rq3_results(data: dict) -> dict:
    sim  = data['simulation_results']
    out  = {
        'rq':       'RQ3',
        'title':    'Cost-Quality Trade-off Under Confidence-Based Gating',
        'timestamp': data.get('timestamp', ''),
        'table15':  {},
    }

    table_strategies = ['Random', 'Uncertainty', 'Novelty', 'Combined']
    target_budgets   = [0.1, 0.2, 0.5]

    for ds in ['BGL', 'HDFS']:
        out['table15'][ds] = {}
        for strat in table_strategies:
            out['table15'][ds][strat] = {}
            for b in target_budgets:
                row = next(
                    (r for r in sim if r['strategy'] == strat
                     and r['budget'] == b and r['dataset'] == ds),
                    None
                )
                if row:
                    out['table15'][ds][strat][str(b)] = round(row['coverage'] * 100, 1)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sim_path = INPUT_DIR / 'rq3_gating_simulation_results.json'
    if not sim_path.exists():
        print(f'[ERROR] Input not found: {sim_path}', file=sys.stderr)
        sys.exit(1)

    data = _load_json(sim_path)
    sim_results = data['simulation_results']

    # Table 15
    t15 = build_table15(sim_results)
    print(t15)
    _write_txt(OUT_DIR / 'table15_signature_coverage.txt', t15)
    print('[OK] Saved table15_signature_coverage.txt')

    # Figure 15
    fig_path = OUT_DIR / 'figure15_pareto_curves.png'
    build_figure15(sim_results, fig_path)
    print('[OK] Saved figure15_pareto_curves.png')

    # Summary JSON
    rq3_out = build_rq3_results(data)
    with open(OUT_DIR / 'rq3_results.json', 'w', encoding='utf-8') as f:
        json.dump(rq3_out, f, indent=2)
    print('[OK] Saved rq3_results.json')

    print()
    print(f'[OK] All outputs saved to results/rq3/')


if __name__ == '__main__':
    main()
