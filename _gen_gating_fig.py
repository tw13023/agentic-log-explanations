"""Generate Gating analysis figure for advisor presentation slide."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

records = []
with open('results/explanations_BGL_20260301_112519.jsonl') as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))
print(f'[OK] Loaded {len(records)} records')

margins = [r['screener']['margin'] for r in records]
labels  = [r.get('label', 0) for r in records]
tp_margins = [m for m, l in zip(margins, labels) if l == 1]
fp_margins = [m for m, l in zip(margins, labels) if l == 0]
print(f'     TP={len(tp_margins)}  FP={len(fp_margins)}')

fig = plt.figure(figsize=(13, 5.5), facecolor='white')
gs  = GridSpec(1, 2, figure=fig, width_ratios=[1.1, 0.9], wspace=0.35)

# ── Left: Margin histogram ──────────────────────────────────────────────────
ax1  = fig.add_subplot(gs[0])
bins = np.linspace(0, 1, 41)
ax1.hist(tp_margins, bins=bins, color='#444444', alpha=0.85,
         label=f'True Positive  (n={len(tp_margins):,})')
ax1.hist(fp_margins, bins=bins, color='#bbbbbb', alpha=0.85,
         label=f'False Positive (n={len(fp_margins):,})')
ax1.axvline(0.8, color='black', linestyle='--', linewidth=1.2)
ax1.annotate(
    'margin < 0.8\n47 sessions\n100% FP',
    xy=(0.795, 600), xytext=(0.58, 650), fontsize=8.5,
    arrowprops=dict(arrowstyle='->', color='black', lw=1.1),
    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='black', lw=0.8),
)
ax1.set_xlabel('Confidence Margin  ( |p_anomaly - p_normal| )', fontsize=10)
ax1.set_ylabel('Number of Sessions', fontsize=10)
ax1.set_title(
    'Screener Confidence Margin Distribution\n(BGL Test Set - 6,295 Predicted Anomalies)',
    fontsize=10, fontweight='bold')
ax1.legend(fontsize=9, framealpha=0.9)
ax1.set_xlim(0, 1.02)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Inset: zoom on low-margin tail [0, 0.8)
ax_in = ax1.inset_axes([0.03, 0.46, 0.36, 0.42])
ax_in.hist(tp_margins, bins=np.linspace(0, 0.8, 17), color='#444444', alpha=0.85)
ax_in.hist(fp_margins, bins=np.linspace(0, 0.8, 17), color='#bbbbbb', alpha=0.85)
ax_in.set_xlim(0, 0.8)
ax_in.set_title('Zoom: margin < 0.8', fontsize=7.5)
ax_in.tick_params(labelsize=7)
ax_in.spines['top'].set_visible(False)
ax_in.spines['right'].set_visible(False)

# ── Right: Gating simulation table ─────────────────────────────────────────
ax2 = fig.add_subplot(gs[1])
ax2.axis('off')

total_tp   = 5844
total_cost = 44.92
sorted_recs = sorted(zip(margins, labels), key=lambda x: x[0])  # low margin first

rows = []
for pct in [10, 20, 30, 50, 70, 100]:
    n   = int(len(sorted_recs) * pct / 100)
    sub = sorted_recs[:n]
    tp  = sum(1 for _, l in sub if l == 1)
    fp  = sum(1 for _, l in sub if l == 0)
    cost = total_cost * pct / 100
    rows.append([
        f'{pct}%',
        f'{n:,}',
        f'{tp:,}',
        f'{tp / total_tp * 100:.0f}%',
        f'{fp:,}',
        f'${cost:.1f}',
    ])

col_labels = [
    'Budget\n(Top-K%)', 'Sessions\nExplained',
    'TPs\nCovered', 'TP\nRecovery',
    'FPs\nIncluded', 'Est. Cost',
]

tbl = ax2.table(cellText=rows, colLabels=col_labels,
                loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
tbl.scale(1.0, 1.65)

ncols = len(col_labels)
# Header row
for j in range(ncols):
    tbl[0, j].set_facecolor('#333333')
    tbl[0, j].set_text_props(color='white', fontweight='bold')
# 100% row (explain-all baseline)
for j in range(ncols):
    tbl[len(rows), j].set_facecolor('#dddddd')
# 50% row (sweet spot)
for j in range(ncols):
    tbl[4, j].set_facecolor('#f2f2f2')

ax2.set_title(
    'Mode b: Uncertainty-First Gating\nBudget vs. TP Coverage (BGL Full Run)',
    fontsize=10, fontweight='bold', pad=14)
ax2.text(
    0.5, 0.07,
    'Sessions ordered by ascending margin (uncertain-first).\n'
    'Shaded row = Explain-All baseline (Mode a, $44.92).',
    ha='center', va='center', fontsize=7.5, transform=ax2.transAxes,
    bbox=dict(boxstyle='round', fc='#f0f0f0', ec='#999999', lw=0.7))

for dpi, suffix in [(150, ''), (600, '_600dpi')]:
    out = f'results/gating_analysis_BGL{suffix}.png'
    plt.savefig(out, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f'[OK] Saved: {out}')
plt.close()
