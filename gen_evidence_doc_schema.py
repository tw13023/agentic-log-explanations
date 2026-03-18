"""Generate a publication-quality EvidenceDoc schema diagram (grayscale, 600 dpi)."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(9, 7.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")

# ── colours (grayscale, white background) ──
TITLE_BG = "#3a3a3a"
TITLE_FG = "white"
FIELD_BG = "white"
FIELD_FG = "#222222"
SUB_FG   = "#666666"
BORDER   = "#888888"
CODE_BG  = "#f0f0f0"

# ── layout constants ──
LEFT = 0.4
RIGHT = 9.6
WIDTH = RIGHT - LEFT
COL1 = LEFT + 0.15          # field name x
COL2 = LEFT + 2.6           # value x

row_h = 0.0  # will be set per row

def draw_row(y_top, height, field, lines, sub_lines=None, code_block=None):
    """Draw one row with field name on left, content on right."""
    # row background
    rect = FancyBboxPatch((LEFT, y_top - height), WIDTH, height,
                          boxstyle="square,pad=0", facecolor=FIELD_BG,
                          edgecolor=BORDER, linewidth=0.8)
    ax.add_patch(rect)
    # vertical divider
    ax.plot([LEFT + 2.4, LEFT + 2.4], [y_top, y_top - height],
            color=BORDER, linewidth=0.8)
    # field name
    ax.text(COL1, y_top - 0.28, field, fontsize=9, fontweight="bold",
            fontfamily="monospace", color=FIELD_FG, va="top")
    # value lines
    y = y_top - 0.25
    for line in lines:
        ax.text(COL2, y, line, fontsize=8.5, fontfamily="monospace",
                color=FIELD_FG, va="top")
        y -= 0.30
    # sub-annotation lines (grey, smaller)
    if sub_lines:
        for sl in sub_lines:
            ax.text(COL2 + 0.25, y, sl, fontsize=7.5, fontfamily="monospace",
                    color=SUB_FG, va="top", style="italic")
            y -= 0.26
    # code block (darker background inset)
    if code_block:
        cb_top = y + 0.05
        cb_h = len(code_block) * 0.26 + 0.12
        cb_rect = FancyBboxPatch((COL2 - 0.05, cb_top - cb_h), 6.2, cb_h,
                                 boxstyle="round,pad=0.06", facecolor=CODE_BG,
                                 edgecolor=BORDER, linewidth=0.5)
        ax.add_patch(cb_rect)
        cy = cb_top - 0.10
        for cl in code_block:
            ax.text(COL2 + 0.05, cy, cl, fontsize=7.5, fontfamily="monospace",
                    color=FIELD_FG, va="top")
            cy -= 0.26
    return y_top - height

# ── title bar ──
title_h = 0.55
title_rect = FancyBboxPatch((LEFT, 10 - title_h), WIDTH, title_h,
                            boxstyle="round,pad=0.08", facecolor=TITLE_BG,
                            edgecolor=TITLE_BG, linewidth=1.2)
ax.add_patch(title_rect)
ax.text((LEFT + RIGHT) / 2, 10 - title_h / 2, "EvidenceDoc",
        fontsize=13, fontweight="bold", fontfamily="monospace",
        color=TITLE_FG, ha="center", va="center")

# ── rows ──
y = 10 - title_h

y = draw_row(y, 0.55, "evidence_id", ['"E_BGL_03700010"'])

y = draw_row(y, 0.55, "session_id", ['"BGL_03700010"'])

y = draw_row(y, 0.55, "evidence_type", ['"session"'])

y = draw_row(y, 3.55, "text\n(BM25 indexed)", [
    '10 normalized lines, joined by \\n',
], sub_lines=[
    'Example:',
], code_block=[
    '"<IPV6> <NUM>.<NUM> <NODE> RAS KERNEL INFO',
    '  1 torus receiver x- input pipe error(s)',
    '  (dcr <HEX>) detected\\n<IPV6> <NUM>..."',
    '',
    'Replacements:',
    '  IP address  ->  <IPV6>',
    '  numbers     ->  <NUM>',
    '  node id     ->  <NODE>',
    '  hex value   ->  <HEX>',
])

y = draw_row(y, 2.80, "metadata", [
    'label            : 0  (0=normal, 1=anomaly)',
    'dataset          : "BGL"',
    'num_lines        : 10',
    'original_length  : 1,757  (chars)',
    'normalized_length: 1,677  (chars)',
    'param_stats      : { NODE: 20, IPV6: 10,',
    '                     HEX: 9,  NUM: 103 }',
])

# ── outer border ──
outer = FancyBboxPatch((LEFT, y), WIDTH, 10 - title_h - y,
                       boxstyle="square,pad=0", facecolor="none",
                       edgecolor=BORDER, linewidth=1.2)
# just redraw top/bottom borders (rows already have edges)

plt.tight_layout()
out = "results/evidence_doc_schema.png"
fig.savefig(out, dpi=600, bbox_inches="tight", facecolor="white")
print(f"Saved to {out}")
plt.close()
