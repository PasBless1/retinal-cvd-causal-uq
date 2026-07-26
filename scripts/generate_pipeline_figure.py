#!/usr/bin/env python3
"""Generate the four-stage pipeline overview figure for the paper.

Run from the project root:
    python scripts/generate_pipeline_figure.py
Output: outputs/pipeline_overview.png
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os

os.makedirs('outputs', exist_ok=True)

W, H = 7.0, 10.5
fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis('off')
fig.patch.set_facecolor('white')

CX    = W / 2
BOX_W = 5.8
BL    = CX - BOX_W / 2

COLOURS = [
    ('#E3F2FD', '#0D47A1'),  # input  — deep blue
    ('#E8F5E9', '#1B5E20'),  # stage1 — deep green
    ('#F3E5F5', '#4A148C'),  # stage2 — deep purple
    ('#FFF3E0', '#BF360C'),  # stage3 — deep orange
    ('#FCE4EC', '#880E4F'),  # stage4 — deep rose
    ('#FFFDE7', '#E65100'),  # output — amber
]

# (centre_y, half_height, colour_index, title, detail, output_or_None)
BOXES = [
    (10.0, 0.38, 0,
     'Retinal Fundus Image',
     r'RFMiD  $\cdot$  1,920 training images  $\cdot$  512$\times$512 px',
     None),

    (8.75, 0.62, 1,
     'Stage 1   Bayesian Vessel Segmentation',
     r'CLAHE preprocessing  $\cdot$  DRIVE-pretrained Bayesian U-Net'
     r'  $\cdot$  $S=20$ MC passes',
     r'$\rightarrow$  Vessel mask  +  per-pixel epistemic uncertainty'
     r'  $\hat{\sigma}$'),

    (6.95, 0.62, 2,
     'Stage 2   Vascular Biomarker Extraction',
     r'DRIVE-calibrated threshold  $\cdot$  medial-axis skeletonisation'
     r'  $\cdot$  box-counting $D_f$  $\cdot$  arc-to-chord $\kappa$',
     r'$\rightarrow$  10 features: $\rho,\;A,\;D_f,\;\kappa,\;B,\;W,'
     r'\;C,\;L,\;\beta,\;\hat{\sigma}$'),

    (5.15, 0.62, 3,
     'Stage 3   Causal Mediation Analysis',
     r'Two-model OLS  $\cdot$  sequential ignorability'
     r'  $\cdot$  bootstrap CIs ($B=1{,}000$)',
     r'$\rightarrow$  NDE,  NIE,  ATE,  proportion mediated'),

    (3.35, 0.62, 4,
     'Stage 4   Split-Conformal Prediction',
     r'50/35/15 split  $\cdot$  $\alpha=0.10$  (90\% nominal coverage)',
     r'$\rightarrow$  Risk intervals  $\hat{f}(x) \pm \hat{q}$  with'
     r'  $\Pr\!\left(Y \in \mathcal{C}(X)\right) \geq 1-\alpha$'),

    (1.55, 0.38, 5,
     'Report',
     r'Retinal disease-risk classification  $\cdot$  causal pathways  $\cdot$  '
     r'coverage-guaranteed intervals',
     None),
]


def draw_box(cy, hy, ci, title, detail, out_text):
    face, edge = COLOURS[ci]
    ax.add_patch(FancyBboxPatch(
        (BL, cy - hy), BOX_W, 2 * hy,
        boxstyle='round,pad=0.10',
        facecolor=face, edgecolor=edge, linewidth=1.8, zorder=3,
    ))
    if out_text:
        ax.text(CX, cy + hy * 0.46, title,
                ha='center', va='center', fontsize=9.5,
                fontweight='bold', color=edge, zorder=4)
        ax.text(CX, cy, detail,
                ha='center', va='center', fontsize=8.2,
                color='#333333', zorder=4)
        ax.text(CX, cy - hy * 0.50, out_text,
                ha='center', va='center', fontsize=8.2,
                color=edge, style='italic', zorder=4)
    else:
        ax.text(CX, cy + hy * 0.20, title,
                ha='center', va='center', fontsize=10.0,
                fontweight='bold', color=edge, zorder=4)
        ax.text(CX, cy - hy * 0.28, detail,
                ha='center', va='center', fontsize=8.2,
                color='#555555', zorder=4)


for box in BOXES:
    draw_box(*box)

for i in range(len(BOXES) - 1):
    from_y = BOXES[i][0]   - BOXES[i][1]
    to_y   = BOXES[i+1][0] + BOXES[i+1][1]
    ax.annotate('',
                xy=(CX, to_y + 0.04), xytext=(CX, from_y - 0.04),
                arrowprops=dict(arrowstyle='->', color='#555555',
                                lw=1.8, mutation_scale=18),
                zorder=2)

plt.tight_layout(pad=0.3)
out_path = 'outputs/pipeline_overview.png'
plt.savefig(out_path, dpi=180, bbox_inches='tight', facecolor='white')
print(f'Saved to {out_path}')
