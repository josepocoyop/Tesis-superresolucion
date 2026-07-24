"""
Figure: typical degradation in low-cost CCTV footage.

Shows a real surveillance frame next to its degraded low-resolution version
(x4 downscale + blur + JPEG + noise, the same protocol used in the pipeline),
with a zoomed crop of each to make the loss of detail visible.

Output: output/thesis_figures/fig01_degradacion.png

Usage:
    python fig_degradation.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'pipeline'))
from config import (
    FRAMES_HR_DIR, FRAMES_LR_DIR, THESIS_FIGURES_DIR,
    SCALE_FACTOR, FIG_DOUBLE_W, FIG_DPI,
)


def main():
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image

    hr_frames = sorted(FRAMES_HR_DIR.glob('*.png'))
    if not hr_frames:
        print(f'ERROR: no frames in {FRAMES_HR_DIR}; run step2 first.')
        sys.exit(1)
    name = hr_frames[len(hr_frames) // 2].name

    hr = np.array(Image.open(FRAMES_HR_DIR / name).convert('RGB'))
    lr = np.array(Image.open(FRAMES_LR_DIR / name).convert('RGB'))
    lr_up = np.array(Image.fromarray(lr).resize(
        (lr.shape[1] * SCALE_FACTOR, lr.shape[0] * SCALE_FACTOR), Image.NEAREST))

    # central crop for the zoom row
    h, w = hr.shape[:2]
    ch, cw = h // 4, w // 4
    y0, x0 = (h - ch) // 2, (w - cw) // 2

    panels = [
        (hr, 'Fotograma original'),
        (lr_up, 'Version degradada (x4, desenfoque, JPEG, ruido)'),
        (hr[y0:y0 + ch, x0:x0 + cw], 'Detalle original'),
        (lr_up[y0:y0 + ch, x0:x0 + cw], 'Detalle degradado'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(FIG_DOUBLE_W, FIG_DOUBLE_W * 0.66))
    for ax, (img, lbl) in zip(axes.ravel(), panels):
        ax.imshow(img)
        ax.set_title(lbl, fontsize=8)
        ax.axis('off')
    fig.tight_layout(pad=0.4)

    THESIS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = THESIS_FIGURES_DIR / 'fig01_degradacion.png'
    fig.savefig(out, dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved -> {out}')


if __name__ == '__main__':
    main()
