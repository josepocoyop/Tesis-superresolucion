"""
Step 8 — Thesis figures: multi-method comparison and per-condition metrics.

Generates, at 300 DPI in output/thesis_figures/:
    thesis_fig_comparison_<condition>.png
        One comparison strip per lighting condition: LR input, bicubic,
        ESRGAN, SwinIR, ESRGAN fine-tuned (those available) and HR reference.
    thesis_fig_zoom_<condition>.png
        Same panels cropped to a central detail region, where the texture
        differences between methods are most visible.
    thesis_fig_metrics_by_condition.png
        Grouped bar chart of LPIPS per method and condition (falls back to
        PSNR if LPIPS was skipped in step7).
    thesis_fig_training_loss.png
        Fine-tuning loss curves parsed from the basicsr training log
        (skipped if step6 has not been run).
    thesis_fig_rife.png
        Two consecutive frames with the RIFE-interpolated frame between
        them (skipped if step4 has not been run).

Run step7_compute_metrics.py first.

Usage:
    python step8_generate_figures.py
"""

import sys
import subprocess
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    FRAMES_HR_DIR, FRAMES_LR_DIR, FRAMES_BIC_DIR, FRAMES_SR_DIR,
    FRAMES_SWINIR_DIR, FRAMES_SR_FT_DIR, FRAMES_RIFE_DIR, REALESRGAN_DIR,
    METRICS_DIR, THESIS_FIGURES_DIR,
    SCALE_FACTOR, FIG_DOUBLE_W, FIG_DPI, condition_of,
)

METHODS = [
    ('bic',    'Bicúbico',            FRAMES_BIC_DIR),
    ('sr',     'ESRGAN',              FRAMES_SR_DIR),
    ('swinir', 'SwinIR',              FRAMES_SWINIR_DIR),
    ('sr_ft',  'ESRGAN fine-tuned',   FRAMES_SR_FT_DIR),
]


def install_deps():
    pkgs = ['matplotlib', 'pillow', 'numpy', 'pandas']
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q'] + pkgs, check=True)


def load_rgb(path: Path):
    import numpy as np
    from PIL import Image
    return np.array(Image.open(path).convert('RGB'))


ROI_COLOR       = '#D62828'
HIGHLIGHT_LABEL = 'ESRGAN fine-tuned'


def _roi(img_shape):
    """Central region, one quarter of the frame."""
    h, w = img_shape[:2]
    ch, cw = h // 4, w // 4
    return (h - ch) // 2, (w - cw) // 2, ch, cw


def _build_panels(name: str, active):
    import numpy as np
    from PIL import Image

    hr = load_rgb(FRAMES_HR_DIR / name)
    lr = load_rgb(FRAMES_LR_DIR / name)
    lr_up = np.array(Image.fromarray(lr).resize(
        (lr.shape[1] * SCALE_FACTOR, lr.shape[0] * SCALE_FACTOR), Image.NEAREST))

    panels = [(lr_up, 'Entrada LR (ampliada)')]
    panels += [(load_rgb(d / name), lbl) for _, lbl, d in active]
    panels.append((hr, 'Referencia HR'))
    return panels, hr


def comparison_strip(name: str, active, out_path: Path):
    """2-row grid of full frames, each with the analyzed region marked."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    panels, hr = _build_panels(name, active)
    y0, x0, ch, cw = _roi(hr.shape)

    n = len(panels)
    cols = (n + 1) // 2
    ratio = hr.shape[0] / hr.shape[1]
    fig, axes = plt.subplots(2, cols,
                             figsize=(FIG_DOUBLE_W, FIG_DOUBLE_W / cols * ratio * 2.15))
    axes = axes.ravel()
    for ax in axes[n:]:
        ax.axis('off')
    for ax, (img, lbl) in zip(axes, panels):
        ax.imshow(img)
        ax.add_patch(Rectangle((x0, y0), cw, ch, fill=False,
                               edgecolor=ROI_COLOR, linewidth=1.2))
        ax.set_title(lbl, fontsize=7)
        ax.axis('off')
    fig.tight_layout(pad=0.3)
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path.name}')


def zoom_strip(name: str, active, out_path: Path):
    """2-row grid with the marked region enlarged; fine-tuned panel highlighted."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    panels, hr = _build_panels(name, active)
    y0, x0, ch, cw = _roi(hr.shape)

    n = len(panels)
    cols = (n + 1) // 2
    ratio = ch / cw
    fig, axes = plt.subplots(2, cols,
                             figsize=(FIG_DOUBLE_W, FIG_DOUBLE_W / cols * ratio * 2.3))
    axes = axes.ravel()
    for ax in axes[n:]:
        ax.axis('off')
    for ax, (img, lbl) in zip(axes, panels):
        ih, iw = img.shape[:2]
        yy, xx = min(y0, max(ih - ch, 0)), min(x0, max(iw - cw, 0))
        ax.imshow(img[yy:yy + ch, xx:xx + cw])
        if lbl == HIGHLIGHT_LABEL:
            ax.set_title(lbl, fontsize=7, color=ROI_COLOR, fontweight='bold')
            ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                   fill=False, edgecolor=ROI_COLOR,
                                   linewidth=2, clip_on=False))
        else:
            ax.set_title(lbl, fontsize=7)
        ax.axis('off')
    fig.tight_layout(pad=0.3)
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path.name}')


METHOD_COLORS = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']
COND_LABELS   = {'day': 'Día', 'night': 'Noche', 'indoor': 'Interior',
                 'unspecified': 'Sin clasificar'}


def metrics_bar_chart(summary: dict, active, out_path: Path):
    import numpy as np
    import matplotlib.pyplot as plt

    conditions = sorted(summary['by_condition'].keys())
    keys   = [k for k, _, _ in active]
    labels = [lbl for _, lbl, _ in active]

    metrics = [('psnr', 'PSNR (dB)\nmayor es mejor'),
               ('ssim', 'SSIM\nmayor es mejor'),
               ('lpips', 'LPIPS\nmenor es mejor')]
    metrics = [(m, t) for m, t in metrics
               if any(f'{m}_{k}' in summary['overall'] for k in keys)]

    x = np.arange(len(conditions))
    width = 0.8 / len(keys)

    fig, axes = plt.subplots(1, len(metrics),
                             figsize=(FIG_DOUBLE_W, 2.6))
    if len(metrics) == 1:
        axes = [axes]

    for ax, (metric, title) in zip(axes, metrics):
        for i, key in enumerate(keys):
            vals = [summary['by_condition'][c].get(f'{metric}_{key}', float('nan'))
                    for c in conditions]
            pos = x + i * width - 0.4 + width / 2
            bars = ax.bar(pos, vals, width * 0.92,
                          color=METHOD_COLORS[i % len(METHOD_COLORS)],
                          edgecolor='black', linewidth=0.4)
            fmt = '{:.1f}' if metric == 'psnr' else '{:.2f}'
            for b, v in zip(bars, vals):
                if v == v:
                    ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v),
                            ha='center', va='bottom', fontsize=5.5)
        ax.set_title(title, fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([COND_LABELS.get(c, c) for c in conditions], fontsize=8)
        ax.tick_params(axis='y', labelsize=7)
        ax.margins(y=0.15)
        ax.grid(axis='y', alpha=0.3, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)

    from matplotlib.patches import Patch
    handles = [Patch(facecolor=METHOD_COLORS[i % len(METHOD_COLORS)],
                     edgecolor='black', linewidth=0.4)
               for i in range(len(labels))]
    fig.legend(handles, labels, loc='upper center', ncol=len(labels),
               fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path.name}')


def training_curve(out_path: Path):
    """Loss curves of the step6 fine-tuning, parsed from the basicsr log."""
    import re
    import matplotlib.pyplot as plt

    exp_dir = REALESRGAN_DIR / 'experiments' / 'finetune_RealESRGANx4plus_cctv'
    logs = sorted(exp_dir.glob('train_*.log')) if exp_dir.exists() else []
    if not logs:
        print('  Training log not found (run step6 first); loss curve skipped.')
        return
    pat = re.compile(r'iter:\s*([\d,]+).*?l_g_pix:\s*([\d.eE+-]+).*?l_g_percep:\s*([\d.eE+-]+)')
    iters, l_pix, l_percep = [], [], []
    for log in logs:
        for line in log.read_text(encoding='utf-8', errors='ignore').splitlines():
            m = pat.search(line)
            if m:
                iters.append(int(m.group(1).replace(',', '')))
                l_pix.append(float(m.group(2)))
                l_percep.append(float(m.group(3)))
    if not iters:
        print('  No loss entries found in the training log; loss curve skipped.')
        return

    import pandas as pd

    fig, axes = plt.subplots(1, 2, figsize=(FIG_DOUBLE_W, 2.6))
    series = [(l_pix, 'Pérdida L1 (píxel)', '#4C72B0'),
              (l_percep, 'Pérdida perceptual (VGG)', '#DD8452')]
    for ax, (vals, title, color) in zip(axes, series):
        smooth = pd.Series(vals).ewm(alpha=0.08).mean()
        ax.plot(iters, vals, linewidth=0.6, color=color, alpha=0.25)
        ax.plot(iters, smooth, linewidth=1.4, color=color)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel('Iteración', fontsize=8)
        ax.set_ylabel('Pérdida', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path.name}')


def rife_strip(out_path: Path):
    """Two consecutive frames and the RIFE frame interpolated between them."""
    import matplotlib.pyplot as plt

    origs = sorted(FRAMES_RIFE_DIR.glob('frame_*_orig.png'))
    if len(origs) < 2:
        print('  RIFE output not found (run step4 first); RIFE figure skipped.')
        return
    mid = len(origs) // 2
    a, b = origs[mid], origs[mid + 1]
    # the interpolated frame is the odd index between the two originals
    idx = int(a.stem.split('_')[1]) + 1
    interp = FRAMES_RIFE_DIR / f'frame_{idx:06d}_rife.png'
    if not interp.exists():
        print('  Interpolated frame missing; RIFE figure skipped.')
        return

    panels = [(load_rgb(a), 'Fotograma t'),
              (load_rgb(interp), 'Interpolado (RIFE)'),
              (load_rgb(b), 'Fotograma t+1')]
    fig, axes = plt.subplots(1, 3, figsize=(FIG_DOUBLE_W, FIG_DOUBLE_W / 3 * 0.62))
    for ax, (img, lbl) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(lbl, fontsize=7)
        ax.axis('off')
    fig.tight_layout(pad=0.3)
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path.name}')


def main():
    install_deps()
    import pandas as pd

    print('=' * 60)
    print('STEP 8 — Thesis figures')
    print('=' * 60)

    summary_path = METRICS_DIR / 'thesis_summary.json'
    if not summary_path.exists():
        print(f'ERROR: {summary_path} not found. Run step7_compute_metrics.py first.')
        sys.exit(1)
    summary = json.loads(summary_path.read_text())

    active = [(k, lbl, d) for k, lbl, d in METHODS
              if d.exists() and any(d.glob('*.png'))]
    THESIS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # One comparison strip per condition, using a mid-sequence frame
    # common to all methods
    hr_names = sorted(p.name for p in FRAMES_HR_DIR.glob('*.png'))
    common = set(hr_names) & {p.name for p in FRAMES_LR_DIR.glob('*.png')}
    for _, _, d in active:
        common &= {p.name for p in d.glob('*.png')}

    by_cond = {}
    for name in sorted(common):
        by_cond.setdefault(condition_of(name), []).append(name)

    print('\nComparison and zoom strips:')
    for cond, names in sorted(by_cond.items()):
        pick = names[len(names) // 2]
        comparison_strip(pick, active,
                         THESIS_FIGURES_DIR / f'thesis_fig_comparison_{cond}.png')
        zoom_strip(pick, active,
                   THESIS_FIGURES_DIR / f'thesis_fig_zoom_{cond}.png')

    print('\nMetrics chart:')
    metrics_bar_chart(summary, active,
                      THESIS_FIGURES_DIR / 'thesis_fig_metrics_by_condition.png')

    print('\nTraining loss curve:')
    training_curve(THESIS_FIGURES_DIR / 'thesis_fig_training_loss.png')

    print('\nRIFE interpolation figure:')
    rife_strip(THESIS_FIGURES_DIR / 'thesis_fig_rife.png')

    print(f'\nAll thesis figures saved in: {THESIS_FIGURES_DIR}')
    print('Step 8 complete.')


if __name__ == '__main__':
    main()
