"""
Step 6 - Generacion de la figura de comparacion del articulo.

Produce fig4_comparison.png, la figura de dos filas y cuatro columnas que compara
los metodos: arriba los fotogramas completos con la region de interes marcada,
abajo esa misma region ampliada. Se guarda directamente en ColCom Paper/figures/
al ancho de dos columnas de IEEEtran y por encima de 300 DPI efectivos, que es lo
que pide IEEE para contenido fotografico.

La figura temporal (fig5_rife.png) la genera step8_temporal_figure.py, que
necesita fotogramas contiguos y no la lista de fotogramas separados que produce
step2.

Entradas : dataset/frames_hr, frames_lr, output/bicubic, output/sr
Salida   : ColCom Paper/figures/fig4_comparison.png

Uso:
    python step6_generate_figures.py
    python step6_generate_figures.py --frame-idx 5
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    FRAMES_HR_DIR, FRAMES_LR_DIR, FRAMES_BIC_DIR, FRAMES_SR_DIR,
    METRICS_DIR, FIGURES_DIR, FIG_DOUBLE_W, FIG_DPI, SCALE_FACTOR,
)

# tamano del recorte ampliado, en pixeles del fotograma HR
PATCH_W, PATCH_H = 96, 64

RED = '#c0392b'
GREY = '#3a3a3a'


def setup_mpl():
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams.update({
        'font.family':        'serif',
        'font.serif':         ['Times New Roman', 'DejaVu Serif'],
        'font.size':          8,
        'figure.dpi':         FIG_DPI,
        'savefig.dpi':        FIG_DPI,
        'savefig.bbox':       'tight',
        'savefig.pad_inches': 0.03,
        'axes.linewidth':     0.5,
    })


def load_rgb(path):
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f'No se pudo leer: {path}')
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def nearest_resize(img, width, height):
    import cv2
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)


def find_roi(hr, bic, sr):
    """Recorte donde ESRGAN mas se separa del bicubico sobre zona con textura.

    Se pondera a la baja la mitad superior del fotograma, que en esta escena
    contiene avisos y carteles: son faciles de destacar pero poco
    representativos de lo que interesa en vigilancia.
    """
    import numpy as np
    from numpy.lib.stride_tricks import sliding_window_view

    diff = np.abs(sr.astype(float) - bic.astype(float)).mean(axis=2)
    win_diff = sliding_window_view(diff, (PATCH_H, PATCH_W)).mean(axis=(-2, -1))
    win_var = sliding_window_view(hr.mean(axis=2), (PATCH_H, PATCH_W)).var(axis=(-2, -1))

    combined = win_diff * (win_var > 200)
    if combined.max() == 0:
        combined = win_diff

    rows = combined.shape[0]
    weight = np.ones(rows)
    weight[:rows // 2] = 0.35
    weighted = combined * weight[:, np.newaxis]

    if weighted.max() > 0.4 * combined.max():
        ry, rx = np.unravel_index(weighted.argmax(), weighted.shape)
    else:
        ry, rx = np.unravel_index(combined.argmax(), combined.shape)

    h, w = hr.shape[:2]
    return int(min(rx, w - PATCH_W - 2)), int(min(ry, h - PATCH_H - 2))


def build_figure(hr, lr, bic, sr, roi, m):
    """Cuadricula de 2x4: fotogramas completos arriba, region ampliada abajo."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    rx, ry = roi
    h, w = hr.shape[:2]
    lr_full = nearest_resize(lr, w, h)

    lr_patch = nearest_resize(
        lr[ry // SCALE_FACTOR:ry // SCALE_FACTOR + PATCH_H // SCALE_FACTOR,
           rx // SCALE_FACTOR:rx // SCALE_FACTOR + PATCH_W // SCALE_FACTOR],
        PATCH_W, PATCH_H)

    def patch(img):
        return img[ry:ry + PATCH_H, rx:rx + PATCH_W]

    # la altura de cada fila sigue la proporcion de sus imagenes, si no matplotlib
    # centra la imagen en la celda y deja una franja vacia entre las dos filas
    col_w = FIG_DOUBLE_W / 4
    top_h = col_w / (w / h)
    bot_h = col_w / (PATCH_W / PATCH_H)

    fig = plt.figure(figsize=(FIG_DOUBLE_W, top_h + bot_h + 0.62))
    gs = fig.add_gridspec(2, 4, height_ratios=[top_h, bot_h],
                          hspace=0.06, wspace=0.02,
                          left=0.004, right=0.996, top=0.86, bottom=0.005)

    top = [
        (lr_full, f'LR input\n({w // SCALE_FACTOR}$\\times${h // SCALE_FACTOR}, NN)', 'black', False),
        (bic, ('Bicubic $\\times$4\n'
               f'PSNR: {m["psnr_bic"]:.2f} dB | SSIM: {m["ssim_bic"]:.3f}\n'
               f'LPIPS: {m["lpips_bic"]:.3f}'), GREY, False),
        # ESRGAN is not our model: the red frame marks the stage under
        # evaluation, it does not claim authorship of the network
        (sr, ('ESRGAN $\\times$4\n'
              f'PSNR: {m["psnr_sr"]:.2f} dB | SSIM: {m["ssim_sr"]:.3f}\n'
              f'LPIPS: {m["lpips_sr"]:.3f}'), RED, True),
        (hr, f'HR reference\n({w}$\\times${h})', 'black', False),
    ]
    for col, (img, title, color, highlight) in enumerate(top):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(img, interpolation='none')
        ax.set_title(title, color=color, fontsize=6.2, pad=2.5,
                     fontweight='bold' if highlight else 'normal', linespacing=1.35)
        ax.axis('off')
        ax.add_patch(mpatches.Rectangle((rx, ry), PATCH_W, PATCH_H,
                                        linewidth=1.8, edgecolor='#e74c3c',
                                        facecolor='none'))

    bottom = [
        (lr_patch, 'LR (magnified)', GREY, False),
        (patch(bic), 'Bicubic', GREY, False),
        (patch(sr), 'ESRGAN', RED, True),
        (patch(hr), 'HR reference', 'black', False),
    ]
    for col, (img, title, color, highlight) in enumerate(bottom):
        ax = fig.add_subplot(gs[1, col])
        ax.imshow(img, interpolation='none')
        ax.set_title(title, color=color, fontsize=6.5, pad=2.5,
                     fontweight='bold' if highlight else 'normal')
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2.2 if highlight else 0.6)
            spine.set_edgecolor(RED if highlight else '#888888')

    return fig


def main():
    parser = argparse.ArgumentParser(description='Figura de comparacion del articulo')
    parser.add_argument('--frame-idx', type=int, default=0,
                        help='indice del fotograma dentro de dataset/frames_hr')
    args = parser.parse_args()

    import json

    summary_path = METRICS_DIR / 'summary.json'
    if not summary_path.exists():
        summary_path = Path(__file__).resolve().parents[1] / 'results' / 'summary.json'
    if not summary_path.exists():
        print('ERROR: no se encontro summary.json; corre step5_compute_metrics.py')
        sys.exit(1)
    m = json.loads(summary_path.read_text())

    frames = sorted(FRAMES_HR_DIR.glob('*.png'))
    if not frames:
        print(f'ERROR: no hay fotogramas en {FRAMES_HR_DIR}')
        sys.exit(1)
    name = frames[args.frame_idx].name

    print('=' * 60)
    print('Figura de comparacion')
    print('=' * 60)
    print(f'  Fotograma : {name}')

    setup_mpl()
    hr = load_rgb(FRAMES_HR_DIR / name)
    lr = load_rgb(FRAMES_LR_DIR / name)
    bic = load_rgb(FRAMES_BIC_DIR / name)
    sr = load_rgb(FRAMES_SR_DIR / name)

    roi = find_roi(hr, bic, sr)
    print(f'  Region    : x={roi[0]}, y={roi[1]}, {PATCH_W}x{PATCH_H}')

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / 'fig4_comparison.png'
    fig = build_figure(hr, lr, bic, sr, roi, m)
    fig.savefig(str(out))

    import cv2
    px = cv2.imread(str(out)).shape[1]
    print(f'  Salida    : {out.name}  ({px} px, {px / FIG_DOUBLE_W:.0f} DPI efectivos)')


if __name__ == '__main__':
    main()
