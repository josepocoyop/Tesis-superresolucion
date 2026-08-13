"""
Step 9 - Miniaturas para los diagramas de las Figs. 1 a 3.

Los diagramas en TikZ llevan pequenas vinetas que representan fotogramas. En
lugar de dibujar rectangulos vacios, este script recorta fotogramas reales del
material del articulo y los deja en ColCom Paper/figures/ con un tamano
suficiente para imprimirse nitidos a poco menos de un centimetro de ancho.

Miniaturas producidas:
  thumb_lr.png    entrada degradada, ampliada por vecino mas proximo
  thumb_sr.png    salida de ESRGAN
  thumb_hr.png    referencia de alta resolucion
  thumb_t.png     fotograma t
  thumb_mid.png   fotograma t+0.5 sintetizado por RIFE
  thumb_t1.png    fotograma t+1

Entradas : dataset/frames_lr, frames_hr, output/sr, dataset/raw_videos
Salida   : ColCom Paper/figures/thumb_*.png

Uso:
    python step9_diagram_thumbs.py
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    FRAMES_HR_DIR, FRAMES_LR_DIR, FRAMES_SR_DIR, RAW_VIDEO_DIR,
    FIGURES_DIR, SCALE_FACTOR,
)
from step8_temporal_figure import degrade, build_upsampler, interpolate

# ancho en pixeles de cada vineta; a 0.95 cm impreso son mas de 600 DPI
THUMB_W = 260


def save_thumb(img, name, nearest=False):
    import cv2

    h, w = img.shape[:2]
    th = int(round(THUMB_W * h / w))
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_AREA
    out = FIGURES_DIR / name
    cv2.imwrite(str(out), cv2.resize(img, (THUMB_W, th), interpolation=interp))
    print(f'  {name:<16} {THUMB_W}x{th}')


def main():
    parser = argparse.ArgumentParser(description='Miniaturas para los diagramas')
    parser.add_argument('--frame-idx', type=int, default=0,
                        help='fotograma de dataset/frames_hr para las Figs. 1 y 2')
    parser.add_argument('--video', default='day_street.mp4')
    parser.add_argument('--temporal-frame', type=int, default=1269,
                        help='fotograma t para la vineta de la Fig. 3')
    args = parser.parse_args()

    import cv2
    import numpy as np
    import torch

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print('=' * 60)
    print('Miniaturas para los diagramas')
    print('=' * 60)

    frames = sorted(FRAMES_HR_DIR.glob('*.png'))
    if not frames:
        print(f'ERROR: no hay fotogramas en {FRAMES_HR_DIR}')
        sys.exit(1)
    name = frames[args.frame_idx].name
    print(f'Fotograma para Figs. 1 y 2: {name}')

    hr = cv2.imread(str(FRAMES_HR_DIR / name))
    lr = cv2.imread(str(FRAMES_LR_DIR / name))
    sr = cv2.imread(str(FRAMES_SR_DIR / name))
    if hr is None or lr is None or sr is None:
        print('ERROR: faltan fotogramas; corre los pasos 2 y 3 primero.')
        sys.exit(1)

    # la vineta de entrada sale de los pixeles LR sin pasar por un reescalado
    # suave, para que conserve el aspecto degradado que realmente tiene
    save_thumb(lr, 'thumb_lr.png', nearest=True)
    save_thumb(sr, 'thumb_sr.png')
    save_thumb(hr, 'thumb_hr.png')

    video_path = RAW_VIDEO_DIR / args.video
    if not video_path.exists():
        print(f'AVISO: {video_path} no existe, se omiten las vinetas de la Fig. 3')
        return

    print(f'Fotogramas para Fig. 3: {args.temporal_frame} y {args.temporal_frame + 1}')
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.temporal_frame)
    ok_a, frame_a = cap.read()
    ok_b, frame_b = cap.read()
    cap.release()
    if not (ok_a and ok_b):
        print('ERROR: no se pudieron leer los dos fotogramas')
        sys.exit(1)

    hh, ww = frame_a.shape[:2]
    frame_a = frame_a[:hh - hh % SCALE_FACTOR, :ww - ww % SCALE_FACTOR]
    frame_b = frame_b[:hh - hh % SCALE_FACTOR, :ww - ww % SCALE_FACTOR]

    np.random.seed(0)
    upsampler = build_upsampler()
    sr_a, _ = upsampler.enhance(degrade(frame_a), outscale=SCALE_FACTOR)
    sr_b, _ = upsampler.enhance(degrade(frame_b), outscale=SCALE_FACTOR)
    del upsampler
    torch.cuda.empty_cache()

    save_thumb(sr_a, 'thumb_t.png')
    save_thumb(interpolate(sr_a, sr_b), 'thumb_mid.png')
    save_thumb(sr_b, 'thumb_t1.png')


if __name__ == '__main__':
    main()
