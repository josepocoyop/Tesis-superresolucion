"""
Step 4 — RIFE temporal frame interpolation.

Clones the official ECCV2022-RIFE repository, downloads pretrained weights,
and interpolates between consecutive ESRGAN-enhanced frames to produce a
2x-fps output sequence stored in output/rife/.

Inputs  : output/sr/       (ESRGAN-enhanced frames, sorted alphabetically)
Outputs : output/rife/     (original + interpolated frames interleaved)

Usage:
    python step4_rife_inference.py
    python step4_rife_inference.py --pairs 20    # interpolate only first N pairs
"""

import sys
import os
import subprocess
import argparse
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    FRAMES_SR_DIR, FRAMES_RIFE_DIR,
    RIFE_DIR, RIFE_REPO_URL, RIFE_MODEL_GDRIVE,
    BASE_DIR,
)

RIFE_MODEL_DIR = RIFE_DIR / 'train_log'


def install_deps():
    pkgs = ['torch', 'torchvision', 'gdown', 'tqdm', 'numpy', 'opencv-python', 'Pillow']
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q'] + pkgs, check=True)


def clone_rife():
    if RIFE_DIR.exists() and (RIFE_DIR / 'inference_img.py').exists():
        print(f'RIFE repo already cloned at {RIFE_DIR}')
        return
    print(f'Cloning RIFE repository...')
    subprocess.run(['git', 'clone', '--depth', '1', RIFE_REPO_URL, str(RIFE_DIR)],
                   check=True)
    subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '-q', '-r',
         str(RIFE_DIR / 'requirements.txt')],
        check=True,
    )
    print('RIFE cloned and dependencies installed.')


def download_rife_weights():
    if RIFE_MODEL_DIR.exists() and any(RIFE_MODEL_DIR.iterdir()):
        print(f'RIFE weights found at {RIFE_MODEL_DIR}')
        return
    RIFE_MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    print('Downloading RIFE pretrained weights from Google Drive...')
    zip_path = RIFE_DIR / 'train_log.zip'
    subprocess.run(
        [sys.executable, '-m', 'gdown', RIFE_MODEL_GDRIVE, '-O', str(zip_path)],
        check=True,
    )
    shutil.unpack_archive(str(zip_path), str(RIFE_DIR))
    zip_path.unlink(missing_ok=True)
    print('RIFE weights ready.')


def interpolate_pair(frame0_path: Path, frame1_path: Path,
                     out_path: Path, tmp_dir: Path):
    """Run RIFE inference_img.py for one consecutive pair."""
    script = RIFE_DIR / 'inference_img.py'
    tmp_dir.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [sys.executable, str(script),
         '--img', str(frame0_path), str(frame1_path),
         '--exp', '1'],
        cwd=str(RIFE_DIR),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f'RIFE failed: {result.stderr[:400]}')

    # inference_img.py saves output in a local 'output/' subdir of RIFE_DIR
    rife_out_dir = RIFE_DIR / 'output'
    candidates = sorted(rife_out_dir.glob('*.png'))
    if not candidates:
        raise RuntimeError('RIFE produced no output files.')
    # The interpolated frame is the middle one
    mid = candidates[len(candidates) // 2]
    shutil.copy(str(mid), str(out_path))
    # Clean up RIFE's own output dir for next pair
    shutil.rmtree(str(rife_out_dir), ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description='Step 4: RIFE frame interpolation')
    parser.add_argument('--pairs', type=int, default=0,
                        help='Interpolate only first N consecutive pairs (0 = all)')
    args = parser.parse_args()

    install_deps()
    from tqdm import tqdm
    import shutil as _shutil

    print('=' * 60)
    print('STEP 4 — RIFE temporal frame interpolation')
    print('=' * 60)

    clone_rife()
    download_rife_weights()

    sr_frames = sorted(FRAMES_SR_DIR.glob('*.png'))
    if len(sr_frames) < 2:
        print(f'ERROR: Need at least 2 SR frames in {FRAMES_SR_DIR}')
        print('       Run step3_esrgan_inference.py first.')
        sys.exit(1)

    pairs = list(zip(sr_frames[:-1], sr_frames[1:]))
    if args.pairs > 0:
        pairs = pairs[:args.pairs]

    print(f'SR frames available : {len(sr_frames)}')
    print(f'Pairs to interpolate: {len(pairs)}')
    print()

    if FRAMES_RIFE_DIR.exists():
        _shutil.rmtree(str(FRAMES_RIFE_DIR))
    FRAMES_RIFE_DIR.mkdir(parents=True)

    tmp_dir = BASE_DIR / '_rife_tmp'
    errors  = 0
    idx     = 0

    for f0, f1 in tqdm(pairs, unit='pair'):
        # Copy original frame 0
        orig_out = FRAMES_RIFE_DIR / f'frame_{idx:06d}_orig.png'
        _shutil.copy(str(f0), str(orig_out))
        idx += 1

        # Interpolated frame
        interp_out = FRAMES_RIFE_DIR / f'frame_{idx:06d}_rife.png'
        try:
            interpolate_pair(f0, f1, interp_out, tmp_dir)
        except Exception as e:
            print(f'\nWarning: RIFE failed for pair {f0.name}/{f1.name}: {e}')
            errors += 1
        idx += 1

    # Copy last original frame
    last_out = FRAMES_RIFE_DIR / f'frame_{idx:06d}_orig.png'
    _shutil.copy(str(sr_frames[len(pairs)]), str(last_out))

    _shutil.rmtree(str(tmp_dir), ignore_errors=True)

    rife_count = len(list(FRAMES_RIFE_DIR.glob('*.png')))
    print(f'\nRIFE interpolation complete.')
    print(f'  Input pairs    : {len(pairs)}')
    print(f'  Output frames  : {rife_count}  (≈2x frame rate)')
    print(f'  Saved to       : {FRAMES_RIFE_DIR}')
    if errors:
        print(f'  Errors         : {errors} pairs failed')
    print('\nStep 4 complete. Run step5_swinir_inference.py next.')


if __name__ == '__main__':
    main()
