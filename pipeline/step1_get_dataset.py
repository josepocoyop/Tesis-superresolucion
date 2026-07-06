"""
Step 1 — Dataset acquisition.

Place a surveillance-style video file (MP4 / AVI / MKV) inside
    dataset/raw_videos/
then run this script. It validates the file and prints a summary so
you can confirm it is suitable before running Step 2.

Recommended footage (free, no registration):
  - Pexels.com  → search "surveillance camera outdoor" or "parking lot security"
    → Download any HD (1080p or 720p) video → save to dataset/raw_videos/
  - The video should show a FIXED camera angle with pedestrians or vehicles.

Alternatively, pass --pexels-key YOUR_KEY to download automatically.
Get a free Pexels API key at: https://www.pexels.com/api/

Usage:
    python step1_get_dataset.py                    # validate existing video
    python step1_get_dataset.py --pexels-key KEY   # auto-download from Pexels
"""

import sys
import os
import subprocess
import argparse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RAW_VIDEO_DIR, FRAMES_HR_DIR, NUM_FRAMES

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mkv', '.mov', '.webm'}
MIN_WIDTH  = 1024
MIN_HEIGHT = 576
MIN_FRAMES = 120   # at least 4 seconds at 30 fps


def install_deps():
    subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '-q', 'opencv-python', 'requests', 'tqdm'],
        check=True
    )


def find_videos():
    return sorted(
        p for p in RAW_VIDEO_DIR.iterdir()
        if p.suffix.lower() in VIDEO_EXTENSIONS
    )


def validate_video(path: Path) -> dict:
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {'ok': False, 'error': 'Cannot open file'}

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    issues = []
    if width < MIN_WIDTH or height < MIN_HEIGHT:
        issues.append(f'Resolution {width}x{height} is below minimum {MIN_WIDTH}x{MIN_HEIGHT}')
    if total < MIN_FRAMES:
        issues.append(f'Only {total} frames found (need at least {MIN_FRAMES})')
    if fps < 10:
        issues.append(f'Frame rate {fps:.1f} fps is unusually low')

    return {
        'ok':     len(issues) == 0,
        'path':   path,
        'width':  width,
        'height': height,
        'fps':    fps,
        'frames': total,
        'issues': issues,
    }


def download_from_pexels(api_key: str, count: int = 2):
    import requests

    queries = [
        'surveillance camera outdoor',
        'security camera parking lot',
        'CCTV street pedestrian',
    ]
    headers = {'Authorization': api_key}
    downloaded = []

    for query in queries:
        if len(downloaded) >= count:
            break
        print(f'  Searching Pexels: "{query}"...')
        resp = requests.get(
            'https://api.pexels.com/videos/search',
            headers=headers,
            params={'query': query, 'per_page': 5, 'size': 'large'},
            timeout=15,
        )
        if resp.status_code != 200:
            print(f'  Pexels API returned {resp.status_code} — skipping.')
            continue

        for video in resp.json().get('videos', []):
            if len(downloaded) >= count:
                break
            best_file = None
            best_w    = 0
            for f in video.get('video_files', []):
                w = f.get('width', 0)
                if w >= MIN_WIDTH and w > best_w:
                    best_file = f
                    best_w    = w
            if best_file is None:
                continue

            out_path = RAW_VIDEO_DIR / f'cctv_{len(downloaded) + 1:02d}.mp4'
            print(f'  Downloading {out_path.name} '
                  f'({best_file["width"]}x{best_file["height"]})...')
            urllib.request.urlretrieve(best_file['link'], out_path)
            downloaded.append(out_path)
            print(f'  Saved → {out_path}')

    return downloaded


def main():
    parser = argparse.ArgumentParser(description='Step 1: Dataset acquisition')
    parser.add_argument('--pexels-key', default='', help='Pexels API key')
    args = parser.parse_args()

    install_deps()

    print('=' * 60)
    print('STEP 1 — Dataset acquisition')
    print('=' * 60)

    if args.pexels_key:
        print('\nDownloading surveillance footage from Pexels...')
        videos = download_from_pexels(args.pexels_key)
        if not videos:
            print('ERROR: No videos downloaded. Check your API key and internet connection.')
            sys.exit(1)

    videos = find_videos()

    if not videos:
        print(f'\nNo video files found in: {RAW_VIDEO_DIR}')
        print('\nTo get started:')
        print('  Option A (manual):')
        print('    1. Go to https://www.pexels.com')
        print('    2. Search "surveillance camera outdoor" or "parking lot security camera"')
        print('    3. Download a free HD video (720p or 1080p)')
        print('    4. Save it to:')
        print(f'       {RAW_VIDEO_DIR}')
        print('    5. Re-run this script.')
        print()
        print('  Option B (automatic):')
        print('    python step1_get_dataset.py --pexels-key YOUR_FREE_KEY')
        print('    (Get key free at https://www.pexels.com/api/)')
        sys.exit(1)

    print(f'\nFound {len(videos)} video(s) — validating...\n')
    all_ok = True
    for v in videos:
        info = validate_video(v)
        status = 'OK' if info['ok'] else 'WARN'
        print(f'  [{status}] {v.name}')
        print(f'        Resolution : {info["width"]}x{info["height"]}')
        print(f'        Frame rate : {info["fps"]:.1f} fps')
        print(f'        Total frames: {info["frames"]}')
        if info['issues']:
            for issue in info['issues']:
                print(f'        WARNING: {issue}')
            all_ok = False
        print()

    if all_ok:
        print('All videos passed validation.')
        print(f'Will extract {NUM_FRAMES} frames per video.')
        print('\nStep 1 complete. Run step2_prepare_frames.py next.')
    else:
        print('Some videos have issues (see above). You can still proceed,')
        print('but results may be suboptimal. Consider replacing flagged files.')


if __name__ == '__main__':
    main()
