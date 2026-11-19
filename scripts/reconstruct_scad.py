#!/usr/bin/env python3
"""Reconstruct the SCAD clips from independently downloaded source datasets.

Run from the cloned SCAD repository: python scripts/reconstruct_scad.py
The script expects all four source folders directly inside the SCAD repository.
Requires FFmpeg and FFprobe on PATH; Python standard library only.
"""
import argparse
import csv
import json
import subprocess
from collections import Counter
from pathlib import Path

VIDEO_SUFFIXES = {'.mp4', '.mov', '.avi', '.mkv', '.mpeg', '.mpg'}


def call(args):
    p = subprocess.run([str(a) for a in args], capture_output=True, text=True, errors='replace')
    if p.returncode:
        raise RuntimeError((p.stderr or p.stdout or 'Command failed').strip()[-1200:])
    return p.stdout


def metadata(path):
    info = json.loads(call(['ffprobe', '-v', 'error', '-show_entries',
                            'format=duration:stream=codec_type,avg_frame_rate,nb_frames',
                            '-of', 'json', path]))
    video = next(x for x in info['streams'] if x.get('codec_type') == 'video')
    frames = int(video.get('nb_frames') or 0)
    return float(info['format']['duration']), frames


def encode(output, input_args, vf, frames=None):
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.stem + '.partial.mp4')
    tmp.unlink(missing_ok=True)
    args = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
            *input_args, '-an', '-vf', vf]
    if frames is not None:
        args += ['-frames:v', str(frames)]
    args += ['-r', '30', '-fps_mode', 'cfr', '-c:v', 'mpeg4', '-q:v', '3',
             '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(tmp)]
    try:
        call(args)
        _, actual = metadata(tmp)
        if frames is not None and actual != frames:
            raise ValueError(f'Expected {frames} frames, got {actual}: {output.name}')
        tmp.replace(output)
    finally:
        tmp.unlink(missing_ok=True)


def source_frames(root, sid, cache, overwrite):
    folder = next((p for p in [root / 'CADP' / sid,
                               root / 'CADP' / 'extracted_frames' / sid] if p.is_dir()), None)
    if folder is None:
        raise FileNotFoundError(f'CADP folder {sid} not found')
    frames = sorted(((int(p.stem), p) for p in folder.iterdir()
                     if p.suffix.lower() in {'.jpg', '.jpeg'} and p.stem.isdecimal()))
    if not frames or [i for i, _ in frames] != list(range(len(frames))):
        raise ValueError(f'CADP frames missing or not numbered from 0: {folder}')
    extension = frames[0][1].suffix
    if any(p.suffix != extension for _, p in frames):
        raise ValueError(f'Mixed frame file extensions: {folder}')
    output = cache / 'CADP' / f'{sid}.mp4'
    if output.exists() and not overwrite:
        return output
    # CADP source sequences are reconstructed at 30 FPS (same default as validated run).
    encode(output, ['-framerate', '30', '-start_number', '0', '-i',
                    folder / ('%d' + extension)], 'setpts=PTS-STARTPTS', len(frames))
    return output


def retime_source(path, output, overwrite):
    if output.exists() and not overwrite:
        return output
    # Retiming, NOT ordinary 25->30 or ~60->30 frame-rate conversion.
    # Preserve the decoded frame sequence and assign each frame to 1/30s.
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.stem + '.partial.mp4')
    tmp.unlink(missing_ok=True)
    try:
        call(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
              '-i', path, '-map', '0:v:0', '-an',
              '-vf', 'settb=AVTB,setpts=N/(30*TB)', '-r', '30',
              '-fps_mode', 'cfr', '-c:v', 'mpeg4', '-q:v', '3',
              '-pix_fmt', 'yuv420p', '-movflags', '+faststart', tmp])
        duration, frames = metadata(tmp)
        if not frames or abs(duration - frames / 30.0) > 0.05:
            raise ValueError(f'Invalid 30 FPS source: {path}')
        tmp.replace(output)
    finally:
        tmp.unlink(missing_ok=True)
    return output


def extract(source, output, index, overwrite):
    if output.exists() and not overwrite:
        duration, frames = metadata(output)
        if frames == 120 and abs(duration - 4) < 0.05:
            return
    start = (index - 1) * 3
    duration, _ = metadata(source)
    if start >= duration - 0.00001:
        raise ValueError(f'Clip {index} starts after source end: {source.name}')
    # Four seconds at 30 FPS; pad an incomplete tail by freezing last frame.
    vf = ('trim=duration=4,setpts=PTS-STARTPTS,'
          'fps=30:start_time=0,tpad=stop_mode=clone:stop_duration=4,'
          'trim=duration=4,setpts=PTS-STARTPTS,format=yuv420p')
    encode(output, ['-ss', f'{start:.6f}', '-i', source], vf, 120)


def load_manifest(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    required = {'path', 'split', 'label', 'dataset', 'source_id', 'clip_index'}
    if not rows or not required.issubset(rows[0]):
        raise ValueError('Invalid or empty SCAD manifest')
    seen = set()
    for row in rows:
        p = Path(row['path'].replace('\\', '/'))
        if (p.is_absolute() or '..' in p.parts or len(p.parts) != 3
                or p.parts[0] != row['split'] or p.parts[1] != row['label']
                or p.suffix.lower() != '.mp4' or p.as_posix() in seen):
            raise ValueError(f'Invalid output path in manifest: {row["path"]}')
        seen.add(p.as_posix())
    return rows


def index_sources(root):
    locations = {'SO-TAD': root / 'SO-TAD' / 'so_tad',
                 'TU-DAT': root / 'Positive_Vidoes',
                 'UCF-Crime': root / 'RoadAccidents'}
    found = {}
    for dataset, folder in locations.items():
        if folder.is_dir():
            for p in folder.rglob('*'):
                if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES:
                    found.setdefault((dataset, p.stem), []).append(p)
    return found


def main():
    ap = argparse.ArgumentParser(description='Reconstruct SCAD from source datasets')
    ap.add_argument('--overwrite', action='store_true', help='Regenerate existing output clips')
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]
    root = repo
    rows = load_manifest(repo / 'manifests' / 'scad_manifest.csv')
    indexed = index_sources(root)
    cache = repo / 'temp'
    output = repo / 'SCAD_output'
    processed = {}
    successes = Counter()
    failures = []

    for row in rows:
        dataset = row['dataset']
        sid = row['source_id']
        index = int(row['clip_index'])
        # The upstream SO-TAD release removed source 106 as a duplicate of 47.
        # SCAD retained 106_2 (instead of 47_2), so reconstruct from 47 clip 2.
        if dataset == 'SO-TAD' and sid == '106' and index == 2:
            sid = '47'
            index = 2
        key = (dataset, sid)
        try:
            if key not in processed:
                if dataset == 'CADP':
                    processed[key] = source_frames(root, sid, cache, args.overwrite)
                else:
                    matches = indexed.get(key, [])
                    if len(matches) != 1:
                        raise FileNotFoundError(f'Source {dataset}/{sid} missing or ambiguous ({len(matches)} matches)')
                    if dataset in ('SO-TAD', 'TU-DAT'):
                        processed[key] = retime_source(
                            matches[0], cache / dataset / f'{sid}.mp4', args.overwrite)
                    else:
                        processed[key] = matches[0]
            dest = output / Path(row['path'].replace('\\', '/'))
            extract(processed[key], dest, index, args.overwrite)
            successes[dataset] += 1
        except Exception as ex:
            failures.append((row['path'], str(ex)))

    print(f'SCAD reconstructed: {sum(successes.values())}/{len(rows)} clips')
    for dataset in ('CADP', 'SO-TAD', 'TU-DAT', 'UCF-Crime'):
        print(f'  {dataset}: {successes[dataset]}')
    if failures:
        print(f'Missing or failed: {len(failures)}')
        for name, reason in failures:
            print(f'  {name}: {reason}')
        raise SystemExit(1)
    print(f'Output: {output}')


if __name__ == '__main__':
    main()
