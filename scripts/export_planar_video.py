#!/usr/bin/env python3
"""Export recorded planar camera frames as timestamped VFR videos using ffmpeg."""
import argparse
import base64
import json
from pathlib import Path
import shutil
import subprocess
import tempfile


def export(log_path, output):
    if not shutil.which('ffmpeg'):
        raise ValueError('ffmpeg_required')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {}
    # Temporary filenames are generated locally, never interpolated from logs.
    with tempfile.TemporaryDirectory(prefix='planar-video-') as directory:
        directory = Path(directory)
        frames = {'head':[], 'right':[]}
        with Path(log_path).open() as stream:
            for line in stream:
                row = json.loads(line)
                if row.get('event') != 'camera_frame' or row.get('camera') not in frames:
                    continue
                key = row['camera']
                timestamp = int(row['source_ns'])
                if frames[key] and timestamp <= frames[key][-1][0]:
                    raise ValueError('nonmonotonic_camera_recording')
                filename = f'{key}_{len(frames[key]):08d}.jpg'
                (directory/filename).write_bytes(base64.b64decode(row['jpeg_base64'], validate=True))
                frames[key].append((timestamp, filename))
        if any(len(rows)<2 for rows in frames.values()):
            raise ValueError('two_frames_per_camera_required')
        origin = min(rows[0][0] for rows in frames.values())
        for key, rows in frames.items():
            entries = ['ffconcat version 1.0']
            # Pad start with first frame to align the two exported timelines.
            if rows[0][0] > origin:
                entries.extend([f"file '{rows[0][1]}'", f'duration {(rows[0][0]-origin)/1e9:.9f}'])
            for (stamp, filename), (following, _) in zip(rows, rows[1:]):
                entries.extend([f"file '{filename}'",f'duration {(following-stamp)/1e9:.9f}'])
            entries.append(f"file '{rows[-1][1]}'")
            listing = directory/f'{key}.ffconcat'
            listing.write_text('\n'.join(entries)+'\n')
            subprocess.run(['ffmpeg','-nostdin','-v','error','-f','concat','-safe','1','-i',str(listing),
                            '-vsync','vfr','-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2',
                            '-c:v','libx264','-pix_fmt','yuv420p',str(output/f'{key}.mp4')],check=True)
            manifest[key] = dict(frames=len(rows), first_source_ns=rows[0][0], last_source_ns=rows[-1][0],
                                 leading_hold_s=(rows[0][0]-origin)/1e9)
    manifest['common_origin_ns'] = origin
    (output/'timestamps.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('events')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.events,args.output)))


if __name__ == '__main__':
    main()
