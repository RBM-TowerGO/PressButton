#!/usr/bin/env python3
"""Validate recorded camera synchronization and compose a labeled video grid."""
import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import subprocess


LABELS = {
    "overview": "全景 / Overview", "panel": "面板特写 / Panel",
    "front": "面板正面 / Front", "lobby": "电梯厅 / Lobby",
    "head": "头部 / Head", "wrist_left": "左腕 / Left wrist",
    "wrist_right": "右腕 / Right wrist",
}
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def probe(path):
    result = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=width,height,avg_frame_rate,nb_read_frames,duration",
        "-of", "json", str(path)], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)["streams"][0]


def validate(folder):
    report = json.loads((folder / "report.json").read_text())
    names = list(report["camera_paths"])
    fps = Fraction(str(report["config"]["capture_hz"]))
    timing = json.loads((folder / "camera_timing.json").read_text())
    if not timing or not names:
        raise ValueError("No recorded camera frames")
    streams = {name: probe(folder / f"{name}.mp4") for name in names}
    for name, stream in streams.items():
        if (int(stream["nb_read_frames"]) != len(timing)
                or report["video_frames"][name] != len(timing)
                or Fraction(stream["avg_frame_rate"]) != fps):
            raise ValueError(f"Frame count or frame rate mismatch: {name}")
    max_skew, max_lag = 0., 0.
    previous = None
    for row in timing:
        if set(row["cameras"]) != set(names):
            raise ValueError("Missing camera timestamp")
        times = list(row["cameras"].values())
        if not all(math.isfinite(t) and t >= 0 for t in times):
            raise ValueError("Invalid native camera timestamp")
        max_skew = max(max_skew, max(times) - min(times))
        max_lag = max(max_lag, *(abs(t - row["simulation_time"]) for t in times))
        if previous and previous["epoch"] == row["epoch"]:
            for name in names:
                delta = row["cameras"][name] - previous["cameras"][name]
                if abs(delta - 1 / float(fps)) > 1e-5:
                    raise ValueError(f"Stale or skipped frame: {name} at step {row['step']}")
        previous = row
    if max_skew > 1e-5 or max_lag > 1 / float(fps) + 1e-5:
        raise ValueError(f"Unsynchronized cameras: skew={max_skew}, lag={max_lag}")
    audit = dict(status="passed", cameras=streams, frame_count=len(timing),
                 fps=float(fps), duration_s=len(timing) / float(fps),
                 max_camera_skew_s=max_skew, max_camera_physics_lag_s=max_lag,
                 reset_epochs=sorted({row["epoch"] for row in timing}),
                 task_status=report["status"])
    return names, fps, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    args = parser.parse_args()
    folder = args.recording.resolve()
    names, fps, audit = validate(folder)
    output = folder / "all_cameras.mp4"
    if output.exists():
        raise FileExistsError(output)
    columns = min(4, len(names))
    rows = math.ceil(len(names) / columns)
    filters = []
    command = ["ffmpeg", "-n", "-v", "error", "-filter_complex_threads", "1"]
    font = f"fontfile={FONT}:" if Path(FONT).exists() else ""
    for index, name in enumerate(names):
        command += ["-i", str(folder / f"{name}.mp4")]
        label = LABELS.get(name, name)
        filters.append(
            f"[{index}:v]setpts=PTS-STARTPTS,"
            "scale=640:480:force_original_aspect_ratio=decrease,"
            "pad=640:520:(ow-iw)/2:40+(480-ih)/2:color=0x101824,setsar=1,"
            f"drawtext={font}text='{label}':x=14:y=6:fontsize=24:fontcolor=white[v{index}]")
    layout = "|".join(f"{i % columns * 640}_{i // columns * 520}" for i in range(len(names)))
    filters.append("".join(f"[v{i}]" for i in range(len(names))) +
                   f"xstack=inputs={len(names)}:layout={layout}:fill=0x101824[grid]")
    # Use the unused eighth tile for playback context.
    if len(names) == 7:
        filters.append(
            f"[grid]drawtext={font}text='Astribot S1 / VIB-820':x=1950:y=590:fontsize=30:fontcolor=white,"
            f"drawtext={font}text='7 路同步画面 · {float(fps):g} FPS':x=1950:y=650:fontsize=26:fontcolor=white,"
            f"drawtext={font}text='仿真原速 1× · 含返回与重置检查':x=1950:y=710:fontsize=24:fontcolor=white,"
            f"drawtext={font}text='播放时间 %{{pts}}':x=1950:y=770:fontsize=24:fontcolor=white[out]")
    else:
        filters.append("[grid]null[out]")
    command += ["-filter_complex", ";".join(filters), "-map", "[out]", "-an",
                "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-threads", "4",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)]
    subprocess.run(command, check=True)
    composed = probe(output)
    if (int(composed["nb_read_frames"]) != audit["frame_count"]
            or Fraction(composed["avg_frame_rate"]) != fps
            or composed["width"] != columns * 640 or composed["height"] != rows * 520):
        raise ValueError("Composed video does not match recorded streams")
    audit["composite"] = composed
    (folder / "camera_video_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    print(output)


if __name__ == "__main__":
    main()
