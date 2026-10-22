"""Collect evaluated robot footage and make bounded offline GIF previews."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg

try:
    from . import libero_export
except ImportError:
    import libero_export


ROOT = Path(__file__).resolve().parents[1]


def encode(arguments, output, *, overwrite=False):
    output = Path(output).resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f"output already exists use --overwrite to replace {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="media-", dir=output.parent) as staging:
        staged = Path(staging) / output.name
        command = [imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-hide_banner",
                   "-loglevel", "error", "-threads", "2", "-filter_threads", "1",
                   "-filter_complex_threads", "1", *map(str, arguments),
                   "-threads", "2", "-n", str(staged)]
        subprocess.run(command, check=True, timeout=120)
        if not staged.is_file() or not staged.stat().st_size:
            raise ValueError("encoder produced no media")
        if output.suffix.lower() == ".gif" and staged.stat().st_size > 8 * 1024 * 1024:
            raise ValueError("GIF exceeds eight MiB reduce width fps or duration")
        staged.replace(output)
    return output


def video_to_gif(source, output=None, *, start=0, duration=20, speed=1,
                 width=640, fps=12, overwrite=False):
    source = Path(source).resolve()
    output = Path(output).resolve() if output else source.with_suffix(".gif")
    if not source.is_file() or output == source or output.suffix.lower() != ".gif":
        raise ValueError("provide a local video and a different GIF output path")
    if (not all(math.isfinite(value) for value in (start, duration, speed))
            or start < 0 or not 0 < duration <= 60 or not 1 <= speed <= 4
            or duration / speed > 30 or type(width) is not int or not 64 <= width <= 1280
            or type(fps) is not int or not 1 <= fps <= 20):
        raise ValueError("use at most sixty input seconds thirty output seconds and twenty fps")
    if output.exists() and not overwrite:
        raise FileExistsError(f"output already exists use --overwrite to replace {output}")
    inputs = ["-ss", start, "-t", duration, "-i", source]
    filters = (f"setpts=(PTS-STARTPTS)/{speed},fps={fps},"
               f"scale=w=min({width}\\,iw):h=-2:flags=lanczos")
    # two passes keep palette creation from buffering the whole video
    with tempfile.TemporaryDirectory(prefix="gap-gif-") as staging:
        palette = Path(staging) / "palette.png"
        encode([*inputs, "-vf", filters + ",palettegen=stats_mode=diff",
                "-frames:v", "1", "-update", "1", "-an"], palette)
        return encode([*inputs, "-i", palette, "-filter_complex",
                       f"[0:v]{filters}[clip];[clip][1:v]paletteuse=dither=bayer:bayer_scale=3",
                       "-an", "-loop", "0"], output, overwrite=overwrite)


def matching_recording(session, record):
    for path in sorted(Path(session).glob("visualizations/*/replay/receipt.json"), reverse=True):
        receipt = libero_export.read(path)
        if (receipt.get("graph_sha256") != record["graph_sha256"]
                or receipt.get("seed") != record["seed"] or receipt.get("ok") is not True):
            continue
        directory = path.parent / "recording"
        if not (directory / "frames.json").is_file():
            continue
        frames = libero_export.frame_index(directory)
        rows = frames["frames"]
        if (rows[0]["step"] == 0 and rows[-1]["step"] == receipt.get("steps")
                and all(row["file"] == f"frame-{index:05d}.png"
                        for index, row in enumerate(rows))):
            return path, frames
    raise ValueError(f"no complete successful matching recording in {session} export it first")


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def robot_video(replay, frames, output, *, overwrite=False):
    rows, fps = frames["frames"], frames["fps"]
    # keep sparse camera samples at their recorded simulator times
    with tempfile.TemporaryDirectory(prefix="gap-frames-") as staging:
        playlist = Path(staging) / "frames.ffconcat"
        lines = ["ffconcat version 1.0"]
        for index, row in enumerate(rows):
            path = (Path(replay).parent / "recording" / row["file"]).resolve()
            escaped = path.as_posix().replace("'", "'\\''")
            next_step = rows[index + 1]["step"] if index + 1 < len(rows) else row["step"] + 1
            lines += [f"file '{escaped}'", f"duration {(next_step - row['step']) / fps:.9f}"]
        lines.append(lines[-2])
        playlist.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return encode(["-f", "concat", "-safe", "0", "-i", playlist,
                       "-t", (rows[-1]["step"] + 1) / fps, "-r", fps, "-fps_mode", "cfr",
                       "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-an", "-c:v", "libx264",
                       "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
                       "-movflags", "+faststart"], output, overwrite=overwrite)


def collect_sessions(sessions, *, overwrite=False):
    prepared = []
    tasks = set()
    for session in sessions:
        session = Path(session).resolve()
        data, manifest = libero_export.session_history(session)
        record = data["iterations"][-1]
        task_id = manifest["task"]["task_id"]
        if task_id in tasks:
            raise ValueError("select only one session per task for the collection")
        tasks.add(task_id)
        replay, frames = matching_recording(session, record)
        video = libero_export.presentation_output(manifest).resolve()
        graph = video.with_name(video.stem.removesuffix("-process") + "-policy.png")
        if not video.is_file() or not graph.is_file():
            raise ValueError(f"export the graph and process video first for {task_id}")
        robot = video.with_name(video.stem.removesuffix("-process") + "-robot.mp4")
        if robot.exists() and not overwrite:
            raise FileExistsError(f"robot video exists use --overwrite to replace {robot}")
        prepared.append((session, data, manifest, replay, frames, video, graph, robot))
    entries = []
    for session, data, manifest, replay, frames, video, graph, robot in prepared:
        record = data["iterations"][-1]
        best = max(data["iterations"], key=lambda row: row["successes"] / row["trials"])
        print(f"Collecting {manifest['task']['task_id']} from {record['version'].name}", flush=True)
        robot_video(replay, frames, robot, overwrite=overwrite)
        feedback = libero_export.load_feedback(record["version"])
        entries.append({
            "task_id": manifest["task"]["task_id"], "session": relative(session),
            "final_version": record["version"].name,
            "successes": record["successes"], "trials": record["trials"],
            "generation_attempts": len(list(session.glob("v[0-9][0-9][0-9][0-9]"))),
            "best_version": best["version"].name, "best_successes": best["successes"],
            "variation_indices": manifest["task"]["seeds"],
            "failed_variations": [row["seed"] for row in feedback["records"] if not row["success"]],
            "graph_sha256": record["graph_sha256"], "task_sha256": manifest["task_sha256"],
            "feedback": relative(libero_export.feedback_path(record["version"])),
            "history": [{"version": row["version"].name, "successes": row["successes"],
                         "trials": row["trials"]} for row in data["iterations"]],
            "skipped_revisions": data["skipped_iterations"],
            "process_video": relative(video), "policy_image": relative(graph),
            "robot_video": relative(robot), "robot_video_sha256": libero_export.sha256(robot),
            "recording_receipt": relative(replay), "recording_seed": record["seed"],
            "recording_steps": frames["frames"][-1]["step"], "recording_fps": frames["fps"],
        })
    return {"schema_version": 1,
            "scope": "final measured revision per selected session not held out evaluation",
            "video_note": "one successful replay per final graph not the ten trial evaluation",
            "scenarios": entries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    gif = commands.add_parser("gif", help="convert a local video without models or simulation")
    gif.add_argument("video", type=Path)
    gif.add_argument("--output", type=Path)
    gif.add_argument("--start", type=float, default=0)
    gif.add_argument("--duration", type=float, default=20)
    gif.add_argument("--speed", type=float, default=1)
    gif.add_argument("--width", type=int, default=640)
    gif.add_argument("--fps", type=int, default=12)
    gif.add_argument("--overwrite", action="store_true")
    collect = commands.add_parser("collect", help="collect cached successful graph recordings only")
    collect.add_argument("sessions", type=Path, nargs="+")
    collect.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "gif":
            print(video_to_gif(args.video, args.output, start=args.start, duration=args.duration,
                               speed=args.speed, width=args.width, fps=args.fps,
                               overwrite=args.overwrite))
        else:
            output = ROOT / "docs/visualizations/results-summary.json"
            if output.exists() and not args.overwrite:
                raise FileExistsError("results summary exists use --overwrite to replace it")
            summary = collect_sessions(args.sessions, overwrite=args.overwrite)
            output.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="summary-", dir=output.parent) as staging:
                staged = Path(staging) / output.name
                staged.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
                staged.replace(output)
            print(output)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        parser.exit(1, f"Media export stopped {type(error).__name__}: {error}\n")


if __name__ == "__main__":
    main()
