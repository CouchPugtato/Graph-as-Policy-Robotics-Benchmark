"""Measured harness history and one recorded final LIBERO execution."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from bisect import bisect_right
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from gap_harness.evaluations import evaluation_complete, feedback_path, load_feedback

try:
    from . import app
except ImportError:
    import app

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def latest_session():
    candidates = [path for path in (ROOT / "results/harness").glob("*")
                  if (path / "session.json").is_file()
                  and any(load_feedback(version) is not None
                          and evaluation_complete(load_feedback(version))
                          for version in path.glob("v[0-9][0-9][0-9][0-9]"))]
    if not candidates:
        raise ValueError("no evaluated harness session found run harness rehearsal first "
                         "or provide --session PATH")
    return max(candidates, key=lambda path: max(
        feedback_path(version).stat().st_mtime_ns for version in path.glob("v*")
        if feedback_path(version) is not None
        and evaluation_complete(load_feedback(version))))


def session_history(session):
    session = Path(session).resolve()
    manifest = read(session / "session.json")
    iterations, skipped, previous = [], [], None
    for version in sorted(session.glob("v[0-9][0-9][0-9][0-9]")):
        feedback = load_feedback(version)
        if feedback is None or not evaluation_complete(feedback):
            skipped.append({
                "iteration":int(version.name[1:]),
                "reason":"not evaluated" if feedback is None else "incomplete evaluation",
            })
            continue
        graph_path = version / "graph.json"
        digest = sha256(graph_path)
        validation = read(version / "validation.json")
        trials = feedback.get("records", [])
        if (feedback.get("graph_sha256") != digest or not validation.get("ok")
                or validation.get("graph_sha256") != digest
                or feedback.get("task_sha256") != manifest["task_sha256"]):
            raise ValueError(f"revision evidence does not match graph or task in {version.name}")
        if (not trials or len(trials) != feedback["trials"]
                or any(type(trial.get("success")) is not bool for trial in trials)
                or sum(trial["success"] for trial in trials) != feedback["successes"]
                or sorted(trial["seed"] for trial in trials) != sorted(manifest["task"]["seeds"])):
            raise ValueError("iteration rate must come from the complete measured seed suite")
        graph = read(graph_path)
        nodes = {node["id"]: node for node in graph["nodes"]}
        node_kinds = {}
        if (version / "workflow.json").is_file():
            for subgraph in read(version / "workflow.json").get("subgraphs", []):
                node_kinds.update({f"{subgraph['name']}__{node['name']}": node["kind"]
                                   for node in subgraph["nodes"]})
        representative = next((trial for trial in trials if trial["success"]), trials[0])
        summary = feedback_path(version)
        evaluations = version / "evaluation" if summary.parent == version else summary.parent
        receipt_path = evaluations / f"seed-{representative['seed']}" / "receipt.json"
        receipt = read(receipt_path) if receipt_path.is_file() else None
        if receipt and (receipt.get("graph_sha256") != digest
                        or receipt.get("seed") != representative["seed"]):
            raise ValueError("iteration receipt belongs to a different graph or seed")
        record = {"iteration": int(version.name[1:]), "version": version,
                  "successes": feedback["successes"], "trials": feedback["trials"],
                  "policy": graph, "graph_path": graph_path, "graph_sha256": digest,
                  "changed": [] if previous is None else [
                      name for name, node in nodes.items() if previous.get(name) != node],
                  "removed": [] if previous is None else sorted(set(previous) - set(nodes)),
                  "receipt": receipt, "backend": "libero", "node_kinds": node_kinds,
                  "seed": representative["seed"],
                  "video_label": f"revision {int(version.name[1:])} initial scene"}
        initial = receipt_path.parent / "initial/rgb.png"
        if initial.is_file():
            record["scene_image_path"] = initial
        iterations.append(record)
        previous = nodes
    if not iterations:
        raise ValueError("session has no evaluated graph versions")
    return {"iterations": iterations, "skipped_iterations": skipped}, manifest


def frame_index(recording):
    recording = Path(recording).resolve()
    data = read(recording / "frames.json")
    frames = data.get("frames", [])
    fps = data.get("fps")
    if (not isinstance(fps, (int, float)) or not math.isfinite(fps) or not 1 <= fps <= 60
            or not frames or len(frames) > 1001):
        raise ValueError("invalid or over budget simulation recording")
    previous = -1
    for frame in frames:
        path = (recording / frame["file"]).resolve()
        if (path.parent != recording or not path.is_file() or type(frame["step"]) is not int
                or frame["step"] <= previous):
            raise ValueError("recording frames must exist locally with increasing steps")
        previous = frame["step"]
    return data


def record_final(session, record, manifest, destination):
    # reuse only recordings of this exact final graph and representative seed
    for receipt_path in sorted(session.glob("visualizations/*/replay/receipt.json"), reverse=True):
        receipt = read(receipt_path)
        recording = receipt_path.parent / "recording"
        if (receipt.get("graph_sha256") == record["graph_sha256"]
                and receipt.get("seed") == record["seed"]
                and (recording / "frames.json").is_file()):
            frame_index(recording)
            return receipt_path, False
    runner = manifest["task"].get("runner")
    if not runner or "gap_libero.cli" not in runner["command"]:
        raise ValueError("this task needs a LIBERO runner with --record support")
    task_path = manifest.get("task_manifest_path")
    if not task_path:
        raise ValueError("session lacks its task manifest path generate a new harness session")
    cwd = (Path(task_path).parent / runner["cwd"]).resolve()
    substitutions = {"root": str(cwd), "graph": str(record["graph_path"]),
                     "seed": str(record["seed"]), "output": str(destination)}
    command = [argument.format_map(substitutions) for argument in runner["command"]]
    if not any(mode in command for mode in ("compare-grocery", "run", "scenario")):
        raise ValueError("only bounded LIBERO graph run modes can be recorded")
    command.append("--record")
    environment = {key: value for key, value in os.environ.items()
                   if key not in {"OPENAI_API_KEY", "OPENROUTER_API_KEY", "HF_TOKEN",
                                  "HUGGING_FACE_HUB_TOKEN", "GAP_CODEX_PROXY_KEY"}}
    environment.setdefault("LIBERO_CONFIG_PATH", str(cwd / "cache/libero-config"))
    local_vlm = "local-vlm" in command or "scenario" in command
    print("Recording one final LIBERO run keep Isaac and other model workers closed; "
          + ("keep local Qwen running in LM Studio" if local_vlm else "keep LM Studio closed"),
          flush=True)
    with (destination.parent / "recording.log").open("x", encoding="utf-8") as handle:
        subprocess.run(command, cwd=cwd, env=environment, stdout=handle,
                       stdin=subprocess.DEVNULL,
                       stderr=subprocess.STDOUT, check=False,
                       timeout=min(runner.get("timeout_s", 120), 180))
    receipt_path = destination / runner.get("receipt", "receipt.json")
    if not receipt_path.is_file():
        raise RuntimeError("recording worker produced no receipt inspect recording.log")
    receipt = read(receipt_path)
    if (receipt.get("graph_sha256") != record["graph_sha256"]
            or receipt.get("seed") != record["seed"]):
        raise RuntimeError("recording did not complete graph execution inspect its receipt")
    frame_index(destination / "recording")
    return receipt_path, True


def event_failed(event, policy=None):
    if event["outcome"] == "failed":
        return True
    if event["outcome"] in {"succeeded", "true", "found"}:
        return False
    if policy is not None:
        return policy["terminals"].get(event.get("next_node")) == "failed"
    return True


def graph_state(receipt, step, final=False, policy=None):
    trace, trajectory = receipt["graph"]["trace"], receipt.get("trajectory", [])
    if len(trace) != len(trajectory):
        raise ValueError("recorded run requires one timed sample per graph trace event")
    completed, active, failed = [], None, None
    taken = {}
    previous = 0
    for event, sample in zip(trace, trajectory, strict=True):
        start, end = sample.get("start_step", previous), sample["step"]
        if event["action"] != sample["action"] or start < previous or end < start:
            raise ValueError("recorded action timeline does not match the graph trace")
        if step >= end:
            taken[event["node_id"]] = event["outcome"]
            if not event_failed(event, policy):
                completed.append(event["node_id"])
            else:
                failed = event["node_id"]
        elif start <= step < end:
            active = event["node_id"]
        previous = end
    if final and trace and event_failed(trace[-1],policy):
        failed = trace[-1]["node_id"]
    return {"completed_nodes": completed, "active_node": active, "failed_node": failed,
            "taken_routes": taken}


def font(size):
    path = Path("C:/Windows/Fonts/segoeui.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def composite(base, pixels, label, status, color, size, footage_bounds=(.637, .039, .331, .445)):
    image = Image.fromarray(base)
    x, y, w, h = footage_bounds
    left, top = round(x * size[0]), round((1-y-h) * size[1])
    width, height = round(w * size[0]), round(h * size[1])
    scene = ImageOps.pad(Image.fromarray(pixels), (width, height), color="#1b2329")
    image.paste(scene, (left, top))
    draw = ImageDraw.Draw(image)
    face = font(max(12, round(15 * size[0] / 1280)))
    for text, y, fill in ((label, top + 8, "#14202a"),
                          (status, top + height - 33, color)):
        bounds = draw.textbbox((left + 9, y), text, font=face)
        draw.rounded_rectangle((bounds[0]-4, bounds[1]-3, bounds[2]+4, bounds[3]+3),
                               radius=4, fill=fill)
        draw.text((left + 9, y), text, fill="white", font=face)
    return image


def render_movie(data, receipt_path, output, *, fps=60, size=(1280, 720), recap_s=1.25,
                 playback_speed=1):
    if not math.isfinite(playback_speed) or not 1 <= playback_speed <= 4:
        raise ValueError("playback speed must be between one and four")
    receipt_path, output = Path(receipt_path), Path(output)
    receipt = read(receipt_path)
    recording = receipt_path.parent / "recording"
    frames = frame_index(recording)
    records = data["iterations"]
    last = records[-1]
    footage_bounds = app.presentation_layout(last)["footage_axes"]
    if receipt.get("graph_sha256") != last["graph_sha256"]:
        raise ValueError("final footage belongs to a different graph")
    steps = [frame["step"] for frame in frames["frames"]]
    final_step = steps[-1]
    if steps[0] != 0 or final_step != receipt.get("steps"):
        raise ValueError("recording must cover the complete receipt step range")
    duration = (final_step - steps[0] + 1) / frames["fps"]
    graph_state(receipt, steps[0],policy=last["policy"])
    command = [app.ffmpeg(), "-n", "-loglevel", "error", "-f", "rawvideo",
               "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}", "-r", str(fps),
               "-i", "pipe:0", "-an", "-c:v", "libx264", "-threads", "2",
               "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(output)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    bases = {}
    zero_step_events = [(index, event, sample) for index, (event, sample) in enumerate(zip(
        receipt["graph"]["trace"], receipt["trajectory"], strict=True))
        if sample.get("start_step") == sample["step"]]
    presented, pause_seconds = set(), 0.0
    try:
        for index, record in enumerate(records):
            print(f"Rendering measured revision {record['iteration']}", flush=True)
            image = app.render(data, index, size=size)
            for _ in range(max(1, round(recap_s * fps))):
                process.stdin.write(image.tobytes())
        previous_frame, pixels = None, None
        video_frames = math.ceil(duration * fps / playback_speed)
        for number in range(video_frames):
            source_step = (final_step if number == video_frames - 1 else min(
                final_step, steps[0] + number * frames["fps"] * playback_speed / fps))
            index = max(0, bisect_right(steps, source_step) - 1)
            frame = frames["frames"][index]
            if index != previous_frame:
                with Image.open(recording / frame["file"]) as source:
                    pixels = np.asarray(source.convert("RGB"))
                previous_frame = index
            for event_index, event, sample in zero_step_events:
                if event_index in presented or sample["step"] > frame["step"]:
                    continue
                view = app.action_view(last)
                display_members = view["policy"].get("metadata", {}).get("display_members", {})
                group = next((names for names in display_members.values()
                              if event["node_id"] in names), None)
                if group and event["node_id"] != group[-1] and event["outcome"] == "succeeded":
                    presented.add(event_index)
                    continue
                prefix = receipt["graph"]["trace"][:event_index]
                display = dict(data, iterations=list(records))
                display["iterations"][-1] = dict(last, changed=[],
                    completed_nodes=[item["node_id"] for item in prefix
                                     if not event_failed(item,last["policy"])],
                    active_node=event["node_id"], failed_node=None,
                    node_outcome=event["outcome"],
                    taken_routes={item["node_id"]: item["outcome"] for item in prefix})
                base = app.render(display, len(records)-1, size=size)
                projected = app.action_view(display["iterations"][-1])
                label = next(node["action"] for node in projected["policy"]["nodes"]
                             if node["id"] == projected["active_node"])
                evidence = composite(base, pixels, "recorded preparation and decision step",
                    f"{label}  {event['outcome']}", app.BLUE, size, footage_bounds)
                count = max(1, round(1.1 * fps))
                for _ in range(count):
                    process.stdin.write(evidence.tobytes())
                pause_seconds += count / fps
                presented.add(event_index)
            state = graph_state(receipt, frame["step"], final=frame["step"] == final_step,
                                policy=last["policy"])
            key = (state["active_node"], tuple(state["completed_nodes"]), state["failed_node"],
                   tuple(state["taken_routes"].items()))
            if key not in bases:
                if len(bases) >= 4:
                    bases.pop(next(iter(bases)))
                display = dict(data, iterations=list(records))
                display["iterations"][-1] = dict(last, **state, changed=[],
                    video_label="final policy recording", run_label="running")
                bases[key] = app.render(display, len(records)-1, size=size)
            done = frame["step"] == final_step
            status = ("recorded run succeeded" if receipt.get("ok") else "recorded run failed")
            image = composite(bases[key], pixels,
                f"final LIBERO run  variation {last['seed']}",
                status if done else "running", app.GREEN if done and receipt.get("ok")
                else app.RED if done else "#525c66", size, footage_bounds)
            process.stdin.write(image.tobytes())
        # retain the final measured state without looping the simulation
        for _ in range(fps):
            process.stdin.write(image.tobytes())
        process.stdin.close()
        if process.wait(timeout=30):
            raise RuntimeError("video encoder failed")
    finally:
        if not process.stdin.closed:
            process.stdin.close()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
    return {"fps": fps, "native_simulation_fps": frames["fps"], "dimensions": list(size),
            "simulation_duration_s": duration, "recap_seconds_per_revision": recap_s,
            "playback_speed": playback_speed,
            "motion_playback_duration_s": video_frames / fps,
            "zero_step_presentation_pause_s": pause_seconds,
            "recorded_run_ok": receipt.get("ok"),
            "footage_receipt": str(receipt_path.resolve()),
            "footage_receipt_sha256": sha256(receipt_path),
            "rates": [{key: record[key] for key in ("iteration", "successes", "trials")}
                      for record in records],
            "graph_sha256": last["graph_sha256"],
            "graph_detail": last.get("graph_detail", "tools"),
            "note": "zero step actions use presentation holds not extra simulation time; "
                    "fresh final run recording does not change measured iteration rates"}


def export_session(session=None, output=None, *, size=(1280, 720), graph_detail="tools",
                   playback_speed=2):
    if not math.isfinite(playback_speed) or not 1 <= playback_speed <= 4:
        raise ValueError("playback speed must be between one and four")
    session = Path(session).resolve() if session else latest_session()
    data, manifest = session_history(session)
    print("Presenting evaluated graph revisions: " + ", ".join(
        f"v{record['iteration']:04d}" for record in data["iterations"]),flush=True)
    for skipped in data["skipped_iterations"]:
        print(f"Skipping v{skipped['iteration']:04d}: {skipped['reason']}",flush=True)
    for record in data["iterations"]:
        record["graph_detail"] = graph_detail
    identifier = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    destination = session / "visualizations" / identifier
    destination.mkdir(parents=True, exist_ok=False)
    canonical = (ROOT / "docs/visualizations/visualizer_final_graph.mp4").resolve()
    output = Path(output).resolve() if output else canonical
    if output.exists() and output != canonical:
        raise FileExistsError("existing video was not overwritten")
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt, _ = record_final(session, data["iterations"][-1], manifest,
                              destination / "replay")
    print("LIBERO stopped encoding video with two CPU threads", flush=True)
    staged = output.with_name(f".{output.stem}-{identifier}.mp4")
    render_movie(data, receipt, staged, size=size, playback_speed=playback_speed)
    staged.replace(output)
    app.export_graph_image(data["iterations"][-1], output.parent / "policy-graph.png")
    print(output, flush=True)
    return output


if __name__ == "__main__":
    export_session(Path(sys.argv[1]) if len(sys.argv) > 1 else None)
