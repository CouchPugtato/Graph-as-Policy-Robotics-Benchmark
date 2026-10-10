"""Offline policy presentation viewer and bounded cpu video renderer."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
BG, FG, MUTED = "#0b1115", "#e8eef2", "#a2abb3"
GREEN, RED, ORANGE, BLUE = "#31804a", "#bb423c", "#d88b28", "#3079b5"


def resolve(base, value):
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def load_history(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or not data.get("iterations"):
        raise ValueError("history must declare schema version one and iterations")
    duration = data.get("seconds_per_iteration", 2)
    if not isinstance(duration, (int, float)) or not math.isfinite(duration) or not 0.5 <= duration <= 30:
        raise ValueError("seconds per iteration must be between half a second and thirty")
    previous = None
    previous_number = 0
    for record in data["iterations"]:
        number, successes, trials = (record[key] for key in ("iteration", "successes", "trials"))
        if any(type(value) is not int for value in (number, successes, trials)):
            raise ValueError("iteration numbers and counts must be integers")
        if number <= previous_number or not 0 <= successes <= trials or trials < 1:
            raise ValueError("iteration numbers must increase and success counts must be valid")
        graph_path = resolve(path.parent, record.get("graph", data["graph"]))
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        nodes = {node["id"]: node for node in graph["nodes"]}
        if len(nodes) != len(graph["nodes"]) or graph["start"] not in nodes:
            raise ValueError("graph requires unique nodes and a valid start")
        known = set(nodes) | set(graph["terminals"])
        if any(route not in known for node in nodes.values() for route in node["routes"].values()):
            raise ValueError("graph contains a route to an unknown node")
        record["policy"] = graph
        record["graph_path"] = graph_path
        record["changed"] = ([] if previous is None else
                             [key for key, value in nodes.items() if previous.get(key) != value])
        record["removed"] = [] if previous is None else sorted(set(previous) - set(nodes))
        record["receipt"] = None
        if record.get("result"):
            receipt = json.loads(resolve(path.parent, record["result"]).read_text(encoding="utf-8"))
            expected = receipt.get("graph_sha256")
            if expected and expected != hashlib.sha256(graph_path.read_bytes()).hexdigest():
                raise ValueError("run receipt belongs to a different graph")
            record["receipt"] = receipt
        for key in ("footage", "scene_image"):
            value = record.get(key, data.get(key))
            if value:
                record[key + "_path"] = resolve(path.parent, value)
                if not record[key + "_path"].is_file():
                    raise FileNotFoundError(record[key + "_path"])
        for key, default in (("clip_start_s", 0), ("playback_speed", 1)):
            value = record.get(key, default)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("clip offsets and playback speed must be finite and nonnegative")
            if key == "playback_speed" and not 0 < value <= 16:
                raise ValueError("playback speed must be positive and at most sixteen")
        previous, previous_number = nodes, number
    return data


def node_box(ax, x, y, label, kind="script", highlighted=False, width=2.2, height=.52):
    colors = {"script": ("#d6efd6", "#3f8f47"), "end": ("#ececec", "#999999"),
              "success": ("#d6efd6", GREEN), "failure": ("#f7d6d6", RED)}
    fill, edge = colors[kind]
    ax.add_patch(FancyBboxPatch((x-width/2, y-height/2), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.07", facecolor=fill,
        edgecolor=ORANGE if highlighted else edge, linewidth=2 if highlighted else 1, zorder=3))
    # wrap long action names instead of clipping their meaning
    label = "\n".join(textwrap.wrap(label.replace("_", " "), width=19))
    ax.text(x, y, label, ha="center", va="center", fontsize=8, color="#263039", zorder=4)


def route(ax, start, end, color, curve=0):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=8,
        connectionstyle=f"arc3,rad={curve}", lw=.9, color=color, zorder=1))


ACTION_DETAILS = {
    "open_gripper": "local gripper",
    "align_over_target": "lula ik and alignment guard",
    "descend_to_grasp": "guarded cartesian descent",
    "close_gripper": "contact and force preload",
    "lift_grasped": "lift and hold checks",
    "move_over_basket": "bounded transport",
    "position_for_release": "release pose",
    "release_object": "open and release checks",
    "retract_from_basket": "bounded retraction",
    "settle_release": "settle and containment check",
}


def policy_lanes(nodes):
    lanes = []
    releasing = False
    for node in nodes:
        if node["action"] == "release_object":
            releasing = True
        title = node["skill"]
        if title == "transporting-objects":
            title += " / release" if releasing else " / transport"
        if not lanes or lanes[-1][0] != title or len(lanes[-1][1]) >= 4:
            lanes.append((title, []))
        lanes[-1][1].append(node)
    return lanes


def policy_bubble(ax, x, y, label, subtitle, kind="script", changed=False):
    fill, edge = {"script": ("#d6efd6", "#3f8f47"),
                  "success": ("#d6efd6", GREEN), "failure": ("#f7d6d6", RED)}[kind]
    ax.add_patch(FancyBboxPatch((x-1.325,y-.32),2.65,.64,
        boxstyle="round,pad=0.02,rounding_size=0.09", facecolor=fill,
        edgecolor=ORANGE if changed else edge, linewidth=1.6 if changed else 1.1, zorder=5))
    ax.text(x,y+.105,label,ha="center",va="center",fontsize=9,color="#263039",zorder=6)
    ax.text(x,y-.135,f"({subtitle})",ha="center",va="center",
            fontsize=6.1 if len(subtitle) > 27 else 6.9,
            color="#555560",zorder=6)


def graph_panel(ax, record):
    graph = record["policy"]
    lanes = policy_lanes(graph["nodes"])
    top = 1.5 + (len(lanes)-1)*2.2
    ax.set(xlim=(0,14.3),ylim=(-1.45,top+1.25))
    ax.axis("off")
    positions = {}
    # lanes show skill phases without changing the executable flat policy
    for index, (title, members) in enumerate(lanes):
        y = top-index*2.2
        ax.add_patch(FancyBboxPatch((.35,y-.62),len(members)*3.1+.3,1.25,
            boxstyle="round,pad=0.02,rounding_size=0.12", facecolor="#fbfbfd",
            edgecolor=ORANGE if any(n["id"] in record["changed"] for n in members) else "#c8c8d0",
            linewidth=1,zorder=0))
        ax.text(.52,y+.45,title,fontsize=8,color="#777782",va="center")
        for column,node in enumerate(members):
            x = 1.95+column*3.1
            positions[node["id"]] = (x,y)
            policy_bubble(ax,x,y,node["id"],ACTION_DETAILS.get(node["action"],node["skill"]),
                          changed=node["id"] in record["changed"])
    failures = [key for key,status in graph["terminals"].items() if status != "succeeded"]
    successes = [key for key,status in graph["terminals"].items() if status == "succeeded"]
    for names,base_x,kind in ((successes,6,"success"),(failures,11.9,"failure")):
        for index,name in enumerate(names):
            y = -.65-index*.8
            positions[name] = (base_x,y)
            policy_bubble(ax,base_x,y,"done" if kind == "success" else "abort",name,kind)
    ax.set_ylim(bottom=-1.4-max(0,max(len(failures),len(successes))-1)*.8)
    for node in graph["nodes"]:
        x,y = positions[node["id"]]
        for outcome,target in node["routes"].items():
            tx,ty = positions[target]
            if outcome == "failed" and target in failures:
                bus_y = y-.92
                route(ax,(x,y-.35),(x,bus_y),RED)
                ax.plot([x,13.6,13.6],[bus_y,bus_y,ty+.23],color=RED,lw=.75,zorder=1)
                route(ax,(13.6,ty+.23),(tx+1.36,ty),RED)
            elif y == ty:
                route(ax,(x+1.36,y),(tx-1.36,ty),GREEN)
            else:
                route(ax,(x,y-.35),(tx,ty+.35),GREEN,.08)


def panel(fig, bounds):
    fig.patches.append(FancyBboxPatch((bounds[0], bounds[1]), bounds[2], bounds[3],
        boxstyle="round,pad=0.004,rounding_size=0.012", transform=fig.transFigure,
        facecolor="#fafafa", edgecolor="#c8cdd1", linewidth=1, zorder=-1))


def render(data, index, progress=1, size=(1920, 1080), video_frame=None):
    record = data["iterations"][index]
    fig = plt.figure(figsize=(size[0]/100, size[1]/100), dpi=100, facecolor=BG)
    try:
        panel(fig, (.025, .03, .58, .94))
        panel(fig, (.63, .535, .345, .435))
        panel(fig, (.63, .03, .345, .465))
        graph_ax = fig.add_axes((.035, .07, .565, .85))
        graph_panel(graph_ax, record)
        chart = fig.add_axes((.681, .595, .275, .30), facecolor="#fafafa")
        records = data["iterations"]
        xs = [r["iteration"] for r in records]
        ys = [100*r["successes"]/r["trials"] for r in records]
        visible_x, visible_y = xs[:index+1], ys[:index+1]
        if index > 0 and progress < 1:
            visible_x = xs[:index] + [xs[index-1] + progress*(xs[index]-xs[index-1])]
            visible_y = ys[:index] + [ys[index-1] + progress*(ys[index]-ys[index-1])]
        chart.plot(visible_x, visible_y, color=BLUE, lw=2.1)
        chart.scatter(visible_x, visible_y, color=BLUE, s=37, zorder=3)
        chart.set(xlim=(xs[0]-.4, xs[-1]+.4), ylim=(0, 100), xlabel="iteration",
                  ylabel="success rate (%)", xticks=xs[::max(1, len(xs)//10)], yticks=[0,25,50,75,100])
        chart.tick_params(labelsize=9, colors="#747b82", length=0)
        chart.xaxis.label.set_color("#747b82")
        chart.yaxis.label.set_color("#747b82")
        chart.grid(axis="y", color="#dfe3e6", lw=.6)
        for spine in chart.spines.values():
            spine.set_visible(False)
        fig.text(.651, .929, f"{ys[index]:g}%    {record['successes']} / {record['trials']} variations",
                 color=BLUE, fontsize=15)
        footage_ax = fig.add_axes((.637, .039, .331, .445))
        footage_ax.axis("off")
        if video_frame is not None:
            footage_ax.imshow(video_frame)
        elif record.get("scene_image_path"):
            with Image.open(record["scene_image_path"]) as image:
                footage_ax.imshow(ImageOps.pad(image.convert("RGB"), (640,360), color="#1b2329"))
        else:
            footage_ax.set_facecolor("#1b2329")
            footage_ax.text(.5, .5, "add simulation footage", transform=footage_ax.transAxes,
                            ha="center", color=MUTED)
        video_label = ("simulation replay" if video_frame is not None else
                       "static scene preview  no recording attached")
        footage_ax.text(.025, .94, video_label, transform=footage_ax.transAxes, fontsize=9,
                        color=FG, bbox=dict(boxstyle="round,pad=.4", fc="#14202a", ec="#68707a"))
        receipt = record["receipt"]
        if receipt:
            status = receipt.get("evaluation", {}).get("status", "unknown")
            label = "attached run evaluator  " + status
            color = GREEN if status == "succeeded" else RED
        else:
            label, color = "no run receipt attached", "#525c66"
        footage_ax.text(.025, .04, label, transform=footage_ax.transAxes, fontsize=10,
                        color="white", bbox=dict(boxstyle="round,pad=.45", fc=color, ec="none"))
        fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    finally:
        plt.close(fig)


def ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


class Clip:
    def __init__(self, record, fps):
        self.process = None
        if record.get("footage_path"):
            speed = record.get("playback_speed", 1)
            filters = (f"setpts=PTS/{speed},fps={fps},scale=640:480:force_original_aspect_ratio=decrease,"
                       "pad=640:480:(ow-iw)/2:(oh-ih)/2")
            self.process = subprocess.Popen([ffmpeg(), "-nostdin", "-loglevel", "error", "-threads", "2",
                "-protocol_whitelist", "file,pipe",
                "-stream_loop", "-1", "-ss", str(record.get("clip_start_s", 0)),
                "-i", str(record["footage_path"]), "-an", "-vf", filters,
                "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def frame(self):
        if self.process is None:
            return None
        frame = self.process.stdout.read(640*480*3)
        if len(frame) != 640*480*3:
            raise RuntimeError("could not decode the attached simulation recording")
        return np.frombuffer(frame, dtype=np.uint8).reshape(480,640,3)

    def close(self):
        if self.process:
            self.process.stdout.close()
            if self.process.poll() is None:
                self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)


def export(data, output, fps, size, seconds=None, replay=None):
    duration = data.get("seconds_per_iteration", 2) if seconds is None else seconds
    frames = max(1, round(duration*fps))
    output.parent.mkdir(parents=True, exist_ok=True)
    # encode on the cpu with a small thread budget and no simulator processes
    command = [ffmpeg(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{size[0]}x{size[1]}", "-r", str(fps), "-i", "pipe:0", "-an",
               "-c:v", "libx264", "-threads", "2", "-preset", "fast", "-crf", "20",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        records = data["iterations"]
        if replay:
            replay_record = dict(records[-1], footage_path=replay, clip_start_s=0, playback_speed=1)
            clip = Clip(replay_record, fps)
            try:
                # keep the scene replay continuous while the history advances
                for frame in range(frames):
                    position = frame * len(records) / frames
                    index = min(int(position), len(records) - 1)
                    display = dict(data)
                    display["iterations"] = list(records)
                    display["iterations"][index] = dict(records[index], receipt=replay_record["receipt"])
                    image = render(display, index, min(1, (position-index)/.65), size, clip.frame())
                    process.stdin.write(image.tobytes())
            finally:
                clip.close()
        for index, record in enumerate([] if replay else records):
            print(f"rendering iteration {record['iteration']}", flush=True)
            clip = Clip(record, fps)
            try:
                for frame in range(frames):
                    progress = min(1, (frame+1)/max(1, frames*.65))
                    image = render(data, index, progress, size, clip.frame())
                    process.stdin.write(image.tobytes())
            finally:
                clip.close()
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
    output.with_suffix(".manifest.json").write_text(json.dumps({
        "schema_version": 1, "rates": [{key: r[key] for key in
        ("iteration", "successes", "trials")} for r in data["iterations"]],
        "graphs": [str(r["graph_path"]) for r in data["iterations"]],
        "fps": fps, "dimensions": size, "seconds_per_iteration": duration,
        "continuous_replay": str(replay) if replay else None,
        "simulation_started": False, "models_loaded": False,
    }, indent=2), encoding="utf-8")
    print(output)


def viewer(data):
    import tkinter as tk
    from PIL import ImageTk
    root = tk.Tk()
    root.title("Graph policy visualizer")
    root.configure(bg=BG)
    label = tk.Label(root, bg=BG)
    label.pack()
    playing = False
    position = 0
    duration = data.get("seconds_per_iteration", 2)
    clip = None
    active_index = None

    def update(value):
        nonlocal clip, active_index
        index = int(float(value))
        if active_index != index:
            if clip:
                clip.close()
            clip = Clip(data["iterations"][index], 5)
            active_index = index
        image = render(data, index, size=(1280,720), video_frame=clip.frame())
        photo = ImageTk.PhotoImage(Image.fromarray(image))
        label.configure(image=photo)
        label.image = photo

    slider = tk.Scale(root, from_=0, to=len(data["iterations"])-1, orient="horizontal",
                      command=update, length=1120, bg=BG, fg=FG, label="iteration index")
    slider.pack()

    def tick():
        nonlocal position
        if playing:
            position += .2/duration
            if position >= len(data["iterations"]):
                position = 0
            if slider.get() != int(position):
                slider.set(int(position))
            else:
                update(int(position))
        root.after(200, tick)

    def toggle():
        nonlocal playing, position
        position = float(slider.get())
        playing = not playing
        button.configure(text="Pause" if playing else "Play")

    button = tk.Button(root, text="Play", command=toggle)
    button.pack()

    def close():
        if clip:
            clip.close()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", close)
    update(0)
    tick()
    root.mainloop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["preview", "export", "view", "validate"])
    parser.add_argument("--history", type=Path, default=HERE/"history.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--iteration", type=int, default=10, help="iteration number for preview")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--width", type=int, choices=[1280,1920], default=1920)
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--replay-footage", type=Path, help="continuous scene video during history playback")
    args = parser.parse_args()
    if not 1 <= args.fps <= 60 or (args.seconds is not None and
            (not math.isfinite(args.seconds) or not .5 <= args.seconds <= 30)):
        parser.error("use fps from one to sixty and seconds from half to thirty")
    data = load_history(args.history)
    if args.replay_footage and (args.mode != "export" or not args.replay_footage.is_file()):
        parser.error("replay footage requires export mode and an existing video")
    size = (args.width, args.width*9//16)
    if args.mode == "validate":
        print(json.dumps({"ok": True, "iterations": len(data["iterations"]), "models_loaded": False}))
    elif args.mode == "view":
        viewer(data)
    elif args.mode == "preview":
        index = next((i for i,r in enumerate(data["iterations"]) if r["iteration"] == args.iteration), None)
        if index is None:
            parser.error("requested iteration is not in the history")
        output = args.output or HERE/"output/preview.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        clip = Clip(data["iterations"][index], args.fps)
        try:
            Image.fromarray(render(data, index, size=size, video_frame=clip.frame())).save(output)
        finally:
            clip.close()
        print(output)
    else:
        export(data, args.output or HERE/"output/process.mp4", args.fps, size, args.seconds,
               args.replay_footage.resolve() if args.replay_footage else None)


if __name__ == "__main__":
    main()
