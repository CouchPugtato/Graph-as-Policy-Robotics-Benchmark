"""Offline policy presentation viewer and bounded cpu video renderer."""
from __future__ import annotations

import argparse
import copy
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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon
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
    label = "\n".join(textwrap.wrap(label.replace("_", " "), width=19))
    ax.text(x, y, label, ha="center", va="center", fontsize=8, color="#263039", zorder=4)


def route(ax, start, end, color, curve=0, linewidth=.9):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=8,
        connectionstyle=f"arc3,rad={curve}", lw=linewidth, color=color, zorder=1))


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
VISION_DETAILS = {
    "observe_scene": "calibrated RGB-D camera",
    "detect_target": "color object detection",
    "detect_container": "color object detection",
    "identify_target": "local VLM identification",
    "identify_container": "local VLM identification",
    "segment_target": "binary foreground mask",
    "segment_container": "binary foreground mask",
    "estimate_target": "depth and cylinder fit",
    "estimate_container": "depth and basket fit",
}
DECISION_LABELS = {"check_target": "target found", "check_container": "container found",
                   "check_grasp": "grasp verified", "check_placement": "placement verified"}
SHORT_LABELS = {
    "observe_scene": "observe", "detect_target": "detect\ntarget",
    "identify_target": "identify\ntarget", "segment_target": "segment\ntarget",
    "estimate_target": "fit\ntarget", "check_target": "target\nfound",
    "detect_container": "detect\nbasket", "identify_container": "identify\nbasket",
    "segment_container": "segment\nbasket", "estimate_container": "fit\nbasket",
    "check_container": "basket\nfound", "open_gripper": "open", "align_over_target": "align",
    "descend_to_grasp": "descend", "close_gripper": "close", "check_grasp": "held",
    "lift_grasped": "lift", "move_over_basket": "transport", "position_for_release": "lower",
    "release_object": "release", "retract_from_basket": "retract", "settle_release": "settle",
    "check_placement": "placed",
}
POPUP_DETAILS = {
    "observe_scene": "Capture a calibrated color image and depth map of the scene.",
    "detect_target": "Propose separate colored object regions for target identification.",
    "detect_container": "Propose separate colored object regions for basket identification.",
    "identify_target": "Ask the local vision model to select the designated target region.",
    "identify_container": "Ask the local vision model to select the basket region.",
    "segment_target": "Create the selected target mask for depth reconstruction.",
    "segment_container": "Create the selected basket mask for depth reconstruction.",
    "estimate_target": "Fit the cylinder geometry from the masked depth measurements.",
    "estimate_container": "Fit the basket walls and rim from masked depth measurements.",
    "check_target": "Continue only if target identification and geometry are verified.",
    "check_container": "Continue only if basket identification and geometry are verified.",
    "open_gripper": "Open the fingers before approaching the target.",
    "align_over_target": "Move above the perceived target and verify alignment.",
    "descend_to_grasp": "Descend toward the grasp pose with motion and collision checks.",
    "close_gripper": "Close the fingers and apply the bounded grasp preload.",
    "check_grasp": "Route on the measured two finger grasp verification result.",
    "lift_grasped": "Lift the target while checking that the grasp remains stable.",
    "move_over_basket": "Carry the held target above the perceived basket.",
    "position_for_release": "Lower the target into the release position.",
    "release_object": "Open the fingers to release the target.",
    "retract_from_basket": "Move the gripper clear of the basket after release.",
    "settle_release": "Allow the released object to settle before checking placement.",
    "check_placement": "Verify that the target is released and contained in the basket.",
}


def action_view(record):
    if record.get("graph_detail", "tools") == "tools":
        return record
    result = copy.deepcopy(record)
    graph = result["policy"]
    metadata = graph.setdefault("metadata", {})
    membership = metadata.get("subgraph_membership", {})
    nodes = graph["nodes"]
    incoming = {}
    for node in nodes:
        for target in node["routes"].values():
            incoming.setdefault(target, set()).add(node["id"])
    groups, current = [], []
    for node in nodes:
        if node["action"] not in VISION_DETAILS:
            if current:
                groups.append(current)
                current = []
            continue
        if current and not (
            membership.get(current[-1]["id"], current[-1]["skill"]) ==
            membership.get(node["id"], node["skill"]) and
            current[-1]["routes"].get("succeeded") == node["id"] and
            incoming.get(node["id"]) == {current[-1]["id"]} and
            graph.get("start") != node["id"]
        ):
            groups.append(current)
            current = []
        current.append(node)
    if current:
        groups.append(current)
    mapping, folded, replacements = {}, set(), {}
    for members in groups:
        if len(members) < 2 or any(
            set(node["routes"]) != {"succeeded", "failed"} for node in members
        ) or len({node["routes"]["failed"] for node in members}) != 1:
            continue
        first, last = members[0], members[-1]
        identifier = first["id"]
        ids = {node["id"] for node in members}
        if first["routes"]["failed"] in ids:
            continue
        label = "locate basket" if any("container" in n["action"] for n in members) else "locate target"
        replacements[identifier] = dict(first, action=label, routes=dict(last["routes"]))
        mapping.update({name: identifier for name in ids})
        folded.update(ids - {identifier})
        metadata.setdefault("node_kinds", {})[identifier] = "subgraph"
        metadata.setdefault("display_members", {})[identifier] = [n["id"] for n in members]
        completed = set(record.get("completed_nodes", []))
        result["completed_nodes"] = [name for name in result.get("completed_nodes", []) if name not in ids]
        if ids <= completed:
            result["completed_nodes"].append(identifier)
        routes = result.setdefault("taken_routes", {})
        outcome = ("failed" if any(routes.get(name) == "failed" for name in ids)
                   else routes.get(last["id"]))
        for name in ids:
            routes.pop(name, None)
        if outcome:
            routes[identifier] = outcome
    graph["nodes"] = [replacements.get(n["id"], n) for n in nodes if n["id"] not in folded]
    for node in graph["nodes"]:
        node["routes"] = {key: mapping.get(value, value) for key, value in node["routes"].items()}
    if "start" in graph:
        graph["start"] = mapping.get(graph["start"], graph["start"])
    result["changed"] = list(dict.fromkeys(mapping.get(name, name) for name in record.get("changed", [])))
    for key in ("active_node", "failed_node"):
        result[key] = mapping.get(record.get(key), record.get(key))
    return result


def policy_lanes(nodes, membership=None):
    if membership:
        groups = {}
        for node in nodes:
            title = membership.get(node["id"], node["skill"])
            groups.setdefault(title, []).append(node)
        return list(groups.items())
    lanes = []
    releasing = False
    for node in nodes:
        if node["action"] == "release_object":
            releasing = True
        title = membership.get(node["id"], node["skill"]) if membership else node["skill"]
        if not membership and title == "transporting-objects":
            title += " / release" if releasing else " / transport"
        limit = 4
        if not lanes or lanes[-1][0] != title or len(lanes[-1][1]) >= limit:
            lanes.append((title, []))
        lanes[-1][1].append(node)
    return lanes


def policy_bubble(ax, x, y, label, subtitle, kind="script", changed=False,
                  active=False, completed=False, failed=False, text_scale=1, compact=False):
    pixels = ax.get_window_extent().width * 2.65 / (ax.get_xlim()[1] - ax.get_xlim()[0])
    font_scale = min(1, pixels / 145) * text_scale
    fill, edge = {"script": ("#d6efd6", "#3f8f47"),
                  "tool": ("#dcecff", BLUE),
                  "subgraph": ("#e9e1f4", "#8563ae"),
                  "router": ("#eeeeee", "#888888"),
                  "success": ("#d6efd6", GREEN), "failure": ("#f7d6d6", RED)}[kind]
    if completed:
        fill, edge = "#c4e7c9", GREEN
    if active:
        fill, edge = "#ffe4a5", ORANGE
    if failed:
        fill, edge = "#f7d6d6", RED
    style = dict(facecolor=fill,
                 edgecolor=ORANGE if changed and not (active or completed or failed) else edge,
                 linewidth=2.4 if active or failed else 1.6 if changed else 1.1, zorder=5)
    if kind == "router":
        ax.add_patch(Polygon([(x-1.325,y),(x,y+.4),(x+1.325,y),(x,y-.4)], **style))
    else:
        ax.add_patch(FancyBboxPatch((x-1.325,y-.32),2.65,.64,
            boxstyle="round,pad=0.02,rounding_size=0.09", **style))
    if compact:
        scale = min(1, pixels / 80) * text_scale
        ax.text(x, y, label.replace("_", " "), ha="center", va="center",
                fontsize=12*scale, color="#263039", zorder=6, linespacing=1.1)
        return
    label = "\n".join(textwrap.wrap(label.replace("_", " "), width=22))
    ax.text(x,y+.105,label,ha="center",va="center",fontsize=8*font_scale,
            color="#263039",zorder=6)
    ax.text(x,y-.135,f"({subtitle})",ha="center",va="center",
            fontsize=(6.1 if len(subtitle) > 27 else 6.9)*font_scale,
            color="#555560",zorder=6)


def graph_panel(ax, record):
    record = action_view(record)
    graph = record["policy"]
    membership = graph.get("metadata", {}).get("subgraph_membership")
    lanes = policy_lanes(graph["nodes"], membership)
    top = 1.5 + (len(lanes)-1)*2.2
    right = max(14.3, max(len(members) for _, members in lanes)*3.1+.9)
    ax.set(xlim=(0,right),ylim=(-1.45,top+1.25))
    ax.axis("off")
    positions = {}
    text_scale = record.get("graph_text_scale", 1)
    # lanes show skill phases without changing the executable flat policy
    for index, (title, members) in enumerate(lanes):
        y = top-index*2.2
        ax.add_patch(FancyBboxPatch((.35,y-.62),len(members)*3.1+.3,1.25,
            boxstyle="round,pad=0.02,rounding_size=0.12", facecolor="#fbfbfd",
            edgecolor=ORANGE if any(n["id"] in record["changed"] for n in members) else "#c8c8d0",
            linewidth=1,zorder=0))
        ax.text(.52,y+.45,title,fontsize=8*text_scale,color="#777782",va="center")
        for column,node in enumerate(members):
            x = 1.95+column*3.1
            positions[node["id"]] = (x,y)
            kind = graph.get("metadata", {}).get("node_kinds", {}).get(
                node["id"], record.get("node_kinds", {}).get(node["id"], "script"))
            detail = ("perception subgraph" if kind == "subgraph" else
                      "true or false" if kind == "router" else
                      VISION_DETAILS[node["action"]] if node["action"] in VISION_DETAILS else
                      "bounded libero action" if record.get("backend") == "libero" else
                      ACTION_DETAILS.get(node["action"],node["skill"]))
            label = SHORT_LABELS.get(node["action"],
                                     DECISION_LABELS.get(node["action"], node["action"]))
            policy_bubble(ax,x,y,label if membership else node["id"], detail,
                          kind=kind,
                          changed=node["id"] in record["changed"],
                          active=node["id"] == record.get("active_node"),
                          completed=node["id"] in record.get("completed_nodes", []),
                          failed=node["id"] == record.get("failed_node"), text_scale=text_scale,
                          compact=True)
    failures = [key for key,status in graph["terminals"].items() if status != "succeeded"]
    successes = [key for key,status in graph["terminals"].items() if status == "succeeded"]
    for names,base_x,kind in ((successes,right*.42,"success"),
                             (failures,right*.83,"failure")):
        for index,name in enumerate(names):
            y = -.65-index*.8
            positions[name] = (base_x,y)
            policy_bubble(ax,base_x,y,"done" if kind == "success" else "abort",name,kind,
                          text_scale=text_scale, compact=True)
    ax.set_ylim(bottom=-1.4-max(0,max(len(failures),len(successes))-1)*.8)
    failure_labels = set()
    for node in graph["nodes"]:
        x,y = positions[node["id"]]
        targets = {}
        for outcome, target in node["routes"].items():
            targets.setdefault(target, []).append(outcome)
        for target, outcomes in targets.items():
            tx,ty = positions[target]
            negative = all(outcome in {"failed", "false", "not_found", "rejected"}
                           for outcome in outcomes)
            color = RED if negative else GREEN
            taken = record.get("taken_routes", {}).get(node["id"]) in outcomes
            linewidth = 2 if taken else .7
            label = " / ".join(outcomes) if outcomes != ["succeeded"] else ""
            if negative and target in failures:
                bus_y = y-.92
                route(ax,(x,y-.35),(x,bus_y),RED,linewidth=linewidth)
                ax.plot([x,right-.7,right-.7],[bus_y,bus_y,ty+.23],color=RED,lw=.75,zorder=1)
                route(ax,(right-.7,ty+.23),(tx+1.36,ty),RED)
                key = (y, target, label)
                if key not in failure_labels:
                    ax.text(x+.05,y-.66,label,color=RED,fontsize=5.5*text_scale,zorder=7)
                    failure_labels.add(key)
            elif y == ty:
                route(ax,(x+1.36,y),(tx-1.36,ty),color,linewidth=linewidth)
                ax.text((x+tx)/2,y+.08,label,color=color,fontsize=5.5*text_scale,ha="center",zorder=7)
            else:
                route(ax,(x,y-.35),(tx,ty+.35),color,.08,linewidth=linewidth)
                ax.text((x+tx)/2,(y+ty)/2,label,color=color,fontsize=5.5*text_scale,zorder=7,
                        bbox=dict(facecolor="#fbfbfd",edgecolor="none",pad=.5))
    active = record.get("active_node") or record.get("failed_node")
    node = next((node for node in graph["nodes"] if node["id"] == active), None)
    if node is not None:
        action = node["action"]
        title = action.replace("_", " ")
        detail = POPUP_DETAILS.get(action, ACTION_DETAILS.get(action, action.replace("_", " ")))
        text = title + "\n" + "\n".join(textwrap.wrap(detail, width=47))
        outcome = record.get("node_outcome") or record.get("taken_routes", {}).get(active)
        if outcome:
            text += "\nroute  " + outcome
        spare = next(((index, members) for index, (_, members) in enumerate(lanes)
                      if len(members)*3.1 < right*.45), None)
        if spare:
            index, _ = spare
            location = (right*.52, top-index*2.2)
        else:
            location = (right*.38, top+.85)
        ax.annotate(text, xy=positions[active], xytext=location,
                    ha="left", va="center", fontsize=12, color="#263039", linespacing=1.5,
                    bbox=dict(boxstyle="round,pad=.65", fc="#fff8e6", ec=ORANGE, lw=1.5),
                    arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2,
                                    connectionstyle="arc3,rad=.08"), zorder=20)


def export_graph_image(record, output):
    graph = action_view(record)["policy"]
    lanes = policy_lanes(graph["nodes"], graph.get("metadata", {}).get("subgraph_membership"))
    width = max(20, max(len(nodes) for _, nodes in lanes) * 3.6)
    figure, axis = plt.subplots(figsize=(width, max(11, len(lanes) * 4)), dpi=160)
    try:
        graph_panel(axis, dict(record, graph_text_scale=2.5))
        figure.tight_layout()
        figure.savefig(output, facecolor="white")
    finally:
        plt.close(figure)


def panel(fig, bounds):
    fig.patches.append(FancyBboxPatch((bounds[0], bounds[1]), bounds[2], bounds[3],
        boxstyle="round,pad=0.004,rounding_size=0.012", transform=fig.transFigure,
        facecolor="#fafafa", edgecolor="#c8cdd1", linewidth=1, zorder=-1))


def presentation_layout(record):
    graph = action_view(record)["policy"]
    membership = graph.get("metadata", {}).get("subgraph_membership")
    wide = membership and max(len(nodes) for _, nodes in policy_lanes(graph["nodes"], membership)) > 8
    if wide:
        return {"graph_panel": (.025, .51, .95, .46), "graph_axes": (.035, .535, .93, .40),
                "chart_panel": (.025, .03, .345, .43), "chart_axes": (.076, .09, .275, .29),
                "rate": (.046, .419), "footage_panel": (.395, .03, .58, .43),
                "footage_axes": (.402, .039, .566, .412)}
    return {"graph_panel": (.025, .03, .58, .94), "graph_axes": (.035, .07, .565, .85),
            "chart_panel": (.63, .535, .345, .435), "chart_axes": (.681, .595, .275, .30),
            "rate": (.651, .929), "footage_panel": (.63, .03, .345, .465),
            "footage_axes": (.637, .039, .331, .445)}


def render(data, index, progress=1, size=(1920, 1080), video_frame=None):
    record = data["iterations"][index]
    layout = presentation_layout(record)
    fig = plt.figure(figsize=(size[0]/100, size[1]/100), dpi=100, facecolor=BG)
    try:
        for name in ("graph_panel", "chart_panel", "footage_panel"):
            panel(fig, layout[name])
        graph_ax = fig.add_axes(layout["graph_axes"])
        graph_panel(graph_ax, record)
        chart = fig.add_axes(layout["chart_axes"], facecolor="#fafafa")
        records = data["iterations"]
        xs = [r["iteration"] for r in records]
        ys = [100*r["successes"]/r["trials"] for r in records]
        visible_x, visible_y = xs[:index+1], ys[:index+1]
        if index > 0 and progress < 1:
            visible_x = xs[:index] + [xs[index-1] + progress*(xs[index]-xs[index-1])]
            visible_y = ys[:index] + [ys[index-1] + progress*(ys[index]-ys[index-1])]
        chart.plot(visible_x, visible_y, color=BLUE, lw=2.1)
        chart.scatter(visible_x, visible_y, color=BLUE, s=37, zorder=3)
        chart.set(xlim=(xs[0]-.4, xs[-1]+.4), ylim=(0, 105), xlabel="iteration",
                  ylabel="success rate (%)", xticks=xs[::max(1, len(xs)//10)], yticks=[0,25,50,75,100])
        chart.tick_params(labelsize=9, colors="#747b82", length=0)
        chart.xaxis.label.set_color("#747b82")
        chart.yaxis.label.set_color("#747b82")
        chart.grid(axis="y", color="#dfe3e6", lw=.6)
        for spine in chart.spines.values():
            spine.set_visible(False)
        fig.text(*layout["rate"], f"{ys[index]:g}%    {record['successes']} / {record['trials']} variations",
                 color=BLUE, fontsize=15)
        footage_ax = fig.add_axes(layout["footage_axes"])
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
        video_label = record.get("video_label") or (
            "simulation replay" if video_frame is not None else
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
        label, color = record.get("run_label", label), record.get("run_color", color)
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
    parser.add_argument("mode", choices=["preview", "export", "view", "validate", "libero"])
    parser.add_argument("--session", type=Path, help="harness session for libero video")
    parser.add_argument("--graph-detail", choices=["actions", "tools"], default="tools")
    parser.add_argument("--playback-speed", type=float, default=2,
                        help="LIBERO footage speed from one to four without changing simulation")
    parser.add_argument("--history", type=Path, default=HERE/"history.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--iteration", type=int, default=10, help="iteration number for preview")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--width", type=int, choices=[1280,1920], default=1920)
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--replay-footage", type=Path, help="continuous scene video during history playback")
    args = parser.parse_args()
    if args.mode == "libero":
        from libero_export import export_session

        try:
            export_session(args.session, args.output, size=(args.width, args.width*9//16),
                           graph_detail=args.graph_detail, playback_speed=args.playback_speed)
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
            parser.exit(1, f"LIBERO video stopped: {error}\n")
        return
    if not 1 <= args.fps <= 60 or (args.seconds is not None and
            (not math.isfinite(args.seconds) or not .5 <= args.seconds <= 30)):
        parser.error("use fps from one to sixty and seconds from half to thirty")
    data = load_history(args.history)
    for record in data["iterations"]:
        record["graph_detail"] = args.graph_detail
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
