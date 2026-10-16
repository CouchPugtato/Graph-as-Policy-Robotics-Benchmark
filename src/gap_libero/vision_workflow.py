"""Create a separately versioned image driven adaptation without model requests."""

from __future__ import annotations

import argparse
import copy
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from gap_harness.engine import Harness, dump, read
from gap_harness.models import CheckpointSet, Subgraph, Task, Topology
from gap_harness.workflow import compile_workflow

ROOT = Path(__file__).resolve().parents[2]
TASK = ROOT / "harness/tasks/libero-grocery-vision.json"
VLM_TASK = ROOT / "harness/tasks/libero-grocery-local-vlm.json"


def vision_task(original, backend="rgbd-color"):
    task = copy.deepcopy(original)
    if any(action["name"] == "observe_scene" for skill in task["skills"]
           for action in skill["actions"]):
        for skill in task["skills"]:
            if any(action["name"] == "observe_scene" for action in skill["actions"]):
                skill["shared_tools"] = True
        return local_vlm_task(task) if backend == "local-vlm" else task
    task["task_id"] = "libero-grocery-vision"
    task["instruction"] = (
        "Observe RGB-D and localize the blue alphabet_soup cylinder and green basket from "
        "image evidence before motion. Use explicit true false routers for target found "
        "container found grasp verified and placement verified. Abort on missing geometry. "
        "Pick the target and place it into the basket. No simulator object pose fallback.")
    actions = [{"name": "observe_scene", "description": "Capture calibrated RGB-D"}]
    for prefix in ("target", "container"):
        previous = "observe_scene"
        for verb, description in (("detect", "Detect the configured color in camera pixels"),
                                  ("segment", "Produce and save a binary image mask"),
                                  ("estimate", "Reconstruct metric geometry from masked depth")):
            name = f"{verb}_{prefix}"
            actions.append({"name": name, "description": description, "requires": [previous]})
            previous = name
        actions.append({"name": f"check_{prefix}", "kind": "router",
                        "description": "Return true only for valid image derived geometry",
                        "outcomes": ["true", "false", "failed"], "requires": [previous]})
    task["skills"].insert(0, {"name": "perceiving-proxy-objects", "shared_tools": True,
        "description": "CPU color segmentation and calibrated depth geometry for colored proxies",
        "instructions": "Execute observe detect segment estimate check. This is not DINO or SAM. "
                        "Rejected geometry produces a false route not a successful detection.",
        "actions": actions})
    for skill in task["skills"][1:]:
        for action in skill["actions"]:
            if action["name"] == "open_gripper":
                action["requires"] = ["check_target", "check_container"]
            if action["name"] == "lift_grasped":
                action["requires"] = ["check_grasp"]
        if any(action["name"] == "close_gripper" for action in skill["actions"]):
            skill["actions"].append({"name": "check_grasp", "kind": "router",
                "description": "Route on verified two finger contact after closing",
                "requires": ["close_gripper"], "outcomes": ["true", "false", "failed"]})
        if any(action["name"] == "settle_release" for action in skill["actions"]):
            skill["actions"].append({"name": "check_placement", "kind": "router",
                "description": "Route on the unchanged containment and release predicate",
                "requires": ["settle_release"], "outcomes": ["true", "false", "failed"]})
    task["required_actions"].extend([action["name"] for action in actions]
                                    + ["check_grasp", "check_placement"])
    for path in ("perception.target_found", "perception.container_found"):
        task["required_true"].append(path)
        task["checkpoint_evidence"].append(path)
    task["runner"]["command"].extend(["--perception-backend", "rgbd-color"])
    task["limitations"] = [
        "Custom proxy grocery task not an official LIBERO benchmark",
        "Color specific identification not neural semantic perception",
        "Known cylinder dimensions basket dimensions and support plane are priors",
        "Privileged simulator state remains confined to the evaluator and contact checks",
        "Adapted graph is hand authored and not a new model generated revision"]
    return local_vlm_task(task) if backend == "local-vlm" else task


def local_vlm_task(task):
    task["task_id"] = "libero-grocery-local-vlm"
    task["instruction"] = (
        "Use local Qwen through LM Studio to identify the designated blue upright cylinder "
        "alphabet_soup proxy and green open basket from enlarged candidate image crops. "
        "Execute observe detect identify segment estimate check before motion. "
        "Use explicit true false routes for found grasp verified and placement verified. "
        "Abort when semantic selection or geometry is missing. No cloud or oracle fallback.")
    for skill in task["skills"]:
        if any(action["name"] == "observe_scene" for action in skill["actions"]):
            skill["shared_tools"] = True
            skill["description"] = "Local VLM identification and calibrated RGB-D proxy geometry"
            skill["instructions"] = (
                "The detector proposes up to six colored regions without choosing an object. "
                "identify_target and identify_container invoke the loaded local Qwen model "
                "and choose a region by its requested description. Their selection controls "
                "the mask and depth geometry. Two total local inference calls per episode. "
                "There is no color identity fallback and no hosted VLM provider.")
            for prefix in ("target", "container"):
                identify = f"identify_{prefix}"
                if not any(action["name"] == identify for action in skill["actions"]):
                    skill["actions"].append({"name": identify,
                        "description": "Use local Qwen to choose a candidate by visual description",
                        "requires": [f"detect_{prefix}"]})
                next(action for action in skill["actions"] if action["name"] ==
                     f"segment_{prefix}")["requires"] = [identify]
                if identify not in task["required_actions"]:
                    task["required_actions"].append(identify)
    command = task["runner"]["command"]
    command[command.index("--perception-backend") + 1] = "local-vlm"
    task["runner"]["timeout_s"] = 180
    for path in ("perception.target_identified", "perception.container_identified"):
        for field in ("required_true", "checkpoint_evidence"):
            if path not in task[field]:
                task[field].append(path)
    task["limitations"] = [
        "Custom proxy grocery task not an official LIBERO benchmark",
        "Color region proposals are not a general object detector or DINO",
        "Local Qwen identifies candidates while masks remain image color components not SAM",
        "Known cylinder basket dimensions and support plane remain geometry priors",
        "LM Studio must be started manually with the local vision model loaded",
        "Prepared graph adaptation is hand authored not a new model generated revision"]
    return task


def adapt_workflow(workflow, backend="rgbd-color"):
    result = copy.deepcopy(workflow)
    if any(node["action"] == "observe_scene" for subgraph in result["subgraphs"]
           for node in subgraph["nodes"]):
        return embed_vision(add_vlm_nodes(result) if backend == "local-vlm" else result)
    topology = result["topology"]
    abort = next(node["name"] for node in topology["terminals"] if node["status"] == "failed")
    original_start = topology["start"]
    topology["start"] = "vision_target"
    for prefix, destination, evidence in (
            ("target", "vision_container", "perception.target_found"),
            ("container", original_start, "perception.container_found")):
        name = f"vision_{prefix}"
        topology["subgraphs"].insert(0 if prefix == "target" else 1, {
            "name": name, "skill": "perceiving-proxy-objects", "description": "Image evidence",
            "inputs": [], "outputs": [], "exits": ["found"], "on_error": "not_found",
            "routes": [{"outcome": "found", "target": destination},
                       {"outcome": "not_found", "target": abort}]})
        names = (["observe_scene"] if prefix == "target" else []) + [
            f"detect_{prefix}", f"segment_{prefix}", f"estimate_{prefix}", f"check_{prefix}"]
        nodes = []
        for index, action in enumerate(names):
            router = action.startswith("check_")
            nodes.append({"name": action, "kind": "router" if router else "tool",
                "action": action, "routes": [
                    {"outcome": "true" if router else "succeeded",
                     "target": "found" if router else names[index + 1]},
                    *([{"outcome": "false", "target": "not_found"}] if router else []),
                    {"outcome": "failed", "target": "not_found"}]})
        result["subgraphs"].insert(0 if prefix == "target" else 1,
            {"name": name, "rationale": "Real RGB-D tools and explicit routing",
             "start": names[0], "nodes": nodes})
        result["checkpoints"].append({"subgraph": name, "name": f"{prefix}_localized",
            "rationale": "Geometry must come from observed pixels", "evidence": evidence,
            "expected": "true"})
    specifications = {item["name"]: item for item in topology["subgraphs"]}
    for subgraph in result["subgraphs"]:
        for node in list(subgraph["nodes"]):
            action = {"close_gripper": "check_grasp", "settle_release": "check_placement"}.get(
                node["action"])
            if action is None:
                continue
            edge = next(route for route in node["routes"] if route["outcome"] == "succeeded")
            destination = edge["target"]
            name = node["name"] + "_verified"
            edge["target"] = name
            error = specifications[subgraph["name"]]["on_error"]
            index = subgraph["nodes"].index(node)
            subgraph["nodes"].insert(index + 1, {"name": name, "kind": "router", "action": action,
                "routes": [{"outcome": "true", "target": destination},
                           {"outcome": "false", "target": error},
                           {"outcome": "failed", "target": error}]})
    return embed_vision(add_vlm_nodes(result) if backend == "local-vlm" else result)


def embed_vision(workflow):
    topology = workflow["topology"]
    specs = {item["name"]: item for item in topology["subgraphs"]}
    graphs = {item["name"]: item for item in workflow["subgraphs"]}
    names = [name for name in ("vision_target", "vision_container") if name in specs]
    if not names:
        return workflow
    destination = topology["start"]
    while destination in names:
        spec = specs[destination]
        destination = next(route["target"] for route in spec["routes"]
                           if route["outcome"] in spec["exits"])
    owner, owner_spec = graphs[destination], specs[destination]
    entry = owner["start"]
    for name in reversed(names):
        source, spec = graphs[name], specs[name]
        local = {node["name"] for node in source["nodes"]}
        if local & {node["name"] for node in owner["nodes"]}:
            raise ValueError("perception node names collide with physical action nodes")
        for node in source["nodes"]:
            for route in node["routes"]:
                if route["target"] in spec["exits"]:
                    route["target"] = entry
                elif route["target"] == spec["on_error"]:
                    route["target"] = owner_spec["on_error"]
        owner["nodes"] = source["nodes"] + owner["nodes"]
        entry = source["start"]
        for check in workflow["checkpoints"]:
            if check["subgraph"] == name:
                check["subgraph"] = destination
    owner["start"] = entry
    for spec in topology["subgraphs"]:
        for route in spec["routes"]:
            if route["target"] in names:
                route["target"] = destination
    topology["start"] = destination if topology["start"] in names else topology["start"]
    topology["subgraphs"] = [item for item in topology["subgraphs"] if item["name"] not in names]
    workflow["subgraphs"] = [item for item in workflow["subgraphs"] if item["name"] not in names]
    return workflow


def add_vlm_nodes(workflow):
    for subgraph in workflow["subgraphs"]:
        for prefix in ("target", "container"):
            detect = next((node for node in subgraph["nodes"] if node["action"] ==
                           f"detect_{prefix}"), None)
            if detect is None or any(node["action"] == f"identify_{prefix}"
                                     for node in subgraph["nodes"]):
                continue
            name = detect["name"] + "_identify"
            edge = next(route for route in detect["routes"] if route["outcome"] == "succeeded")
            destination = edge["target"]
            error = next(route["target"] for route in detect["routes"]
                         if route["outcome"] == "failed")
            edge["target"] = name
            subgraph["nodes"].insert(subgraph["nodes"].index(detect) + 1, {
                "name": name, "kind": "tool", "action": f"identify_{prefix}", "routes": [
                    {"outcome": "succeeded", "target": destination},
                    {"outcome": "failed", "target": error}]})
            workflow["checkpoints"].append({"subgraph": subgraph["name"],
                "name": f"{prefix}_identity_selected",
                "rationale": "A local VLM selected the region",
                "evidence": f"perception.{prefix}_identified", "expected": "true"})
    return workflow


def prepare(source, session, backend="rgbd-color"):
    source, session = Path(source).resolve(), Path(session).resolve()
    task = vision_task(read(source.parent / "session.json")["task"], backend)
    task_path = VLM_TASK if backend == "local-vlm" else TASK
    if not task_path.exists():
        dump(task_path, task)
    elif (Task.model_validate(read(task_path)).model_dump()
          != Task.model_validate(task).model_dump()):
        raise ValueError("existing vision task differs use an explicit catalog update")
    harness = Harness(task_path, session)
    harness.prepare()
    workflow = adapt_workflow(read(source / "workflow.json"), backend)
    topology = Topology.model_validate(workflow["topology"])
    subgraphs = [Subgraph.model_validate(item) for item in workflow["subgraphs"]]
    checkpoints = CheckpointSet.model_validate({"checkpoints": workflow["checkpoints"]})
    graph = compile_workflow(topology, subgraphs, harness.task, checkpoints)
    graph["metadata"].update(perception_backend=backend, adaptation="hand-authored",
        source_graph_sha256=hashlib.sha256((source / "graph.json").read_bytes()).hexdigest())
    version = session / "v0001"
    version.mkdir()
    (version / "calls").mkdir()
    for name, value in (("workflow", workflow), ("topology", topology.model_dump()),
                        ("checkpoints", checkpoints.model_dump()), ("graph", graph)):
        dump(version / f"{name}.json", value)
    dump(version / "lineage.json", {"parent": None, "task_sha256": harness.task_sha256,
        "source_version": str(source), "model": None, "adaptation": "hand-authored"})
    dump(version / "validation.json", {"ok": True, "execution_validated": False,
        "graph_sha256": hashlib.sha256((version / "graph.json").read_bytes()).hexdigest()})
    return version


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-version", required=True, type=Path)
    parser.add_argument("--session", type=Path)
    parser.add_argument("--backend", choices=("rgbd-color", "local-vlm"), default="local-vlm")
    args = parser.parse_args()
    session = args.session or ROOT / "results/harness" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-vision-" + uuid4().hex[:8])
    print(prepare(args.source_version, session, args.backend))


if __name__ == "__main__":
    main()
