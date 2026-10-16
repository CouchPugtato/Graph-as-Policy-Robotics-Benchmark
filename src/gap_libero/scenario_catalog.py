"""Task catalogs and transparent reference policies for custom LIBERO scenarios."""

import json
from pathlib import Path

from gap_isaac.graph_policy import GraphPolicy

ROOT = Path(__file__).resolve().parents[2]
PERCEPTION = (
    "observe_scene",
    "detect_target",
    "identify_target",
    "segment_target",
    "estimate_target",
    "check_target",
    "detect_container",
    "identify_container",
    "segment_container",
    "estimate_container",
    "check_container",
)
GRASP = (
    "open_gripper",
    "align_over_target",
    "descend_to_grasp",
    "close_gripper",
    "check_grasp",
    "lift_grasped",
)
PLACE = (
    "move_over_basket",
    "position_for_release",
    "release_object",
    "retract_from_basket",
    "settle_release",
    "check_placement",
)
ROUTERS = {
    "check_target",
    "check_container",
    "check_grasp",
    "check_placement",
    "select_next_item",
    "check_all_packed",
    "check_popcorn_complete",
}
STATE_ACTIONS = {
    "select_next_item",
    "select_knob_on",
    "select_knob_off",
    "check_all_packed",
    "check_popcorn_complete",
}
PREREQUISITES = {
    "detect_target": ["observe_scene"],
    "identify_target": ["detect_target"],
    "segment_target": ["identify_target"],
    "estimate_target": ["segment_target"],
    "check_target": ["estimate_target"],
    "detect_container": ["observe_scene"],
    "identify_container": ["detect_container"],
    "segment_container": ["identify_container"],
    "estimate_container": ["segment_container"],
    "check_container": ["estimate_container"],
    "open_gripper": ["check_target"],
    "align_over_target": ["check_target", "open_gripper"],
    "descend_to_grasp": ["align_over_target"],
    "close_gripper": ["descend_to_grasp"],
    "check_grasp": ["close_gripper"],
    "lift_grasped": ["check_grasp"],
    "move_over_basket": ["lift_grasped", "check_container"],
    "position_for_release": ["move_over_basket"],
    "release_object": ["close_gripper"],
    "retract_from_basket": ["release_object"],
    "settle_release": ["retract_from_basket"],
    "check_placement": ["settle_release"],
    "return_ready": ["retract_from_basket"],
    "select_knob_on": ["check_placement"],
    "select_knob_off": ["wait_heating"],
    "turn_stove_on": ["check_grasp"],
    "turn_stove_off": ["check_grasp", "wait_heating"],
    "wait_heating": ["turn_stove_on", "release_object", "return_ready"],
    "check_popcorn_complete": ["turn_stove_off", "wait_heating"],
}


def sequence(scenario):
    transfer = [*PERCEPTION, *GRASP, *PLACE]
    if scenario == "fulfillment":
        return transfer
    if scenario == "packing":
        return ["select_next_item", *transfer, "return_ready", "check_all_packed"]
    if scenario != "popcorn":
        raise ValueError("unknown deployed scenario")
    turn = [*PERCEPTION[:6], *GRASP[:5]]
    return [
        *transfer,
        "return_ready",
        "select_knob_on",
        *turn,
        "turn_stove_on",
        "release_object",
        "retract_from_basket",
        "return_ready",
        "wait_heating",
        "select_knob_off",
        *turn,
        "turn_stove_off",
        "release_object",
        "retract_from_basket",
        "return_ready",
        "check_popcorn_complete",
    ]


def reference_graph(scenario):
    actions = sequence(scenario)
    nodes = []
    for index, action in enumerate(actions):
        identifier = f"step_{index:02d}_{action}"
        next_node = (
            f"step_{index + 1:02d}_{actions[index + 1]}" if index + 1 < len(actions) else "done"
        )
        routes = {"true" if action in ROUTERS else "succeeded": next_node, "failed": "abort"}
        if action in ROUTERS:
            routes["false"] = "abort"
        nodes.append(
            {"id": identifier, "action": action, "skill": "local-libero-actions", "routes": routes}
        )
    if scenario == "packing":
        nodes[0]["routes"]["false"] = nodes[-1]["id"]
        nodes[-2]["routes"]["succeeded"] = nodes[0]["id"]
    graph = {
        "schema_version": 1,
        "graph_id": f"libero-{scenario}-reference",
        "start": nodes[0]["id"],
        "nodes": nodes,
        "terminals": {"done": "succeeded", "abort": "failed"},
        "metadata": {
            "scenario": scenario,
            "perception_backend": "local-vlm",
            "reference_policy": True,
            "model_generated": False,
            "node_kinds": {n["id"]: "router" if n["action"] in ROUTERS else "tool" for n in nodes},
        },
    }
    GraphPolicy.from_dict(graph)
    return graph


def task_catalog(scenario):
    descriptions = {
        "fulfillment": (
            "Pick the designated grocery object and place it in the basket. The requested object "
            "varies among blue and red cylinders and brown and yellow boxes. Test position "
            "variation basket swaps and object permutations together. Identify the requested "
            "object using local Qwen and derive all motion geometry from RGB-D."
        ),
        "packing": (
            "Pack both requested grocery objects into the basket using a bounded item loop. "
            "The two item set and positions vary by episode. Use select_next_item true to process "
            "an item and false to check_all_packed. Verify each placement before selecting another "
            "item and return_ready before continuing. Use local Qwen for every item and basket "
            "identification."
        ),
        "popcorn": (
            "Move the pan by its blue handle onto the purple stove burner. Then physically grasp "
            "and turn the green knob on wait through the bounded heating abstraction and turn it "
            "off. Use select_knob_on and select_knob_off to change the active target before "
            "observing it. Turn actions require a verified knob grasp. Release and retract after "
            "each operation. Finish with check_popcorn_complete. This is a physical proxy with no "
            "thermal or food simulation. All object geometry must come from local Qwen selected "
            "RGB-D evidence."
        ),
    }
    if scenario not in descriptions:
        raise ValueError("unknown scenario")
    deployed = sorted(set(sequence(scenario)))
    skills = []
    for name, predicate, shared in (
        ("perceiving-proxy-objects", lambda a: a in PERCEPTION, True),
        ("managing-task-state", lambda a: a in STATE_ACTIONS, True),
        (
            "libero-physical-actions",
            lambda a: a not in PERCEPTION and a not in STATE_ACTIONS,
            False,
        ),
    ):
        actions = [
            {
                "name": a,
                "description": a.replace("_", " "),
                "requires": PREREQUISITES.get(a, []),
                "kind": "router" if a in ROUTERS else "tool",
                "outcomes": ["true", "false", "failed"]
                if a in ROUTERS
                else ["succeeded", "failed"],
            }
            for a in deployed
            if predicate(a)
        ]
        if actions:
            skills.append(
                {
                    "name": name,
                    "shared_tools": shared,
                    "description": name.replace("-", " "),
                    "instructions": (
                        "Use only deployed aliases. Respect physical guards and local vision call "
                        "limits. Observe detect identify segment estimate and check the active "
                        "target before motion. A destination also needs identification and "
                        "geometry before transport. No simulator poses no teleports no cloud "
                        "fallback and no arbitrary code."
                    ),
                    "actions": actions,
                }
            )
    return {
        "schema_version": 1,
        "task_id": f"libero-{scenario}-local-qwen",
        "instruction": descriptions[scenario],
        "skills": skills,
        "seeds": list(range(10)),
        "checkpoint_evidence": [
            "evaluation.task_success",
            "evaluation.checkpoints.target_grasped",
            "evaluation.checkpoints.target_lifted",
            "evaluation.checkpoints.target_contained",
            "evaluation.checkpoints.target_released",
            *(
                [
                    "evaluation.scenario_evidence.all_requested_contained",
                    "evaluation.scenario_evidence.all_requested_released",
                ]
                if scenario == "packing"
                else []
            ),
            *(
                [
                    "evaluation.scenario_evidence.pan_placed",
                    "evaluation.scenario_evidence.stove_turned_on",
                    "evaluation.scenario_evidence.heating_wait_completed",
                    "evaluation.scenario_evidence.stove_turned_off",
                ]
                if scenario == "popcorn"
                else []
            ),
        ],
        "required_true": ["evaluation.task_success"],
        "required_actions": ["check_all_packed"] if scenario == "packing" else deployed,
        "required_executed_actions": deployed,
        "max_trace_steps": 160,
        "runner": {
            "command": [
                "{root}/.venv-libero/Scripts/python.exe",
                "-m",
                "gap_libero.cli",
                "scenario",
                "--scenario",
                scenario,
                "--variation",
                "mixed",
                "--seed",
                "{seed}",
                "--graph",
                "{graph}",
                "--output",
                "{output}",
                "--enable-motion",
            ],
            "cwd": "../..",
            "timeout_s": 300,
            "receipt": "receipt.json",
        },
        "limitations": [
            "Custom LIBERO adaptation not the official paper benchmark",
            "Local Qwen on loopback with at most eight inference calls per episode",
            "Known proxy shape dimensions support geometry fitting without simulator pose fallback",
            "At most four thousand controller steps and one worker per evaluation",
            "Packing uses two objects per episode with varying item identities",
            "Packing uses a declared 0.32 by 0.26 by 0.15 meter bin with unchanged wall clearance",
            "Popcorn uses physical proxy pan and knob with a timed heating abstraction",
        ],
    }


def main():
    for scenario in ("fulfillment", "packing", "popcorn"):
        print(
            json.dumps(
                {
                    "path": f"harness/tasks/libero-{scenario}-local-qwen.json",
                    "task": task_catalog(scenario),
                }
            )
        )


def reference_workflow(scenario):
    from gap_harness.models import CheckpointSet, Subgraph, Task, Topology
    from gap_harness.workflow import compile_workflow

    reference = reference_graph(scenario)
    task = Task.model_validate(task_catalog(scenario))
    name = "perform_physical_task"
    topology = Topology.model_validate(
        {
            "graph_id": reference["graph_id"],
            "rationale": "hand authored integration reference",
            "start": name,
            "terminals": [
                {"name": "done", "status": "succeeded"},
                {"name": "abort", "status": "failed"},
            ],
            "subgraphs": [
                {
                    "name": name,
                    "skill": "libero-physical-actions",
                    "description": task.instruction,
                    "inputs": [],
                    "outputs": [],
                    "exits": ["ready"],
                    "on_error": "error",
                    "routes": [
                        {"outcome": "ready", "target": "done"},
                        {"outcome": "error", "target": "abort"},
                    ],
                }
            ],
        }
    )
    subgraph = Subgraph.model_validate(
        {
            "name": name,
            "rationale": "hand authored reference not model generation",
            "start": reference["start"],
            "nodes": [
                {
                    "name": node["id"],
                    "kind": reference["metadata"]["node_kinds"][node["id"]],
                    "action": node["action"],
                    "routes": [
                        {
                            "outcome": outcome,
                            "target": {"done": "ready", "abort": "error"}.get(target, target),
                        }
                        for outcome, target in node["routes"].items()
                    ],
                }
                for node in reference["nodes"]
            ],
        }
    )
    checks = CheckpointSet.model_validate(
        {
            "checkpoints": [
                {
                    "subgraph": name,
                    "name": "task_complete",
                    "rationale": "independent final task check",
                    "evidence": "evaluation.task_success",
                    "expected": "true",
                }
            ]
        }
    )
    graph = compile_workflow(topology, [subgraph], task, checks)
    graph["metadata"].update(
        reference_policy=True,
        model_generated=False,
        perception_backend="local-vlm",
        scenario=scenario,
    )
    return {
        "topology": topology.model_dump(),
        "subgraphs": [subgraph.model_dump()],
        "checkpoints": checks.model_dump()["checkpoints"],
        "graph": graph,
    }


if __name__ == "__main__":
    main()
