"""Offline Codex generation package and strict acceptance gate; standard library only."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from gap_isaac.graph_policy import GraphPolicy

SKILLS_REVISION = "752dfbd96968d4d08379f8111765efbf6457800b"
AUTHORING = "codex-assisted-constrained-generation"
GEOMETRY_SOURCE = "saved-dino-sam3-rgbd-model-assisted"
MODEL = "gpt-6-astra"
DEFAULT_INSTRUCTION = "Pick up alphabet_soup and place it in the basket."
# Dependencies belong to the deployed action contracts, not to LLM configuration.
# generated policies may only call the actions deployed in this harness
ACTION_CONTRACTS = {
    "open_gripper": ("grasping-direct-ik", None, "Open before approaching the target."),
    "align_over_target": ("grasping-direct-ik", "open_gripper", "Align above target."),
    "descend_to_grasp": ("grasping-direct-ik", "align_over_target", "Descend aligned."),
    "close_gripper": ("grasping-direct-ik", "descend_to_grasp", "Close and verify hold."),
    "lift_grasped": ("transporting-objects", "close_gripper", "Lift verified held target."),
    "move_over_basket": ("transporting-objects", "lift_grasped", "Carry above basket."),
    "position_for_release": (
        "transporting-objects", "move_over_basket", "Reach bounded release position."
    ),
    "release_object": ("transporting-objects", "position_for_release", "Open and release."),
    "retract_from_basket": ("transporting-objects", "release_object", "Retract after release."),
    "settle_release": (
        "transporting-objects", "retract_from_basket", "Wait and check final evaluation."
    ),
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def metadata() -> dict:
    return {
        "graph_authoring": AUTHORING,
        "geometry_source": GEOMETRY_SOURCE,
        "open_robot_skills_commit": SKILLS_REVISION,
        "target_id": "alphabet_soup",
        "container_id": "basket",
    }


def graph_schema() -> dict:
    """Schema for Codex structured output; local validation remains authoritative."""
    def obj(properties: dict) -> dict:
        return {"type": "object", "properties": properties,
                "required": list(properties), "additionalProperties": False}

    return obj({
        "schema_version": {"type": "integer", "enum": [1]},
        "graph_id": {"type": "string"},
        "start": {"type": "string"},
        "metadata": obj({key: {"type": "string", "enum": [value]}
                         for key, value in metadata().items()}),
        "terminals": obj({"SUCCEEDED": {"type": "string", "enum": ["succeeded"]},
                          "FAILED": {"type": "string", "enum": ["failed"]}}),
        "nodes": {"type": "array", "minItems": 10, "maxItems": 10, "items": obj({
            "id": {"type": "string"},
            "skill": {"type": "string", "enum": ["grasping-direct-ik", "transporting-objects"]},
            "action": {"type": "string", "enum": list(ACTION_CONTRACTS)},
            "routes": obj({"succeeded": {"type": "string"},
                           "failed": {"type": "string", "enum": ["FAILED"]}}),
        })},
    })


def exact_keys(value: object, keys: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{label} must contain exactly {sorted(keys)}")


def validate_generated_graph(payload: dict) -> GraphPolicy:
    exact_keys(payload, {"schema_version", "graph_id", "start", "metadata",
                         "terminals", "nodes"}, "graph")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("schema_version must be integer 1")
    identifier = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}\Z")
    for key in ("graph_id", "start"):
        if not isinstance(payload[key], str) or not identifier.fullmatch(payload[key]):
            raise ValueError(f"invalid {key}")
    if payload["metadata"] != metadata():
        raise ValueError("metadata must match fixed task, skills revision, and authoring contract")
    if payload["terminals"] != {"SUCCEEDED": "succeeded", "FAILED": "failed"}:
        raise ValueError("terminal statuses must match fixed success/failure contract")
    nodes = payload["nodes"]
    if not isinstance(nodes, list) or len(nodes) != len(ACTION_CONTRACTS):
        raise ValueError("graph requires exactly ten deployed actions")
    seen_actions: set[str] = set()
    for node in nodes:
        exact_keys(node, {"id", "skill", "action", "routes"}, "node")
        if not isinstance(node["id"], str) or not identifier.fullmatch(node["id"]):
            raise ValueError("invalid node id")
        if node["id"] in payload["terminals"]:
            raise ValueError("node id overlaps a terminal")
        action = node["action"]
        if not isinstance(action, str) or action not in ACTION_CONTRACTS:
            raise ValueError("unknown action")
        if action in seen_actions:
            raise ValueError("duplicate action")
        seen_actions.add(action)
        if node["skill"] != ACTION_CONTRACTS[action][0]:
            raise ValueError("action/skill mismatch")
        exact_keys(node["routes"], {"succeeded", "failed"}, "routes")
        if node["routes"]["failed"] != "FAILED":
            raise ValueError("every failure must abort immediately")
        if not isinstance(node["routes"]["succeeded"], str):
            raise ValueError("route target must be a string")
    policy = GraphPolicy.from_dict(payload)
    current = policy.start
    visited: set[str] = set()
    completed: set[str] = set()
    while current in policy.nodes:
        if current in visited:
            raise ValueError("cycles are not allowed")
        visited.add(current)
        node = policy.nodes[current]
        prerequisite = ACTION_CONTRACTS[node.action][1]
        if prerequisite and prerequisite not in completed:
            raise ValueError(f"{node.action} requires {prerequisite} first")
        completed.add(node.action)
        current = node.routes["succeeded"]
    if current != "SUCCEEDED" or visited != set(policy.nodes):
        raise ValueError("all actions must be reachable on the complete success path")
    return policy


def load_json(path: Path) -> dict:
    if path.stat().st_size > 256 * 1024:
        raise ValueError("JSON exceeds 256 KiB limit")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=unique)


def write_new(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def verify_package(bundle: Path) -> dict:
    manifest = load_json(bundle / "package.json")
    if manifest.get("requested_model") != MODEL or manifest.get("schema_version") != 1:
        raise ValueError("unsupported generation package")
    expected = {"prompt.md", "schema.json", "skill-definitions.md"}
    if set(manifest.get("file_sha256", {})) != expected:
        raise ValueError("generation package must pin prompt, schema, and skill definitions")
    for name, digest in manifest["file_sha256"].items():
        if sha256((bundle / name).read_bytes()) != digest:
            raise ValueError(f"generation package changed: {name}")
    return manifest


def prepare(root: Path) -> Path:
    skills = root / "cache/upstream/open-robot-skills"
    revision = subprocess.run(["git", "-C", str(skills), "rev-parse", "HEAD"],
                              check=True, capture_output=True, text=True).stdout.strip()
    if revision != SKILLS_REVISION:
        raise ValueError("upstream skills checkout must match the pinned revision")
    definitions = []
    for skill in ("grasping-direct-ik", "transporting-objects"):
        path = skills / "skills" / skill / "SKILL.md"
        relative = path.relative_to(skills).as_posix()
        pinned = subprocess.run(["git", "-C", str(skills), "show",
                                 f"{SKILLS_REVISION}:{relative}"],
                                check=True, capture_output=True).stdout
        if path.read_bytes().replace(b"\r\n", b"\n") != pinned.replace(b"\r\n", b"\n"):
            raise ValueError(f"pinned skill definition modified: {skill}")
        definitions.append(f"# {skill}\n\n" + pinned.decode("utf-8"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    bundle = root / "graphs/generated" / f"codex-{stamp}-{uuid4().hex[:8]}"
    bundle.mkdir(parents=True)
    (bundle / "skill-definitions.md").write_text("\n\n".join(definitions), encoding="utf-8")
    write_new(bundle / "schema.json", graph_schema())
    contracts = "\n".join(
        f"- {action}: skill={skill}; requires={prerequisite or 'none'}; {description}"
        for action, (skill, prerequisite, description) in ACTION_CONTRACTS.items()
    )
    prompt = f"""Generate one policy graph for this fixed instruction: {DEFAULT_INSTRUCTION}

Return ONLY a JSON object conforming to schema.json. Read skill-definitions.md
for the pinned upstream skills. Do not read reference graphs, prior results,
other repository files, or earlier conversations. Do not use tools beyond
reading these generation-package inputs (read-only file-access commands are allowed).
Do not run application/code commands, edit code,
launch Isaac, load models, make API calls, or generate Python/scripts.

This is a constrained project-local generator, NOT the upstream generator.
The upstream skill definitions supply context; ONLY these deployed local
action contracts are executable (canonical upstream scripts are not deployed):
{contracts}

Use each action exactly once. Choose node identifiers and explicit routes.
Every failure must go directly to FAILED. All nodes must be reachable on the
success path, prerequisites must hold, no loops, and final success must go to
SUCCEEDED. Schema fixes metadata and terminal status values. No action
parameters, numeric poses, code, safety settings, or invented skills are allowed.
Perception runs separately; saved RGB-D geometry supplies the poses. Existing
local actions enforce grasp/slip/collision/timeouts and final containment.
This narrow task has strongly constrained ordering; do not claim general planning.
"""
    (bundle / "prompt.md").write_text(prompt, encoding="utf-8")
    write_new(bundle / "package.json", {
        "schema_version": 1, "requested_model": MODEL, "instruction": DEFAULT_INSTRUCTION,
        "skills_revision": SKILLS_REVISION, "created_at": stamp,
        "file_sha256": {name: sha256((bundle / name).read_bytes())
                        for name in ("prompt.md", "schema.json", "skill-definitions.md")},
        "models_loaded": False, "network_requests": 0,
    })
    return bundle


def accept(bundle: Path, model: str, session_note: str) -> Path:
    if model != MODEL:
        raise ValueError("this workflow accepts only declared gpt-6-astra generation")
    package = verify_package(bundle)
    candidate = bundle / "candidate.json"
    payload = load_json(candidate)
    raw = candidate.read_bytes()
    policy = validate_generated_graph(payload)
    accepted = bundle / "accepted.json"
    receipt = bundle / "accepted.manifest.json"
    if accepted.exists() or receipt.exists():
        raise ValueError("acceptance artifacts already exist; prepare a fresh package")
    write_new(accepted, payload)
    write_new(receipt, {
        "schema_version": 1, "ok": True, "graph_id": policy.graph_id,
        "graph_sha256": sha256(accepted.read_bytes()), "candidate_sha256": sha256(raw),
        "package_sha256": sha256((bundle / "package.json").read_bytes()),
        "prompt_sha256": package["file_sha256"]["prompt.md"],
        "declared_model": model, "model_verified": False,
        "model_provenance": "operator declaration; not independently attested",
        "session_note": session_note, "graph_authoring": AUTHORING,
        "accepted_at": datetime.now(timezone.utc).isoformat(),
        "node_count": len(policy.nodes), "motion_enabled": False,
        "models_loaded": False, "network_requests": 0,
    })
    return accepted


def verify_accepted(path: Path) -> dict:
    """Revalidate artifact at execution time; receipts are provenance, not signatures."""
    if path.name != "accepted.json":
        raise ValueError("generated graph must be the accepted.json artifact")
    validate_generated_graph(load_json(path))
    verify_package(path.parent)
    receipt = load_json(path.with_name("accepted.manifest.json"))
    checks = {
        "graph_sha256": sha256(path.read_bytes()),
        "candidate_sha256": sha256(path.with_name("candidate.json").read_bytes()),
        "package_sha256": sha256(path.with_name("package.json").read_bytes()),
        "prompt_sha256": sha256(path.with_name("prompt.md").read_bytes()),
        "declared_model": MODEL, "graph_authoring": AUTHORING,
    }
    if receipt.get("ok") is not True or any(receipt.get(k) != v for k, v in checks.items()):
        raise ValueError("accepted graph provenance mismatch")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    acceptance = commands.add_parser("accept")
    acceptance.add_argument("--bundle", type=Path, required=True)
    acceptance.add_argument("--model", required=True)
    acceptance.add_argument("--session-note", required=True)
    verification = commands.add_parser("verify")
    verification.add_argument("--graph", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            result = {"ok": True, "bundle": str(prepare(args.root.resolve()))}
        elif args.command == "accept":
            result = {"ok": True, "accepted_graph": str(accept(
                args.bundle.resolve(), args.model, args.session_note))}
        else:
            result = verify_accepted(args.graph.resolve())
        print(json.dumps(result, indent=2))
        return 0
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(json.dumps({"ok": False, "error": str(error), "motion_enabled": False}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
