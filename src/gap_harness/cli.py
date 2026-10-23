"""Generate versioned policies or rehearse them with explicit API and motion opt ins."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, ValidationError

from gap_harness.config import (
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    load_environment,
)
from gap_harness.engine import MIN_GENERATED_SUBGRAPHS, Harness
from gap_harness.evaluations import evaluation_complete, load_feedback
from gap_harness.iterations import iterate
from gap_harness.preflight import check_runner
from gap_harness.provider import OpenAIResponses

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TASK = ROOT / "harness/tasks/libero-grocery-local-vlm.json"


def failure_summary(error):
    if isinstance(error, ValidationError):
        issues = error.errors(include_input=False, include_context=False, include_url=False)
        locations = [
            ".".join(str(part) for part in issue["loc"]) or "response" for issue in issues[:3]
        ]
        return "structured output validation failed at " + ", ".join(locations)
    return str(error)


def main():
    load_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["check", "api-check", "generate", "evaluate", "iterate"])
    parser.add_argument(
        "--scenario",
        choices=(
            "grocery",
            "fulfillment",
            "packing",
            "popcorn",
            "native-object",
            "published-fulfillment",
            "published-packing",
            "published-popcorn",
        ),
        help="select a deployed local Qwen LIBERO task",
    )
    parser.add_argument("--session", type=Path)
    parser.add_argument("--version", type=Path)
    parser.add_argument("--live", action="store_true", help="allow hosted model usage")
    parser.add_argument("--execute", action="store_true", help="allow configured simulator motion")
    parser.add_argument(
        "--max-attempts",
        type=int,
        choices=range(1, 11),
        default=10,
        help="cap total saved generation attempts in an iteration session",
    )
    args = parser.parse_args()
    model = os.environ.get("OPENAI_MODEL") or DEFAULT_MODEL
    reasoning_effort = os.environ.get("OPENAI_REASONING_EFFORT") or DEFAULT_REASONING_EFFORT
    if args.mode in ("api-check", "generate", "iterate") and not args.live:
        parser.error("model requests require --live and use hosted service allowance or billing")
    if args.mode in ("evaluate", "iterate") and not args.execute:
        parser.error(
            "evaluation requires --execute; stop Isaac and unrelated model workers "
            "but keep LM Studio running for a local VLM task"
        )
    if args.mode == "evaluate" and args.version is None:
        parser.error("evaluation requires --version")
    session = args.session or (
        args.version.resolve().parent
        if args.version
        else ROOT
        / "results/harness"
        / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    )
    try:
        provider = OpenAIResponses(model, reasoning_effort=reasoning_effort) if args.live else None
        if args.mode == "api-check":

            class Probe(BaseModel):
                model_config = ConfigDict(extra="forbid", strict=True)
                ok: bool

            probe = OpenAIResponses(model, max_output_tokens=512, reasoning_effort=reasoning_effort)
            response = probe.complete(
                "Return only the requested JSON with ok true.",
                {"task": "connection and structured output check"},
                Probe,
            )
            parsed = Probe.model_validate_json(response["text"])
            if response.get("status") != "completed" or parsed.ok is not True:
                raise ValueError("structured API probe did not complete successfully")
            print(
                json.dumps(
                    {
                        "ok": True,
                        **probe.public_settings(),
                        "response_model": response["model"],
                        "usage": {
                            key: (response.get("usage") or {}).get(key)
                            for key in ("input_tokens", "output_tokens", "total_tokens")
                        },
                        "structured_outputs_validated": True,
                        "simulator_started": False,
                    },
                    indent=2,
                )
            )
            return
        manifest = session / "session.json"
        saved = json.loads(manifest.read_text(encoding="utf-8")) if manifest.is_file() else {}
        max_subgraphs = saved.get("max_subgraphs", 8)
        max_calls = saved.get("max_calls", args.max_attempts * 10)
        selected_task = None
        if args.scenario:
            if args.scenario.startswith("published-"):
                from gap_libero.benchmarks.vab import task_manifest

                selected_task = task_manifest(args.scenario.removeprefix("published-"))
            else:
                selected_task = (
                    DEFAULT_TASK
                    if args.scenario == "grocery"
                    else ROOT / f"harness/tasks/libero-{args.scenario}-local-qwen.json"
                )
        task_path = selected_task or saved.get("task_manifest_path", DEFAULT_TASK)
        harness = Harness(
            task_path, session, provider, max_subgraphs=max_subgraphs, max_calls=max_calls
        )
        if args.mode == "check":
            result = {
                "ok": True,
                "task": harness.task.task_id,
                "available_skills": [skill.name for skill in harness.task.skills],
                "skill_agent_count": "chosen by orchestrator",
                "max_subgraphs": max_subgraphs,
                "min_subgraphs": MIN_GENERATED_SUBGRAPHS,
                "max_calls": max_calls,
                "seeds": harness.task.seeds,
                "model": model,
                "api": OpenAIResponses(model, reasoning_effort=reasoning_effort).public_settings(),
                "network_requests": 0,
                "simulator_started": False,
                "runner_configured": harness.task.runner is not None,
                "limitations": harness.task.limitations,
            }
        elif args.mode == "evaluate":
            result = harness.evaluate(args.version)
        else:
            version = None
            rounds = args.max_attempts if args.mode == "iterate" else 1
            harness.progress.event(
                f"task {harness.task.task_id} | model {model} | "
                f"reasoning {reasoning_effort} | "
                f"session {session}"
            )
            harness.progress.event(
                f"limits: up to {rounds} revision attempt(s); "
                f"{MIN_GENERATED_SUBGRAPHS} to {max_subgraphs} physical skill agents per revision; "
                f"{max_calls} generation calls per session; one request at a time"
            )
            if args.mode == "iterate":
                result = iterate(
                    harness,
                    rounds,
                    args.max_attempts,
                    before_generate=lambda: check_runner(harness),
                )
            else:
                harness.progress.event("revision attempt 1/1")
                version = harness.generate()
                result = {
                    "session": str(session),
                    "version": str(version),
                    "accepted": bool(
                        load_feedback(version)
                        and evaluation_complete(load_feedback(version))
                        and load_feedback(version)["successes"] == load_feedback(version)["trials"]
                    ),
                    "note": "generated is not the same as validated in simulation",
                }
        print(json.dumps(result, indent=2))
    except Exception as error:
        parser.exit(1, f"Harness stopped {type(error).__name__}: {failure_summary(error)}\n")


if __name__ == "__main__":
    main()
