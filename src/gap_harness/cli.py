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
    REASONING_EFFORTS,
    load_environment,
)
from gap_harness.engine import Harness
from gap_harness.evaluations import evaluation_complete, load_feedback
from gap_harness.provider import OpenAIResponses

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TASK = ROOT / "harness/tasks/libero-grocery.json"


def failure_summary(error):
    if isinstance(error, ValidationError):
        issues = error.errors(include_input=False, include_context=False, include_url=False)
        locations = [".".join(str(part) for part in issue["loc"]) or "response"
                     for issue in issues[:3]]
        return "structured output validation failed at " + ", ".join(locations)
    return str(error)


def main():
    load_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["check", "api-check", "generate", "evaluate", "iterate"])
    parser.add_argument("--task", type=Path, default=DEFAULT_TASK)
    parser.add_argument("--session", type=Path)
    parser.add_argument("--version", type=Path)
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL") or DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", choices=REASONING_EFFORTS,
                        default=os.environ.get("OPENAI_REASONING_EFFORT")
                        or DEFAULT_REASONING_EFFORT)
    parser.add_argument("--live", action="store_true", help="allow hosted model usage")
    parser.add_argument("--execute", action="store_true", help="allow configured simulator motion")
    parser.add_argument("--rounds", type=int, choices=range(1, 6), default=3)
    parser.add_argument("--max-subgraphs", type=int, choices=range(1, 17), metavar="N", default=8)
    parser.add_argument("--max-calls", type=int, choices=range(1, 129), metavar="N", default=48)
    args = parser.parse_args()
    if args.mode in ("api-check", "generate", "iterate") and not args.live:
        parser.error("model requests require --live and use hosted service allowance or billing")
    if args.mode in ("evaluate", "iterate") and not args.execute:
        parser.error("evaluation requires --execute and all other simulators and models stopped")
    if args.mode == "evaluate" and args.version is None:
        parser.error("evaluation requires --version")
    session = args.session or (args.version.resolve().parent if args.version else
        ROOT / "results/harness" / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]))
    try:
        provider = (OpenAIResponses(args.model, reasoning_effort=args.reasoning_effort)
                    if args.live else None)
        if args.mode == "api-check":
            class Probe(BaseModel):
                model_config = ConfigDict(extra="forbid", strict=True)
                ok: bool

            probe = OpenAIResponses(args.model, max_output_tokens=512,
                                    reasoning_effort=args.reasoning_effort)
            response = probe.complete("Return only the requested JSON with ok true.",
                                      {"task": "connection and structured output check"}, Probe)
            parsed = Probe.model_validate_json(response["text"])
            if response.get("status") != "completed" or parsed.ok is not True:
                raise ValueError("structured API probe did not complete successfully")
            print(json.dumps({"ok": True, **probe.public_settings(),
                              "response_model": response["model"],
                              "usage": {key: (response.get("usage") or {}).get(key)
                                        for key in ("input_tokens", "output_tokens",
                                                    "total_tokens")},
                              "structured_outputs_validated": True,
                              "simulator_started": False}, indent=2))
            return
        harness = Harness(args.task, session, provider, max_subgraphs=args.max_subgraphs,
                          max_calls=args.max_calls)
        if args.mode == "check":
            result = {"ok": True, "task": harness.task.task_id,
                      "available_skills": [skill.name for skill in harness.task.skills],
                      "skill_agent_count": "chosen by orchestrator",
                      "max_subgraphs": args.max_subgraphs, "max_calls": args.max_calls,
                      "seeds": harness.task.seeds, "model": args.model,
                      "api": OpenAIResponses(args.model,
                          reasoning_effort=args.reasoning_effort).public_settings(),
                      "network_requests": 0, "simulator_started": False,
                      "runner_configured": harness.task.runner is not None,
                      "limitations": harness.task.limitations}
        elif args.mode == "evaluate":
            result = harness.evaluate(args.version)
        else:
            version = None
            rounds = args.rounds if args.mode == "iterate" else 1
            harness.progress.event(f"task {harness.task.task_id} | model {args.model} | "
                                   f"reasoning {args.reasoning_effort} | "
                                   f"session {session}")
            harness.progress.event(f"limits: up to {rounds} revision attempt(s); "
                f"{args.max_subgraphs} skill agents per revision; "
                f"{args.max_calls} generation calls per session; one request at a time")
            for round_index in range(1, rounds+1):
                harness.progress.event(f"revision attempt {round_index}/{rounds}")
                try:
                    version = harness.generate()
                except ValueError:
                    if args.mode != "iterate":
                        raise
                    harness.progress.event("candidate rejected; see saved validation and responses")
                    continue
                if args.mode == "iterate":
                    feedback = harness.evaluate(version)
                    if not feedback["evaluation_complete"]:
                        raise ValueError("runner infrastructure failed; fix it and reevaluate "
                                         "the same graph before continuing")
                    if feedback["successes"] == feedback["trials"]:
                        harness.progress.event("all configured variations passed; stopping early")
                        break
            result = {"session": str(session), "version": str(version) if version else None,
                      "accepted": bool(version and load_feedback(version)
                                       and evaluation_complete(load_feedback(version))
                                       and load_feedback(version)["successes"]
                                       == load_feedback(version)["trials"]),
                      "note": "generated is not the same as validated in simulation"}
        print(json.dumps(result, indent=2))
    except Exception as error:
        parser.exit(1, f"Harness stopped {type(error).__name__}: {failure_summary(error)}\n")


if __name__ == "__main__":
    main()
