"""Generate versioned policies or rehearse them with explicit API and motion opt ins."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from gap_harness.engine import Harness
from gap_harness.provider import OpenAIResponses

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TASK = ROOT / "harness/tasks/libero-grocery.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["check", "generate", "evaluate", "iterate"])
    parser.add_argument("--task", type=Path, default=DEFAULT_TASK)
    parser.add_argument("--session", type=Path)
    parser.add_argument("--version", type=Path)
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--live", action="store_true", help="allow paid OpenAI requests")
    parser.add_argument("--execute", action="store_true", help="allow configured simulator motion")
    parser.add_argument("--rounds", type=int, choices=range(1, 6), default=3)
    parser.add_argument("--max-subgraphs", type=int, choices=range(1, 17), metavar="N", default=8)
    parser.add_argument("--max-calls", type=int, choices=range(1, 129), metavar="N", default=48)
    args = parser.parse_args()
    if args.mode in ("generate", "iterate") and not args.live:
        parser.error("generation requires --live and sends task context to OpenAI "
                     "for paid inference")
    if args.mode in ("evaluate", "iterate") and not args.execute:
        parser.error("evaluation requires --execute and all other simulators and models stopped")
    if args.mode == "evaluate" and args.version is None:
        parser.error("evaluation requires --version")
    session = args.session or (args.version.resolve().parent if args.version else
        ROOT / "results/harness" / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]))
    provider = OpenAIResponses(args.model) if args.live else None
    try:
        harness = Harness(args.task, session, provider, max_subgraphs=args.max_subgraphs,
                          max_calls=args.max_calls)
        if args.mode == "check":
            result = {"ok": True, "task": harness.task.task_id,
                      "available_skills": [skill.name for skill in harness.task.skills],
                      "skill_agent_count": "chosen by orchestrator",
                      "max_subgraphs": args.max_subgraphs, "max_calls": args.max_calls,
                      "seeds": harness.task.seeds, "model": args.model,
                      "network_requests": 0, "simulator_started": False,
                      "runner_configured": harness.task.runner is not None,
                      "limitations": harness.task.limitations}
        elif args.mode == "evaluate":
            result = harness.evaluate(args.version)
        else:
            version = None
            for _ in range(args.rounds if args.mode == "iterate" else 1):
                print(f"Generating policy in {session}", flush=True)
                try:
                    version = harness.generate()
                except ValueError:
                    if args.mode != "iterate":
                        raise
                    print("Candidate rejected see saved validation and agent responses", flush=True)
                    continue
                if args.mode == "iterate":
                    feedback = harness.evaluate(version)
                    print(f"Rehearsal {feedback['successes']}/{feedback['trials']}", flush=True)
                    if feedback["successes"] == feedback["trials"]:
                        break
            result = {"session": str(session), "version": str(version) if version else None,
                      "accepted": bool(version and (version / "accepted.json").exists()),
                      "note": "generated is not the same as validated in simulation"}
        print(json.dumps(result, indent=2))
    except Exception as error:
        parser.exit(1, f"Harness stopped {type(error).__name__}: {error}\n")


if __name__ == "__main__":
    main()
