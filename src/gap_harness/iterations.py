"""Bounded feedback driven generation with immutable candidate history."""

from datetime import datetime, timezone
from uuid import uuid4

from pydantic import ValidationError

from gap_harness.engine import CandidateRejected, digest, dump, session_lock
from gap_harness.evaluations import evaluation_complete, load_feedback


def passed(feedback, trial_count):
    return bool(
        feedback
        and evaluation_complete(feedback)
        and feedback.get("configured_trials", trial_count) == trial_count
        and feedback.get("trials") == trial_count
        and feedback.get("successes") == trial_count
    )


def iterate(harness, rounds, max_attempts=None, *, before_generate=None):
    if not 1 <= rounds <= 10 or (max_attempts is not None and not 1 <= max_attempts <= 10):
        raise ValueError("iteration limits must be between one and ten")
    harness.prepare()
    with session_lock(harness.session, ".iteration.lock"):
        return iterate_locked(harness, rounds, max_attempts, before_generate=before_generate)


def iterate_locked(harness, rounds, max_attempts, *, before_generate=None):
    existing = harness.versions()
    version = existing[-1] if existing else None
    accepted = False
    stop_reason = "round_limit"
    attempts = 0
    for candidate in reversed(existing):
        feedback = load_feedback(candidate)
        if passed(feedback, len(harness.task.seeds)):
            if (
                feedback.get("graph_sha256") != digest(candidate / "graph.json")
                or feedback.get("task_sha256") != harness.task_sha256
            ):
                raise ValueError("successful candidate evidence changed or belongs to another task")
            version, accepted, stop_reason = candidate, True, "accepted"
            break
    remaining = (
        rounds if max_attempts is None else min(rounds, max(0, max_attempts - len(existing)))
    )
    if not accepted and remaining and before_generate is not None:
        before_generate()
    if not accepted:
        for index in range(remaining):
            attempts += 1
            harness.progress.event(
                f"revision attempt {len(existing) + index + 1} | "
                f"up to {remaining} new attempt(s) this run"
            )
            try:
                version = harness.generate()
            except (CandidateRejected, ValidationError):
                harness.progress.event("candidate rejected; saved validation counts as an attempt")
                continue
            feedback = harness.evaluate(version)
            if not feedback.get("evaluation_complete") or feedback.get("trials") != len(
                harness.task.seeds
            ):
                raise ValueError(
                    "runner infrastructure failed; fix it and reevaluate "
                    "the same graph before continuing"
                )
            if passed(feedback, len(harness.task.seeds)):
                accepted, stop_reason = True, "accepted"
                harness.progress.event("all configured variations passed; stopping early")
                break
        if not accepted and max_attempts is not None and len(harness.versions()) >= max_attempts:
            stop_reason = "attempt_limit"
    scored = [
        (candidate, feedback)
        for candidate in harness.versions()
        if (feedback := load_feedback(candidate))
        and evaluation_complete(feedback)
        and feedback.get("trials") == len(harness.task.seeds)
    ]
    best = max(scored, key=lambda item: item[1]["successes"]) if scored else None
    harness.progress.event(
        f"iteration stopped: {stop_reason} | "
        f"{len(harness.versions())} total candidates | best score "
        f"{best[1]['successes'] if best else 'not evaluated'}/{len(harness.task.seeds)}"
    )
    result = {
        "session": str(harness.session),
        "version": str(version) if version else None,
        "accepted": accepted,
        "attempts_this_run": attempts,
        "total_attempts": len(harness.versions()),
        "max_attempts": max_attempts,
        "stop_reason": stop_reason,
        "best_version": str(best[0]) if best else None,
        "best_successes": best[1]["successes"] if best else None,
        "trials": len(harness.task.seeds),
        "note": "generated is not the same as validated in simulation",
    }
    dump(
        harness.session
        / (
            "iteration-run-"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-"
            + uuid4().hex[:8]
            + ".json"
        ),
        result,
    )
    return result
