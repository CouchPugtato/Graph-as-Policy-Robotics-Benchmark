"""Summarize all retained attempts and contact/tracking order without pose correction."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def summarize_result(path):
    result = json.loads(Path(path).read_text())
    diagnostics = result.get("diagnostics", {})
    descent = diagnostics.get("cartesian_descent", {})
    summary = {"result": path, "ok": result.get("ok"), "revision": result.get("run_revision"),
               "error": result.get("error"), "descent_status": descent.get("status"),
               "final_position_error_m": descent.get("final_position_error_m"),
               "descent_trace": diagnostics.get("descent_trace_file")}
    start = diagnostics.get("descent_start", {})
    if start:
        summary["descent_start_xy_error_m"] = math.dist(
            start["position_m"][:2], start["destination_m"][:2])
    trace_path = diagnostics.get("descent_trace_file")
    if trace_path:
        frames = [json.loads(line) for line in Path(trace_path).read_text().splitlines()]
        def impulse(frame):
            return sum(c["impulse_ns"] for c in frame["contacts"]
                       if c["body_b"].endswith("/alphabet_soup") and "finger" in c["body_a"])
        first_contact = next((f for f in frames if impulse(f) > 1e-9), None)
        first_substantial = next((f for f in frames if impulse(f) > .001), None)
        first_pause = next((f for f in frames if f["phase"] == "path"
                            and f["progress_allowed"] is False), None)
        summary.update(frame_count=len(frames),
                       maximum_finger_target_impulse_ns=max(map(impulse, frames), default=0),
                       paused_path_frames=sum(f["phase"] == "path" and
                                              f["progress_allowed"] is False for f in frames),
                       contact_frames=sum(impulse(f) > 1e-9 for f in frames),
                       max_xy_error_m=max((f["xy_error_m"] for f in frames), default=0),
                       max_orientation_error_rad=max((f["orientation_error_rad"]
                                                      for f in frames), default=0))
        for name, frame in (("first_contact", first_contact),
                            ("first_substantial_contact", first_substantial),
                            ("first_pause", first_pause)):
            summary[name] = None if frame is None else {
                key: frame[key] for key in ("elapsed_s", "descent_elapsed_s", "path_time_s",
                                            "tcp_position_m", "xy_error_m",
                                            "orientation_error_rad", "pause_reasons")}
        if first_contact:
            before = [f for f in frames if f["elapsed_s"] < first_contact["elapsed_s"]]
            summary["before_contact_max_xy_m"] = max((f["xy_error_m"] for f in before), default=0)
            summary["before_contact_max_angle_rad"] = max(
                (f["orientation_error_rad"] for f in before), default=0)
        if frames:
            final = frames[-1]
            summary["final_path_time_s"] = final["path_time_s"]
            summary["descent_elapsed_s"] = final["descent_elapsed_s"]
    if "evaluation" in result:
        summary["evaluation"] = result["evaluation"]
        summary["graph_sha256"] = result.get("graph_sha256")
        summary["successful_nodes"] = sum(t["outcome"] == "succeeded"
                                           for t in result["graph"]["trace"])
    return summary


def analyze(directory):
    campaign = json.loads((directory / "campaign.json").read_text())
    attempts = []
    for trial_path in campaign["trials"]:
        trial = json.loads(Path(trial_path).read_text())
        attempt = {"trial": trial_path, "seed": trial["seed"], "status": trial["status"],
                   "revision": trial["run_revision"], "error": trial.get("error"),
                   "infrastructure_retries": trial["infrastructure_retries"],
                   "steps": trial["steps"]}
        for mode in ("grasp", "graph"):
            if mode in trial["artifacts"]:
                attempt[mode] = summarize_result(trial["artifacts"][mode])
        attempts.append(attempt)
    return {"campaign": str(directory / "campaign.json"), "completed": campaign.get("completed"),
            "attempts": attempts,
            "first_attempt_successes": sum(a["status"] == "succeeded" and
                                           a["infrastructure_retries"] == 0 for a in attempts),
            "successes_including_infrastructure_retries": sum(a["status"] == "succeeded"
                                                              for a in attempts),
            "infrastructure_retry_count": sum(a["infrastructure_retries"] for a in attempts)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = analyze(args.directory)
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
