"""Find immutable evaluation attempts without replacing historical evidence."""

import json
from pathlib import Path


def feedback_path(version):
    version = Path(version)
    attempts = sorted(version.glob("evaluation-[0-9][0-9][0-9][0-9]/feedback.json"))
    if attempts:
        return attempts[-1]
    original = version / "feedback.json"
    return original if original.is_file() else None


def evaluation_complete(feedback):
    if "evaluation_complete" in feedback:
        return feedback["evaluation_complete"] is True
    # older startup timeouts must not become policy feedback
    return all(record.get("graph_bound") is not False
               and record.get("seed_bound") is not False
               and record.get("known_trace") is not False
               for record in feedback.get("records", []))


def load_feedback(version):
    path = feedback_path(version)
    return json.loads(path.read_text(encoding="utf-8")) if path else None


def attempt_directory(version):
    version = Path(version)
    if not (version / "evaluation").exists() and not (version / "feedback.json").exists():
        return version / "evaluation", version
    for number in range(2, 10000):
        directory = version / f"evaluation-{number:04d}"
        if not directory.exists():
            return directory, directory
    raise ValueError("evaluation attempt limit exceeded")
