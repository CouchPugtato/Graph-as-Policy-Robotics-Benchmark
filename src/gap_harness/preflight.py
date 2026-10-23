"""Check required local services before combined generation and rehearsal."""

import json
import subprocess


def check_runner(harness):
    runner = harness.task.runner
    actions = {action.name for skill in harness.task.skills for action in skill.actions}
    if runner is None or "gap_libero.cli" not in runner.command or "identify_target" not in actions:
        return
    root = (harness.task_path.parent / runner.cwd).resolve()
    command = [runner.command[0].format(root=str(root)), "-m", "gap_libero.cli", "preflight"]
    native = "--native-object" in runner.command
    if native:
        command.append("--native-object")
    published = "published" in runner.command
    if published:
        command += ["--published-case", runner.command[runner.command.index("--scenario") + 1]]
    harness.progress.event("checking local vision server and renderer before generation")
    try:
        result = subprocess.run(
            command, cwd=root, capture_output=True, text=True, timeout=45, check=False,
            stdin=subprocess.DEVNULL,
        )
        report = json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        raise RuntimeError(
            "local vision preflight failed before generation; start LM Studio with Qwen "
            "on port 1234 and check the LIBERO environment"
        ) from error
    if result.returncode or report.get("ok") is not True:
        raise RuntimeError("local vision preflight failed before generation: " +
                           str(report.get("detail", "local services are not ready"))[:1500])
    if (native or published) and report.get("native_task_files_present") is not True:
        raise RuntimeError("native task readiness was not verified before generation")
    harness.progress.event("local vision preflight passed; no inference or simulation performed")
