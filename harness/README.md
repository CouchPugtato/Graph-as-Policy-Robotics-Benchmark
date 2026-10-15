# Thin multi agent graph harness

The hierarchy follows GaP and its pinned generator: a coordinator chooses the number
of subgraphs and their skill contracts, one skill agent fills each subgraph, and a
separate whole workflow checkpoint agent authors postconditions. Ordinary local code
assembles, validates, rehearses and supplies feedback. The accepted graph executes
without agents or API calls.

This is a local implementation of that architecture, not an invocation of the upstream
generator. It reuses selected upstream skill instructions. Its structured JSON contracts
and compiler differ from upstream workflow v3 and generated Python builder scripts.
Script nodes refer only to existing bindings and generated Python is never executed.
This restriction keeps the integration small and auditable.

## Commands

From the repository root the existing core environment is sufficient. No new model
weights or packages are needed. The transport uses the installed httpx and Pydantic.

```powershell
.\.venv\Scripts\python.exe -m gap_harness.cli check
```

Set OPENAI_API_KEY privately in the shell that runs generation. Do not paste it into
chat, task JSON or a command that will be committed. API requests send the instruction,
catalog, selected skill references and revision feedback to OpenAI. Images are not
uploaded automatically. Generation is billed separately from a Codex subscription.

```powershell
.\.venv\Scripts\python.exe -m gap_harness.cli generate --live
```

The command prints its session and version path. To rehearse that version, close
Isaac, LM Studio and perception workers first. Replace SESSION with the printed path.

```powershell
.\.venv\Scripts\python.exe -m gap_harness.cli evaluate --version SESSION\v0001 --execute
```

To generate a revision using the saved feedback, reuse the same session.

```powershell
.\.venv\Scripts\python.exe -m gap_harness.cli generate --session SESSION --live
```

Or start a bounded generation and rehearsal loop with at most three candidate versions.

```powershell
.\.venv\Scripts\python.exe -m gap_harness.cli iterate --live --execute
```

The default task uses the existing LIBERO grocery bindings and seeds zero through nine.
The orchestrator chooses its decomposition within eight subgraphs. API calls and
simulators run sequentially. There are at most forty eight API attempts per session,
4096 output tokens per request, a bounded input size, a request timeout and a separate
simulator timeout. These are resource bounds, not a guaranteed dollar ceiling. No
automatic API retries are made. Static failures are preserved and can inform the next
version. A simulator timeout stops its direct worker; runners must not launch detached
simulator descendants. Close other simulator instances manually before executing.

## Version storage

Each session is a new directory under results/harness. Exclusive file creation and a
session lock prevent overwriting candidates or concurrent writes.

```text
session.json                   frozen task catalog references and limits
v0001/
    lineage.json               parent version task fingerprint and model
    calls/                     each request response usage and failure
    topology.json              orchestrator output
    subgraph NAME.json         each skill agent output
    checkpoints.json           independent checkpoint agent output
    workflow.json              assembled hierarchical policy
    graph.json                 compiled graph for existing executors
    validation.json            accepted structure or rejection reason
    evaluation/                commands logs original receipts and per seed feedback
    feedback.json              aggregate evidence for the next revision
    accepted.json              present only after every configured seed passes
    selection.json             best evaluated version without replacing prior artifacts
v0002/
    ...
```

Rejected or interrupted versions retain all completed outputs and raw responses.
There is no destructive cleanup and no replacement of the original accepted graph.
The existing simulator runner keeps its receipt format but writes into the candidate
evaluation directory. Earlier standalone LIBERO and Isaac results are untouched.
Task or skill reference changes require a new session. A stale lock after a killed
process requires checking that its process is stopped before removing the lock.
Rehearsal is intentionally once per version to preserve evidence. Generate a new
version rather than overwriting an old evaluation.

## Feedback protocol and plan

GaP Appendix E describes recording object state and contact differences around phases,
analyzing failed rehearsals and refining the graph across sampled variations. Here:

1. Retain the original receipt and graph digest for every seed.
2. Retain action traces and map each failed node to its parent subgraph.
3. Collect before and after end effector and object state, grasp state and contact
   pairs for each phase when the runner supplies them. Missing evidence is explicit.
4. Evaluate generated postconditions against allowlisted evidence. Fixed task success
   predicates remain authoritative and generated checks can only add constraints.
5. Aggregate success counts, failed phases and state changes across the full seed suite.
6. Give the coordinator the previous workflow, static errors and aggregate feedback.
   Give assigned skill agents that previous evidence along with their new contracts.
   Large packets retain the first failing phase per trial in API context while the
   complete unabridged evidence stays on disk. If still oversized the packet explicitly
   marks omitted state evidence rather than silently claiming it was analyzed.
7. Regenerate checkpoints in the independent pass, then validate and rerun all seeds.
Keep every revision including regressions. An accepted marker requires all seeds.

The current implementation regenerates the assigned subgraphs each round rather than
patching source files. The coordinator may preserve or change topology. Physics,
evaluator definitions, task criteria and runner commands are outside model control.
Checkpoint evidence is checked from receipts after rehearsal, not as an additional
online control signal. Historical milestone flags remain historical and should not
be interpreted as a continuously maintained grasp.

Next extensions toward the paper are per subgraph output versus privileged world
comparisons, optional selected images for visual failure analysis, targeted reuse of
unchanged subgraphs, and a separate held out variation suite. Do not silently replace
missing measurements with a model judgment or treat tuning seeds as held out tests.

## Other tasks and simulators

Use --task PATH with a new task manifest. It supplies the instruction, available
skills and action contracts, reference files, variations, evidence paths, fixed
success predicates and trusted runner command. The orchestrator is not hardcoded
to a grocery task or a fixed agent count. An inspection schema example is included:

```powershell
.\.venv\Scripts\python.exe -m gap_harness.cli check --task harness/tasks/inspection-example.json
```

That example is not an implemented simulation. A new task still needs real actions and
an evaluator. The runner command uses argument substitution for graph, seed, output
and root without a shell. It must write receipt.json with seed, graph_sha256, ok,
graph status and trace, plus the evidence paths declared in its task. Paths for cwd
and skill references are relative to the task manifest. Commands are trusted local
configuration and must never come from model output.

An Isaac task can use its existing wrapper through a task specific receipt adapter.
Only the LIBERO grocery adapter is configured out of the box. This does not make an
arbitrary generated action executable and does not establish perception parity.
The flat compiler preserves executable route behavior and parent subgraph membership;
ports describe shared state contracts, not an upstream typed RPC implementation.

References

- https://arxiv.org/html/2607.05369v1
- https://developers.openai.com/api/docs/guides/structured-outputs
- cache/upstream/graph-as-policy/gap/agent/multi_agent.py
