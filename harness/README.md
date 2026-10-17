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

From the repository root the core environment is sufficient. No model weights are
needed. The transport uses httpx and Pydantic with python dotenv for local settings.

```powershell
.\.venv\Scripts\python.exe -m gap_harness.cli check
```

The CLI loads the repository root .env automatically without variable expansion.
Existing shell settings take precedence. Only OPENAI_API_KEY, OPENAI_BASE_URL,
OPENAI_MODEL, OPENAI_REASONING_EFFORT, GAP_CODEX_PROXY_KEY, GAP_VLM_BASE_URL,
GAP_VLM_MODEL and GAP_VLM_API_KEY are read from the file.
Never paste credentials
into chat, task JSON or a command that will be committed. Requests send the instruction,
catalog, selected skill references and revision feedback to the configured service.
Images are not uploaded automatically.

New sessions default to `harness/tasks/libero-grocery-local-vlm.json`. It deploys
RGB-D capture, color region proposals, local Qwen identification, binary masks,
metric shape fitting and explicit boolean routers. Load `qwen3-vl-4b-instruct` in
LM Studio and start Local Server on `http://127.0.0.1:1234/v1` before evaluating.
The VLM chooses regions from enlarged labeled crops and its choice controls the
mask and geometry. There is no known color identity fallback or cloud fallback.
This remains a colored proxy detector and shape fitting backend, not DINO or SAM.
Motion requires both identification and geometry checks to pass and cannot fall
back to simulator object poses. Two local inference calls are allowed per episode,
with 128 output tokens per call, 35 second client timeouts and no retries.
Ten variations therefore make at most twenty local inference calls, separate from
the hosted graph generation call budget. No weights are loaded by the worker.
Keep Isaac and other large models stopped, but keep Qwen in LM Studio running.

```powershell
.\.venv\Scripts\python.exe -m gap_libero.local_vlm check
```

This only checks endpoint model availability and makes no inference call.
Use `GAP_VLM_BASE_URL`, `GAP_VLM_MODEL` and optional `GAP_VLM_API_KEY` to override
the local settings. Numeric loopback endpoints are mandatory; OpenAI or OpenRouter
keys are never forwarded. Old sessions automatically retain their original task
manifest and budgets. Advanced custom catalogs remain accessible through the
Python Harness class rather than extra command line options.

Router nodes declare `true`, `false` and `failed` routes. Their actual outputs are
retained per trace event. The visualizer draws their real predicates as diamonds
and labels edges with the declared routing values rather than inventing checks.

An existing generated workflow can be adapted without a model call:

```powershell
.\.venv\Scripts\python.exe -m gap_libero.vision_workflow --source-version PATH\v0001
```

The command creates a separate vision session with explicit hand-authored adaptation
provenance. It does not rewrite or reclassify the original generated graph. Evaluate
its printed version path with the usual `evaluate --version PATH --execute` command.

### Local Codex compatibility proxy

The current .env selects the loopback proxy and gpt-6.1-sol with medium reasoning.
Requests explicitly send reasoning.effort=medium. OPENAI_API_KEY stays empty.
Use OPENAI_MODEL and OPENAI_REASONING_EFFORT in .env to change these settings.
Effective settings are saved in each call request and version lineage without
rewriting historical graph versions.
The pinned unofficial server uses the existing Codex ChatGPT login and plan allowance.
It is not a locally running model and does not bypass plan limits or guarantee free
usage. Keep it on loopback. Never copy Codex tokens into .env or share auth.json.

Install optional tools on a new machine, then start the server:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[harness-tools]"
# run codex login first if this machine is not already signed in
.\.venv\Scripts\python.exe -m gap_harness.proxy start
.\.venv\Scripts\python.exe -m gap_harness.cli api-check --live
```

The start helper uses uvx openai-api-server-via-codex==0.2.2 and the checked in
codex-proxy.toml. One concurrent request is allowed and public response storage is
disabled. Daemon state and logs live under cache/codex-proxy/run. No GPU models or
simulators are started. The API check sends one small structured output request.

```powershell
.\.venv\Scripts\python.exe -m gap_harness.proxy status
.\.venv\Scripts\python.exe -m gap_harness.proxy stop
```

The project stop helper verifies the daemon process identity and stops its supervisor
and matching worker on Windows because the upstream daemon stop can leave a worker
running. It does not stop unrelated applications or edit Codex authentication.

GAP_CODEX_PROXY_KEY is optional client to proxy authentication only. If enabling it,
configure the same incoming api_key in the local proxy server. Platform OPENAI_API_KEY
is never forwarded to the proxy. Redirects and environment HTTP proxies are disabled.
Only the official HTTPS Platform endpoint or a loopback base URL ending in /v1 is
accepted. There is no automatic fallback from the proxy to paid Platform API calls.

### Direct Platform API

Set OPENAI_BASE_URL=https://api.openai.com/v1 and provide OPENAI_API_KEY privately in
.env or the shell. Direct Platform usage is billed separately from Codex subscription
allowance. GAP_CODEX_PROXY_KEY is not used for the Platform API.

```powershell
.\.venv\Scripts\python.exe -m gap_harness.cli generate --live
```

The command prints its session and version path.

Generation progress is printed immediately to stderr with elapsed time. It identifies
the current revision, coordinator, skill agent index and assigned skill, checkpoint
pass, validation stages and saved graph path. Each completed call reports its duration,
input and output tokens, cumulative reported session tokens and used request count.
A status only heartbeat appears every fifteen seconds while a model request is pending.
It does not stream model reasoning, print prompts or responses, retry requests or extend
timeouts. Missing usage is marked as incomplete rather than treated as zero cost.
The client timeout shown is a configuration value, not a guaranteed remote cancellation
deadline. Token totals are informational and do not add an aggregate token budget.
Prior saved request usage is included when resuming the same session.

When iterating, progress also shows the current revision attempt, variation index,
worker log path, each seed result and the running rehearsal score. Normal completion
keeps the final JSON summary on stdout so it remains usable by other scripts.

To rehearse that version, close Isaac and unrelated model workers first.
Keep Qwen and the LM Studio local server running for a local VLM task.
Replace SESSION with the printed path.

```powershell
.\.venv\Scripts\python.exe -m gap_harness.cli evaluate --version SESSION\v0001 --execute
```

To generate a revision using the saved feedback, reuse the same session.

```powershell
.\.venv\Scripts\python.exe -m gap_harness.cli generate --session SESSION --live
```

Or start a bounded generation and rehearsal loop with at most ten candidate versions.

```powershell
.\.venv\Scripts\python.exe -m gap_harness.cli iterate --scenario packing --live --execute
```

The default task uses the existing LIBERO grocery bindings and seeds zero through nine.
The orchestrator chooses its decomposition within three to eight physical subgraphs.
The CLI keeps scenario session version max-attempts live and execute options only.
Use --max-attempts to lower the default ten attempt limit. API calls and
simulators run sequentially. New sessions allow at most ten calls per permitted attempt,
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

Reevaluating the same version preserves the original evaluation and feedback.
Subsequent attempts live in `evaluation-0002`, `evaluation-0003` and so on, with
their own feedback and selection files. The harness and visualizer read the latest
completed attempt. Missing or mismatched execution evidence stops the suite and
blocks policy revision until the runner is repaired and the same graph reevaluated.
Incomplete attempts have no policy success rate and do not blame skill checkpoints.
LIBERO workers use the prepared `cache/libero-config` directory by default, honor
an explicit `LIBERO_CONFIG_PATH`, and have standard input disconnected so setup
prompts cannot hang an evaluation. No model requests are made by `evaluate`.

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

Use the Python Harness class with a new task manifest. It supplies the instruction, available
skills and action contracts, reference files, variations, evidence paths, fixed
success predicates and trusted runner command. The orchestrator is not hardcoded
to a grocery task or a fixed agent count above the three subgraph minimum.
The CLI selects deployed scenarios; for example:

```powershell
.\.venv\Scripts\python.exe -m gap_harness.cli check --scenario packing
```

The separate inspection manifest is only a schema example. A new task still needs real actions and
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
