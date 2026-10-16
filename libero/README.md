# LIBERO alternative backend

This optional backend shares `GraphPolicyExecutor`, the saved observation contract,
and the existing offline perception tools. Isaac scripts and accepted graphs remain unchanged.
It uses LIBERO's Panda OSC pose controller, not Lula IK or Isaac joint trajectories.
It does not train a policy or download demonstration datasets or model weights.
CPU PyTorch is needed because the official benchmark reads its initial state files with torch.

## Setup on native Windows

Keep Isaac and unrelated model workers closed. Local VLM graphs need Qwen in LM Studio.
From PowerShell 7 in the repository root

```powershell
.\scripts\windows\prepare-libero.ps1
.\scripts\windows\run-libero.ps1 -Mode check
.\scripts\windows\run-libero.ps1
```

Setup uses `.venv-libero` with Python 3.10, a pinned official LIBERO revision,
robosuite 1.4.1 and MuJoCo 2.3.7. This is a native Windows compatibility candidate,
not a claim that upstream officially validates Windows. WSL is not assumed necessary.
The metadata check does not import the simulator or prove rendering works.
The worker has a 120 second watchdog and one environment with one 320 by 240 RGB-D camera.
No Isaac or perception environment is modified.
The installer uses editable compatibility mode for LIBERO's namespace package.
Within `.venv-libero` only, it copies the installed MuJoCo DLL to the location
expected by robosuite 1.4.1 and adds private macros selecting GLFW on Windows.
Existing conflicting DLLs or private macros are not overwritten.
The runner defaults to capture of libero_spatial task zero, initial state zero,
seed zero. Suite, TaskId, InitStateId and Seed are optional overrides.
Output directories are generated automatically. There are no manual stopped
confirmation switches; keep Isaac and unrelated model workers closed before capture or motion.

## First task

Capture creates `observation/rgb.png`, raw RGB and metric depth, a calibrated
`observation/observation.json`, and `receipt.json`. The bundle can go through the
existing sequential detection, segmentation and geometry pipeline after LIBERO exits.
Do not reuse the blue cylinder and green basket color mapping for real LIBERO objects.
Identify the selected task objects first. The receipt lists their simulator names.
Initial state IDs select official benchmark states; changing a seed does not replace
the selected fixed initial state with a new benchmark variation.

## Graph execution

### Matched grocery comparison

The custom `grocery_ground` task recreates the existing ground-level Isaac scene
using the exact shared sampler, object dimensions, colors, masses and basket wall
dimensions. Both the arm and groceries are on the ground. Seeds zero through nine
refer to the same sampled layouts as Isaac, not the official LIBERO initial states.
It is registered through LIBERO's task wrapper but is not an official LIBERO suite task.

Capture the matching scene without policy motion

```powershell
.\scripts\windows\run-libero.ps1 -Mode grocery
```

Execute the original accepted graph with new LIBERO action bindings

```powershell
.\scripts\windows\run-libero.ps1 -Mode grocery -EnableMotion
```

Use `-Seed 1` to select another variation. The default graph is the original
`graphs/generated/codex-20261009T145742Z-f28cf576/accepted.json` and its bytes are
not modified. A different graph may be selected with `-Graph`.
The worker saves initial and final RGB-D bundles, the scene manifest, resolved
action plan, per-node trace, grasp and lift checkpoints, contact aborts, and receipt.
Graph and implementation hashes are recorded for reproducibility.

The bindings use bounded Cartesian OSC motion with grip-site orientation feedback,
two-finger contact plus slip checks during transport, gripper preload and controlled
release. Equivalent jaw orientations are selected modulo a half turn relative to
the arm heading toward the basket. Gripper actions hold the last commanded pose
with feedback rather than following position drift as the payload is released.
The final containment predicate is imported directly from the Isaac evaluator.
No object is attached, welded or teleported during execution.

This first transfer test uses known scene geometry rather than running DINO and SAM3.
It tests graph and control portability, not full perception equivalence. Meshes,
controller gains, force limits, friction and contact solver differ from Isaac.
Contact aborts use geometry pairs, not the Isaac impulse threshold.

Run a sequential ten-variation campaign after closing Isaac and models

```powershell
$env:LIBERO_CONFIG_PATH = "$PWD\cache\libero-config"
.\.venv-libero\Scripts\python.exe -m gap_libero.grocery_campaign `
    --output .\results\libero\my-grocery-comparison
```

The output directory must be new. Each child worker has a ninety second timeout.
The summary also links historical Isaac trials whose graph hash matches, when
available. Historical perceived-geometry trials and oracle-geometry LIBERO trials
are not a controlled ranking of the two simulators.

The recorded October 15 transfer campaign is
`results/libero/grocery-transfer-20261015-supported-release/summary.json`.
It completed all ten variations with one consistent implementation and the
unchanged graph hash `f1c0ac910604a3d8949796dea6d17676c1f60331c99223089ef1220fc4a07af5`.
LIBERO succeeded on six variations, seeds zero, three, four, six, seven and nine.
Seeds one and two failed wrist orientation tracking during transport, seed five
stopped on a hand to basket contact, and seed eight failed descent tracking.
The linked historical Isaac v24 receipts show eight successes out of ten.
This historical baseline is separate from later reported Isaac results.

The later jaw orientation and pose holding fix was rerun on all ten seeds in
`results/libero/grocery-transfer-20261015-jaw-heading-final/summary.json`.
That consistent implementation passed seven out of ten, including seed one,
without changing the graph. Seeds two, five and eight remain failed because of
wrist tracking, hand to basket contact and descent tracking respectively.

### Other LIBERO tasks

Create a task-specific JSON plan with suite, task_id, init_state_id, seed,
target_object, workspace_m, max_episode_steps and an actions mapping keyed by
the action names in the existing graph schema. Each action declares kind
`move`, `open`, `close`, `hold`, or `verify`. Moves require position_m in world
meters. Actions may supply max_steps, tolerance_m and require_grasp.

```json
{
  "schema_version": 1,
  "suite": "libero_spatial",
  "task_id": 0,
  "init_state_id": 0,
  "seed": 0,
  "target_object": "replace_with_name_from_capture_receipt",
  "workspace_m": [[-0.6, -0.6, 0.7], [0.6, 0.6, 1.6]],
  "max_episode_steps": 400,
  "actions": {
    "open_gripper": {"kind": "open", "max_steps": 20},
    "align_over_target": {"kind": "move", "position_m": [0.0, 0.0, 1.2]},
    "close_gripper": {"kind": "close", "max_steps": 40},
    "lift_grasped": {"kind": "move", "position_m": [0.0, 0.0, 1.3], "require_grasp": true},
    "settle_release": {"kind": "verify"}
  }
}
```

This illustrates the format, not a validated task solution. Fill every action
in the selected graph and derive waypoints from the new scene, not Isaac heights.
Moves may also specify a normalized `orientation_xyzw` quaternion. Without one,
the controller retains its existing orientation. Position, orientation and total
episode steps are bounded independently.

```powershell
.\scripts\windows\run-libero.ps1 -Graph .\path\to\task-graph.json `
    -Plan .\path\to\libero-plan.json -EnableMotion
```

Motion requires explicit opt-in and workspace bounds. Commands and graph traversal
are step bounded. Providing Graph and Plan selects graph execution automatically;
EnableMotion is still required. Calling the runner without arguments never commands motion.
Closing requires two-finger target contact. Guarded transport
also monitors target motion relative to the gripper. `forbidden_geom_pairs` can
list pairs of MuJoCo geometry names that cause abort on contact. This is not full
collision planning or the Isaac collision evaluator. Task completion always
requires LIBERO's task success predicate, not just a successful graph terminal.
The receipt includes graph and plan hashes and is compatible with the visualizer's
graph receipt attachment. Recording LIBERO footage is not yet implemented.

## Validation status

Adapter and graph execution are tested with an injected fake environment.
Native Windows rendering and RGB-D capture passed for libero_spatial task zero,
initial state zero on October 15 2026, with motion disabled and all depth pixels valid.
A real physical grasp and completion of an official LIBERO bowl task still need validation.
The matched custom grocery task has separately completed a physical grasp,
transport and release using all ten nodes of the unchanged accepted graph.
This does not establish performance on official LIBERO bowl tasks.
No Isaac benchmark score applies to LIBERO.

## Executable image perception

The separate vision task captures 640 by 480 RGB-D observations and executes
color detection, binary segmentation, masked depth reconstruction and bounded
cylinder or basket fitting as graph nodes. Shape dimensions and the ground plane
are declared priors, not measured simulator object poses. The hand-authored vision
adapter is marked separately from its source model-generated workflow.

Target found and container found predicates return explicit true or false outputs.
False branches abort before motion if geometry is missing or unreliable. Grasp and
placement predicates also appear as actual router nodes. Detection boxes, masks,
world point clouds, fit reports, observation hashes and routing outputs are saved.
There is no automatic simulator pose fallback. Privileged object state remains in
the independent evaluation receipt and is not passed to the perception functions.

This backend is lightweight CPU color perception, not DINO VLM SAM inference and
not unrestricted semantic recognition. Camera intrinsics use integer pixel center
coordinates, including the half pixel correction for MuJoCo rendered images.

### Local vision language identification

Select the `local-vlm` backend to insert real `identify_target` and
`identify_container` tool nodes between detection and segmentation. The detector
proposes up to six colored connected components without identifying the target.
Local Qwen selects an ID from enlarged labeled crops. That selection determines
the mask used for depth fitting; null or invalid selections cannot fall back to
the known blue or green identity. Unsupported geometry still fails closed.

LM Studio uses the same `qwen3-vl-4b-instruct` model and port 1234 as the earlier
local vision tests. No hosted vision API, new model download or model loading in
the LIBERO process is required. The loaded Qwen server must remain running while
other large model workers and Isaac are stopped. There are at most two inference
requests per episode, 128 output tokens each, no retries and no cloud fallback.
Candidate sheets, replies, model identifiers and selection outputs are saved.
Before starting the renderer this mode verifies GPU zero is the RTX 5080 and has
at least four GiB of free VRAM. Reduce the LM Studio context or stop other models
if the headroom check fails. No automatic model loading or unloading is performed.

`python -m gap_libero.vision_workflow --source-version PATH\v0001` now prepares
a separate local VLM adaptation by default. Add `--backend rgbd-color` to retain
the CPU only catalog. Previously generated and evaluated graphs are not changed.

Grocery bindings derive the downward grasp heading from the target bearing plus
a quarter turn, keeping the Panda wrist away from its end stop. Basket headings
use the same radial reference while preserving parallel jaw alignment with the
basket. These are shared geometry rules, not seed specific corrections. Collision,
slip, containment and motion budgets are unchanged. Receipts include implementation
hashes and phase joint positions so control changes remain distinguishable from
generated graph revisions. Compare graph versions with identical binding hashes
when attributing an improvement to generation alone.

## Additional local Qwen scenarios

Keep Qwen loaded in LM Studio on port 1234. Close Isaac and unrelated model
workers. These commands execute one bounded, explicitly hand-authored reference
graph each; they do not call the graph generator or a hosted model.

```powershell
.\scripts\windows\run-libero.ps1 -Mode fulfillment -Seed 0 -EnableMotion
.\scripts\windows\run-libero.ps1 -Mode packing -Seed 1 -EnableMotion
.\scripts\windows\run-libero.ps1 -Mode popcorn -Seed 0 -EnableMotion
```

Omit `-EnableMotion` for scene capture only. Add `-Record` to save simulation
frames for export. Every invocation creates a separate directory under
`results/libero`, including the graph, receipt, initial and final images,
perception evidence, plans and per-action diagnostics. Existing runs are not
overwritten. Receipts distinguish reference graphs from model-generated graphs
through the saved graph metadata and include implementation hashes.

Fulfillment varies the requested grocery shape, item positions, basket positions
and item permutation. `-Variation` accepts `xy`, `basket-swap`, `permutation` or
`mixed`. Packing requests two groceries, verifies each item independently and
requires both to remain contained and released at the final check. Its declared
bin is 0.32 by 0.26 by 0.15 meters, allowing two groceries and gripper clearance;
the original single-item grocery basket is unchanged. Popcorn moves
a dynamic pan by its handle onto a burner, physically turns a hinged knob on,
waits forty controller steps and turns it off. Popcorn seeds vary the initial pan
position and heading; its variation selector does not add grocery permutations.

These are custom proxy tasks, not official paper or LIBERO benchmark replicas.
Popcorn does not simulate food, heat or cooking. Shape dimensions are declared
priors; positions and headings for planning come from local Qwen selected RGB-D
masks, never evaluator poses. The new task evaluator uses full 3D box and cylinder
support extents, including tilt, with the existing wall clearance and vertical
tolerance. The original grocery evaluator is unchanged. Packing checks that all
finger contacts clear before retraction and holds the measured aperture using
actuator commands rather than changing joint state.

Each new episode permits at most eight local inference requests, four thousand
controller steps and 160 graph visits. The PowerShell wrapper imposes a 180 second
wall clock limit. Workers run one at a time; there is no automatic retry or cloud
fallback. A successful reference run is an integration check, not evidence that
a generated graph or every seed succeeds.

The generator task catalogs are separate files under `harness/tasks` named
`libero-fulfillment-local-qwen.json`, `libero-packing-local-qwen.json` and
`libero-popcorn-local-qwen.json`. They expose perception as shared tools for
physical-action subgraphs and support seeds zero through nine. Generate and
evaluate one task at a time after reviewing reference results. Static checks
enforce action order and deployed capabilities. Stateful loop actions are also
required to complete in the actual execution trace; a zero-item shortcut cannot
claim success. Normal item-loop exhaustion is not reported as a failed decision.
Reaching a graph visit limit produces bounded policy feedback rather than losing
the trace and blocking refinement as a runner error.

### Generate and refine a policy

A perfect reference score is not required before generating a first policy.
Readiness means the catalog compiles, physical operations work in integration,
and all configured trials produce trustworthy feedback. Failed trials remain
visible and do not count as success. Reference policies are explicitly marked
hand authored, not model generated.

With the existing Codex proxy and local Qwen server running, start one generation
and evaluation cycle with either `fulfillment`, `packing` or `popcorn`.

```powershell
.\.venv\Scripts\python.exe -m gap_harness.cli iterate --scenario packing --live --execute --rounds 1
```

This creates a new versioned session, generates the hierarchy through the existing
orchestrator, skill and checkpoint agents, and evaluates all ten seeds serially.
Model allowance is consumed by generation, not by the reference validator. Keep
the printed session path to resume with `--session` for feedback-based refinement.
The configured model remains `gpt-6.1-sol` with medium reasoning. The orchestrator
still chooses the number and boundaries of physical-action subgraphs; shared
vision tools stay inside those subgraphs.

For generation alone, use `generate --scenario packing --live`. For a metadata
and catalog check without inference or simulation, use `check --scenario packing`.

To repeat generation and ten seed evaluation until success or ten total attempts:

```powershell
.\scripts\windows\run-until-success.ps1 packing
```

The first argument also accepts `fulfillment`, `popcorn` or the original `grocery`.
Use `-MaxTries 3` for a lower cap. Use `-Session` with the printed session path
to resume; earlier saved candidates count toward the same total limit. Each new
revision receives previous evaluation feedback, and rejected candidates also
count as attempts. All versions and evaluations are preserved. An already
successful or exhausted session makes no further generation calls.

The wrapper allows at most one hundred model calls for a fresh ten-attempt
session, covering an orchestrator, up to eight skill agents and one checkpoint
agent per candidate. Resuming preserves the existing session call budget instead
of expanding it. Infrastructure errors, usage limits and API failures stop the
run immediately. A final summary reports the stopping reason and best result;
reaching the attempt cap is not reported as success. Keep Qwen and the Codex proxy
running, and stop unrelated simulator or model workers. The script does not
guarantee that graph refinement can repair low level physics or perception issues.

The no-generation reference validator exercises the same compiler, subprocess
runner, trace binding and feedback builder as model-generated graphs.

```powershell
.\.venv\Scripts\python.exe -m gap_libero.scenario_validation
```

Use `--scenario packing` for one scenario, `--seeds 0 1` for an explicit subset,
or `--check-only` to validate compilation without starting models or simulation.
Every run has a new output directory under `results/scenario-validation`, with
immutable per-seed receipts, feedback and a readiness summary. A check-only run
does not claim that physical execution has been validated.

### Reference validation status

The latest complete local Qwen reference sweeps produced these integration
baselines. Each sweep ran seeds 0 through 9 with zero runner infrastructure
failures. These are hand-authored reference results, not generated policy scores.

| Scenario | Reference successes | Feedback directory under results/scenario-validation |
| --- | --- | --- |
| Fulfillment | 9 of 10 | 20261016T024559Z-13759e44/fulfillment/v0001 |
| Packing | 3 of 10 | 20261016T025638Z-38809e77/packing/v0001 |
| Popcorn | 10 of 10 | 20261016T024559Z-13759e44/popcorn/v0001 |

Packing remains physically less reliable. Its failures include release settling,
grasp stability and incomplete image geometry; do not attribute those failures
automatically to the generated hierarchy or promise that graph refinement alone
will repair them. The original grocery graph also passed a seed zero regression
with its original evaluator.

Reference results are saved per invocation rather than being treated as a fixed
benchmark score. Packing receipts include every requested item's final evidence
so a successful earlier placement cannot mask a later failure. Popcorn exposes
persistent verified pan-placement, stove-on, heating-wait and stove-off events for
the checkpoint agent; a later stove-off action does not erase the earlier on event.
Compare scores only under the same scene and implementation hashes. Do not
attribute improvements caused by controller or scene changes to graph generation.

## Upstream interfaces

- [LIBERO environment wrapper](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/8f1084e3132a39270c3a13ebe37270a43ece2a01/libero/libero/envs/env_wrapper.py)
- [LIBERO benchmark initial states](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/8f1084e3132a39270c3a13ebe37270a43ece2a01/libero/libero/benchmark/__init__.py)
- [Robosuite camera calibration and depth](https://github.com/ARISE-Initiative/robosuite/blob/v1.4.1/robosuite/utils/camera_utils.py)
