# LIBERO alternative backend

This optional backend shares `GraphPolicyExecutor`, the saved observation contract,
and the existing offline perception tools. Isaac scripts and accepted graphs remain unchanged.
It uses LIBERO's Panda OSC pose controller, not Lula IK or Isaac joint trajectories.
It does not train a policy or download demonstration datasets or model weights.
CPU PyTorch is needed because the official benchmark reads its initial state files with torch.

## Setup on native Windows

Keep Isaac and perception models closed. From PowerShell 7 in the repository root

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
confirmation switches; keep Isaac and perception models closed before capture or motion.

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

## Upstream interfaces

- [LIBERO environment wrapper](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/8f1084e3132a39270c3a13ebe37270a43ece2a01/libero/libero/envs/env_wrapper.py)
- [LIBERO benchmark initial states](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/8f1084e3132a39270c3a13ebe37270a43ece2a01/libero/libero/benchmark/__init__.py)
- [Robosuite camera calibration and depth](https://github.com/ARISE-Initiative/robosuite/blob/v1.4.1/robosuite/utils/camera_utils.py)
