# Policy process visualizer

Standalone, offline presentation renderer for this project. No Isaac startup,
model loading, network calls, or changes to benchmark execution. It uses
Matplotlib, Pillow, and a bundled FFmpeg binary with two encoder threads.

The layout follows the supplied reference video: a dark background, grouped
policy lanes on the left, benchmark progress at upper right, and simulation
footage at lower right. The policy has horizontal grasp, transport, and release
groups matching the standalone diagram. These are presentation groups of the
flat policy, not newly implemented executable subgraphs. Policy labels stay
inside the group frames and action bubbles. The viewer has play/pause and an
iteration scrubber.

## Run from the repository root

### Offline robot clips and GIF previews

`media.py` collects complete successful cached recordings for the final evaluated
graph in each selected session. It validates graph/task evidence through the existing
session reader, checks replay hashes and seed correspondence, and preserves sparse
camera sample timing. It never starts simulation, model inference, or hosted calls.
It requires existing process MP4 and policy PNG exports and writes robot-only MP4s
beside them plus `docs/visualizations/results-summary.json`.

```powershell
.\.venv\Scripts\python.exe .\visualizer\media.py collect `
    .\results\harness\20261019T043212Z-edbce0ed `
    .\results\harness\grocery-20261019-003946 `
    .\results\harness\fulfillment-20261019-003946 `
    .\results\harness\packing-20261019-003946 `
    .\results\harness\popcorn-20261019-003946

.\.venv\Scripts\python.exe .\visualizer\media.py gif `
    .\docs\visualizations\libero-native-object-local-qwen\libero-native-object-local-qwen-robot.mp4 `
    --duration 60 --speed 3 --width 640 --fps 12
```

Use `--overwrite` only when replacing existing collection outputs or a GIF.
The converter accepts any local video and an optional `--output` GIF path.
`--start` and `--duration` select input seconds; `--speed` changes preview playback
without changing simulation evidence. The defaults are 20 input seconds, normal
speed, 640-pixel width and 12 FPS. Bounds are 60 input seconds, 30 output seconds,
20 FPS, 1280-pixel width and an eight MiB GIF. Two-pass palette conversion limits
memory use; encoding uses two CPU threads and a 120-second timeout per pass.

The checked-in collection is indexed in
[docs/visualizations](../docs/visualizations/README.md). A single successful replay
does not replace the measured evaluation score or prove an untested graph works.

### Automatic LIBERO harness video

```powershell
.\.venv\Scripts\python.exe .\visualizer\app.py libero
```

This single mode argument selects the most recently evaluated session under
results/harness. It reads actual per version feedback and graphs rather than the
manual sample history. If the final policy has no matching recording, it runs one
bounded representative LIBERO variation and records camera observations. Keep Isaac,
and unrelated perception workers closed. For a local VLM task keep Qwen and the
LM Studio local server running; otherwise keep LM Studio closed. Local VLM graphs
make bounded loopback identification requests, not hosted model requests.

The video first reveals each evaluated revision and its measured success rate, then
plays the final policy continuously while highlighting active and completed graph
nodes using recorded simulator steps. Subgraph frames come from actual agent generated
membership. Earlier graphs are not presented as controlling the final recording.
Missing or unevaluated revisions are not assigned invented success rates.

Graphs with executable router nodes show diamond predicates and labeled true false
and error routes. Highlighting follows the recorded outputs. Perception and predicate
actions that consume no simulation steps receive short presentation holds so they
remain visible in the video. The manifest records this extra presentation time;
these holds do not represent extra simulator motion or fabricated inference timing.

The representative variation is the first successful trial of the final evaluated
revision, or its first trial if none succeeded. A new recording does not alter the
historical campaign rate. Its actual success or failure is shown at the end. Subsequent
exports reuse matching footage. The default output is the single
scenario named process video under docs/visualizations with its matching
policy PNG alongside it. Grocery order fulfillment packing and popcorn each
have their own folder and descriptive filenames. No manifest or preview image
is exported. Each revision stays visible for five seconds by default and the
top corner shows the current iteration. Use --seconds to change the hold time.
Successful exports replace this
presentation video atomically. Original graphs feedback and receipts stay unchanged.
Explicit alternate video paths still refuse overwrites. The app exports 1080p at 60 fps;
use `--width 1280` for 720p. Native camera
motion is recorded at 20 fps and frames are held for display rather than interpolated.
Only a small image cache and two CPU encoder threads are used after the simulator exits.

Double speed footage is the default without changing the controller or measured
success rates. Use `--playback-speed 1` for normal speed. Speed is not shown on the
video. Decision presentation holds remain. Each actual
subgraph stays on one horizontal row. Long subgraphs use a full width graph above
the chart and footage so individual tools remain visible.

Boxes use short action labels without repeated subtitles. During playback an
active node gets a large description popup connected to its box. Router popups
include the actual outcome when that result is being presented. Recorded zero step
actions pause for readability without adding simulated execution time. The graph
PNG uses the same compact labels without a permanent key or inactive popups.

To select a particular session using one positional argument instead:

```powershell
.\.venv\Scripts\python.exe .\visualizer\libero_export.py .\results\harness\SESSION
```

The app also supports libero --session PATH and --output PATH. At least one completed
harness evaluation is required. An empty session or historical Isaac sample history
cannot supply LIBERO iteration rates. The supported recorder works with the existing
gap_libero.cli compare-grocery and run task adapters; other adapters need their own
timed receipt and recording support.

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\visualizer\requirements.txt

# check paths and history without rendering or loading models
.\.venv\Scripts\python.exe .\visualizer\app.py validate

# interactive viewer
.\.venv\Scripts\python.exe .\visualizer\app.py view

# presentation screenshot
.\.venv\Scripts\python.exe .\visualizer\app.py preview --iteration 10

# 20 second widescreen h264 video
.\.venv\Scripts\python.exe .\visualizer\app.py export

# lighter 720p export
.\.venv\Scripts\python.exe .\visualizer\app.py export --width 1280 --fps 12
```

Outputs default to `visualizer/output/preview.png` and `process.mp4`. The video
also gets a provenance manifest. `--output`, `--history`, and `--seconds` override
the defaults. MP4 export is silent. Only the Tk viewer opens a window.

## Current example and evidence limits

`history.json` contains the ten rates supplied in conversation. Iteration three
at five out of ten was inferred from the wording. This is presentation history,
not ten reconstructed and independently audited campaigns.

All sample iterations reference `policy.json`, a snapshot of the accepted graph.
Therefore no orange
policy edits are invented. Improvements to controllers or perception are not
automatically described as changes to the policy graph. The scene is currently
a **static screenshot**, not a recorded execution, and has no attached run receipt.

## Attach actual recordings and revision artifacts

Paths are relative to the history JSON file, unless absolute. Add fields to each
iteration record as appropriate. For example:

```json
{
  "iteration": 10,
  "successes": 9,
  "trials": 10,
  "graph": "../graphs/generated/codex-20261009T145742Z-f28cf576/accepted.json",
  "footage": "recordings/iteration-10.mp4",
  "clip_start_s": 0,
  "playback_speed": 2,
  "result": "../results/raw/YOUR_MATCHING_RUN.json",
  "note": "Describe the actual controller or perception change here.",
  "focus_node": "move_over_basket"
}
```

`footage` replaces the static scene with decoded frames, looping the clip if
necessary. Recording inputs stay local. Provide matching footage and a run
receipt for each iteration rather than implying one recording represents all
ten campaigns. The rate is still supplied explicitly; a receipt for one seed
does not prove the aggregate success count.

`result` displays that receipt's evaluator status. When a receipt contains a
graph hash, it must match the configured graph. No success banner is fabricated
when a receipt is absent. Receipt and footage correspondence is the author's
responsibility; this tool does not verify what happened in a video.

Give each iteration its actual `graph` snapshot to highlight modified nodes in
orange. Graph identifiers alone do not count as node edits. The initial graph
has no comparison baseline. Removed nodes are listed in revision details.

For code changes without policy edits, use `note` and `focus_node`. Notes are
manual annotations, not changes detected by the visualizer.

## Presentation workflow

1. Record an Isaac execution with your screen recorder and save an MP4 locally.
2. Preserve the matching accepted graph and raw evaluator result.
3. Update an iteration record with those paths and its aggregate success count.
4. Run `validate`, inspect `view` or `preview`, then run `export`.

This tool does not record or rerun Isaac itself. It composes existing artifacts.

## Harness revision replay

For a real harness session, run:

```powershell
.\.venv\Scripts\python.exe .\visualizer\app.py libero --session "actual session path"
```

Each generation attempt has its own version directory. Valid candidates keep
their own `graph.json`; rejected candidates retain their partial outputs and
validation errors rather than inventing an executable graph.

The LIBERO video reads each fully evaluated version's own graph snapshot in
version order. Its opening recap switches graphs and progressively adds that
version's measured success rate. Changed nodes are compared with the preceding
evaluated graph, not with the final graph. It then plays the final evaluated
graph's simulation with timed node and decision routing highlights. A normal
false branch such as item-loop exhaustion is not marked as a failed node.

Unevaluated, rejected or incomplete attempts are disclosed as skipped and are
not assigned fabricated rates. Their artifacts remain on disk. Graph and task
hash checks still prevent mismatched evaluation results from being presented.
The only presentation outputs remain the video and final graph PNG.

## Tests

The default graph view shows every tool and decision node within its actual
subgraph. New generation groups perception tools inside physical action goals
rather than standalone vision subgraphs. Use `--graph-detail actions` only when
an optional compact projection is wanted. That projection folds linear perception
chains without modifying executable graphs or evaluation receipts.

```powershell
.\.venv\Scripts\python.exe -m pytest .\visualizer\test_app.py
```

The folder is independent of the harness. Exported media and recordings are
ignored by its `.gitignore`. Markdown is ignored by the repository's existing
rules; this local guide remains readable on disk.
