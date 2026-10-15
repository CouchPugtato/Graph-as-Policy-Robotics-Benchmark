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

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest .\visualizer\test_app.py
```

The folder is independent of the harness. Exported media and recordings are
ignored by its `.gitignore`. Markdown is ignored by the repository's existing
rules; this local guide remains readable on disk.
