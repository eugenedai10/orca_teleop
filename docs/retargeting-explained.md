# Retargeting Explained

This document expands on the "Retargeting" section of the top-level `README.md`,
defining the terms and explaining how
`src/orca_teleop/retargeting/configs/adaptive_analytical_orca.yaml` drives the
default `adaptive_analytical` retargeter
(`src/orca_teleop/retargeting/adaptive_analytical.py`).

## What is "retargeting"?

A human hand and the ORCA robot hand have different geometry (finger lengths,
joint layout). **Retargeting** is the algorithm that converts a tracked human
hand pose into robot joint angles, so the robot hand reproduces the human's
hand shape/gesture as closely as its own kinematics allow. This runs every
frame, in real time.

## The two retargeters

### `adaptive_analytical` (default)

An Orca-native port of the "Wuji-style" analytical retargeting strategy
(referring to a known approach for analytical dexterous-hand retargeting).
Per frame, it:

1. Takes the full MediaPipe hand skeleton (21 3D landmarks: wrist, knuckles,
   fingertips, etc.) from the tracked human hand.
2. Looks up which robot URDF frame corresponds to each relevant human
   landmark, via the YAML config's explicit frame mappings.
3. Uses **Pinocchio** (a robotics library) to compute:
   - **Forward kinematics** — given candidate joint angles, where do the
     robot's frames land in 3D space?
   - **Jacobians** — how does nudging each joint change a frame's position?
     (Tells the optimizer which joints to move, and by how much.)
4. Runs **`nlopt`** (a nonlinear optimization library), specifically the
   SLSQP solver, to find joint angles — within the robot's physical joint
   limits ("bounded") — that best match the robot's frames to the human's
   landmarks, according to a weighted cost function (see below).

### Legacy fingertip key-vector retargeter (`rmsprop`)

An older, simpler method, kept for comparison via
`--retargeter rmsprop`. Instead of matching a full skeleton, it tracks
vectors from the palm/wrist to each fingertip ("key vectors") and iteratively
adjusts robot joint angles using **RMSProp** (a gradient-descent variant) so
those vectors are reproduced on the robot hand.

## Finger joint terminology: PIP, DIP, TIP

Standard hand-anatomy abbreviations, used as keypoint names throughout the
config and code:

- **MCP** — Metacarpophalangeal joint: the knuckle at the base of the finger,
  where it meets the palm.
- **PIP** — Proximal Interphalangeal joint: the middle knuckle.
- **DIP** — Distal Interphalangeal joint: the knuckle closest to the
  fingertip.
- **TIP** — the fingertip itself (not a joint — the very end of the finger).

Order from palm to fingertip: **MCP → PIP → DIP → TIP**.

Two repo-specific wrinkles:

- **Thumb naming differs anatomically.** The thumb has one fewer bone/joint
  than the other fingers, so its base joint is a **CMC** (carpometacarpal)
  joint rather than an MCP. The retargeter's code uniformly calls this
  position `pip` for all fingers as a naming convention, so
  `joint_name_aliases` maps the thumb's `pip` to the URDF's real joint name,
  `thumb_cmc`.
- **The "v2" hand hardware has no DIP frame.** Its four-finger URDF chains
  define only two joints (`mcp`, `pip`) per finger, not three. The
  `robot_frames_by_version.v2` mapping compensates by shifting the
  retargeter's `pip`/`dip` keypoint names onto the URDF's `mcp`/`pip` frames
  respectively — reusing the two joints that do exist, rather than matching
  anatomically.

## The config file: `adaptive_analytical_orca.yaml`

Loaded as a plain nested dict (via `yaml.safe_load`, no schema/validation),
so unknown keys are silently ignored and missing required keys only raise
errors when accessed. This file is also the **packaged default** — loaded
automatically via `importlib.resources` when no `--retarget-config` is
passed, not merely an example.

### `optimizer.type: adaptive_analytical`

Descriptive only — **not read by any code**. The retargeter backend is
actually selected via a Python-level argument
(`Retargeter.from_paths(backend=...)`), not by this field. Changing it here
has no effect.

### `retarget:` — optimization tuning knobs

These shape the per-frame cost function the `nlopt` solver minimizes.

| Key | Meaning |
| --- | --- |
| `huber_delta_cm` | [Huber-loss](https://en.wikipedia.org/wiki/Huber_loss) transition threshold (cm) for *position* error terms (fingertip-to-palm vector, full-hand pip/dip/tip positions). Below this, error is penalized quadratically; beyond it, linearly — robust to noisy/outlier tracking frames. |
| `huber_delta_dir` | Same Huber-loss idea, but for the fingertip *direction* error (unit vector dip→tip). Kept separate from `huber_delta_cm` because direction residuals are unitless while position residuals are in cm. |
| `norm_delta` | Weight on an L2 regularization term penalizing joint angles from drifting far from the previous frame's solution — a smoothness/damping term in the objective (not a post-hoc filter). |
| `lp_alpha` | Exponential-smoothing coefficient for a low-pass filter applied to the *solved* joint angles after optimization. `1.0` = no smoothing; lower values smooth more (at the cost of added lag). Must be in `(0, 1]`. |
| `w_pos` | Weight on the fingertip **position** error term. |
| `w_dir` | Weight on the fingertip **direction** error term. Default (`10.0`) is 10x `w_pos` — direction alignment is prioritized well above raw position matching. |
| `w_full_hand` | Weight on the "full hand" pip/dip/tip position-matching term — keeps the whole finger's curl shape consistent, not just the tip. |
| `scaling` | Global multiplier scaling the human hand's reach before comparing to robot geometry (accounts for human vs. robot hand size). |
| `segment_scaling` | Per-finger, per-segment (proximal/middle/distal) scale corrections applied to "full hand" target vectors — e.g. the default config lengthens index/pinky distal segments (1.03–1.15x) to compensate for shape mismatches between MediaPipe's skeleton and the robot's kinematics. |
| `pinch_thresholds_cm` | Per-finger `d1` (close) / `d2` (far) thumb-to-fingertip distance thresholds (cm), used to detect pinch gestures. As the thumb approaches a finger, the optimizer smoothly shifts weighting from "match overall hand shape" (`w_full_hand`) toward "match precise tip position/direction" (`w_pos`/`w_dir`), so pinches register accurately. |
| `mediapipe_rotation` | Optional x/y/z rotation (degrees) applied to raw MediaPipe landmarks before retargeting, to correct for an off-axis-mounted camera. All zeros (default) = no correction applied. |

Two additional solver settings exist in code with hardcoded defaults, but
aren't currently exposed in this YAML: `maxeval` (default 50) and
`ftol_abs` (default 1e-4) — SLSQP stopping criteria.

### `robot_frames:`

For each hand (`left`/`right`), maps each finger's `pip`/`dip`/`tip` plus a
`palm` entry to the corresponding **URDF frame name**. The optimizer uses
these names to query Pinocchio for "where is this frame, and how do joint
angles move it?"

### `robot_frames_by_version:`

An override of `robot_frames` keyed by hand hardware/URDF revision (e.g.
`v2`), consulted first; if no entry exists for the detected version, the
base `robot_frames` table is used instead. The version is auto-detected from
the URDF file path (looking for a `v1`/`v2` path segment) — it is not an
explicit config field.

### `joint_name_aliases:`

Renames the retargeter's internal joint identifiers to the URDF's actual
joint names where they differ. In the default config, only
`thumb_pip -> thumb_cmc` is aliased (both hands), matching the anatomical
note above.

## Other implementation notes

- The `adaptive_analytical` backend requires optional dependencies `nlopt`,
  `pinocchio`, and `pyyaml` (installed via `uv sync --extra adaptive`);
  missing them raises a clear install hint.
- `_PinocchioOrcaRobot` assumes one degree of freedom per joint
  (`model.nq == model.nv`) and raises `NotImplementedError` otherwise — this
  backend does not support multi-DoF (e.g. spherical) joints.
