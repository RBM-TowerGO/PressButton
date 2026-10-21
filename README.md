# PressB — Elevator Button Pressing with VLA-JEPA and Online Residual RL

**English** | [简体中文](README.zh-CN.md)

**RL update, 2026-10-20:** The first 20k residual run scored 11/40 on validation, below the frozen 23/40 baseline. Both stabilized 50k seeds resumed from checkpoints around 17:56 China time after camera-pose validation interruptions; the primary seed is past 20k with updates, and the replica is collecting warmup experience. [Interruption evidence, configuration and live progress](docs/astribot_round2_20261020.md). This supersedes the older “no formal long run” status below.

**Status: 2026-10-19.** One Isaac Sim codebase, two task lines. The active line is an **Astribot S1 humanoid pressing the VIB-820 hall-call button**: 400 expert episodes were collected and exported to LeRobot v3, VLA-JEPA was fine-tuned to **step 8000**, the frozen test split scored **17/40 (42.5%)**, and a paired validation comparison against a conservatively continued **step 17,000** scored **23/40 vs 19/40** (McNemar p = 0.3438, not significant). Between those two checkpoints offline action error improved while closed-loop success did not. Online residual SAC is now wired to this task and passes 20 CPU contract tests plus a GPU/Isaac smoke run; no formal long run has been executed yet. The earlier **PiPER 12-button line** is complete (best 119/120 = 99.17%) and is kept below as history.

[Current results](#current-results) · [Quick start](#quick-start) · [Reproducing the active line](#reproducing-the-active-line) · [Online residual RL](#online-residual-rl) · [Legacy PiPER line](#legacy-piper-line) · [Project handover](docs/HANDOVER.md)

The validated environment is **Isaac Sim 5.0 / Isaac Lab 2.2 / Python 3.11** on RTX 5090 and RTX PRO 6000 Blackwell. The repository contains scene construction, scripted expert collection, per-episode physical audit, LeRobot export, a frozen VLA inference service, synchronous closed-loop evaluation, and residual reinforcement learning. Historical Isaac Sim 4.5 / Isaac Lab 2.0.2 entry points are retained but are not the current line.

## Two task lines

| | Astribot S1 / VIB-820 (active) | PiPER / 12-button wall panel (legacy) |
|---|---|---|
| Robot | Astribot S1 humanoid; fixed base, torso, head and left arm; right arm and original closed gripper act | AgileX PiPER arm holding an 8 mm-gripped pressing rod |
| Task | One downward hall-call button; instruction `Press the down hall-call button.` | 12 floor buttons; `Press 24 floor.` … `Press 35 floor.` |
| State / action | 32 joints recorded; 7 right-arm joints + right gripper learned (8-D), expanded to 32 | 8-D TCP pose + gripper width, in `base_link` |
| Cameras | 7 views recorded; head 1280×720 + right wrist 640×480 feed the policy | 2 D435 views, 640×480 |
| Expert data | 400 episodes, 245,983 frames, 30 Hz | 1,200 episodes, 306,830 frames, 30 Hz |
| Policy | VLA-JEPA DiT-B, 7-step horizon, step 8000 | VLA-JEPA step 10600 + residual SAC |
| Status | 40-scene closed loop evaluated; online RL wired | Complete; best 119/120 (99.17%) |

Everything below the legacy heading refers to the PiPER line; do not mix its numbers with the S1 results.

## Current results

The active task is button `0` in the VIB-820 lobby scene: a single downward hall-call button on a 400 × 95 mm brushed-steel panel, 1,200 mm above the floor, with 2 mm simulated travel. The S1 keeps a fixed base, torso, head and left arm, and presses with its original closed right gripper. The policy input is the head RGB, the right-wrist RGB, the 32 measured joint angles and a fixed instruction; scene coordinates, button geometry and episode indices are never given to the model.

| Stage | Result |
|---|---|
| Expert collection | **400/400** accepted episodes (2026-10-16 → 10-17) from 528 recorded attempts; 30 `physical_failure` and 22 `infrastructure_error` attempts kept as evidence; 7 camera views, 30 Hz sampling, 120 Hz physics |
| LeRobot v3 export | 400 episodes / 245,983 frames / 13 GiB / `robot_type=astribot`; official reader and full video decode pass; `release_report.json` reports `status=passed` |
| Fine-tune | VLA-JEPA on a frozen 320/40/40 train/val/test split; best **step 8000**, 10,268,108,194 bytes, SHA-256 `db98b7c5…`; contracts, normalization statistics, split and training source are stored with the weights |
| Closed loop, test split | **17/40 = 42.5%**; 23 timeouts (16 moved without contact, 7 stalled near the start pose); 0 unexpected contact |
| Paired, validation split | **23/40 = 57.5%** (step 8000) vs **19/40 = 47.5%** (step 17,000); exact McNemar **p = 0.3438** |
| Online residual SAC | End-to-end wiring verified by 20 CPU contract tests and a GPU/Isaac smoke run; ~2.1–2.4 transitions/s in a single environment, 20k transitions ≈ 2.5 h; no formal run yet |

The test and validation splits contain **different scenarios**, so 42.5% and 57.5% must not be compared with each other.

### Paired comparison: offline error improved, closed loop did not

Step 17,000 was produced by conservative continuation of step 8000 (about one third of the original learning rate, selected by offline score). Both checkpoints were run once on each of the 40 validation scenarios, with matched seeds and verified paired initial conditions.

| Checkpoint | Success | Paired outcome |
|---|---:|---|
| step 8000 | 23/40 (57.5%) | both succeed 16, both fail 14 |
| step 17,000 | 19/40 (47.5%) | improved 3, regressed 7 |

| Offline metric (same 120 validation anchors) | step 8000 | step 17,000 | Direction |
|---|---:|---:|---|
| arm MAE (rad) | 0.002696 | 0.002620 | better |
| gripper MAE (rad) | 0.003259 | 0.001995 | better |
| selection score (rad) | 0.003190 | 0.003053 | better |
| flow loss | 0.004782 | 0.006078 | worse |

The metric used to select checkpoints disagrees with the closed-loop objective: better offline MAE came with a lower success rate. Failures concentrate in the approach and contact phases rather than in per-step action error, so the useful direction is contact-phase behaviour, termination policy and approach trajectories — not further compression of offline MAE.

| Failure class (mutually exclusive, in order) | step 8000 | step 17,000 |
|---|---:|---:|
| Moved but never entered the button neighbourhood (≤ 20 mm) | 4 | 9 |
| Stalled near the initial pose | 6 | 4 |
| Inside 20 mm but no valid contact | 5 | 4 |
| Still approaching at timeout | 2 | 1 |
| Approached, then left | 0 | 2 |
| `unexpected_contact` | 0 | 1 |
| **Success** | **23** | **19** |

For context, the scripted IK expert (no learning) completes 25/30 = 83.3% over a 40 cm × 40 cm base-shift grid, with all 30 key poses solvable and 5 unexpected contacts. That is a different protocol and only indicates how much room remains for the learned policy.

## Quick start

Requirements: Linux x86_64, Conda, an Isaac-Sim-compatible NVIDIA RTX GPU and driver (Blackwell validated with driver 580.178.04), Vulkan, GPU rendering even when headless, and system `git`, `ffmpeg`/`ffprobe` (with `libx264`) and DejaVu fonts.

```bash
git clone https://github.com/RBM-TowerGO/PressButton.git
cd PressButton

sudo apt-get install ffmpeg fonts-dejavu-core          # if not already installed
bash scripts/setup_env_5090.sh                        # Python 3.11 / Isaac Sim 5.0 / Isaac Lab 2.2
bash scripts/setup_dataset_env.sh                     # Python 3.12 / LeRobot 0.6.1 / CPU PyTorch

# Inference overlay used by the policy server; leaves the Isaac environment untouched
.conda/envs/pressb/bin/python -m venv --system-site-packages .venv-astribot-inference
```

Keep three environments separate and never install `requirements-dataset.txt` into the Isaac environment:

| Environment | Purpose |
|---|---|
| `.conda/envs/pressb` | Simulation, collection, evaluation, online RL |
| `.conda/envs/lerobot` | LeRobot conversion, official reader, data audits |
| `.venv-astribot-inference` | VLA-JEPA policy server only |

Fetch the pinned assets and build the active scene:

```bash
source scripts/activate_local.sh                      # Conda + CUDA_VISIBLE_DEVICES=0
python scripts/fetch_astribot_assets.py               # S1 USD/URDF/meshes, ~502 MB, SHA-256 pinned
PYTHONPATH=.cache/usd-inspect python scripts/prepare_astribot_asset.py
python scripts/build_vib820.py --output outputs/vib820_model
```

`scripts/activate_local.sh` and the `scripts/*_astribot_*.sh` wrappers encode this workspace's absolute paths and Conda location; on a fresh clone call the underlying `scripts/*.py` entry points directly. `activate_local.sh` exposes GPU 0 by default; export `CUDA_VISIBLE_DEVICES=0,1` and pass `--gpu 1` to use the second card.

Verify the scripted expert before collecting or evaluating anything. Output directories must not already exist:

```bash
python scripts/run_astribot_press.py --headless --video \
  --config configs/astribot_vib820.json --buttons 0 --repeats 3 \
  --output outputs/vib820_press
python scripts/audit_astribot_press.py outputs/vib820_press
```

Run the test suite (CPU only; 959 passed, 14 skipped, 969 collected as of 2026-10-19):

```bash
.conda/envs/pressb/bin/python -m pytest -q
```

## Reproducing the active line

### 1. Expert collection

The 400 episodes were produced on two GPUs with two resident single-environment worker processes per GPU (CPU physics, GPU rendering, NVENC), under an explicit user authorization file rather than the original multi-environment throughput gate. No speedup ratio is claimed for that configuration. Candidates were frozen first (`seed=20261014`, base X ∈ [-0.30, -0.14] m, Y ∈ [0, 0.40] m, yaw ∈ [0°, 10°]); the first 400 successful candidates in ID order form the set; after a full audit 147/378 entries were moved out and replaced by the separately collected candidates 515/516.

Entry points are `scripts/run_astribot_resident_pipeline.py` and `scripts/collect_astribot_dataset.py`. The exact production commands, authorization file, resume rules and replacement evidence are in the [collection runbook](docs/plan/vib820_gpu_collection_400/runbook.md); the frozen parameters and acceptance gates are in the [collection plan](docs/plan/vib820_gpu_collection_400/plan.md) and the [experiment specification](docs/vib820_experiment_spec/README.md). Always use a new output directory; existing committed evidence is immutable.

### 2. LeRobot v3 export

```bash
.conda/envs/lerobot/bin/python scripts/export_astribot_lerobot.py \
  --selection outputs/vib820_gpu_collection_400/selection.json \
  --output outputs/vib820_lerobot_400 --gpu 1
```

`observation.state` is 32 measured joint angles in radians and `action` is the 32-dimensional absolute joint target at the next 30 Hz sample; the final frame repeats the previous target rather than a retraction. Seven `observation.images.<camera>` features keep native resolution, intrinsics and extrinsics. The official reader checks first/middle/last frames and field shapes, and the finalizer re-decodes every training video and runs stratified action replays:

```bash
.conda/envs/pressb/bin/python scripts/finalize_astribot_collection.py \
  --raw outputs/vib820_gpu_collection_400 \
  --export outputs/vib820_lerobot_400 \
  --replay-runs outputs/vib820_gpu_collection_400_replays/gpu0 \
                outputs/vib820_gpu_collection_400_replays/gpu1 \
  --output outputs/vib820_gpu_collection_400_release --gpu 0
```

Fine-tuning itself is **not** reproduced on this machine: it ran on a separate HPC cluster with VLA-JEPA and the `astribot_lerobot` adapter. This repository keeps the resulting weights, the action/training contracts, the frozen split, normalization statistics, `config.yaml` and a training-source snapshot under `models/astribot_head_right_joint8_ft_20261018/`; the policy server loads that snapshot and the relocated config.

### 3. Policy server and closed-loop evaluation

The server validates the checkpoint SHA-256 and every data/action/training contract, loads all 1617 tensors, and listens only on `127.0.0.1:8794`:

```bash
bash scripts/start_astribot_policy_server.sh            # GPU 1

# Interface smoke (2 s per scenario), then the frozen test split
bash scripts/run_astribot_closedloop.sh smoke outputs/astribot_closedloop/NEW_smoke
bash scripts/run_astribot_closedloop.sh test  outputs/astribot_closedloop/NEW_test40
```

Each run writes `protocol.json`, `summary.json` and one directory per scenario with `report.json`, `trajectory.npz`, `policy_requests.json`, camera timing/calibration and four videos. Only `complete=true` means all scheduled scenarios ran. The wrappers hard-code this workspace; on a fresh clone use `scripts/astribot_policy_server.py` and `scripts/eval_astribot_closedloop.py` with the same arguments.

### 4. Paired checkpoint comparison

```bash
.conda/envs/pressb/bin/python scripts/run_astribot_pair.py
```

`run_astribot_pair.py` pins `outputs/astribot_paired_val_20261019/`, verifies both weight and source hashes, starts two policy services, schedules four simulation instances (two per GPU) and merges the earlier serial scenarios with the parallel dispatch without double counting. It refuses to overwrite an existing `status.json`. Re-summarize results without re-running the simulation with `scripts/astribot_pair_analysis.py`.

### 5. Online residual RL

```bash
# Smoke: 6 s per episode, 8 transitions, batch 4
.conda/envs/pressb/bin/python scripts/run_astribot_online_rl.py \
  --model-run models/astribot_head_right_joint8_ft_20261018 --checkpoint-step 8000 \
  --output outputs/astribot_online_rl/NEW_smoke --smoke
```

The formal recipe (20,000 transitions), resume and evaluation commands are in [online residual RL](docs/astribot_online_rl.md).

## Online residual RL

The frozen VLA is never updated. A small SAC actor corrects its 8-D joint targets, so the recipe matches what a real deployment could compute from joint state alone.

| Item | Value |
|---|---|
| Base policy | `models/astribot_head_right_joint8_ft_20261018`, step 8000, SHA-256 checked by the service |
| Base action | 7 steps × 8 absolute joint targets (right arm 7 joints + right gripper) |
| Residual | 7 × 8, normalised to [-1, 1], scaled by 0.02 rad per right-arm joint and 0.05 for the gripper |
| Observation | 64-D: 32 measured joint angles + 32 joint velocities; no scene coordinates, button geometry or task id |
| Reward | Sparse physical success (travel ≥ 1 mm, right tip force > 0.02 N, light latched, no unexpected contact), discounted per executed chunk step |
| Safety termination | Non-finite or out-of-limit joints, unexpected contact, base drift > 1 mm, inactive-joint drift > 0.02 rad, residual pushing a target out of tolerance |
| Data split | Training can only load the frozen `train` split (320 scenarios); `--eval-only` evaluation uses `val` (default) or `test` |
| Timing | 30 Hz decisions over 120 Hz physics, 4 interpolated physics steps per target; physics pauses during inference |
| Episode budget | 20 s simulation time for training, 40 s for evaluation |

Only the action-residual method is wired to the S1 task: `AstribotPolicy.predict` exposes `action8`/`action32` and has no interface for flow initial noise. The smoke run confirmed real cameras, the frozen RPC, physical success detection, sparse reward and per-chunk discounting, replay writes, SAC updates, atomic checkpoint save and `--resume --eval-only` over validation scenarios, with 0 unexpected contacts across 8 episodes. The actor received only 8 updates, so residuals stay near zero and the smoke success rate is essentially the frozen policy — it is not an RL result.

## What the numbers mean

- Both closed-loop experiments are **synchronous simulation**: physics pauses while the model runs. They measure policy capability, not real-time control latency, and say nothing about the real robot.
- Each scenario was run **once per checkpoint**. There is no cross-seed stability estimate and no independent generalisation test set.
- The 400-episode dataset is **success-filtered**, so it cannot serve as a denominator for a success rate. Failed attempts are kept as evidence but are not training data.
- The collection batch ran under an explicit authorization rather than the planned ≥ 2× throughput gate; no acceleration claim is made, and the single-GPU diagnostic numbers are not production throughput.
- Failure classification is descriptive. Distance-to-button thresholds are geometric proxies, not collision-surface distances, and do not prove a cause.
- Scene coordinates are used only to initialise the physics world and to score outcomes, never as policy input or control target.

## Legacy PiPER line

A black PiPER arm sits at the table edge holding a pressing rod with a fixed 8 mm gripper opening. A 2 × 6 panel holds floors 24–29 on the left and 30–35 on the right; contact force depresses a button and its orange edge light latches on. A wrist D435 and a tabletop D435 record 640×480 RGB. Collection covers panel offsets of ±25 mm left/right and ±10 mm front/back with a stratified 10 × 10 grid per floor, 100 episodes each, 1,200 in total; training keeps only the prefix up to the first successful light, while retraction and homing stay in the raw diagnostic records.

The base VLA-JEPA was fine-tuned on those 1,200 episodes (step 10600) and frozen for the online stage. Two methods were learned separately — action-output residuals and flow initial noise — then the trained noise policy was frozen while a new action residual was trained from scratch. The table is one fixed-condition protocol: 12 buttons × 5 panel positions (centre and four corners) × 2 repeats = 120 trials, seed 20260930, parameters frozen.

| Method | Online training | Success | Rate |
|---|---|---:|---:|
| Base VLA-JEPA | no RL | 25/120 | 20.83% |
| Independent action residual | 1M | 81/120 | 67.50% |
| Independent initial-noise policy | 1M | 81/120 | 67.50% |
| Both independently trained policies | two 1M checkpoints, no joint retraining | 96/120 | 80.00% |
| Fixed noise, retrained 9-D residual, γ=0.99 | 400k | 114/120 | 95.00% |
| Fixed noise, retrained 9-D residual, γ=0.995 | 400k | 118/120 | 98.33% |
| Previous setting, residual scale halved | 400k | 87/120 | 72.50% |
| **Fixed noise, retrained XYZ residual, γ=0.995** | **400k** | **119/120** | **99.17%** |

The latest XYZ actor emits only 7 × 3 = 21 residual dimensions with position scale 0.02 m; rotation and gripper keep the frozen model's outputs. That run took about 7 h 25 min for 400k transitions and 398k updates, and 119/120 is a single fixed-condition result, not a claim about all seeds. Smoothing experiments (1,320 evaluations) were rejected: XYZ exponential smoothing with α=0.8 cut measured joint jerk by ~14% and end-effector jerk by ~30% and passed screening at 119/120, but independent-seed checks landed at 117/120 and no candidate met every predefined threshold, so the default remains **no extra post-processing**. Details and reproduction: [residual training over frozen noise](docs/residual_on_frozen_noise.md), [result JSON](docs/results/online_rl_20261009.json), [result CSV](docs/results/online_rl_20261009.csv), [smoothing summary](docs/results/smoothing_20261009.csv).

Demo videos cover all 12 buttons at 1× simulation time with the global view above the wrist view; a labelled freeze-frame marks each task's end. Click a preview to play or download. These demos are not the 120-trial evaluation, which includes panel offsets.

| Base VLA → fixed noise + 9-D residual | 9-D residual → XYZ residual | XYZ smoothing experiment (not adopted) |
|---|---|---|
| [![Before and after reinforcement learning](docs/media/before-vs-residual.jpg)](docs/media/before-vs-residual.mp4) | [![9-D and XYZ residuals](docs/media/pose9-vs-xyz.jpg)](docs/media/pose9-vs-xyz.mp4) | [![XYZ smoothing experiment](docs/media/xyz-smoothing-experimental.jpg)](docs/media/xyz-smoothing-experimental.mp4) |
| Left: 4/12; right: 12/12 (γ=0.995, 400k) | Left: 12/12; right: 11/12, including one 24→25 wrong press | Both sides 12/12; full concurrent evaluations were 116/120 and 117/120 |

See the [media notes](docs/media/README.md) for sources, compression settings, frame counts and audit records.

The PiPER line introduced the three-service architecture — an Isaac simulation node, a frozen VLA-JEPA inference node and a SAC learner node speaking HTTP — with 64 parallel environments, batched inference and independent episode resets. A 1M-transition run measured about 19.81 h for the action residual and 15.16 h for the initial-noise policy; the latest XYZ 400k run took about 7.43 h. One transition is a ≤ 7-step action chunk in one environment, not an episode. The S1 runner is a single-process variant because multi-environment camera isolation has not passed acceptance; see [online RL](docs/online_rl.md), [high-throughput runner](docs/online_rl_fast.md) and the [protocol](docs/online_rl_protocol.md). Video smoothness comes from replayed measured trajectories; no acceleration or frame interpolation is used to create differences.

## Repository layout

| Directory | Contents |
|---|---|
| `src/pressb/` | PiPER scene, kinematics, planning and cameras; VIB-820 scene, materials and S1 press control; dataset semantics |
| `src/pressb/astribot_collection/` | Resident/pipeline collectors, per-episode audit, NVENC and render controls |
| `src/pressb/online_rl/` | SAC learner, three-node protocol, batched simulation/inference, residual-on-noise and S1 residual runner |
| `scripts/` | Setup, asset preparation, collection, export, replay, evaluation, online RL and audit entry points |
| `configs/` | VIB-820 / Astribot sensor configs, PiPER panel layouts, stratified collection and online RL configs |
| `tests/` | CPU unit tests and optional USD scene integration tests (65 test files, 969 tests collected) |
| `assets/*.json` | Pinned asset sources, SHA-256 checksums and installation alignment metadata |
| `docs/` | Current workflow guides, experiment records and the project handover |
| `docs/results/`, `docs/media/` | Small result summaries and selected compressed PiPER demos shipped with the repository |
| `models/` | Local VLA weights, contracts, splits and source snapshots (ignored by Git) |
| `datasets/`, `outputs/`, `logs/`, `.conda/`, `.cache/`, `vendor/` | Local data, evidence, environments, caches and downloads (ignored by Git) |
| `src/VLA-JEPA/`, `src/ZPRL/` | Separately cloned dependency/reference repositories (ignored by Git) |

## Tests

```bash
.conda/envs/pressb/bin/python -m pytest -q

# USD integration tests need an explicit PiPER scene snapshot
PRESSB_TEST_SNAPSHOT=outputs/edge_feedback/scene.usda \
  PYTHONPATH=.cache/usd-inspect:src \
  .conda/envs/pressb/bin/python -m pytest -q tests/test_dataset_scene.py

# Data/video audits run in the separate LeRobot environment
.conda/envs/lerobot/bin/python -m pip install pytest
.conda/envs/lerobot/bin/python -m pytest -q tests/test_press_dataset_audit.py tests/test_export_fps.py
```

A clone without a snapshot skips the USD integration tests; naming a missing snapshot is an error. Kinematics tests that depend on downloaded assets must run after the asset step. When the Isaac environment lacks PyAV, the relevant video tests are skipped explicitly rather than silently passing.

## Documentation map

Active line: [VIB-820 scene](docs/vib820.md) · [S1 asset](docs/astribot_asset.md) · [S1 press control](docs/astribot_press.md) · [sensors](docs/astribot_sensors.md) · [initial pose](docs/astribot_initial_pose.md) · [camera mounts](docs/astribot_wrist_camera_audit.md) · [camera inventory](docs/astribot_camera_inventory.md) · [closed-loop evaluation](docs/astribot_closedloop_20261018.md) · [paired comparison](docs/astribot_paired_val_20261019.md) · [online residual RL](docs/astribot_online_rl.md) · [collection plan](docs/plan/vib820_gpu_collection_400/plan.md) · [collection runbook](docs/plan/vib820_gpu_collection_400/runbook.md) · [experiment specification](docs/vib820_experiment_spec/README.md).

Legacy PiPER line: [scene details](docs/scene_details.md) · [dataset](docs/dataset.md) · [30 Hz data](docs/dataset_30hz.md) · [first-light trimming](docs/dataset_press_only.md) · [panel randomization](docs/dataset_panel_randomization.md) · [stratified coverage](docs/dataset_panel_stratified.md) · [replay](docs/replay.md) · [policy evaluation](docs/policy_eval.md) · [smoothing](docs/policy_smoothing.md) · [online RL](docs/online_rl.md) · [high-throughput RL](docs/online_rl_fast.md) · [residual on frozen noise](docs/residual_on_frozen_noise.md).

Setup and operations: [RTX 5090 guide](docs/rtx5090_setup.md) · [PRO 6000 setup](docs/PRO6000_ENV.md) · [local deployment](docs/local_deployment.md) · [project handover](docs/HANDOVER.md).

## Data, weights, and licenses

Git holds source, configuration, asset manifests, result summaries and the selected PiPER demos only. Datasets, VLA/RL weights, scene snapshots, raw recordings and Conda environments are local and ignored: the 400-episode export lives on the local data disk (documented root `/data/shared/js/PressB/vib820_gpu_collection_400_20261015`), with the raw collection under `outputs/vib820_gpu_collection_400_pro6000/raw` and the fine-tuned weights under `models/`. The S1 description and PiPER assets are downloaded at pinned commits and verified by SHA-256 (`assets/sources.lock.json`, `assets/astribot_sources.lock.json`); VIB-820 materials are CC0 ambientCG/Poly Haven derivatives documented in `assets/materials/vib820/`. NVIDIA software and assets remain subject to their own licenses — see [THIRD_PARTY.md](THIRD_PARTY.md). This repository does not yet declare a license for its own code; public visibility is not a grant.
