"""Online residual SAC for Astribot S1 pressing the VIB-820 hall-call button.

One resident Isaac Sim environment, one frozen VLA policy server, one SAC
actor-critic.  The frozen policy proposes seven-step joint8 chunks; the SAC
actor adds a bounded normalized correction to every step, the corrected chunk
is expanded to the 32-joint contract, and the scenario's physical press signal
supplies a sparse reward.

Enforced by code, not by convention:

* Training scenarios come from the frozen ``train`` split only; validation and
  test ids are rejected so no evaluation scene leaks into training.
* The actor sees deployable proprioception only (32 measured positions and 32
  velocities) plus the frozen base chunk.  Scene coordinates, button geometry
  and task ids never enter the policy.
* Success uses the evaluator's physics rule: button travel and fingertip force
  at the same tick, latched lamp, no unexpected contact.  A residual cannot be
  credited for anything closed-loop evaluation would not count.
* Inference pauses physics; this is a synchronous simulation loop with no
  real-time robot claim.

Ready for a first run, but it has only been unit-tested on CPU here: the GPU /
Isaac steps must be executed on the workstation.  See docs/astribot_online_rl.md.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import traceback
import urllib.request

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from astribot_closedloop_control import FIXED, interpolate_target, physical_hit, validate_prediction
from pressb.online_rl.astribot import (BASE_DIM, DEFAULT_GAMMA, DEFAULT_SCALE, HORIZON, OBS_DIM,
                                        ChunkJournal, compose_joint8, expand_joint8, load_replay,
                                        observation, save_replay, summarize_episode, validate_scale,
                                        within_limits)
from pressb.online_rl.learner import LearnerConfig, ReplayBuffer, SACLearner


class EpisodeStop(Exception):
    """A physical or contract failure that terminates the episode."""

    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def atomic_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_contract(model_run: Path, split: str):
    """Read a frozen split; training is separately restricted to ``train``."""
    if split not in ("train", "val", "test"):
        raise ValueError("split must be train, val or test")
    contract = json.loads((model_run / "data_contract.json").read_text())
    ids = contract.get("split", {}).get(f"{split}_source_episode_ids")
    if not ids or any(not isinstance(candidate, int) for candidate in ids):
        raise ValueError(f"Frozen {split} episode ids are missing or malformed")
    conditions = contract.get("scene_conditions") or {}
    missing = [candidate for candidate in ids if str(candidate) not in conditions]
    if missing:
        raise ValueError(f"Scene conditions missing for candidates {missing[:5]}")
    if not np.isfinite(np.asarray(contract["fixed_action32"], dtype=np.float32)).all():
        raise ValueError("fixed_action32 must be finite")
    return contract, list(ids)


def predict(endpoint, images, state, seed):
    """One frozen-policy request; identical wire format to closed-loop eval."""
    payload = io.BytesIO()
    np.savez(payload, head=images["head"], wrist_right=images["wrist_right"],
             state=state, seed=np.array(seed))
    request = urllib.request.Request(endpoint + "/predict", data=payload.getvalue(),
                                     headers={"Content-Type": "application/octet-stream"})
    with urllib.request.urlopen(request, timeout=300) as response:
        with np.load(io.BytesIO(response.read()), allow_pickle=False) as result:
            return {key: result[key].copy() for key in result.files}


def check_health(endpoint, model_run: Path, step: int):
    manifest = json.loads((model_run / f"checkpoints/step_{step:06d}/manifest.json").read_text())
    with urllib.request.urlopen(endpoint + "/health", timeout=30) as response:
        health = json.load(response)
    if not health.get("ready") or health.get("model_sha256") != manifest["model_sha256"]:
        raise RuntimeError(f"Policy server identity mismatch: {health}")
    return manifest, health


class Recorder:
    """Per-tick physical record plus the evaluator's safety checks."""

    def __init__(self, runtime, cfg, contract):
        self.runtime = runtime
        self.cfg = cfg
        self.env = runtime.envs[0]
        self.world = runtime.world
        self.fixed = np.asarray(contract["fixed_action32"], dtype=np.float32)
        self.limits = np.c_[self.env.robot.dof_properties["lower"],
                            self.env.robot.dof_properties["upper"]]
        self.command = runtime.targets[0].copy()
        self.button = self.env.scene["buttons"][0]
        self.tips = {self.env.meta["robot_path"] + "/astribot_gripper_right_Link_" + s
                     for s in ("L11", "R11")}
        self.forces = [0.0]
        self.unexpected = []
        self.records = []
        self.physics_step = 0
        self.light = False
        self.hit_step = None
        self.base_initial = self.env.base_body.get_world_pose()[0].copy()
        self.subscription = None

    def contact(self, headers, data, actor_paths):
        for header in headers:
            actors = [actor_paths[x] for x in (header.actor0, header.actor1)]
            force = sum(float(np.linalg.norm(data[i].impulse)) * 120
                        for i in range(header.contact_data_offset,
                                       header.contact_data_offset + header.num_contact_data))
            if self.button["body_path"] in actors and any(p in self.tips for p in actors):
                self.forces[0] += force
                continue
            if (force > 0.1 and any(p.startswith(self.env.meta["robot_path"] + "/") for p in actors)
                    and not all(p in self.tips for p in actors)):
                if len(self.unexpected) < 500:
                    self.unexpected.append(dict(step=self.physics_step, actors=actors, force_n=force))

    def start(self):
        self.subscription = self.env.subscribe_contacts(self.contact)

    def close(self):
        if self.subscription is not None:
            self.runtime.contact_router.callbacks.pop(0, None)
            self.subscription = None

    def observe(self):
        return observation(self.runtime.q[0], self.runtime.qd[0])

    def tick(self, target):
        from pressb.astribot_collection.lights import set_button_light
        self.forces[0] = 0.0
        self.runtime.targets[0] = target
        self.runtime.view.set_joint_position_targets(self.runtime.tensor(self.runtime.targets))
        previous_time = float(self.world.current_time)
        self.world.step(render=False)
        self.runtime.refresh()
        if abs(float(self.world.current_time) - previous_time - 1 / 120) > 2e-6:
            raise EpisodeStop("physics_timestep_contract")
        self.command = np.asarray(target, dtype=np.float32).copy()
        self.physics_step += 1
        q = self.runtime.q[0].copy()
        travel = float(self.env.buttons[0].get_world_pose()[0][0] - self.button["rest_x"])
        hit = physical_hit(travel, self.forces[0], self.cfg)
        if hit and self.hit_step is None:
            self.hit_step = self.physics_step
        if hit and not self.light:
            self.light = True
            set_button_light(self.world.stage, self.button, True)
        actual_light = bool(self.world.stage.GetPrimAtPath(self.button["body_path"])
                            .GetAttribute("pressb:illuminated").Get())
        base = self.env.base_body.get_world_pose()[0].copy()
        self.records.append(dict(q=q, qd=self.runtime.qd[0].copy(), target=self.command.copy(),
                                 travel=travel, force=self.forces[0], light=actual_light,
                                 physical_hit=hit, base=base))
        if not np.isfinite(q).all() or np.any(q < self.limits[:, 0] - 0.02) \
                or np.any(q > self.limits[:, 1] + 0.02):
            raise EpisodeStop("nonfinite_or_out_of_limit_state")
        if self.unexpected:
            raise EpisodeStop("unexpected_contact")
        if float(np.linalg.norm(base - self.base_initial)) > 0.001:
            raise EpisodeStop("base_drift_over_1mm")
        if np.max(np.abs(q[FIXED] - self.fixed[FIXED])) > 0.02:
            raise EpisodeStop("inactive_joint_drift_over_0.02rad")
        if self.light != actual_light:
            raise RuntimeError("Button light state mismatch")
        return hit


class EpisodeCounter:
    """Resume-safe episode directory numbering."""

    def __init__(self, root: Path):
        self.root = root / "episodes"
        self.next = len(list(self.root.glob("episode_*"))) if self.root.exists() else 0

    def take(self) -> Path:
        self.next += 1
        return self.root / f"episode_{self.next:06d}"


def run_episode(runtime, cfg, contract, candidate, args, learner, replay, counter,
                train=True):
    """Run one episode; ``train`` adds transitions, evaluation never writes."""
    from pressb.astribot_validation import initial_overlap_check
    out = counter.take()
    runtime.prepare_wave([cfg], [out])
    recorder = Recorder(runtime, cfg, contract)
    recorder.start()
    journal = ChunkJournal(replay, train=train, gamma=args.gamma)
    images = runtime.last_images[0]
    reason = "timeout"
    success = False
    failure = None
    control_index = 0
    started = time.perf_counter()
    requested = 0
    try:
        overlap = initial_overlap_check(runtime.world.stage, recorder.env.meta,
                                        floor_path=recorder.env.prefix + "/Environment/Floor")
        if overlap["status"] != "clear":
            raise EpisodeStop("initial_overlap_" + overlap["status"])
        recorder.tick(recorder.command)
        max_control = round(args.max_seconds * 30)
        while control_index < max_control:
            state = runtime.q[0].copy()
            observation_time = float(runtime.world.current_time)
            result = predict(args.endpoint, images, state, 420000 + candidate * 100003 + requested)
            if runtime.world.current_time != observation_time or not np.array_equal(runtime.q[0], state):
                raise RuntimeError("Physics advanced during inference")
            try:
                validate_prediction(result["action8"], result["action32"],
                                    contract["fixed_action32"], recorder.limits)
            except (ValueError, AssertionError) as exc:
                raise EpisodeStop("invalid_policy_command") from exc
            base8 = np.asarray(result["action8"], dtype=np.float32)
            obs = recorder.observe()
            if journal.pending is not None:
                journal.flush(obs, base8)
            action = learner.act(obs[None, :], base8[None, :, :], deterministic=not train,
                                 exploration_probability=args.exploration_probability)[0]
            joint8 = compose_joint8(base8, action, args.scale)
            action32 = expand_joint8(joint8, contract["fixed_action32"])
            if not within_limits(action32, recorder.limits, 0.02):
                raise EpisodeStop("residual_exceeds_joint_limits")
            row = journal.start(obs, action, base8)
            steps = 0
            try:
                for target in action32[:args.execute_horizon]:
                    if control_index >= max_control:
                        break
                    for interpolated in interpolate_target(recorder.command, target):
                        if recorder.tick(interpolated):
                            success = True
                    steps += 1
                    control_index += 1
                    if success:
                        break
            except EpisodeStop as exc:
                failure = exc.reason
            requested += 1
            images = runtime.render()[0]
            if failure:
                journal.commit(row, next_obs=recorder.observe(),
                               next_base=np.zeros((HORIZON, BASE_DIM)),
                               steps=steps, failure_reason=failure)
                reason = failure
                break
            if success:
                flags = journal.commit(row, next_obs=recorder.observe(),
                                       next_base=np.zeros((HORIZON, BASE_DIM)),
                                       steps=steps, success=True)
                reason = flags["reason"]
                break
            journal.defer(row, steps)
            if requested % args.status_every_requests == 0:
                print(json.dumps(dict(candidate=candidate, control_index=control_index,
                                      max_travel_m=max(r["travel"] for r in recorder.records),
                                      max_force_n=max(r["force"] for r in recorder.records))), flush=True)
        if not success and not failure and journal.pending is not None:
            journal.commit(journal.pending, next_obs=recorder.observe(),
                           next_base=np.zeros((HORIZON, BASE_DIM)),
                           steps=journal.pending_steps, timed_out=True)
            journal.pending = None
            reason = "timeout"
    except EpisodeStop as exc:
        failure = exc.reason
        if journal.pending is not None:
            journal.commit(journal.pending, next_obs=recorder.observe(),
                           next_base=np.zeros((HORIZON, BASE_DIM)),
                           steps=journal.pending_steps, failure_reason=failure)
            journal.pending = None
        reason = failure
    finally:
        recorder.close()
        summary = summarize_episode(recorder.records)
        episode = dict(kind="train" if train else "eval", candidate_id=candidate,
                       status="success" if success else ("failed" if failure else "timeout"),
                       reason=reason, control_steps=control_index,
                       physics_steps=recorder.physics_step,
                       wall_seconds=time.perf_counter() - started,
                       transitions=journal.transitions,
                       unexpected_contacts=len(recorder.unexpected), **summary)
        atomic_json(out / "report.json", episode)
    return episode


def build_learner(args):
    config = LearnerConfig(method="action_residual", obs_dim=OBS_DIM, base_dim=BASE_DIM,
                           residual_scale=args.scale, batch_size=args.batch_size,
                           seed=args.seed)
    return SACLearner(config, device="cpu")


def save_state(path: Path, learner, replay, status):
    learner.save(path / "learner.pt", extra=dict(status=status))
    save_replay(path / "replay.npz", replay)


def describe(learner, replay, status, args, recent):
    return dict(kind="astribot_online_residual_sac", split="train",
                model_sha256=status["model_sha256"], checkpoint_step=status["checkpoint_step"],
                transitions=int(len(replay)), episodes=status["episodes"],
                updates=int(learner.updates), successes=status["successes"],
                rolling_success_rate=(sum(e["status"] == "success" for e in recent) / len(recent)
                                      if recent else None),
                residual_scale=list(args.scale), gamma=args.gamma,
                exploration_probability=args.exploration_probability,
                max_seconds=args.max_seconds, last_episodes=recent,
                replay_capacity=getattr(replay, "capacity", None),
                updated_unix_s=time.time())


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-run", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path,
                        default=ROOT / "docs/vib820_experiment_spec/config_v1_2.json")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8794")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-transitions", type=int, default=20_000)
    parser.add_argument("--warmup-transitions", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--training-frequency", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--scale", type=float, nargs=8, default=list(DEFAULT_SCALE))
    parser.add_argument("--max-seconds", type=float, default=20.0)
    parser.add_argument("--execute-horizon", type=int, default=HORIZON)
    parser.add_argument("--exploration-probability", type=float, default=1.0)
    parser.add_argument("--checkpoint-every", type=int, default=1_000)
    parser.add_argument("--status-every-requests", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval-episodes", type=int, default=0,
                        help="After training, evaluate deterministically on a frozen split")
    parser.add_argument("--eval-split", default="val", choices=("val", "test"))
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training and only run the deterministic evaluation")
    parser.add_argument("--eval-max-seconds", type=float, default=40.0,
                        help="Episode limit for evaluation, matching closed-loop eval")
    parser.add_argument("--smoke", action="store_true",
                        help="Short smoke run: six simulated seconds, tiny learner batch")
    args = parser.parse_args()

    args.scale = validate_scale(args.scale)
    if not 0 <= args.exploration_probability <= 1:
        raise SystemExit("exploration probability must be in [0,1]")
    if args.execute_horizon not in range(1, HORIZON + 1):
        raise SystemExit(f"execute horizon must be in [1,{HORIZON}]")
    if args.eval_episodes:
        if not args.resume:
            raise SystemExit("--eval-episodes requires --resume with a trained learner")
        if args.eval_episodes < 1:
            raise SystemExit("eval episodes must be positive")
    if args.eval_only and not args.eval_episodes:
        raise SystemExit("--eval-only requires --eval-episodes")
    if not 0 < args.eval_max_seconds <= 120:
        raise SystemExit("eval max seconds must be in (0,120]")
    if args.smoke:
        args.max_seconds = min(args.max_seconds, 6.0)
        args.max_transitions = 8
        args.warmup_transitions = 2
        args.batch_size = 4
        args.training_frequency = 1
        args.checkpoint_every = 4
    if not 0 < args.max_seconds <= 120:
        raise SystemExit("max seconds must be in (0,120]")
    if min(args.max_transitions, args.warmup_transitions, args.batch_size,
           args.training_frequency, args.checkpoint_every) < 1:
        raise SystemExit("transition, batch and checkpoint sizes must be positive")
    if args.warmup_transitions > args.max_transitions:
        raise SystemExit("warmup cannot exceed the total transition budget")

    contract, candidates = load_contract(args.model_run, "train")
    manifest, health = check_health(args.endpoint, args.model_run, args.checkpoint_step)
    try:
        args.output.mkdir(parents=True, exist_ok=args.resume)
    except FileExistsError:
        raise SystemExit(f"{args.output} already exists; pass --resume or choose a new directory")
    counter = EpisodeCounter(args.output)
    learner_path, replay_path = args.output / "learner.pt", args.output / "replay.npz"
    if args.resume:
        if not (learner_path.exists() and replay_path.exists()):
            raise SystemExit("--resume requires learner.pt and replay.npz")
        learner = SACLearner.load(learner_path, device="cpu")
        if learner.config.obs_dim != OBS_DIM or learner.config.base_dim != BASE_DIM:
            raise SystemExit("Checkpoint dimensions do not match joint8 proprioception")
        status = json.loads((args.output / "status.json").read_text())
        capacity = int(status.get("replay_capacity", args.max_transitions))
        if args.max_transitions != capacity:
            raise SystemExit(f"--max-transitions must match the checkpoint replay capacity ({capacity})")
        replay = load_replay(replay_path, capacity=capacity, obs_dim=OBS_DIM,
                             action_dim=learner.action_dim, base_dim=BASE_DIM)
        status["episodes"] = counter.next
        status["history"] = status_history(args)
    else:
        learner = build_learner(args)
        replay = ReplayBuffer(args.max_transitions, OBS_DIM, learner.action_dim, args.seed,
                              base_dim=BASE_DIM)
        status = dict(episodes=0, successes=0, history=[],
                      replay_capacity=args.max_transitions,
                      model_sha256=manifest["model_sha256"],
                      checkpoint_step=args.checkpoint_step,
                      source_sha256={"run_astribot_online_rl.py": sha256(Path(__file__)),
                                     "online_rl/astribot.py": sha256(ROOT / "src/pressb/online_rl/astribot.py"),
                                     "online_rl/learner.py": sha256(ROOT / "src/pressb/online_rl/learner.py"),
                                     "astribot_closedloop_control.py":
                                         sha256(ROOT / "scripts/astribot_closedloop_control.py")},
                      split="train", inference=health)
    atomic_json(args.output / "protocol.json", dict(describe(learner, replay, status, args, []),
                                                    contract_sha256=sha256(args.model_run / "data_contract.json"),
                                                    policy_server=health))

    from isaacsim import SimulationApp
    app = SimulationApp(dict(headless=True, create_new_stage=False, active_gpu=args.gpu,
                             physics_gpu=args.gpu, multi_gpu=False, limit_cpu_threads=8,
                             renderer="RayTracedLighting", anti_aliasing=3, width=960, height=600,
                             disable_viewport_updates=True,
                             extra_args=["--/app/asyncRendering=false",
                                         "--/rtx/ecoMode/enabled=false",
                                         "--/validate/p2p/enabled=false",
                                         "--/validate/iommu/enabled=false",
                                         "--/app/extensions/fsWatcherEnabled=false"]))
    runtime = None
    try:
        from astribot_closedloop_runtime import LiveRuntime as Runtime
        base_cfg = json.loads(args.config.read_text())
        runtime = Runtime(app, base_cfg, args.output / "shared", 1, args.gpu, "gpu", 1, "individual")
        order = np.random.default_rng(args.seed).permutation(len(candidates))
        index = status["episodes"] % len(candidates)
        next_checkpoint = (int(learner.updates) // max(1, args.checkpoint_every) + 1) * args.checkpoint_every
        metrics_path = args.output / "metrics.jsonl"
        while not args.eval_only and len(replay) < args.max_transitions:
            candidate = candidates[int(order[index % len(order)])]
            index += 1
            condition = contract["scene_conditions"][str(candidate)]
            cfg = deepcopy(base_cfg)
            cfg.update(robot_base_x=condition["x_m"], robot_base_y=condition["y_m"],
                       robot_base_yaw_rad=condition["yaw_rad"])
            episode = run_episode(runtime, cfg, contract, candidate, args, learner, replay,
                                  counter, train=True)
            status["episodes"] += 1
            status["successes"] += int(episode["status"] == "success")
            status["history"] = (status["history"] + [episode])[-20:]
            print(json.dumps(episode, ensure_ascii=False), flush=True)
            while (len(replay) >= args.warmup_transitions and len(replay) >= args.batch_size
                   and int(learner.updates) < len(replay) * args.training_frequency):
                metrics = learner.update(replay.sample(args.batch_size))
                with metrics_path.open("a") as stream:
                    stream.write(json.dumps(metrics) + "\n")
            atomic_json(args.output / "status.json", describe(learner, replay, status, args,
                                                              status["history"]))
            while int(learner.updates) >= next_checkpoint:
                save_state(args.output, learner, replay, status)
                next_checkpoint += args.checkpoint_every
            if args.smoke:
                break
        if args.eval_episodes:
            evaluate(runtime, args, learner, base_cfg)
    except BaseException:
        atomic_json(args.output / "infrastructure_error.json", dict(error=traceback.format_exc()))
        raise
    finally:
        if runtime is not None:
            runtime.close()
        save_state(args.output, learner, replay, status)
        atomic_json(args.output / "status.json", describe(learner, replay, status, args,
                                                          status_history(args)))
        app.app.post_quit()
        app.close()


def evaluate(runtime, args, learner, base_cfg):
    """Deterministic residual evaluation on a held-out frozen split."""
    eval_contract, candidates = load_contract(args.model_run, args.eval_split)
    counter = EpisodeCounter(args.output / "eval")
    count = min(args.eval_episodes, len(candidates))
    rows = []
    original_seconds = args.max_seconds
    args.max_seconds = args.eval_max_seconds
    for candidate in candidates[:count]:
        condition = eval_contract["scene_conditions"][str(candidate)]
        cfg = deepcopy(base_cfg)
        cfg.update(robot_base_x=condition["x_m"], robot_base_y=condition["y_m"],
                   robot_base_yaw_rad=condition["yaw_rad"])
        episode = run_episode(runtime, cfg, eval_contract, candidate, args, learner, None,
                              counter, train=False)
        rows.append(episode)
        print(json.dumps(episode, ensure_ascii=False), flush=True)
    args.max_seconds = original_seconds
    summary = dict(kind="astribot_online_residual_sac_eval", split=args.eval_split,
                   requested=args.eval_episodes, completed=len(rows),
                   successes=sum(row["status"] == "success" for row in rows),
                   episodes=rows, deterministic=True,
                   note="Held-out validation scenes; each scene once; synchronous simulation")
    atomic_json(args.output / "eval" / "summary.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "episodes"}),
          flush=True)


def status_history(args):
    path = args.output / "episodes"
    rows = []
    for report in sorted(path.glob("episode_*/report.json"))[-20:]:
        try:
            rows.append(json.loads(report.read_text()))
        except (OSError, ValueError):
            continue
    return rows


if __name__ == "__main__":
    main()
