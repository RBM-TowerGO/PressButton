"""Run the authorized, gated Astribot campaign on one local GPU.

Smoke -> both fresh 10k trials -> six fresh 100k runs and paired val-40.
A failed gate stops the campaign and preserves all evidence. No auto-resume,
reward changes, or silent protocol adjustments are performed.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MODES = ("state_only", "state_latent")
SEEDS = (20261024, 20261025, 20261026)
CAMERAS = ["head", "overview", "wrist_left", "wrist_right", "panel", "front", "lobby"]
SOURCES = ["scripts/run_astribot_online_rl.py", "scripts/astribot_policy_server.py",
           "scripts/validate_astribot_latent_engineering.py", "scripts/run_astribot_latent_campaign.py",
           "src/pressb/online_rl/learner.py", "src/pressb/online_rl/astribot.py",
           "src/pressb/online_rl/astribot_policy.py", "src/pressb/online_rl/astribot_features.py"]


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def metrics_summary(path):
    """Stream rather than retaining 100k metric dictionaries in RAM."""
    count = actor_updates = zeros = 0
    longest = streak = 0
    max_boundary = max_outside = 0.
    first = last = None
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            if not all(isinstance(v, (float, int)) and math.isfinite(v) for v in row.values()):
                raise RuntimeError(f"Nonfinite/malformed metrics: {path}")
            first = first or row
            last = row
            count += 1
            actor_updates += row.get("actor_updated", 0) == 1
            zero = row["critic_grad_norm"] == 0
            zeros += zero
            streak = streak + 1 if zero and row["critic_loss"] > 1e-6 else 0
            longest = max(longest, streak)
            max_boundary = max(max_boundary, *(v for k, v in row.items() if k.endswith("_exact_boundary_fraction")))
            max_outside = max(max_outside, row["q_target_outside_unit_fraction"])
    if not count or not actor_updates or longest >= 3:
        raise RuntimeError(f"Learning gate failed: {path}")
    return dict(updates=count, actor_updates=actor_updates, zero_critic_gradient_updates=zeros,
                longest_zero_gradient_with_error=longest, max_exact_boundary_fraction=max_boundary,
                max_target_outside_unit_fraction=max_outside, first=first, last=last)


def episode_summary(run):
    reports = [read_json(path) for path in sorted((run / "episodes").glob("*/report.json"))]
    if any(row is None or row["status"] == "infrastructure_error" for row in reports):
        raise RuntimeError(f"Incomplete/infrastructure episode: {run}")
    timings = {}
    for row in reports:
        for key, value in row["timings"].items():
            timings[key] = timings.get(key, 0.) + value
    return dict(episodes=len(reports), actual_transitions=sum(r["transitions"] for r in reports),
                control_steps=sum(r["control_steps"] for r in reports),
                physics_steps=sum(r["physics_steps"] for r in reports),
                failures=dict(Counter(r["status"] + ":" + r["reason"] for r in reports)),
                unexpected_contacts=sum(r["unexpected_contacts"] for r in reports), timings=timings,
                max_residual_saturation_fraction=max((r["residual_saturation_fraction"] for r in reports), default=0.))


def paired_results(output):
    pairs = []
    for seed in SEEDS:
        groups = {mode: read_json(output / "formal" / f"{mode}_seed{seed}" / "eval/summary.json") for mode in MODES}
        a, b = ({r["candidate_id"]: r["status"] == "success" for r in groups[mode]["episodes"]} for mode in MODES)
        if len(a) != 40 or a.keys() != b.keys():
            raise RuntimeError("Paired evaluation scenes differ")
        improved = sum(not a[key] and b[key] for key in a)
        regressed = sum(a[key] and not b[key] for key in a)
        n = improved + regressed
        p = min(1., 2 * sum(math.comb(n, k) for k in range(min(improved, regressed) + 1)) / 2**n)
        success_times = {mode: [r["wall_seconds"] for r in groups[mode]["episodes"] if r["status"] == "success"] for mode in MODES}
        pairs.append(dict(seed=seed, a_successes=sum(a.values()), b_successes=sum(b.values()),
                          both_success=sum(a[k] and b[k] for k in a), both_failed=sum(not a[k] and not b[k] for k in a),
                          improved=improved, regressed=regressed, rate_difference=(sum(b.values())-sum(a.values()))/40,
                          exact_mcnemar_p=p, success_wall_seconds=success_times,
                          per_scene={str(k): dict(A=a[k], B=b[k]) for k in sorted(a)}))
    differences = [r["rate_difference"] for r in pairs]
    result = dict(seeds=pairs, mean_rate_difference=sum(differences)/3,
                  rate_difference_range=[min(differences), max(differences)],
                  uncertainty="Three training seeds; repeated scenes are not 120 independent scenes. No pooled independence test.",
                  algorithm_conclusion="Review all seeds, contacts and paired uncertainty; no automated efficacy claim.")
    write_json(output / "paired_results.json", result)
    return result


class Campaign:
    def __init__(self, output, port):
        self.output = output
        self.port = port
        self.endpoint = f"http://127.0.0.1:{port}"
        self.server = self.child = None
        self.source = {name: sha(ROOT / name) for name in SOURCES}
        self.env = os.environ.copy()
        for key in ("PYTHONHOME", "PYTHONPATH"):
            self.env.pop(key, None)
        self.env.update(OMNI_KIT_ACCEPT_EULA="YES", PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1",
                        OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        self.job = dict(state="starting", supervisor_pid=os.getpid(), started_unix_s=time.time(),
                        output=str(output), endpoint=self.endpoint, completed=[], source_sha256=self.source,
                        protocol_version="astribot_latent_fan_in_q3e-5_exact_budget_v2",
                        configuration=dict(critic_init="fan_in", critic_lr=3e-5, actor_lr=1e-5,
                            formal_seeds=SEEDS, trial_seed=SEEDS[0], warmup=5000, batch=256,
                            trial_budget=10000, formal_budget=100000, initialization="fresh_each_run",
                            reward_and_q_output="unchanged", bootstrap_truncated=False),
                        gate=dict(sustained_stall_batches=3, nonzero_loss_threshold=1e-6,
                                  failures="stop and preserve; never auto-start later stage after failure"))
        write_json(output / "protocol.json", self.job)
        snapshot = output / "source"
        snapshot.mkdir()
        for name in SOURCES:
            target = snapshot / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / name).read_bytes())
        self.publish()

    def publish(self, run=None):
        self.job["updated_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
        if run is not None:
            status = read_json(run / "status.json", {})
            self.job["active_status"] = {k: status.get(k) for k in (
                "transitions", "updates", "episodes", "rolling_success_rate", "critic_init", "q_lr")}
        write_json(self.output / "progress.json", self.job)

    def check_source(self):
        if any(sha(ROOT / name) != checksum for name, checksum in self.source.items()):
            raise RuntimeError("Source changed during campaign; stop to avoid mixed experiment versions")

    def resources(self):
        try:
            gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu,power.draw", "--format=csv,noheader,nounits"], text=True, timeout=10).strip()
            available = next(row.split()[1] for row in Path("/proc/meminfo").read_text().splitlines() if row.startswith("MemAvailable:"))
            with (self.output / "resources.jsonl").open("a") as stream:
                stream.write(json.dumps(dict(unix_s=time.time(), phase=self.job["state"], gpu=gpu,
                                             available_memory_kib=int(available))) + "\n")
        except (OSError, subprocess.SubprocessError):
            pass

    def command(self, mode, seed, budget, run):
        return [sys.executable, str(ROOT / "scripts/run_astribot_online_rl.py"),
                "--model-run", str(ROOT / "models/astribot_head_right_joint8_ft_20261018"),
                "--checkpoint-step", "8000", "--output", str(run), "--observation-mode", mode,
                "--endpoint", self.endpoint, "--gpu", "0", "--physics-backend", "gpu",
                "--max-transitions", str(budget), "--warmup-transitions", "5000", "--batch-size", "256",
                "--training-frequency", "1", "--checkpoint-every", "2000", "--max-seconds", "40",
                "--policy-lr", "1e-5", "--critic-init", "fan_in", "--q-lr", "3e-5",
                "--init-alpha", ".005", "--no-auto-alpha", "--action-penalty", "1", "--shaping", "1",
                "--scale", *([".01"] * 7), ".025", "--seed", str(seed)]

    def worker(self, command, label, run=None):
        self.check_source()
        self.job.update(state=label, active_run=str(run) if run else None, active_status=None)
        self.publish(run)
        write_json(self.output / (label.replace("/", "_") + "_command.json"), command)
        log_path = self.output / (label.replace("/", "_") + ".log")
        started = time.perf_counter()
        with log_path.open("w") as log:
            self.child = subprocess.Popen(command, cwd=ROOT, env=self.env, stdout=log, stderr=subprocess.STDOUT)
            self.job["worker_pid"] = self.child.pid
            while self.child.poll() is None:
                if self.server and self.server.poll() is not None:
                    raise RuntimeError("Frozen policy server exited")
                self.resources()
                self.publish(run)
                try:
                    self.child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
            if self.child.returncode:
                raise RuntimeError(f"{label} exited {self.child.returncode}; inspect {log_path}")
        if run is not None:
            for marker in ("infrastructure_error.json", "learning_error.json"):
                if (run / marker).exists():
                    raise RuntimeError(f"{label} failed: {run / marker}")
        self.job["completed"].append(dict(phase=label, wall_seconds=time.perf_counter()-started))
        self.child = None
        self.publish(run)

    def gate(self, run, budget):
        status = read_json(run / "status.json")
        if status["transitions"] != budget or status["updates"] != budget:
            raise RuntimeError(f"Budget/update gate failed: {run}")
        if status["critic_init"] != "fan_in" or status["q_lr"] != 3e-5:
            raise RuntimeError("A/B critic configuration mismatch")
        episodes = episode_summary(run)
        if episodes["actual_transitions"] != budget:
            raise RuntimeError("Actual transitions differ from the replay/budget")
        summary = metrics_summary(run / "metrics.jsonl")
        if summary["updates"] != budget:
            raise RuntimeError("Metric update count differs from budget")
        write_json(run / "training_gate.json", dict(passed=True, metrics=summary, episodes=episodes,
            saved_bytes={name: (run / name).stat().st_size for name in ("learner.pt", "replay.npz")},
            caveat="Finite/stall/resource engineering gate; success rate is not an efficacy gate"))
        return summary

    def start_server(self):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", self.port))
        with (self.output / "policy_server.log").open("w") as log:
            self.server = subprocess.Popen([str(ROOT / ".venv-astribot-inference/bin/python"),
                str(ROOT / "scripts/astribot_policy_server.py"), "--run", str(ROOT / "models/astribot_head_right_joint8_ft_20261018"),
                "--device", "cuda:0", "--port", str(self.port)], cwd=ROOT, env=self.env, stdout=log, stderr=subprocess.STDOUT)
        self.job["policy_pid"] = self.server.pid
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            if self.server.poll() is not None:
                raise RuntimeError("Policy startup failed; inspect policy_server.log")
            try:
                with urllib.request.urlopen(self.endpoint + "/health", timeout=2) as response:
                    health = json.load(response)
                if health.get("ready"):
                    write_json(self.output / "policy_health.json", health)
                    return
            except OSError:
                pass
            self.publish()
            time.sleep(2)
        raise TimeoutError("Policy server startup timed out")

    def run(self):
        self.start_server()
        smoke = self.output / "smoke"
        self.worker([sys.executable, str(ROOT / "scripts/validate_astribot_latent_engineering.py"),
                     "--output", str(smoke), "--endpoint", self.endpoint], "smoke")
        smoke_result = read_json(smoke / "smoke_results.json")
        if not smoke_result or not smoke_result.get("smoke_learning_gate_passed"):
            raise RuntimeError("Repaired online smoke gate failed")
        for mode in MODES:
            metrics_summary(smoke / mode / "metrics.jsonl")
        for mode in MODES:
            run = self.output / "trials" / f"{mode}_seed{SEEDS[0]}"
            command = self.command(mode, SEEDS[0], 10000, run)
            self.worker(command + ["--render-cameras", "head", "wrist_right"], f"trial_{mode}_train", run)
            self.gate(run, 10000)
            # Exercise the actual training restore path including replay and
            # optimizer, with no new interaction after the exact budget.
            self.worker(command + ["--resume", "--render-cameras", "head", "wrist_right"], f"trial_{mode}_restore", run)
            self.gate(run, 10000)
        write_json(self.output / "trial_gate.json", dict(passed=True, both_modes=True,
                   budget_each=10000, critic_init="fan_in", q_lr=3e-5,
                   formal_initialization="fresh networks, targets, optimizers, replay; no --resume"))
        for seed in SEEDS:
            for mode in MODES:
                run = self.output / "formal" / f"{mode}_seed{seed}"
                command = self.command(mode, seed, 100000, run)
                self.worker(command + ["--render-cameras", "head", "wrist_right"], f"formal_{seed}_{mode}_train", run)
                self.gate(run, 100000)
                before = {name: sha(run / name) for name in ("learner.pt", "replay.npz", "status.json", "protocol.json")}
                self.worker(command + ["--resume", "--eval-only", "--eval-episodes", "40", "--eval-split", "val",
                            "--eval-max-seconds", "40", "--render-cameras", *CAMERAS], f"formal_{seed}_{mode}_val40", run)
                summary = read_json(run / "eval/summary.json")
                if not summary or summary["completed"] != 40 or not summary["parameters_unchanged"]:
                    raise RuntimeError("Incomplete/mutating formal evaluation")
                contract = read_json(ROOT / "models/astribot_head_right_joint8_ft_20261018/data_contract.json")
                if {r["candidate_id"] for r in summary["episodes"]} != set(contract["split"]["val_source_episode_ids"]):
                    raise RuntimeError("Evaluation IDs differ from frozen val-40")
                if before != {name: sha(run / name) for name in before}:
                    raise RuntimeError("Evaluation changed training artifacts")
        paired_results(self.output)
        self.job["state"] = "complete"

    def close(self):
        for process in (self.child, self.server):
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
        self.publish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8798)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    campaign = Campaign(output, args.port)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Supervisor received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        campaign.run()
    except BaseException:
        campaign.job.update(state="stopped_with_error", error=traceback.format_exc())
        write_json(output / "campaign_error.json", campaign.job)
        raise
    finally:
        if campaign.job["state"] == "complete":
            campaign.job["finished_unix_s"] = time.time()
        campaign.close()


if __name__ == "__main__":
    main()
