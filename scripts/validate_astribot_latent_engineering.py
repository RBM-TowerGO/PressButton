"""Serial local engineering smoke. Requires a separately running policy server.

No 10k trials, formal training or full validation evaluation is launched.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8797")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env.update(OMNI_KIT_ACCEPT_EULA="YES", PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    results = {}
    for mode in ("state_only", "state_latent"):
        run = args.output / mode
        common = [sys.executable, str(ROOT / "scripts/run_astribot_online_rl.py"),
                  "--model-run", str(ROOT / "models/astribot_head_right_joint8_ft_20261018"),
                  "--checkpoint-step", "8000", "--output", str(run), "--observation-mode", mode,
                  "--endpoint", args.endpoint, "--gpu", "0", "--physics-backend", "gpu",
                  "--smoke", "--max-seconds", "2", "--policy-lr", "1e-5", "--init-alpha", ".005",
                  "--no-auto-alpha", "--action-penalty", "1", "--shaping", "1",
                  "--scale", *([".01"] * 7), ".025", "--seed", "20261024"]
        phases = [("train", ["--render-cameras", "head", "wrist_right"]),
                  ("eval", ["--resume", "--eval-only", "--eval-episodes", "1", "--eval-max-seconds", "2",
                            "--render-cameras", "head", "overview", "wrist_left", "wrist_right", "panel", "front", "lobby"])]
        before = None
        for phase, extra in phases:
            started = time.perf_counter()
            command = common + extra
            (args.output / f"{mode}_{phase}_command.json").write_text(json.dumps(command, indent=2))
            with (args.output / f"{mode}_{phase}.log").open("w") as log:
                child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
                while child.poll() is None:
                    try:
                        gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True, timeout=10).strip()
                        memory = Path("/proc/meminfo").read_text().splitlines()
                        with (args.output / "resources.jsonl").open("a") as stream:
                            stream.write(json.dumps(dict(time=time.time(), mode=mode, phase=phase, gpu=gpu,
                                mem_available_kib=int(next(row.split()[1] for row in memory if row.startswith("MemAvailable:"))))) + "\n")
                    except (OSError, subprocess.SubprocessError):
                        pass
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
            if child.returncode or (run / "infrastructure_error.json").exists():
                raise RuntimeError(f"{mode}/{phase} failed; inspect preserved log and infrastructure_error.json")
            results[f"{mode}_{phase}_seconds"] = time.perf_counter() - started
            if phase == "train":
                status = json.loads((run / "status.json").read_text())
                assert status["updates"] >= 1 and status["transitions"] >= 4
                metrics = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
                assert any(row.get("actor_updated") == 1 for row in metrics)
                # Replay restoration is CPU work, also used by GPU training.
                sys.path.insert(0, str(ROOT / "src"))
                from pressb.online_rl.astribot import load_replay
                import numpy as np
                restore_started = time.perf_counter()
                replay = load_replay(run / "replay.npz", capacity=8,
                                     obs_dim=64 if mode == "state_only" else 2112, action_dim=56, base_dim=8,
                                     expected_contract=status["restore_contract"])
                assert len(replay) == status["transitions"]
                with np.load(run / "replay.npz", allow_pickle=False) as saved:
                    for key, values in replay.state_dict()["arrays"].items():
                        np.testing.assert_array_equal(values, saved[key])
                results[f"{mode}_replay_restore_seconds"] = time.perf_counter() - restore_started
                results[f"{mode}_zero_critic_gradient_updates"] = [int(row["updates"]) for row in metrics
                                                                 if row["critic_grad_norm"] == 0]
                results[f"{mode}_learning_stability"] = ("requires_diagnosis_critic_saturation"
                    if results[f"{mode}_zero_critic_gradient_updates"] else "not_assessed_by_short_smoke")
                results[f"{mode}_updates"] = status["updates"]
                results[f"{mode}_transitions"] = status["transitions"]
                before = {name: digest(run / name) for name in ("learner.pt", "replay.npz", "status.json", "protocol.json")}
            else:
                assert before == {name: digest(run / name) for name in before}, "Evaluation modified training artifacts"
                summary = json.loads((run / "eval/summary.json").read_text())
                assert summary["completed"] == 1 and summary["parameters_unchanged"] and summary["updates_unchanged"]
                results[f"{mode}_eval_training_files_unchanged"] = True
                results[f"{mode}_evaluation"] = summary
            (args.output / "smoke_results.json").write_text(json.dumps(results, indent=2) + "\n")
            print(f"{mode}/{phase} passed in {time.perf_counter()-started:.1f}s", flush=True)
    results["engineering_data_path_passed"] = True
    results["learning_stability_passed"] = False
    results["scope"] = "engineering only; 2s train episode and one 2s val episode per mode"
    (args.output / "smoke_results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
