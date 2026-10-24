"""Adopt live B training, then B val-40, resume formal A, then A val-40."""
from __future__ import annotations
import argparse
from datetime import datetime
from pathlib import Path
import signal
import time
import traceback
from zoneinfo import ZoneInfo

from run_astribot_latent_campaign import (Campaign, ExistingProcess, ROOT, CAMERAS,
                                         paired_results, read_json, sha, write_json)


def verify_val40(run, expected_ids, before):
    summary = read_json(run / "eval/summary.json", {})
    rows = summary.get("episodes", [])
    if (summary.get("completed") != 40 or len(rows) != 40
            or {row["candidate_id"] for row in rows} != set(expected_ids)
            or any(row["status"] not in ("success", "failed", "timeout") for row in rows)
            or not summary.get("deterministic")
            or not summary.get("parameters_unchanged") or not summary.get("updates_unchanged")):
        raise RuntimeError("Val-40 incomplete, invalid, or mutated learner")
    if before != {name: sha(run / name) for name in before}:
        raise RuntimeError("Val-40 modified training artifacts")
    return summary


class SequentialJob(Campaign):
    def __init__(self, output, port, campaign_root, prior):
        self.campaign_root = campaign_root
        super().__init__(output, port)
        self.adopted = prior
        # Preserve every previous source identity plus this orchestrator.
        for name, checksum in prior["source_sha256"].items():
            self.source[name] = checksum
        name = "scripts/run_astribot_b_then_a_job.py"
        self.source[name] = sha(ROOT / name)
        for name in self.source:
            target = output / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / name).read_bytes())
        self.job.update(requested_scope="B 100k -> B val-40 -> A 100k -> A val-40",
                        automatic_A=True, automatic_evaluation=True,
                        sequence=["B_100k", "B_val40", "A_100k", "A_val40"],
                        A_initialization="resume prior fresh formal checkpoint; never inherit 10k state",
                        adopted_worker_pid=prior["worker_pid"], policy_pid=prior["policy_pid"],
                        replaced_supervisor_pid=prior["supervisor_pid"],
                        continued_B_job_started_unix_s=prior["started_unix_s"])
        write_json(output / "protocol.json", self.job)
        self.publish()

    def publish(self, run=None):
        super().publish(run)
        write_json(self.campaign_root / "progress.json", dict(self.job, job_output=str(self.output)))

    def evaluate_one(self, mode):
        run = self.campaign_root / "formal" / f"{mode}_seed20261024"
        if (run / "eval/summary.json").exists():
            raise FileExistsError("Never overwrite an existing formal evaluation")
        before = {name: sha(run / name) for name in ("learner.pt", "replay.npz", "status.json", "protocol.json")}
        command = self.command(mode, 20261024, 100000, run)
        self.worker(command + ["--resume", "--eval-only", "--eval-episodes", "40", "--eval-split", "val",
                              "--eval-max-seconds", "40", "--render-cameras", *CAMERAS],
                    f"formal_20261024_{mode}_val40", run)
        contract = read_json(ROOT / "models/astribot_head_right_joint8_ft_20261018/data_contract.json")
        summary = verify_val40(run, contract["split"]["val_source_episode_ids"], before)
        write_json(run / "evaluation_gate.json", dict(passed=True, completed=40,
                   successes=summary["successes"], training_files_unchanged=True))

    def run(self):
        self.server = ExistingProcess(self.adopted["policy_pid"])
        b = self.campaign_root / "formal/state_latent_seed20261024"
        self.worker(self.command("state_latent", 20261024, 100000, b) + ["--render-cameras", "head", "wrist_right"],
                    "formal_20261024_state_latent_train", b, adopt_existing=True)
        self.gate(b, 100000)
        self.evaluate_one("state_latent")
        a = self.campaign_root / "formal/state_only_seed20261024"
        # The 488-transition formal checkpoint was freshly initialized for this
        # exact protocol; restoring it preserves prior work and RNG/optimizer.
        self.worker(self.command("state_only", 20261024, 100000, a)
                    + ["--resume", "--render-cameras", "head", "wrist_right"],
                    "formal_20261024_state_only_train", a)
        self.gate(a, 100000)
        self.evaluate_one("state_only")
        paired_results(self.campaign_root)
        self.job.update(state="complete", result="B and A each completed 100k plus full val-40",
                        paired_results=str(self.campaign_root / "paired_results.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8798)
    args = parser.parse_args()
    root, output = args.campaign_root.resolve(), args.output.resolve()
    prior = read_json(root / "progress.json")
    if prior.get("state") != "formal_20261024_state_latent_train":
        raise RuntimeError("Adoption requires live B formal training")
    for key, identifier in (("worker_pid", b"run_astribot_online_rl.py"), ("policy_pid", b"astribot_policy_server.py")):
        cmdline = Path(f"/proc/{prior[key]}/cmdline").read_bytes()
        if identifier not in cmdline:
            raise RuntimeError("Adopted process identity mismatch")
    worker_command = Path(f"/proc/{prior['worker_pid']}/cmdline").read_bytes()
    if str(root / "formal/state_latent_seed20261024").encode() not in worker_command:
        raise RuntimeError("B output identity mismatch")
    for name, checksum in prior["source_sha256"].items():
        if sha(ROOT / name) != checksum:
            raise RuntimeError(f"Source changed during live B training: {name}")
    a = root / "formal/state_only_seed20261024"
    for filename in ("learner.pt", "replay.npz", "status.json"):
        if not (a / filename).exists():
            raise RuntimeError("Prior formal A checkpoint is incomplete")
    status = read_json(a / "status.json")
    contract = status["restore_contract"]
    if (status["replay_capacity"] != 100000 or contract["observation"]["mode"] != "state_only"
            or contract["critic_init"] != "fan_in" or contract["q_lr"] != 3e-5
            or contract["seed"] != 20261024):
        raise RuntimeError("A formal checkpoint configuration differs")
    if any(sha(ROOT / name) != checksum for name, checksum in contract["source_sha256"].items()):
        raise RuntimeError("A formal checkpoint training source differs")
    for run in (a, root / "formal/state_latent_seed20261024"):
        if any((run / name).exists() for name in ("learning_error.json", "infrastructure_error.json", "eval/summary.json")):
            raise RuntimeError("Existing error/evaluation evidence requires review before continuing")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "prior_B_progress.json", prior)
    write_json(output / "authorization.json", dict(
        requested_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        instruction="把B训练和val40都做完，完成后再训练A和val40", preserved_B_pid=prior["worker_pid"],
        A_saved_transitions=status["transitions"], A_saved_updates=status["updates"]))
    job = SequentialJob(output, args.port, root, prior)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Sequential supervisor received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        job.run()
    except BaseException:
        job.job.update(state="stopped_with_error", error=traceback.format_exc())
        write_json(output / "campaign_error.json", job.job)
        raise
    finally:
        if job.job["state"] == "complete":
            job.job["finished_unix_s"] = time.time()
        job.close()


if __name__ == "__main__":
    main()
