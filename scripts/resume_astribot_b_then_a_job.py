"""Reviewed one-time infrastructure recovery preserving the original learner contract."""
import argparse
import os
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import shutil
import signal
import time
import traceback
from collections import Counter

from run_astribot_b_then_a_job import SequentialJob
from run_astribot_latent_campaign import (ROOT, Campaign, metrics_summary, paired_results,
                                         read_json, sha, write_json)


def acknowledged_episodes(run, acknowledged):
    rows = []
    for path in sorted((run / "episodes").glob("*/report.json")):
        row = read_json(path)
        if row is None:
            raise RuntimeError("Unreadable episode report")
        if row["status"] == "infrastructure_error":
            identity = str(path.relative_to(run))
            if (acknowledged.get(identity, {}).get("sha256") != sha(path)
                    or acknowledged[identity]["transitions"] != row["transitions"]):
                raise RuntimeError("Unreviewed infrastructure interruption")
        rows.append(row)
    timings = {}
    for row in rows:
        for key, value in row["timings"].items():
            timings[key] = timings.get(key, 0.) + value
    return dict(episodes=len(rows), actual_transitions=sum(r["transitions"] for r in rows),
                control_steps=sum(r["control_steps"] for r in rows),
                physics_steps=sum(r["physics_steps"] for r in rows),
                failures=dict(Counter(r["status"]+":"+r["reason"] for r in rows)),
                infrastructure_interrupted_episodes=sum(r["status"] == "infrastructure_error" for r in rows),
                unexpected_contacts=sum(r["unexpected_contacts"] for r in rows), timings=timings)


class RecoveryJob(SequentialJob):
    def __init__(self, output, port, root, prior, acknowledged):
        super().__init__(output, port, root, prior)
        self.acknowledged = acknowledged
        name = "scripts/resume_astribot_b_then_a_job.py"
        self.source[name] = sha(ROOT / name)
        (output / "source" / name).write_bytes((ROOT / name).read_bytes())
        self.job.update(recovery="Reviewed infrastructure interruption; unchanged training config",
                        saved_B_transitions=27133, saved_B_updates=27050,
                        acknowledged_interruptions=acknowledged, adopted_worker_pid=None)
        write_json(output / "protocol.json", self.job)
        self.publish()

    def gate(self, run, budget):
        if run.name != "state_latent_seed20261024":
            return Campaign.gate(self, run, budget)
        status = read_json(run / "status.json")
        if (status["transitions"] != budget or status["updates"] != budget
                or status["critic_init"] != "fan_in" or status["q_lr"] != 3e-5):
            raise RuntimeError("Recovered training budget/config mismatch")
        episodes = acknowledged_episodes(run, self.acknowledged)
        metrics = metrics_summary(run / "metrics.jsonl")
        if episodes["actual_transitions"] != budget or metrics["updates"] != budget:
            raise RuntimeError("Recovered data/update accounting differs from budget")
        write_json(run / "training_gate.json", dict(passed=True, metrics=metrics, episodes=episodes,
                   acknowledged_interruptions=self.acknowledged,
                   caveat="Only hash-identified reviewed infrastructure reports accepted; valid saved rows retained, incomplete pending row discarded"))

    def close(self):
        if self.server is not None:
            self.job["observed_policy_exit_code_before_cleanup"] = self.server.poll()
        if self.child is not None and self.child.poll() is None:
            # Give the runner a chance to finish its checkpoint before cleanup.
            os.kill(self.child.pid, signal.SIGINT)
            for _ in range(2):
                try:
                    self.child.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    pass
        super().close()

    def run(self):
        self.start_server()
        b = self.campaign_root / "formal/state_latent_seed20261024"
        self.worker(self.command("state_latent", 20261024, 100000, b)
                    + ["--resume", "--render-cameras", "head", "wrist_right"],
                    "formal_20261024_state_latent_train", b)
        self.gate(b, 100000)
        self.evaluate_one("state_latent")
        a = self.campaign_root / "formal/state_only_seed20261024"
        self.worker(self.command("state_only", 20261024, 100000, a)
                    + ["--resume", "--render-cameras", "head", "wrist_right"],
                    "formal_20261024_state_only_train", a)
        self.gate(a, 100000)
        self.evaluate_one("state_only")
        paired_results(self.campaign_root)
        self.job.update(state="complete", result="B and A completed 100k and val-40 after reviewed infrastructure recovery")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8798)
    args = parser.parse_args()
    root, output = args.campaign_root.resolve(), args.output.resolve()
    prior = read_json(root / "progress.json")
    if prior["state"] != "stopped_with_error":
        raise RuntimeError("Recovery requires a stopped failed campaign")
    if any(sha(ROOT / name) != h for name, h in prior["source_sha256"].items()):
        raise RuntimeError("Cannot recover under changed source identities")
    b = root / "formal/state_latent_seed20261024"
    if (b / "learning_error.json").exists():
        raise RuntimeError("Learning failure cannot be treated as infrastructure recovery")
    error = read_json(b / "infrastructure_error.json", {}).get("error", "")
    if "Connection reset by peer" not in error:
        raise RuntimeError("This recovery is only for the reviewed connection-reset interruption")
    status = read_json(b / "status.json")
    if (status["transitions"], status["updates"]) != (27133, 27050):
        raise RuntimeError("Saved checkpoint changed since review")
    acknowledged = {}
    for path in (b / "episodes").glob("*/report.json"):
        row = read_json(path)
        if row["status"] == "infrastructure_error":
            acknowledged[str(path.relative_to(b))] = dict(sha256=sha(path), transitions=row["transitions"],
                status="infrastructure_interruption, not physical failure", pending_incomplete_transition="not saved")
    if len(acknowledged) != 1 or sum(v["transitions"] for v in acknowledged.values()) != 83:
        raise RuntimeError("Interruption/data accounting differs from the reviewed report")
    assert acknowledged_episodes(b, acknowledged)["actual_transitions"] == status["transitions"]
    output.mkdir(parents=True, exist_ok=False)
    archive = b / "recovery" / datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S")
    archive.mkdir(parents=True)
    for name in ("learner.pt", "replay.npz", "status.json", "protocol.json", "infrastructure_error.json"):
        shutil.copy2(b / name, archive / name)
    write_json(archive / "manifest.json", {name: sha(archive / name) for name in (
        "learner.pt", "replay.npz", "status.json", "protocol.json", "infrastructure_error.json")})
    write_json(output / "prior_failed_progress.json", prior)
    write_json(output / "recovery_review.json", dict(
        reviewed_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        cause="Policy connection reset; server ended without Python traceback; exact exit cause unconfirmed",
        memory_evidence="Pre-failure sampled MemAvailable >=37.7 GiB; no kernel event available",
        checkpoint_verified=True, unchanged_training_contract=True, preserved_transitions=27133,
        preserved_updates=27050, archive=str(archive), acknowledged_reports=acknowledged,
        retry_policy="One reviewed recovery; another error stops without automatic retries"))
    # Original error evidence and its checksum remain in the immutable archive.
    (b / "infrastructure_error.json").unlink()
    job = RecoveryJob(output, args.port, root, prior, acknowledged)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Recovery supervisor received signal {signum}")
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
