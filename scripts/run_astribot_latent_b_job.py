"""Run only the authorized fresh B 100k job; do not restart A or auto-evaluate."""
from __future__ import annotations
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import signal
import time
import traceback

from run_astribot_latent_campaign import Campaign, ROOT, read_json, sha, write_json


class BJob(Campaign):
    def __init__(self, output, port, campaign_root):
        self.campaign_root = campaign_root
        super().__init__(output, port)
        name = "scripts/run_astribot_latent_b_job.py"
        self.source[name] = sha(ROOT / name)
        (output / "source" / name).write_bytes((ROOT / name).read_bytes())
        self.job.update(requested_scope="B-only fresh 100k training", automatic_A=False,
                        automatic_evaluation=False, formal_run=str(campaign_root / "formal/state_latent_seed20261024"))
        write_json(output / "protocol.json", self.job)
        self.publish()

    def publish(self, run=None):
        super().publish(run)
        write_json(self.campaign_root / "progress.json", dict(self.job, job_output=str(self.output)))

    def run(self):
        self.start_server()
        run = self.campaign_root / "formal/state_latent_seed20261024"
        command = self.command("state_latent", 20261024, 100000, run)
        if "--resume" in command:
            raise RuntimeError("B formal training must be freshly initialized")
        self.worker(command + ["--render-cameras", "head", "wrist_right"],
                    "formal_20261024_state_latent_train", run)
        self.gate(run, 100000)
        self.job.update(state="complete", result="B 100k training complete; A remains stopped; evaluation not started")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8798)
    args = parser.parse_args()
    campaign_root = args.campaign_root.resolve()
    output = args.output.resolve()
    trial = campaign_root / "trials/state_latent_seed20261024"
    if not read_json(trial / "training_gate.json", {}).get("passed"):
        raise RuntimeError("B 10k stability gate has not passed")
    if not read_json(trial / "restore_metrics.json", {}).get("replay_loaded"):
        raise RuntimeError("B training restore has not passed")
    if (campaign_root / "formal/state_latent_seed20261024").exists():
        raise FileExistsError("B formal output already exists; never overwrite or silently resume")
    contract = read_json(trial / "status.json")["restore_contract"]
    if any(sha(ROOT / name) != checksum for name, checksum in contract["source_sha256"].items()):
        raise RuntimeError("Training source differs from validated B trial")
    prior = read_json(campaign_root / "progress.json")
    if prior.get("state") not in ("stopped_by_user", "complete"):
        raise RuntimeError("Campaign is already active; do not launch competing GPU workers")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "prior_campaign_progress.json", prior)
    write_json(output / "authorization.json", dict(
        requested_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        instruction="先启动B组正式100k", fresh_initialization=True, automatic_A=False, automatic_evaluation=False))
    campaign = BJob(output, args.port, campaign_root)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"B supervisor received signal {signum}")
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
