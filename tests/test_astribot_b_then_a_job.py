"""Ensure authorized order and complete, immutable held-out evaluation gates."""
import json
from pathlib import Path
import pytest
import run_astribot_b_then_a_job as module


def make_job(tmp_path, monkeypatch):
    events = []
    job = module.SequentialJob.__new__(module.SequentialJob)
    job.campaign_root = tmp_path
    job.endpoint = "http://127.0.0.1:8798"
    job.adopted = {"policy_pid": 1}
    job.job = {}
    monkeypatch.setattr(module, "ExistingProcess", lambda pid: object())
    def worker(command, label, run, adopt_existing=False):
        events.append((label, "--resume" in command, adopt_existing))
    job.worker = worker
    job.gate = lambda run, budget: events.append(("gate", run.name, budget))
    job.evaluate_one = lambda mode: events.append(("eval", mode))
    monkeypatch.setattr(module, "paired_results", lambda root: events.append(("paired",)))
    return job, events


def test_live_B_then_val_then_resumed_A_then_val(tmp_path, monkeypatch):
    job, events = make_job(tmp_path, monkeypatch)
    job.run()
    assert events == [
        ("formal_20261024_state_latent_train", False, True),
        ("gate", "state_latent_seed20261024", 100000),
        ("eval", "state_latent"),
        ("formal_20261024_state_only_train", True, False),
        ("gate", "state_only_seed20261024", 100000),
        ("eval", "state_only"), ("paired",)]
    assert job.job["state"] == "complete"


def test_failed_B_evaluation_blocks_A(tmp_path, monkeypatch):
    job, events = make_job(tmp_path, monkeypatch)
    def fail(mode):
        raise RuntimeError("B incomplete")
    job.evaluate_one = fail
    with pytest.raises(RuntimeError, match="B incomplete"):
        job.run()
    assert len(events) == 2
    assert "state_only" not in str(events)


@pytest.mark.parametrize("failure", ["none", "missing_scene", "parameters", "updates", "file_mutation"])
def test_val40_gate(tmp_path, failure):
    (tmp_path / "eval").mkdir()
    (tmp_path / "learner.pt").write_bytes(b"preserved checkpoint")
    before = {"learner.pt": module.sha(tmp_path / "learner.pt")}
    summary = dict(completed=40, deterministic=True, parameters_unchanged=True, updates_unchanged=True,
                   episodes=[dict(candidate_id=i, status="timeout") for i in range(40)])
    if failure == "missing_scene":
        summary["episodes"][-1]["candidate_id"] = 0
    elif failure == "parameters":
        summary["parameters_unchanged"] = False
    elif failure == "updates":
        summary["updates_unchanged"] = False
    elif failure == "file_mutation":
        (tmp_path / "learner.pt").write_bytes(b"modified")
    (tmp_path / "eval/summary.json").write_text(json.dumps(summary))
    if failure == "none":
        assert module.verify_val40(tmp_path, range(40), before)["completed"] == 40
    else:
        with pytest.raises(RuntimeError):
            module.verify_val40(tmp_path, range(40), before)
