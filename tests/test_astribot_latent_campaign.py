"""Prevent expensive later stages from starting on invalid learning evidence."""
import json
from pathlib import Path

import pytest
import run_astribot_latent_campaign as campaign


def write_metrics(path, gradients):
    path.write_text("\n".join(json.dumps(dict(updates=i+1, actor_updated=1.,
        critic_grad_norm=gradient, critic_loss=.1, q_target_outside_unit_fraction=0.,
        q0_exact_boundary_fraction=0.)) for i, gradient in enumerate(gradients)))


def test_gate_rejects_stuck_critic_and_nonfinite_metrics(tmp_path):
    path = tmp_path / "metrics.jsonl"
    write_metrics(path, [1., 0., 0., 0.])
    with pytest.raises(RuntimeError, match="Learning gate"):
        campaign.metrics_summary(path)
    write_metrics(path, [1., float("nan")])
    with pytest.raises(RuntimeError, match="Nonfinite"):
        campaign.metrics_summary(path)
    write_metrics(path, [1., 0., 1., .1])
    assert campaign.metrics_summary(path)["longest_zero_gradient_with_error"] == 1


def test_pairing_counts_scenes_and_reports_each_seed(tmp_path):
    seeds = (20261024, 20261025, 20261026)
    for seed in seeds:
        for mode in campaign.MODES:
            path = tmp_path / "formal" / f"{mode}_seed{seed}" / "eval"
            path.mkdir(parents=True)
            successes = set(range(20)) if mode == "state_only" else set(range(1, 23))
            rows = [dict(candidate_id=i, status="success" if i in successes else "timeout", wall_seconds=1.) for i in range(40)]
            (path / "summary.json").write_text(json.dumps(dict(episodes=rows)))
    result = campaign.paired_results(tmp_path, seeds=seeds)
    assert len(result["seeds"]) == 3
    assert result["mean_rate_difference"] == pytest.approx(.05)
    for row in result["seeds"]:
        assert (row["improved"], row["regressed"]) == (3, 1)
        assert row["exact_mcnemar_p"] == .625


def test_engineering_default_runs_only_one_seed():
    assert campaign.SEEDS == (20261024,)
