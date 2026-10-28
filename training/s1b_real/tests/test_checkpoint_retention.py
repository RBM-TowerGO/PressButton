from starVLA.training.train_s1b_real import prune_checkpoints


def test_milestones_survive_best_and_last_rotation(tmp_path):
    for step in (500, 1000, 1500, 2000, 4500, 5000, 5500, 6000):
        (tmp_path / f"step_{step:06d}").mkdir()
    (tmp_path / "step_006500.tmp").mkdir()
    (tmp_path / "best").symlink_to("step_004500")
    (tmp_path / "last").symlink_to("step_005500")
    prune_checkpoints(tmp_path)
    assert {p.name for p in tmp_path.glob("step_*")} == {
        "step_001000", "step_002000", "step_004500", "step_005000",
        "step_005500", "step_006000", "step_006500.tmp",
    }
    (tmp_path / "last").unlink()
    (tmp_path / "last").symlink_to("step_006000")
    prune_checkpoints(tmp_path)
    assert not (tmp_path / "step_005500").exists()
    assert (tmp_path / "best").resolve().is_dir()
    assert (tmp_path / "step_005000").is_dir()
