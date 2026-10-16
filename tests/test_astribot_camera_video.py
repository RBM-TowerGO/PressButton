import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "compose_astribot", Path(__file__).parents[1] / "scripts/compose_astribot_cameras.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("fault", [None, "missing_frame", "stale", "skew", "lag"])
def test_sync_audit_checks_each_stream_and_reset(tmp_path, monkeypatch, fault):
    names = ["overview", "head"]
    report = dict(camera_paths=dict.fromkeys(names, "unused"),
                  config=dict(capture_hz=30), video_frames=dict.fromkeys(names, 4), status="passed")
    timing = [dict(epoch=epoch, step=i * 4, simulation_time=t,
                   cameras=dict.fromkeys(names, t))
              for i, (epoch, t) in enumerate([(0, 1 / 30), (0, 2 / 30),
                                             (1, 1 / 30), (1, 2 / 30)])]
    if fault == "stale":
        timing[1]["cameras"]["head"] = timing[0]["cameras"]["head"]
    if fault == "skew":
        for row in timing:
            row["cameras"]["head"] += .01
    if fault == "lag":
        for row in timing:
            row["simulation_time"] += 1
    (tmp_path / "report.json").write_text(json.dumps(report))
    (tmp_path / "camera_timing.json").write_text(json.dumps(timing))
    monkeypatch.setattr(module, "probe", lambda p: dict(
        nb_read_frames="3" if fault == "missing_frame" and p.stem == "head" else "4",
        avg_frame_rate="30/1"))
    if fault:
        with pytest.raises(ValueError):
            module.validate(tmp_path)
    else:
        _, fps, audit = module.validate(tmp_path)
        assert fps == 30
        assert audit["max_camera_skew_s"] == 0
        assert audit["reset_epochs"] == [0, 1]
