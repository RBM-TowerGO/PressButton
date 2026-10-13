#!/usr/bin/env python3
"""Independently check S1 button presses from recorded physical arrays."""
import argparse
import json
from pathlib import Path

import numpy as np


def audit(directory):
    directory = Path(directory)
    report = json.loads((directory/"report.json").read_text())
    with np.load(directory/"trajectory.npz", allow_pickle=False) as archive:
        trace = {key:archive[key] for key in archive.files}
    config = report["config"]
    button_ids = report.get("button_ids", list(range(24,36)))
    button_index = {f:i for i,f in enumerate(button_ids)}
    dt = 1/config["physics_hz"]
    checks = {
        "completed": report["status"] == "passed",
        "finite": all(np.isfinite(trace[k]).all() for k in ["q", "target", "tip", "travel", "force", "base", "base_quat", "time"]),
        "physical_timing": bool(np.allclose(np.diff(trace["time"]),dt,atol=1e-10)),
        "no_unexpected_contacts": not report["unexpected_contacts"],
        "episode_count": len(report["episodes"]) == len(report["sequence"])*report["repeats"],
        "closed_gap": -.0001 <= report["measured_closed_gap_m"] <= .001,
        "tcp_fk_agreement": report["max_fk_error_m"] < .004,
    }
    base_drift = np.max(np.linalg.norm(trace["base"]-trace["base"][0],axis=1))
    quat_similarity = np.min(np.abs(trace["base_quat"]@trace["base_quat"][0]))
    checks["fixed_base_translation"] = bool(base_drift<.00001)
    checks["fixed_base_rotation"] = bool(quat_similarity>1.-1e-6)
    names = report["joint_names"]
    inactive = [i for i,n in enumerate(names) if "arm_right" not in n and "gripper_right" not in n]
    right = [i for i,n in enumerate(names) if "arm_right" in n]
    grip = [i for i,n in enumerate(names) if "gripper_right" in n]
    checks["stationary_torso_head_left_arm"] = bool(np.max(np.abs(trace["q"][:,inactive]))<.02)
    active = np.isin(trace["phase"],["approach","press","retract","return_home"])
    signs = {"L1": -1, "L2": -1, "L11": 1, "R1": 1, "R2": 1, "R11": -1}
    closed = np.array([signs[names[i].rsplit("_",1)[-1]]*report["hand"]["closed_angle_rad"] for i in grip])
    checks["gripper_held_closed"] = bool(active.any() and
        np.max(np.abs(trace["q"][active][:,grip]-closed))<.02 and
        np.max(np.abs(trace["target"][active][:,grip]-closed))<1e-5)
    rows = []
    for ep in report["episodes"]:
        start, stop = ep["start_step"], ep["end_step"]
        index = button_index[ep["floor"]]
        travel, force = trace["travel"][start:stop], trace["force"][start:stop]
        pressed = (travel[:,index]>=config["press_threshold"]) & (force[:,index]>config["contact_threshold_n"])
        others = np.delete(travel,index,axis=1)
        wrong = bool(others.size and others.max()>=config["press_threshold"])
        released = travel[-1,index]<config["release_threshold"]
        home_error = np.max(np.abs(trace["q"][stop-1,right]-np.array(report["home_arm_rad"])))
        passed = bool(pressed.any() and released and not wrong and home_error<.025)
        if config.get("button_light_mode") == "latched":
            lights = trace["light"][start:stop,index]
            hits = np.flatnonzero(pressed)
            latch_ok = bool(len(hits) and not lights[:hits[0]].any() and lights[hits[0]:].all())
            checks[f"latched_request_{start}"] = latch_ok
            passed = passed and latch_ok
        rows.append(dict(floor=ep["floor"],repeat=ep["repeat"],passed=passed,
                         max_travel_m=float(travel[:,index].max()),max_force_n=float(force[:,index].max()),
                         returned_home_error_rad=float(home_error)))
    checks["all_measured_episodes_pass"] = bool(rows and all(row["passed"] for row in rows))
    expected_order = [(f, r) for r in range(report["repeats"]) for f in report["sequence"]]
    checks["requested_sequence"] = [(row["floor"],row["repeat"]) for row in rows] == expected_order
    if "reset" in report:
        reset_rows = trace["epoch"]==1
        checks["reset_recovered"] = bool(reset_rows.any() and
            np.max(np.abs(trace["q"][reset_rows][-1]-trace["target"][reset_rows][-1]))<.025 and
            np.max(trace["travel"][reset_rows][-1])<config["release_threshold"])
    if config.get("scene_type") == "vib820":
        checks["single_down_button"] = button_ids == [0] and trace["travel"].shape[1] == 1
        checks["travel_limit"] = bool(trace["travel"].max() <= config["button_travel"] + .0002)
        reset_rows = trace["phase"] == "reset_check"
        checks["request_cleared_on_reset"] = bool(reset_rows.any() and not trace["light"][reset_rows].any())
    if "simulation_time" in trace:
        checks["simulator_clock"] = all(np.allclose(np.diff(trace["simulation_time"][trace["epoch"]==e]),dt,atol=1e-7)
                                        for e in np.unique(trace["epoch"]))
    result = dict(status="passed" if all(checks.values()) else "failed",checks=checks,
                  episodes=rows,base_drift_m=float(base_drift),physical_samples=len(trace["time"]))
    (directory/"independent_audit.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    args=parser.parse_args()
    result=audit(args.directory)
    print(json.dumps({k:v for k,v in result.items() if k!="episodes"},indent=2))
    raise SystemExit(0 if result["status"]=="passed" else 1)
