import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("audit_s1", Path(__file__).parents[1]/"scripts/audit_astribot_press.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def recording(tmp_path):
    names = [f"astribot_arm_right_joint_{i}" for i in range(1,8)] + ["astribot_arm_left_joint_1"]
    names += ["astribot_gripper_right_joint_"+s for s in ["L1","L2","L11","R1","R2","R11"]]
    q = np.zeros((5,len(names)))
    q[:,8:] = [-1,-1,1,1,1,-1]
    travel = np.zeros((5,12)); travel[1:3,8]=.0016
    force = np.zeros((5,12)); force[1:3,8]=.5
    trace = dict(q=q,target=q.copy(),tip=np.zeros((5,3)),travel=travel,force=force,
        base=np.zeros((5,3)),base_quat=np.tile([1.,0,0,0],(5,1)),time=np.arange(1,6)/120,
        phase=np.array(["approach","press","press","retract","return_home"]))
    report = dict(status="passed",config=dict(physics_hz=120,press_threshold=.0015,
        release_threshold=.0008,contact_threshold_n=.02),unexpected_contacts=[],sequence=[32],repeats=1,
        measured_closed_gap_m=.0003,max_fk_error_m=.0002,joint_names=names,home_arm_rad=[0.]*7,
        hand=dict(closed_angle_rad=1.),episodes=[dict(start_step=0,end_step=5,floor=32,repeat=0)])
    (tmp_path/"report.json").write_text(json.dumps(report))
    return tmp_path, trace


def test_audit_accepts_measured_press(recording):
    path,trace = recording
    np.savez(path/"trajectory.npz",**trace)
    assert module.audit(path)["status"] == "passed"


@pytest.mark.parametrize('fault',[None,'root','anchor','physics'])
def test_requested_base_yaw_requires_actual_rotation(recording,fault):
    path,trace=recording
    report=json.loads((path/'report.json').read_text())
    yaw=np.radians(10.)
    quat=[np.cos(yaw/2),0,0,np.sin(yaw/2)]
    report['config']['robot_base_yaw_rad']=yaw
    report['initial_poses']=dict(requested_yaw_rad=yaw,actual_root_yaw_rad=yaw,fixed_joint_world_quat_wxyz=quat)
    trace['base_quat'][:]=quat
    if fault=='root':report['initial_poses']['actual_root_yaw_rad']=0.
    if fault=='anchor':report['initial_poses']['fixed_joint_world_quat_wxyz']=[1,0,0,0]
    if fault=='physics':trace['base_quat'][:]=[1,0,0,0]
    (path/'report.json').write_text(json.dumps(report))
    np.savez(path/'trajectory.npz',**trace)
    assert (module.audit(path)['status']=='passed')==(fault is None)


@pytest.mark.parametrize("fault", [None, "head_drift", "wrong_target", "wrong_angle"])
def test_requested_head_pitch_is_held_during_press(recording, fault):
    path, trace = recording
    report = json.loads((path / "report.json").read_text())
    report["joint_names"].append("astribot_head_joint_2")
    report["config"]["head_pitch_deg"] = 30.
    for key in ("q", "target"):
        trace[key] = np.column_stack([trace[key], np.full(5, np.radians(30.))])
    if fault == "head_drift":
        trace["q"][2, -1] += .03
    elif fault == "wrong_target":
        trace["target"][2, -1] = 0.
    elif fault == "wrong_angle":
        report["config"]["head_pitch_deg"] = 0.
    (path / "report.json").write_text(json.dumps(report))
    np.savez(path / "trajectory.npz", **trace)
    assert (module.audit(path)["status"] == "passed") == (fault is None)


@pytest.mark.parametrize("fault", [None, "left_drift", "changed_hold_target", "wrong_initial_right"])
def test_raised_arm_initial_pose_and_left_hold_are_checked(recording, fault):
    path, trace = recording
    report = json.loads((path/"report.json").read_text())
    report["config"]["initial_arm_positions"] = {"astribot_arm_left_joint_1": -.4}
    trace["q"][:, 7] = -.4
    trace["target"][:, 7] = -.4
    if fault == "left_drift":
        trace["q"][2, 7] += .03
    elif fault == "changed_hold_target":
        trace["target"][2, 7] = 0.
    elif fault == "wrong_initial_right":
        trace["q"][0, 0] = .1
    (path/"report.json").write_text(json.dumps(report))
    np.savez(path/"trajectory.npz", **trace)
    assert (module.audit(path)["status"] == "passed") == (fault is None)


@pytest.mark.parametrize("fault",["no_force","wrong_button","moving_base","open_gripper","failed_return"])
def test_audit_rejects_false_success_report(recording,fault):
    path,trace = recording
    if fault=="no_force": trace["force"][:]=0
    if fault=="wrong_button": trace["travel"][2,7]=.002
    if fault=="moving_base": trace["base"][-1,0]=.002
    if fault=="open_gripper":
        trace["q"][:,8:]=0
        trace["target"][:,8:]=0
    if fault=="failed_return": trace["q"][-1,0]=.1
    np.savez(path/"trajectory.npz",**trace)
    assert module.audit(path)["status"] == "failed"


@pytest.mark.parametrize("reset_error",[0., .1])
def test_reset_is_checked_from_arrays_not_success_flag(recording,reset_error):
    path,trace = recording
    report=json.loads((path/"report.json").read_text())
    report["reset"]={"passed":True,"buttons_released":True}
    (path/"report.json").write_text(json.dumps(report))
    trace={k:np.concatenate([v,v[-1:]],axis=0) for k,v in trace.items()}
    trace["time"][-1]=6/120
    trace["epoch"]=np.array([0,0,0,0,0,1])
    trace["phase"][-1]="reset_check"
    trace["q"][-1,0]=reset_error
    np.savez(path/"trajectory.npz",**trace)
    assert module.audit(path)["checks"]["reset_recovered"] == (reset_error==0.)


@pytest.mark.parametrize("fault",[None,"no_latch","stale_request","uncleared_reset","overtravel"])
def test_single_hall_call_requires_new_latched_contact_and_reset(recording,fault):
    path,trace=recording
    report=json.loads((path/"report.json").read_text())
    report.update(button_ids=[0],sequence=[0],reset={"passed":True})
    report["episodes"][0]["floor"]=0
    report["config"].update(scene_type="vib820",button_light_mode="latched",button_travel=.002)
    trace["travel"]=trace["travel"][:,8:9]
    trace["force"]=trace["force"][:,8:9]
    trace={k:np.concatenate([v,v[-1:]],axis=0) for k,v in trace.items()}
    trace["time"][-1]=6/120
    trace["epoch"]=np.array([0,0,0,0,0,1])
    trace["phase"][-1]="reset_check"
    trace["light"]=np.array([[False],[True],[True],[True],[True],[False]])
    if fault=="no_latch": trace["light"][3:5]=False
    if fault=="stale_request": trace["light"][0]=True
    if fault=="uncleared_reset": trace["light"][-1]=True
    if fault=="overtravel": trace["travel"][1]=.003
    (path/"report.json").write_text(json.dumps(report))
    np.savez(path/"trajectory.npz",**trace)
    assert (module.audit(path)["status"]=="passed") == (fault is None)
