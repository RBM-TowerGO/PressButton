import importlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def test_clip_mode_requires_fresh_explicit_launch_opt_in(monkeypatch,tmp_path):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    launcher=importlib.import_module('s1b_split_launch')
    source=ROOT/'configs/s1b_execution_model_v1.json'
    before=source.read_bytes()
    enabled=launcher.load_execution_config(source,clip_all_targets=True)
    assert enabled['clip_all_targets'] is True
    saved=tmp_path/'execution.json';saved.write_text(json.dumps(enabled))
    # The same snapshot is serializable for the robot, but reusing it next time
    # without the CLI switch must not silently retain the opt-in.
    assert json.loads(saved.read_text())['clip_all_targets'] is True
    assert launcher.load_execution_config(saved)['clip_all_targets'] is False
    assert source.read_bytes()==before


def test_clip_mode_public_entrypoint_help_and_wrong_mode():
    script=str(ROOT/'scripts/start_s1b_inference.py')
    result=subprocess.run([sys.executable,script,'--help'],capture_output=True,text=True)
    assert result.returncode==0 and '--clip-all-targets' in result.stdout
    result=subprocess.run([sys.executable,script,'--clip-all-targets','--mode','shadow'],capture_output=True,text=True)
    assert result.returncode==2 and '--clip-all-targets requires execute' in result.stderr


def test_gripper_override_is_per_launch(monkeypatch,tmp_path):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    launcher=importlib.import_module('s1b_split_launch')
    config=launcher.load_execution_config(ROOT/'configs/s1b_execution_model_v1.json',gripper='open')
    saved=tmp_path/'execution.json';saved.write_text(json.dumps(config))
    assert config['gripper_override']=='open'
    assert launcher.load_execution_config(saved)['gripper_override'] is None
    result=subprocess.run([sys.executable,str(ROOT/'scripts/start_s1b_inference.py'),'--gripper','close','--mode','shadow'],capture_output=True,text=True)
    assert result.returncode==2 and '--gripper requires execute' in result.stderr
