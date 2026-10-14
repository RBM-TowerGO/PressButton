import importlib.util
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_vib820_random_base import samples
from summarize_vib820_random_base import summarize_records

def record(outcome,ik='pass',initial='clear',dynamic='clear',motion=True,pressed=False,full=False):
    return dict(primary_outcome=outcome,ik_status=ik,initial_overlap=initial,dynamic_collision=dynamic,motion_started=motion,press_confirmed=pressed,full_success=full)

def test_frozen_manifest_sampling():
    rows,state=samples(30,20261013)
    assert rows==samples(30,20261013)[0]
    assert len({r['sample_id'] for r in rows})==30
    assert all(-.32<=r['x_m']<=.08 and .08<=r['y_m']<=.48 and r['yaw_rad']==0 for r in rows)
    assert any(abs(r['dx_m'])>.15 for r in rows) and any(abs(r['dy_m'])>.15 for r in rows)
    np.testing.assert_array_equal([[r['dx_m'],r['dy_m']] for r in rows],np.random.default_rng(20261013).uniform(-.2,.2,(30,2)))

def test_denominators_and_press_then_collision():
    rows=[record('success',pressed=True,full=True),
          record('target_ik_failed',ik='fail',dynamic='unknown',motion=False,pressed=None),
          record('initial_collision',initial='collision',dynamic='unknown',motion=False,pressed=None),
          record('unexpected_contact',dynamic='collision',pressed=True),
          record('runtime_error',ik='unknown',initial='unknown',dynamic='unknown',motion=True,pressed=None)]
    s=summarize_records(rows)
    assert s['N_planned']==5 and s['N_ik_pass']==3 and s['N_ik_fail']==1 and s['N_ik_unknown']==1
    assert s['N_attempted']==3 and s['N_press_confirmed']==2 and s['N_full_success']==1
    assert s['metrics']['attempted_press']['fraction']==2/3
    assert s['N_collision']==1 and s['N_clear']==1 and s['N_dynamic_unknown']==3
    assert s['N_dynamic_unknown_started']==1 and s['N_press_unknown_started']==1
    assert s['metrics']['initial_overlap']['denominator']==4
    assert sum(s['outcomes'].values())==5

def test_no_denominator_does_not_become_zero_collision():
    rows=[record('pending',ik='unknown',initial='unknown',dynamic='unknown',motion=False,pressed=None)]
    s=summarize_records(rows)
    for metric in ['key_pose_reachability','attempted_press','initial_overlap','dynamic_collision']:
        assert s['metrics'][metric]['fraction'] is None
    assert s['N_clear']==0 and s['N_dynamic_unknown']==1

import json
import pytest
from summarize_vib820_random_base import classify

@pytest.mark.parametrize('episode,reset,expected',[
    ({'pressed':False,'success':False},None,'press_not_triggered'),
    ({'pressed':True,'success':False},None,'release_or_return_failed'),
    ({'pressed':True,'success':True},{'passed':False},'reset_failed'),
])
def test_runner_physical_failure_is_not_runtime_error(tmp_path,episode,reset,expected):
    report=dict(status='failed',error='RuntimeError: Button task failed physical acceptance; see report.json',episodes=[episode])
    if reset: report['reset']=reset
    (tmp_path/'report.json').write_text(json.dumps(report))
    result=classify(tmp_path,exit_code=1)
    assert result['primary_outcome']==expected
    assert result['press_confirmed'] is None and result['full_success'] is False

def test_interrupted_process_keeps_known_collision(tmp_path):
    report=dict(status='failed',motion_started=True,unexpected_contacts=[dict(actors=['arm','wall'],force_n=.2)],error='RuntimeError: Unexpected physical contact')
    (tmp_path/'report.json').write_text(json.dumps(report))
    r=classify(tmp_path,exit_code=1)
    assert r['primary_outcome']=='unexpected_contact' and r['dynamic_collision']=='collision'
