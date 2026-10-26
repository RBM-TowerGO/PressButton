"""Gate denominators must include missed control slots and reject expired sends."""
import importlib.util
import json
from pathlib import Path
import numpy as np


def analyzer():
    spec=importlib.util.spec_from_file_location('p3_analysis',Path(__file__).resolve().parents[1]/'scripts/analyze_s1b_p3.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def write_run(path,skip=False,late=False):
    requests=[];ticks=[]
    for i in range(2):
        observed=1_000_000_000+i*1_000_000_000
        raw=np.zeros((7,10),np.float32).tolist()
        requests.append(dict(valid=True,scheduler_accepted=True,scheduler_accept_ns=observed+1,session_id='s',observation_id=i,observation_monotonic_ns=observed,received_ns=observed+10_000_000,source_mapped_ns={'head':observed-50_000_000},state10=[0.]*10,action_raw=raw,send_ns=observed+1_000_000,rtt_ns=9_000_000,inference_seconds=.005,preprocess_seconds=.001,observation_to_response_ns=10_000_000))
        tick=observed+20_000_000
        ticks.append(dict(tick_ns=tick,planned_tick_ns=tick,tick_ordinal=i*2 if skip else i,covered=True,state='SHADOW',session_id='s',observation_id=i,observation_monotonic_ns=observed,step_index=0,deadline_ns=observed+33_333_333,simulated_send_ns=observed+34_000_000 if late else tick,sdk_target=None,action_raw=raw[0],response_to_tick_ns=10_000_000))
    # End after second tick so both slots are eligible.
    requests[-1]['received_ns']+=20_000_000
    (path/'summary.json').write_text(json.dumps(dict(step=2000,shadow_thread_error=None)))
    (path/'requests.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in requests))
    (path/'ticks.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in ticks))


def test_skipped_tick_counts_against_coverage(tmp_path):
    write_run(tmp_path,skip=True)
    result=analyzer().analyze(tmp_path,False)
    assert result['qualified_tick_slots']==3
    assert result['skipped_qualified_ticks']==1
    assert result['queue_coverage']==2/3
    assert not result['engineering_gate_pass']


def test_expired_simulated_send_fails_gate(tmp_path):
    write_run(tmp_path,late=True)
    result=analyzer().analyze(tmp_path,False)
    assert result['invalid_simulated_sends']==2
    assert not result['gates']['no_expired_or_modified_send']
