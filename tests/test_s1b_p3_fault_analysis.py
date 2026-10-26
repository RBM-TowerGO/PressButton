import importlib.util
import json
from pathlib import Path


def module():
    spec=importlib.util.spec_from_file_location('fault_analysis',Path(__file__).resolve().parents[1]/'scripts/analyze_s1b_p3_faults.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def fixture(path,covered_fault=False,old_session=False):
    base=10_000_000_000;rows=[]
    for i in range(181):
        ns=base+round(i*1e9/30);seconds=i/30
        fault=(1<=seconds<2) or (3<=seconds<4) or seconds>=5
        session='a' if seconds<2 or old_session and seconds<4 else 'b' if seconds<4 else 'c'
        rows.append(dict(tick_ns=ns,planned_tick_ns=ns,state='FAULT' if fault else 'SHADOW',covered=not fault,session_id=session,deadline_ns=ns+20_000_000,sdk_target=None))
    if covered_fault:rows[35]['covered']=True
    (path/'ticks.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (path/'events.jsonl').write_text(''.join(json.dumps(dict(event='reconnected_shadow',session_id=s))+'\n' for s in ['b','c']))
    summary=dict(scheduler={'queue_length':0},fatal=['RuntimeError: health_changed_requires_new_session'],execution_enabled=False,sdk_imported=False,shadow_thread_error=None)
    (path/'summary.json').write_text(json.dumps(summary))
    markers=[dict(event=name,monotonic_ns=base+int(sec*1e9)) for name,sec in [('server_pause_begin',1),('server_pause_end',1.5),('tunnel_interrupt_begin',3),('tunnel_interrupt_end',3.5),('server_crash_and_model_switch_begin',5)]]
    p=path/'markers.json';p.write_text(json.dumps(markers));return p


def test_fault_audit_accepts_fresh_shadow_sessions(tmp_path):
    markers=fixture(tmp_path);result=module().analyze(tmp_path,markers)
    assert result['passed'] and len(result['fault_segments'])==3


def test_fault_audit_rejects_stale_consume_or_old_session(tmp_path):
    markers=fixture(tmp_path,covered_fault=True)
    assert not module().analyze(tmp_path,markers)['checks']['no_consume_after_fault_or_expiry']
    markers=fixture(tmp_path,old_session=True)
    assert not module().analyze(tmp_path,markers)['checks']['fresh_session_after_recovery']
