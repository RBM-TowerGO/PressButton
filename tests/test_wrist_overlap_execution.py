"""Current wrist containment must protect both model acceptance and HTTP sends."""
import copy
import json
import threading

import pytest

from pressb.agent_loop import client, wrist_overlap_guard as guard
from test_conversation import jpeg
from test_localized_conversation import make, action, observation, overlapping
from test_forward_guard_execution import robot


def missed(oid, unknown=False):
    return guard.validate_raw(dict(coordinate_units='normalized_1000',
        button_visibility='full', button_box=[200,100,300,200],
        contact_visibility='unknown' if unknown else 'full',
        contact_tip=None if unknown else [700,750], note='Current marked tip is away from target.'),oid,[640,360])


@pytest.mark.parametrize('unknown',[False,True])
def test_denied_combined_move_is_never_clipped_or_executed(jpeg,monkeypatch,tmp_path,unknown):
    c,_=make(jpeg,monkeypatch);body=c.prepare_decision('fake',tmp_path/'decision')
    xyz=[0,-20.123456,30.654321]
    replies=iter([action(c,xyz=[15,22,33]),action(c,xyz=xyz)])
    requests=[];checks=[]
    monkeypatch.setattr(client,'request',lambda b,r,body,**kw:requests.append(copy.deepcopy(body)) or next(replies))
    monkeypatch.setattr(guard,'inspect',lambda b,body,oid,d,cancel=None:checks.append(oid) or missed(oid,unknown))
    proposal=client.propose('fake',body,observation(0),c.session,tmp_path/'decision',conversation=c)
    assert proposal['arguments']['xyz_mm']==xyz and len(requests)==2 and checks==['live-0']
    assert c.forward_guard_decision is None
    assert 'CURRENT_WRIST_OVERLAP_FORWARD_CHECK' in json.dumps(requests[1]['messages'])
    assert 'BOTH image axes' in requests[1]['messages'][-1]['content']
    assert 'raw_locations' not in json.dumps(c.messages())
    gate=json.loads((tmp_path/'decision/forward_guard/decision.json').read_text())
    assert not gate['allowed'] and 'authorized_command' not in gate
    c.check_execution(proposal,'live-0')
    r,sent=robot(tmp_path/'robot',monkeypatch);r.forward_guard_policy=guard.POLICY
    r.execute(proposal,observation(0))
    assert len(sent)==1 and sent[0]['xyz_mm']==xyz


@pytest.mark.parametrize('x',[.001,5.0001,25.123456,123.456789])
def test_passed_original_XYZ_has_no_positive_X_cap(jpeg,monkeypatch,tmp_path,x):
    c,_=make(jpeg,monkeypatch);body=c.prepare_decision('fake',tmp_path/'0')
    xyz=[x,-2.718281828,1.414213562]
    monkeypatch.setattr(client,'request',lambda *a,**k:action(c,xyz=xyz))
    monkeypatch.setattr(guard,'inspect',lambda b,body,oid,d,cancel=None:overlapping(oid))
    proposal=client.propose('fake',body,observation(0),c.session,tmp_path/'0',conversation=c)
    c.check_execution(proposal,'live-0')
    r,sent=robot(tmp_path/'robot',monkeypatch);r.forward_guard_policy=guard.POLICY
    r.execute(proposal,observation(0),c.forward_guard_decision)
    assert sent[0]['xyz_mm']==xyz
    assert c.forward_guard_decision['policy']==guard.POLICY
    assert c.decision_context('action_request')['forward_alignment_phase']=='overlap_confirmed'


def test_each_new_observation_needs_fresh_check_and_old_pass_cannot_authorize(jpeg,monkeypatch,tmp_path):
    c,_=make(jpeg,monkeypatch);body=c.prepare_decision('fake',tmp_path/'0')
    replies=iter([action(c,xyz=[12,0,0])]);checks=[]
    monkeypatch.setattr(client,'request',lambda *a,**k:next(replies))
    monkeypatch.setattr(guard,'inspect',lambda b,body,oid,d,cancel=None:checks.append(oid) or overlapping(oid))
    first=client.propose('fake',body,observation(0),c.session,tmp_path/'0',conversation=c)
    old=copy.deepcopy(c.forward_guard_decision)
    c.append_result(first,None,observation(1),jpeg,jpeg)
    assert c.forward_guard_decision is None
    body=c.prepare_decision('fake',tmp_path/'1')
    assert c.decision_context('action_request')['forward_alignment_phase']=='align_yz'
    with pytest.raises(ValueError,match='overlap_permission'):
        client.require_forward_permission(first,observation(1),old,guard.POLICY)
    replies=iter([action(c,xyz=[8,4,3]),action(c,xyz=[0,-12,9])])
    monkeypatch.setattr(guard,'inspect',lambda b,body,oid,d,cancel=None:checks.append(oid) or missed(oid))
    second=client.propose('fake',body,observation(1),c.session,tmp_path/'1',conversation=c)
    assert checks==['live-0','live-1'] and second['arguments']['xyz_mm']==[0,-12,9]


def test_repeated_denied_candidate_reuses_one_check_but_never_passes(jpeg,monkeypatch,tmp_path):
    c,_=make(jpeg,monkeypatch);body=c.prepare_decision('fake',tmp_path)
    replies=iter([action(c,xyz=[6,0,0]),action(c,xyz=[50,0,0]),action(c,xyz=[0,20,20])]);checks=[]
    monkeypatch.setattr(client,'request',lambda *a,**k:next(replies))
    monkeypatch.setattr(guard,'inspect',lambda b,body,oid,d,cancel=None:checks.append(oid) or missed(oid))
    p=client.propose('fake',body,observation(0),c.session,tmp_path,conversation=c)
    assert p['arguments']['xyz_mm']==[0,20,20] and checks==['live-0']
    assert len([m for m in c.messages() if 'CURRENT_WRIST_OVERLAP_FORWARD_CHECK' in str(m)])==1
    assert json.loads((tmp_path/'action_metrics.json').read_text())['validation_failures']==2


@pytest.mark.parametrize('xyz',[[0,30,-40],[-100.123,3,4]])
def test_planar_adjustment_and_retreat_do_not_need_overlap(jpeg,monkeypatch,tmp_path,xyz):
    c,_=make(jpeg,monkeypatch);body=c.prepare_decision('fake',tmp_path)
    monkeypatch.setattr(client,'request',lambda *a,**k:action(c,xyz=xyz))
    monkeypatch.setattr(guard,'inspect',lambda *a,**k:pytest.fail('not a positive-X action'))
    p=client.propose('fake',body,observation(0),c.session,tmp_path,conversation=c)
    c.check_execution(p,'live-0')
    assert p['arguments']['xyz_mm']==xyz and not (tmp_path/'forward_guard').exists()


def test_guard_transport_failure_cannot_be_repaired_into_motion(jpeg,monkeypatch,tmp_path):
    c,_=make(jpeg,monkeypatch);body=c.prepare_decision('fake',tmp_path);calls=[]
    monkeypatch.setattr(client,'request',lambda *a,**k:calls.append(1) or action(c,xyz=[10,0,0]))
    def fail(*a,**k):raise OSError('model unavailable')
    monkeypatch.setattr(guard,'inspect',fail)
    with pytest.raises(RuntimeError,match='forward_guard_failed'):
        client.propose('fake',body,observation(0),c.session,tmp_path,conversation=c)
    assert calls==[1] and c.pending is None and not (tmp_path/'proposal.json').exists()


def test_stop_during_guard_rejects_late_pass(jpeg,monkeypatch,tmp_path):
    c,_=make(jpeg,monkeypatch);body=c.prepare_decision('fake',tmp_path);cancel=threading.Event()
    monkeypatch.setattr(client,'request',lambda *a,**k:action(c,xyz=[10,0,0]))
    def late(b,body,oid,d,cancel=None):cancel.set();return overlapping(oid)
    monkeypatch.setattr(guard,'inspect',late)
    with pytest.raises(InterruptedError):
        client.propose('fake',body,observation(0),c.session,tmp_path,cancel=cancel,conversation=c)
    assert c.pending is None and c.forward_guard_decision is None
    assert not (tmp_path/'proposal.json').exists()


def test_bypass_without_permission_or_with_legacy_permission_fails_at_HTTP(jpeg,monkeypatch,tmp_path):
    from test_forward_guard_execution import judgment
    from pressb.agent_loop import forward_guard
    c,_=make(jpeg,monkeypatch);c.prepare_decision('fake',tmp_path/'0')
    raw=action(c,xyz=[10,1,2]);p=c.parse_response(raw,observation(0),c.session,tmp_path/'0');c.accept(raw)
    with pytest.raises(ValueError,match='overlap_permission'):c.check_execution(p,'live-0')
    old=forward_guard.evaluate(judgment(),'live-0');old['authorized_command']=dict(type='move',xyz_mm=[10,1,2])
    r,sent=robot(tmp_path/'robot',monkeypatch);r.forward_guard_policy=guard.POLICY
    with pytest.raises(ValueError,match='overlap_permission'):r.execute(p,observation(0),old)
    assert not sent
