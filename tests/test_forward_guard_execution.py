"""The independent wrist condition must gate proposals AND HTTP execution."""
import copy
import json
import threading

import pytest

from pressb.agent_loop import client, grounding, forward_guard as guard
from pressb.agent_loop.common import SESSION_SCHEMA
from pressb.agent_loop.conversation import live_message
from test_conversation import jpeg, obs, response


SESSION = dict(schema=SESSION_SCHEMA, session_id='session-test')
BODY = dict(messages=[dict(role='system', content='Choose one command.')])


class StubConversation:
    def __init__(self, jpeg, mode='direct', number=0):
        self.goal='press down-call button'
        self.decision_mode=mode
        self.current_id=obs(number)['observation_id']
        self.observations={self.current_id:live_message(obs(number),jpeg,jpeg)}
        self.accepted=[]
        self.forward_guard_decision=None

    def check_observation(self, oid):
        if oid != self.current_id:
            raise ValueError('decision_observation_mismatch')

    def parse_response(self, raw, observation, session, directory):
        self.check_observation(observation['observation_id'])
        return client.parse_proposal(raw, observation, session)

    def accept(self, raw):
        self.accepted.append(copy.deepcopy(raw))


def judgment(oid='live-0', value='yes'):
    return dict(format=guard.FORMAT,source=guard.SOURCE,observation_id=oid,
                **{field:value for field in guard.FIELDS})


def move(xyz=(3.14159265,-2.71828182,1.41421356), reason='approach'):
    return response('propose_move',dict(xyz_mm=list(xyz),reason=reason))


def model_replies(monkeypatch,replies):
    queue=iter(replies); calls=[]
    def request(base,route,body,**kwargs):
        calls.append(copy.deepcopy(body));return next(queue)
    monkeypatch.setattr(client,'request',request)
    return calls


def visual_replies(monkeypatch,replies):
    queue=iter(replies);calls=[]
    def inspect(base,body,oid,directory,cancel=None):
        calls.append(dict(observation_id=oid,body=copy.deepcopy(body)))
        return next(queue)
    monkeypatch.setattr(guard,'inspect',inspect)
    return calls


def robot(tmp_path,monkeypatch):
    # No constructor or network; exercise the real execute() submit boundary.
    r=client.RobotClient.__new__(client.RobotClient)
    r.output=tmp_path;r.session=SESSION;r.cancel=threading.Event();r.base='fake'
    submitted=[]
    monkeypatch.setattr(r,'submit',lambda route,body:submitted.append(copy.deepcopy(body)) or dict(task_id='task'))
    task=dict(status='completed',capture=dict(observation_id='after'))
    monkeypatch.setattr(r,'wait',lambda receipt:task)
    monkeypatch.setattr(r,'download',lambda oid:(dict(observation_id=oid),tmp_path/'after'))
    return r,submitted


@pytest.mark.parametrize('mode',['direct','separated','grounded'])
def test_denial_repairs_to_original_yz_values_and_never_sends_forward(jpeg,tmp_path,monkeypatch,mode):
    c=StubConversation(jpeg,mode)
    values=[0.0,-2.718281828459,1.414213562373]
    action_calls=model_replies(monkeypatch,[move(),move(values,'correct lateral offset')])
    visual_calls=visual_replies(monkeypatch,[judgment(value='no')])
    proposal=client.propose('fake',BODY,obs(0),SESSION,tmp_path/'decision',conversation=c)
    assert proposal['arguments']['xyz_mm']==values
    assert len(action_calls)==2 and len(visual_calls)==1 and len(c.accepted)==1
    assert c.forward_guard_decision is None
    assert 'NOT executed' in action_calls[1]['messages'][-1]['content']
    assert 'Only horizontal wrist alignment' in action_calls[1]['messages'][-1]['content']
    decision=json.loads((tmp_path/'decision/forward_guard/decision.json').read_text())
    assert decision['allowed'] is False
    r,submitted=robot(tmp_path/'robot',monkeypatch)
    r.execute(proposal,obs(0),forward_check=c.forward_guard_decision)
    assert len(submitted)==1 and submitted[0]['xyz_mm']==values


@pytest.mark.parametrize('value',['no','unknown'])
def test_repeated_positive_candidate_is_denied_until_legal_correction_without_rechecking_frame(jpeg,tmp_path,monkeypatch,value):
    c=StubConversation(jpeg)
    final=move([0,-2,0],'adjust from the current view')
    action_calls=model_replies(monkeypatch,[move(),move([1,0,0],'try again'),move(),move(),final])
    visual_calls=visual_replies(monkeypatch,[judgment(value=value)])
    proposal=client.propose('fake',BODY,obs(0),SESSION,tmp_path/'decision',conversation=c)
    assert len(action_calls)==5 and len(visual_calls)==1
    assert c.accepted==[final] and c.forward_guard_decision is None
    for attempt in range(4):
        decision=json.loads((tmp_path/'decision/forward_guard'/f'candidate-{attempt}.json').read_text())
        assert not decision['decision']['allowed']
        assert (tmp_path/'decision'/f'validation-{attempt}.json').exists()
        assert (tmp_path/'decision'/f'repair-request-{attempt+1}.json').exists()
    r,submitted=robot(tmp_path/'robot',monkeypatch)
    r.execute(proposal,obs(0),forward_check=c.forward_guard_decision)
    assert len(submitted)==1 and submitted[0]['xyz_mm']==[0,-2,0]


def test_unbounded_forward_corrections_are_cancellable_without_accept(jpeg,tmp_path,monkeypatch):
    c=StubConversation(jpeg);cancel=threading.Event();calls=[]
    visual_calls=visual_replies(monkeypatch,[judgment(value='unknown')])
    def request(base,route,body,**kwargs):
        calls.append(copy.deepcopy(body))
        assert c.accepted==[] and c.forward_guard_decision is None
        if len(calls)==5:cancel.set()
        if len(calls)>5:pytest.fail('late cancellation must stop all further requests')
        return move()
    monkeypatch.setattr(client,'request',request)
    with pytest.raises(InterruptedError,match='operator_stopped'):
        client.propose('fake',BODY,obs(0),SESSION,tmp_path,conversation=c,cancel=cancel)
    assert len(calls)==5 and len(visual_calls)==1 and c.accepted==[]
    assert not (tmp_path/'proposal.json').exists() and c.forward_guard_decision is None


@pytest.mark.parametrize('mode',['direct','separated','grounded'])
def test_positive_permission_preserves_exact_xyz_and_original_http_schema(jpeg,tmp_path,monkeypatch,mode):
    c=StubConversation(jpeg,mode)
    raw=move();xyz=json.loads(raw['choices'][0]['message']['tool_calls'][0]['function']['arguments'])['xyz_mm']
    model_replies(monkeypatch,[raw]);visual_calls=visual_replies(monkeypatch,[judgment()])
    p=client.propose('fake',BODY,obs(0),SESSION,tmp_path/'decision',conversation=c)
    assert len(visual_calls)==1 and p['arguments']['xyz_mm']==xyz
    permission=c.forward_guard_decision
    assert permission['allowed'] and permission['authorized_command']==dict(type='move',xyz_mm=xyz)
    assert permission['observation_id']=='live-0'
    r,submitted=robot(tmp_path/'robot',monkeypatch)
    r.execute(p,obs(0),forward_check=permission)
    assert submitted[0]['xyz_mm']==xyz
    assert set(submitted[0])=={'session_id','command_id','observation_id','type','xyz_mm'}
    assert submitted[0]['observation_id']=='live-0'


def test_new_observation_rechecks_instead_of_reusing_allowed_memory(jpeg,tmp_path,monkeypatch):
    c=StubConversation(jpeg)
    model_replies(monkeypatch,[move(),move(),move([0,-1,0])])
    calls=visual_replies(monkeypatch,[judgment(),judgment(oid='live-1',value='unknown')])
    client.propose('fake',BODY,obs(0),SESSION,tmp_path/'0',conversation=c)
    assert c.forward_guard_decision['allowed']
    c.current_id='live-1';c.observations['live-1']=live_message(obs(1),jpeg,jpeg)
    p=client.propose('fake',BODY,obs(1),SESSION,tmp_path/'1',conversation=c)
    assert p['arguments']['xyz_mm']==[0,-1,0] and c.forward_guard_decision is None
    assert [call['observation_id'] for call in calls]==['live-0','live-1']


def test_stale_judgment_cannot_authorize_current_positive_command(jpeg,tmp_path,monkeypatch):
    c=StubConversation(jpeg)
    model_replies(monkeypatch,[move(),move([0,-1,0])])
    visual_replies(monkeypatch,[judgment(oid='old')])
    p=client.propose('fake',BODY,obs(0),SESSION,tmp_path,conversation=c)
    assert p['arguments']['xyz_mm']==[0,-1,0]
    decision=json.loads((tmp_path/'forward_guard/decision.json').read_text())
    assert not decision['allowed'] and 'assessment_observation_mismatch' in decision['reasons']


@pytest.mark.parametrize('exception',[TimeoutError('model timeout'),ValueError('invalid_grounding:bad')])
def test_guard_failure_ends_before_action_repair_or_accept(jpeg,tmp_path,monkeypatch,exception):
    c=StubConversation(jpeg);calls=model_replies(monkeypatch,[move()])
    def inspect(*a,**k):raise exception
    monkeypatch.setattr(guard,'inspect',inspect)
    with pytest.raises(RuntimeError,match='forward_guard_failed'):
        client.propose('fake',BODY,obs(0),SESSION,tmp_path,conversation=c)
    assert len(calls)==1 and c.accepted==[] and c.forward_guard_decision is None
    assert not (tmp_path/'repair-request.json').exists()
    assert json.loads((tmp_path/'error.json').read_text())['stage']=='forward_guard'


def test_real_inspect_many_format_corrections_do_not_consume_action_retries(jpeg,tmp_path,monkeypatch):
    c=StubConversation(jpeg);calls=model_replies(monkeypatch,[move()]);visual_calls=[]
    invalid=response('inspect_forward_alignment',dict(button_visible='yes'))
    valid=response('inspect_forward_alignment',{field:'yes' for field in guard.FIELDS})
    replies=iter([invalid]*3+[valid])
    monkeypatch.setattr(grounding,'request',lambda *a,**k:visual_calls.append(a) or next(replies))
    proposal=client.propose('fake',BODY,obs(0),SESSION,tmp_path,conversation=c)
    assert len(calls)==1 and len(visual_calls)==4 and len(c.accepted)==1
    assert proposal['arguments']['xyz_mm']==[3.14159265,-2.71828182,1.41421356]
    assert c.forward_guard_decision['allowed']
    assert (tmp_path/'forward_guard/repair-request.json').exists()
    assert not (tmp_path/'repair-request.json').exists()


@pytest.mark.parametrize('phase',['action','guard'])
def test_late_response_after_stop_never_accepts_or_authorizes(jpeg,tmp_path,monkeypatch,phase):
    c=StubConversation(jpeg);cancel=threading.Event();visual_calls=[]
    def action_request(*a,**k):
        if phase=='action':cancel.set()
        return move()
    def inspect(*a,**k):
        visual_calls.append(a);cancel.set();return judgment()
    monkeypatch.setattr(client,'request',action_request);monkeypatch.setattr(guard,'inspect',inspect)
    with pytest.raises(InterruptedError,match='operator_stopped'):
        client.propose('fake',BODY,obs(0),SESSION,tmp_path,conversation=c,cancel=cancel)
    assert c.accepted==[] and c.forward_guard_decision is None
    assert len(visual_calls)==(1 if phase=='guard' else 0)
    assert not (tmp_path/'proposal.json').exists()


def test_observation_changed_during_guard_cannot_authorize(jpeg,tmp_path,monkeypatch):
    c=StubConversation(jpeg);model_replies(monkeypatch,[move()])
    def inspect(*a,**k):c.current_id='live-1';return judgment()
    monkeypatch.setattr(guard,'inspect',inspect)
    with pytest.raises(RuntimeError,match='forward_guard_failed:decision_observation_mismatch'):
        client.propose('fake',BODY,obs(0),SESSION,tmp_path,conversation=c)
    assert not c.accepted and c.forward_guard_decision is None


@pytest.mark.parametrize('name,args',[
    ('propose_move',dict(xyz_mm=[0,-1.234567,2.345678],reason='adjust')),
    ('propose_move',dict(xyz_mm=[-1.234567,0,0],reason='retract')),
    ('propose_head',dict(head_deg=[3.141592,28.987654],reason='frame view')),
    ('propose_gripper',dict(key='o',reason='open')),
    ('request_observation',dict(reason='observe')),
    ('report_unable',dict(reason='cannot continue')),
])
def test_nonforward_actions_need_no_visual_request(jpeg,tmp_path,monkeypatch,name,args):
    c=StubConversation(jpeg);model_replies(monkeypatch,[response(name,args)])
    monkeypatch.setattr(guard,'inspect',lambda *a,**k:pytest.fail('no forward check needed'))
    p=client.propose('fake',BODY,obs(0),SESSION,tmp_path/'decision',conversation=c)
    assert p==dict(name=name,arguments=args) and c.forward_guard_decision is None
    if name.startswith('propose_'):
        r,submitted=robot(tmp_path/'robot',monkeypatch)
        r.execute(p,obs(0))
        assert len(submitted)==1


@pytest.mark.parametrize('mutate',[
    lambda d:d.clear(),
    lambda d:d.update(observation_id='old'),
    lambda d:d.update(assessment_observation_id='old'),
    lambda d:d.update(allowed=False),
    lambda d:d.update(policy='none'),
    lambda d:d.update(format='legacy'),
    lambda d:d.update(source='model'),
    lambda d:d.update(authorized_command=dict(type='move',xyz_mm=[1,0,0])),
    lambda d:d['basis'].update(button_between_fingers='unknown'),
])
def test_execute_rejects_missing_stale_or_altered_permission_before_submit(tmp_path,monkeypatch,mutate):
    p=dict(name='propose_move',arguments=dict(xyz_mm=[2.123456,0,0],reason='approach'))
    decision=guard.evaluate(judgment(),'live-0')
    decision['authorized_command']=dict(type='move',xyz_mm=copy.deepcopy(p['arguments']['xyz_mm']))
    mutate(decision)
    r,submitted=robot(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='positive_X_requires_current'):
        r.execute(p,obs(0),forward_check=decision)
    assert not submitted and not (tmp_path/'commands').exists()


def test_positive_execute_without_permission_never_reaches_http(tmp_path,monkeypatch):
    r,submitted=robot(tmp_path,monkeypatch)
    p=dict(name='propose_move',arguments=dict(xyz_mm=[1,0,0],reason='approach'))
    with pytest.raises(ValueError,match='positive_X_requires_current'):
        r.execute(p,obs(0))
    assert not submitted


def test_capture_failure_does_not_resubmit_approved_action(jpeg,tmp_path,monkeypatch):
    c=StubConversation(jpeg);model_replies(monkeypatch,[move()]);visual_replies(monkeypatch,[judgment()])
    p=client.propose('fake',BODY,obs(0),SESSION,tmp_path/'decision',conversation=c)
    r,submitted=robot(tmp_path/'robot',monkeypatch)
    def failed_download(oid):raise ValueError('artifact_hash_mismatch')
    monkeypatch.setattr(r,'download',failed_download)
    with pytest.raises(ValueError,match='artifact_hash_mismatch'):
        r.execute(p,obs(0),forward_check=c.forward_guard_decision)
    assert len(submitted)==1
