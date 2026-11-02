"""Framing and disagreement recovery constrain proposals, never synthesize motion."""
import copy
import json
import threading

import pytest

from pressb.agent_loop import client, conversation, grounding, view_acquisition
from pressb.agent_loop.conversation import Conversation, VIEW_ACTIONS
from test_conversation import jpeg, obs, summary_response
from test_progress_integration import SESSION, action, advance, mock_action, review_answer
from test_view_acquisition import reply, located, unknown


def make(jpeg, **kwargs):
    return Conversation('Press down-call with the WHITE ARROW indicated physical end.',
        obs(0), jpeg, jpeg, SESSION, 'Use the WHITE ARROW.', monitor_progress=True,
        acquire_view=True, **kwargs)


def setup_views(monkeypatch, replies, calls=None):
    queue = iter(replies)
    def request(base, path, body, **kwargs):
        assert path == '/chat/completions' and kwargs['timeout'] == 120
        if calls is not None:
            calls.append(copy.deepcopy(body))
        answer = next(queue)
        if isinstance(answer, Exception):
            raise answer
        return answer
    monkeypatch.setattr(grounding, 'request', request)
    monkeypatch.setattr(conversation, 'tokenize', lambda *args, **kwargs: 2000)


def tools(body):
    return {item['function']['name'] for item in body['tools']}


def finish(c, proposal, n, jpeg):
    result = None
    if proposal['name'].startswith('propose_'):
        command = dict(type=proposal['name'].removeprefix('propose_'), command_id=f'cmd-{n}')
        command.update({key: value for key, value in proposal['arguments'].items() if key != 'reason'})
        result = dict(status='completed', request=command, motion=dict(action_completed=True),
                      capture=dict(observation_id=f'live-{n}'))
    c.append_result(proposal, result, obs(n), jpeg, jpeg)


def conflict(c, jpeg, tmp_path, monkeypatch):
    setup_views(monkeypatch, [reply(head=located())])
    c.prepare_decision('fake', tmp_path/'initial')
    advance(c, jpeg, 1)
    # Independent reviewer says below; original decision says above. Neither
    # statement is accepted as truth by the recovery transition.
    setup_views(monkeypatch, [review_answer(1, wrist_vertical='below', vertical_change='unchanged')])
    mock_action(monkeypatch, [action(1, effect='unknown'),
                             action(1, name='request_observation', effect='unknown')])
    candidate = client.propose('fake', c.body(), obs(1), SESSION, tmp_path/'conflict', conversation=c)
    assert candidate['name'] == 'request_observation'
    return candidate


def test_cropped_initial_view_filters_tools_and_rejects_invented_hand_call(jpeg, tmp_path, monkeypatch):
    c = make(jpeg)
    calls = []
    setup_views(monkeypatch, [reply()], calls)
    body = c.prepare_decision('fake', tmp_path)
    original = copy.deepcopy(body['messages'])
    assert tools(body) == set(VIEW_ACTIONS) and c.view_phase == 'get_view'
    assert len(calls) == 1
    assert 'CURRENT_VIEW_ACQUISITION' in json.dumps(body)
    assert json.loads((tmp_path/'view_state.json').read_text())['observation_id'] == 'live-0'
    # Tool filtering alone is not sufficient; validate even unsolicited move calls.
    decisions = []
    mock_action(monkeypatch, [action(0)]*4+[action(0, name='propose_head', subgoal='get_view')], decisions)
    proposal = client.propose('fake', body, obs(0), SESSION, tmp_path, conversation=c)
    assert len(decisions) == 5 and proposal['name'] == 'propose_head'
    assert proposal['arguments']['head_deg'] == [0, 30]
    assert c.messages() == original and not c.turns
    c.check_execution(proposal, 'live-0')
    assert json.loads((tmp_path/'correction.json').read_text())['corrections'] == 4
    assert len(calls) == 1  # Repairs do not repeat view inference on the same image.


def test_already_usable_head_allows_original_xyz_and_does_not_recheck_each_hand_step(jpeg, tmp_path, monkeypatch):
    c = make(jpeg)
    calls = []
    setup_views(monkeypatch, [reply(head=located())], calls)
    body = c.prepare_decision('fake', tmp_path/'first')
    assert c.view_phase == 'ready' and 'propose_move' in tools(body)
    mock_action(monkeypatch, [action(0, xyz=(0, -17.123456, 4.5))])
    proposal = client.propose('fake', body, obs(0), SESSION, tmp_path/'first', conversation=c)
    assert proposal['arguments']['xyz_mm'] == [0, -17.123456, 4.5]
    c.check_execution(proposal, 'live-0')
    finish(c, proposal, 1, jpeg)
    c.prepare_decision('fake', tmp_path/'next')
    assert len(calls) == 1 and c.view_validated_id == 'live-0'
    state = json.loads((tmp_path/'next/view_state.json').read_text())
    assert state['observation_id'] == 'live-1' and state['assessment_observation_id'] == 'live-0'


def test_head_send_completion_requires_new_image_not_assumed_adequate(jpeg, tmp_path, monkeypatch):
    c = make(jpeg)
    calls = []
    setup_views(monkeypatch, [reply(), reply(oid='live-1'), reply(oid='live-2', head=located())], calls)
    c.prepare_decision('fake', tmp_path/'0')
    prefix = copy.deepcopy(c.messages())
    advance(c, jpeg, 1, name='propose_head')
    assert c.view_phase == 'get_view' and c.view_validated_id is None
    c.prepare_decision('fake', tmp_path/'1')
    assert c.view_phase == 'get_view'
    assert c.messages()[:len(prefix)] == prefix
    advance(c, jpeg, 2, name='propose_head')
    body = c.prepare_decision('fake', tmp_path/'2')
    assert c.view_phase == 'ready' and c.view_validated_id == 'live-2'
    assert len(calls) == 3 and 'propose_move' in tools(body)
    assert all(t['ledger']['result']['command']['type'] == 'head' for t in c.turns)


def test_visual_conflict_requires_new_view_even_if_model_changes_its_claim(jpeg, tmp_path, monkeypatch):
    c = make(jpeg)
    setup_views(monkeypatch, [reply(head=located())])
    c.prepare_decision('fake', tmp_path/'initial')
    advance(c, jpeg, 1)
    prefix = copy.deepcopy(c.messages())
    setup_views(monkeypatch, [review_answer(1, wrist_vertical='below', vertical_change='unchanged')])
    calls = []
    mock_action(monkeypatch, [action(1, effect='unknown'),
                             action(1, xyz=(0, 0, -10), effect='unknown', button_vertical='below'),
                             action(1, name='propose_head', effect='unknown', button_vertical='above')], calls)
    proposal = client.propose('fake', c.body(), obs(1), SESSION, tmp_path/'conflict', conversation=c)
    assert proposal['name'] == 'propose_head' and len(calls) == 3
    assert c.view_phase == 'recover_view'
    assert all(tools(body) == set(VIEW_ACTIONS) for body in calls[1:])
    assert c.messages()[:len(prefix)] == prefix
    record = json.loads((tmp_path/'conflict/progress_review/conflict.json').read_text())
    assert record['source'] == 'model_disagreement_not_visual_fact'
    assert record['model_claim']['button_vertical'] == 'above'
    assert record['review_claim']['wrist_vertical'] == 'below'
    assert record['conflicting_axes'] == ['vertical']
    # A recovery head action may retain its original relation; no forced agreement.
    assert c.progress_records['live-1']['button_vertical'] == 'above'
    c.check_execution(proposal, 'live-1')
    c.prepare_view('fake', tmp_path/'same')  # Same frame never clears recovery.
    assert c.view_phase == 'recover_view' and 'live-1' not in c.view_assessments


def test_only_fresh_usable_localization_resolves_conflict_and_starts_comparison(jpeg, tmp_path, monkeypatch):
    c = make(jpeg)
    proposal = conflict(c, jpeg, tmp_path, monkeypatch)
    finish(c, proposal, 2, jpeg)
    setup_views(monkeypatch, [reply(oid='live-2'), reply(oid='live-3', head=located())])
    body = c.prepare_decision('fake', tmp_path/'still_cropped')
    assert tools(body) == set(VIEW_ACTIONS) and c.progress_comparison_start == 0
    mock_action(monkeypatch, [action(2, name='request_observation', effect='no_motion')])
    proposal = client.propose('fake', body, obs(2), SESSION, tmp_path/'still_cropped', conversation=c)
    finish(c, proposal, 3, jpeg)
    body = c.prepare_decision('fake', tmp_path/'usable')
    assert c.view_phase == 'ready' and c.view_validated_id == 'live-3'
    assert c.progress_comparison_start == 3 and c.progress_unresolved == {}
    assert c._last_progress_move() is None
    assert c.turns[0]['ledger']['result']['command']['xyz_mm'] == [0, 0, 10]
    assert len([t for t in c.turns if t['ledger']['result']['status'] == 'completed']) == 1
    assert 'propose_move' in tools(body)
    assert c.turns[1]['ledger']['visual_conflict']['observation_id'] == 'live-1'


def test_execution_check_rejects_bypassing_restricted_stage(jpeg, tmp_path):
    c = make(jpeg)
    raw = action(0)
    candidate = c.parse_response(raw, obs(0), SESSION, tmp_path)
    c.accept(raw)  # Simulate a caller bypassing propose; final gate still applies.
    with pytest.raises(ValueError, match='view_acquisition_required'):
        c.check_execution(candidate, 'live-0')
    with pytest.raises(ValueError, match='decision_observation_mismatch'):
        c.check_execution(candidate, 'old-observation')


@pytest.mark.parametrize('kind', ['cancel', 'network'])
def test_initial_view_failure_or_late_response_cannot_enable_motion(jpeg, tmp_path, monkeypatch, kind):
    c = make(jpeg)
    cancel = threading.Event()
    def request(*args, **kwargs):
        if kind == 'network':
            raise TimeoutError('view request unavailable')
        cancel.set()
        return reply(head=located())
    monkeypatch.setattr(grounding, 'request', request)
    monkeypatch.setattr(client, 'request', lambda *a, **k: pytest.fail('no action request'))
    with pytest.raises(InterruptedError if kind == 'cancel' else RuntimeError):
        c.prepare_decision('fake', tmp_path, cancel)
    assert c.view_phase == 'get_view' and not c.view_assessments and c.pending is None
    assert json.loads((tmp_path/'decision_context.json').read_text())['stage'] in ('interrupted', 'failed')
    assert not (tmp_path/'view_acquisition/assessment.json').exists()


def test_compaction_keeps_view_provenance_and_current_recovery_gate(jpeg, tmp_path, monkeypatch):
    c = make(jpeg)
    setup_views(monkeypatch, [reply(head=located())])
    c.prepare_decision('fake', tmp_path/'initial')
    for n in range(1, 9):
        advance(c, jpeg, n)
    c._enter_view_recovery(tmp_path/'current', dict(observation_id='live-8', source='model_not_human'),
                          dict(observation_id='live-8', source='model_not_human'), ['vertical'])
    monkeypatch.setattr(conversation, 'request', lambda *a, **k: summary_response())
    c.compact('fake', tmp_path/'compaction')
    ledger = json.loads((tmp_path/'compaction/result.json').read_text())['earlier_action_ledger']
    assert ledger[0]['view_acquisition']['source'] == 'model_not_human'
    assert c.view_phase == 'recover_view' and tools(c.body()) == set(VIEW_ACTIONS)
    assert 'CURRENT_VISUAL_CONFLICT_RECOVERY' in json.dumps(c.body())
    assert len([m for m in c.messages() if m['role'] == 'assistant']) == 5


def test_view_feature_rejects_non_direct_monitor_configuration(jpeg):
    with pytest.raises(ValueError, match='requires_direct_progress_monitor'):
        Conversation('goal', obs(0), jpeg, jpeg, SESSION, 'prompt', acquire_view=True)
