"""Direct decisions retain their tool protocol while progress review can interrupt repetition."""
import copy
import json
import threading
from types import SimpleNamespace

import pytest

from pressb.agent_loop import client, conversation, grounding, progress
from pressb.agent_loop.common import SESSION_SCHEMA
from pressb.agent_loop.conversation import Conversation, image_count
from test_conversation import jpeg, obs, response, summary_response
from test_progress import values, reviewed


SESSION = dict(schema=SESSION_SCHEMA)


def make(jpeg, **kwargs):
    return Conversation('Press down-call button.', obs(0), jpeg, jpeg, SESSION,
                        'Use the WHITE ARROW pointed-to physical contact.', monitor_progress=True, **kwargs)


def action(n, name='propose_move', xyz=(0, 0, 10), effect=None, cid=None, **fields):
    status = values(f'live-{n}', last_effect=effect or ('first' if n == 0 else 'improved'))
    status.update(fields)
    args = dict(reason='UNTRUSTED_OLD_REASON', progress=status)
    if name == 'propose_move':
        args['xyz_mm'] = list(xyz)
    elif name == 'propose_head':
        args['head_deg'] = [0, 30]
    elif name == 'propose_gripper':
        args['key'] = 'o'
    elif name == 'report_done':
        args = dict(evidence='light-on live-1; withdrawal live-2', progress=status)
    return response(name, args, cid or f'call-{n}')


def advance(c, jpeg, n, name='propose_move', xyz=(0, 0, 10), effect=None, **progress_fields):
    raw = action(n-1, name, xyz, effect=effect, **progress_fields)
    proposal = c.parse_response(raw, obs(n-1), SESSION, '.')
    c.accept(raw)
    result = None
    if name.startswith('propose_'):
        command = dict(type=name.removeprefix('propose_'), command_id=f'cmd-{n}')
        command.update({key: value for key, value in proposal['arguments'].items() if key != 'reason'})
        result = dict(status='completed', request=command, motion=dict(action_completed=True),
                      capture=dict(observation_id=f'live-{n}'))
    c.append_result(proposal, result, obs(n), jpeg, jpeg)
    return proposal


def review_answer(n, before=None, **changes):
    return response('review_progress', reviewed(oid=f'live-{n}', before=f'live-{n-1 if before is None else before}', **changes))


def mock_review(monkeypatch, replies, calls=None):
    queue = iter(replies)
    def request(base, path, body, **kwargs):
        if calls is not None:
            calls.append(copy.deepcopy(body))
        assert path == '/chat/completions' and kwargs['timeout'] == 120
        item = next(queue)
        if isinstance(item, Exception):
            raise item
        return item
    monkeypatch.setattr(grounding, 'request', request)


def mock_action(monkeypatch, replies, calls=None):
    queue = iter(replies)
    def request(base, path, body, **kwargs):
        if calls is not None:
            calls.append(copy.deepcopy(body))
        assert path == '/chat/completions'
        return next(queue)
    monkeypatch.setattr(client, 'request', request)


def test_normal_round_one_call_keeps_raw_progress_and_pairs_tool_id(jpeg, tmp_path, monkeypatch):
    c = make(jpeg)
    original = copy.deepcopy(c.body())
    answer = action(0, xyz=(0, 0, 10.123456), cid='raw-tool-id')
    calls = []
    mock_action(monkeypatch, [answer], calls)
    monkeypatch.setattr(grounding, 'request', lambda *a, **k: pytest.fail('ordinary first round needs no review'))
    p = client.propose('fake', c.body(), obs(0), SESSION, tmp_path, conversation=c)
    assert len(calls) == 1
    assert p['arguments'] == dict(xyz_mm=[0, 0, 10.123456], reason='UNTRUSTED_OLD_REASON')
    assert c.pending == answer['choices'][0]['message']
    c.check_execution(p, 'live-0')
    result = dict(status='completed', request=dict(type='move', xyz_mm=p['arguments']['xyz_mm'], command_id='actual'),
                  motion=dict(action_completed=True), capture=dict(observation_id='live-1'))
    c.append_result(p, result, obs(1), jpeg, jpeg)
    messages = c.body()['messages']
    assert messages[:len(original['messages'])] == original['messages']
    assert messages[-3]['tool_calls'][0]['id'] == messages[-2]['tool_call_id'] == 'raw-tool-id'
    assert 'progress' in json.loads(messages[-3]['tool_calls'][0]['function']['arguments'])
    assert c.turns[0]['ledger']['progress']['source'] == 'model_not_human'
    assert c.turns[0]['ledger']['result']['command']['xyz_mm'][2] == 10.123456
    assert json.loads((tmp_path/'proposal.json').read_text()) == p


@pytest.mark.parametrize('invalid', ['missing', 'wrong_observation'])
def test_invalid_progress_repaired_once_before_any_accept(jpeg, tmp_path, monkeypatch, invalid):
    c = make(jpeg)
    bad = action(0)
    args = json.loads(bad['choices'][0]['message']['tool_calls'][0]['function']['arguments'])
    if invalid == 'missing':
        args.pop('progress')
    else:
        args['progress']['observation_id'] = 'old-frame'
    bad['choices'][0]['message']['tool_calls'][0]['function']['arguments'] = json.dumps(args)
    good = action(0, cid='accepted')
    calls = []
    mock_action(monkeypatch, [bad, good], calls)
    client.propose('fake', c.body(), obs(0), SESSION, tmp_path, conversation=c)
    assert len(calls) == 2 and c.pending['tool_calls'][0]['id'] == 'accepted'
    assert len(c.progress_records) == 1 and not c.progress_reviews
    assert (tmp_path/'repair-request.json').exists()


def test_three_completed_moves_review_precedes_next_action_and_keeps_prefix(jpeg, tmp_path, monkeypatch):
    c = make(jpeg)
    for n in range(1, 4):
        advance(c, jpeg, n)
    prefix = copy.deepcopy(c.messages())
    events = []
    def review_request(base, route, body, **kwargs):
        events.append('review')
        assert image_count(body['messages']) == 4
        return review_answer(3)
    def action_request(base, route, body, **kwargs):
        events.append('action')
        assert 'DIRECT_PROGRESS_REVIEW' in json.dumps(body)
        assert image_count(body['messages']) == 8
        return action(3)
    monkeypatch.setattr(grounding, 'request', review_request)
    monkeypatch.setattr(client, 'request', action_request)
    monkeypatch.setattr(conversation, 'tokenize', lambda *a, **k: 4000)
    body = c.prepare_decision('fake', tmp_path)
    assert body['messages'][:len(prefix)] == prefix
    proposal = client.propose('fake', body, obs(3), SESSION, tmp_path, conversation=c)
    assert events == ['review', 'action']
    assert proposal['arguments']['xyz_mm'] == [0, 0, 10]
    assert c.last_review_round == 3
    c.check_execution(proposal, 'live-3')
    record = json.loads((tmp_path/'progress_review/trigger.json').read_text())
    assert record['monitor']['streaks']['Z']['count'] == 3
    assert json.loads((tmp_path/'progress_review/decision.json').read_text())['allowed']


@pytest.mark.parametrize('effect,relation', [('worsened', 'above'), ('improved', 'level')])
def test_early_anomaly_reviews_then_new_action_consumes_independent_evidence(jpeg, tmp_path, monkeypatch, effect, relation):
    c = make(jpeg, demo=SimpleNamespace(content=[dict(type='text', text='SECRET_CASE_TEXT')]))
    advance(c, jpeg, 1)
    original_turns = copy.deepcopy(c.turns)
    calls, reviews = [], []
    mock_action(monkeypatch, [action(1, effect=effect, button_vertical=relation),
                             action(1, name='propose_head', subgoal='get_view', button_vertical='level')], calls)
    mock_review(monkeypatch, [review_answer(1, wrist_vertical='level')], reviews)
    proposal = client.propose('fake', c.body(), obs(1), SESSION, tmp_path, conversation=c)
    assert proposal['name'] == 'propose_head'
    assert len(calls) == 2 and len(reviews) == 1
    assert 'DIRECT_PROGRESS_REVIEW' not in json.dumps(calls[0]['messages'][1:])
    assert 'DIRECT_PROGRESS_REVIEW' in json.dumps(calls[1]['messages'][1:])
    assert image_count(reviews[0]['messages']) == 4
    serialized = json.dumps(reviews[0])
    assert 'SECRET_CASE_TEXT' not in serialized and 'UNTRUSTED_OLD_REASON' not in serialized
    assert 'propose_head' not in serialized and 'propose_move' not in serialized
    assert c.turns == original_turns and c.pending['tool_calls'][0]['function']['name'] == 'propose_head'
    assert json.loads((tmp_path/'progress_review/decision.json').read_text())['allowed']


def test_repeated_bad_candidates_keep_correcting_until_legal_recovery(jpeg, tmp_path, monkeypatch):
    c = make(jpeg)
    advance(c, jpeg, 1)
    calls, reviews = [], []
    bad = [action(1, effect='crossed', button_vertical='below', cid=f'rejected-{n}') for n in range(4)]
    good = action(1, name='propose_head', subgoal='get_view', effect='crossed',
                  view='unknown', button_vertical='unknown', button_horizontal='unknown', cid='recovery')
    mock_action(monkeypatch, bad+[good], calls)
    mock_review(monkeypatch, [review_answer(1, wrist_vertical='below', vertical_change='crossed')], reviews)
    proposal = client.propose('fake', c.body(), obs(1), SESSION, tmp_path, conversation=c)
    assert len(calls) == 5 and len(reviews) == 1
    assert proposal['name'] == 'propose_head'
    assert c.pending['tool_calls'][0]['id'] == 'recovery' and len(c.turns) == 1
    for attempt in range(5):
        archived = json.loads((tmp_path/f'response-{attempt}.json').read_text())['response']
        assert archived == (bad+[good])[attempt]
    for attempt in range(1, 5):
        assert json.loads((tmp_path/f'repair-request-{attempt}.json').read_text()) == calls[attempt]
        assert (tmp_path/f'validation-{attempt-1}.json').exists()
    assert json.loads((tmp_path/'repair-request.json').read_text()) == calls[-1]
    correction = json.loads((tmp_path/'correction.json').read_text())
    assert correction['status'] == 'accepted' and correction['accepted_attempt'] == 4
    metrics = json.loads((tmp_path/'action_metrics.json').read_text())
    assert metrics['attempts'] == 5 and metrics['validation_failures'] == 4 and metrics['status'] == 'validated'


def test_repeated_rejections_remain_cancellable_without_accept_or_execution(jpeg, tmp_path, monkeypatch):
    c = make(jpeg)
    advance(c, jpeg, 1)
    cancel = threading.Event()
    calls, reviews = [], []
    mock_review(monkeypatch, [review_answer(1, wrist_vertical='below', vertical_change='crossed')], reviews)
    def repeatedly_bad(base, route, body, **kwargs):
        assert c.pending is None and len(c.turns) == 1
        calls.append(copy.deepcopy(body))
        if len(calls) == 5:
            cancel.set()
        if len(calls) > 5:
            pytest.fail('must stop requesting after cancellation')
        return action(1, effect='crossed', button_vertical='below', cid=f'rejected-{len(calls)}')
    monkeypatch.setattr(client, 'request', repeatedly_bad)
    monkeypatch.setattr(client.RobotClient, 'execute', lambda *a, **k: pytest.fail('rejected proposal must not execute'))
    with pytest.raises(InterruptedError, match='operator_stopped'):
        client.propose('fake', c.body(), obs(1), SESSION, tmp_path, cancel=cancel, conversation=c)
    assert len(calls) == 5 and len(reviews) == 1
    assert c.pending is None and 'live-1' not in c.progress_records
    assert not (tmp_path/'proposal.json').exists() and not (tmp_path/'assistant.json').exists()
    assert json.loads((tmp_path/'action_metrics.json').read_text())['status'] == 'interrupted'


def test_forward_rejection_then_new_progress_review_reaches_third_informed_decision(jpeg, tmp_path, monkeypatch):
    """Regression for real R11: review produced after correction must reach the model."""
    from pressb.agent_loop import forward_guard
    from test_forward_guard_execution import robot
    c = make(jpeg)
    advance(c, jpeg, 1, xyz=(10, 0, 0))
    raw = [action(1, xyz=(10, 0, 0), subgoal='approach', button_horizontal='center', cid='forward'),
           action(1, xyz=(0, 0, 5), effect='unchanged', button_horizontal='center', cid='raise'),
           action(1, name='propose_head', subgoal='get_view', effect='unchanged',
                  view='unknown', button_vertical='unknown', button_horizontal='unknown', cid='head')]
    calls, reviews, guards = [], [], []
    mock_action(monkeypatch, raw, calls)
    mock_review(monkeypatch, [review_answer(1, wrist_horizontal='right', vertical_change='unchanged')], reviews)
    def deny_forward(base, body, oid, directory, cancel=None):
        guards.append(oid)
        return dict(format=forward_guard.FORMAT, source=forward_guard.SOURCE, observation_id=oid,
                    button_visible='yes', two_fingertips_visible='no', button_between_fingers='unknown')
    monkeypatch.setattr(forward_guard, 'inspect', deny_forward)
    proposal = client.propose('fake', c.body(), obs(1), SESSION, tmp_path/'decision', conversation=c)
    assert len(calls) == 3 and len(reviews) == 1 and guards == ['live-1']
    def current_reviews(body):
        found = []
        for message in body['messages']:
            content = message.get('content')
            if not isinstance(content, list):
                continue
            for item in content:
                if item.get('type') == 'text' and item.get('text', '').startswith('{'):
                    value = json.loads(item['text'])
                    if value.get('context_type') == 'DIRECT_PROGRESS_REVIEW':
                        found.append(value['assessment']['observation_id'])
        return found
    assert current_reviews(calls[0]) == [] and current_reviews(calls[1]) == []
    assert current_reviews(calls[2]) == ['live-1']
    assert proposal['name'] == 'propose_head' and c.pending['tool_calls'][0]['id'] == 'head'
    assert c.forward_guard_decision is None
    for attempt in range(3):
        assert json.loads((tmp_path/'decision'/f'response-{attempt}.json').read_text())['response'] == raw[attempt]
    for attempt in (1, 2):
        assert json.loads((tmp_path/'decision'/f'repair-request-{attempt}.json').read_text()) == calls[attempt]
    correction = json.loads((tmp_path/'decision/correction.json').read_text())
    assert correction['status'] == 'accepted' and correction['accepted_attempt'] == 2
    c.check_execution(proposal, 'live-1')
    executor, submitted = robot(tmp_path/'robot', monkeypatch)
    executor.execute(proposal, obs(1), forward_check=c.forward_guard_decision)
    assert len(submitted) == 1 and submitted[0]['type'] == 'head' and submitted[0]['head_deg'] == [0, 30]
    assert 'xyz_mm' not in submitted[0]


def test_unknown_review_is_valid_and_observation_recovery_does_not_replay_motion(jpeg, tmp_path, monkeypatch):
    c = make(jpeg)
    advance(c, jpeg, 1)
    mock_action(monkeypatch, [action(1, effect='unknown'), action(1, name='request_observation',
                effect='unknown', view='unknown', button_horizontal='unknown', button_vertical='unknown')])
    mock_review(monkeypatch, [review_answer(1, comparison_view='unknown', wrist_vertical='unknown',
                wrist_horizontal='unknown', vertical_change='unknown', horizontal_change='unknown')])
    p = client.propose('fake', c.body(), obs(1), SESSION, tmp_path, conversation=c)
    assert p['name'] == 'request_observation'
    c.append_result(p, None, obs(2), jpeg, jpeg)
    assert c.turns[-1]['ledger']['result'] == dict(status='observed', observation_id='live-2')
    assert c._last_progress_move()['result']['command']['xyz_mm'] == [0, 0, 10]
    assert len([t for t in c.turns if t['ledger']['result']['status'] == 'completed']) == 1
    # A fresh review after an observation refresh spans the actual movement and new frame.
    reviews = []
    mock_review(monkeypatch, [review_answer(2, before=0, wrist_vertical='level')], reviews)
    c._review_progress('fake', tmp_path/'next', None)
    assert c.progress_reviews['live-2']['previous_observation_id'] == 'live-0'
    assert image_count(reviews[0]['messages']) == 4
    actual = json.loads(reviews[0]['messages'][1]['content'][-1]['text'])
    assert actual['command'] == dict(type='move', xyz_mm=[0, 0, 10])


@pytest.mark.parametrize('during_prepare', [False, True])
def test_late_review_after_stop_never_accepts_action(jpeg, tmp_path, monkeypatch, during_prepare):
    c = make(jpeg)
    count = 3 if during_prepare else 1
    for n in range(1, count+1):
        advance(c, jpeg, n)
    cancel = threading.Event()
    action_calls = []
    mock_action(monkeypatch, [action(count, effect='unknown')], action_calls)
    def late(*args, **kwargs):
        cancel.set()
        return review_answer(count)
    monkeypatch.setattr(grounding, 'request', late)
    monkeypatch.setattr(conversation, 'tokenize', lambda *a, **k: 1000)
    with pytest.raises(InterruptedError, match='operator_stopped'):
        if during_prepare:
            c.prepare_decision('fake', tmp_path, cancel)
        else:
            client.propose('fake', c.body(), obs(count), SESSION, tmp_path, cancel=cancel, conversation=c)
    assert len(action_calls) == (0 if during_prepare else 1)
    assert c.pending is None and not c.progress_reviews and not c.progress_contexts
    assert not (tmp_path/'progress_review/assessment.json').exists()
    assert json.loads((tmp_path/'decision_context.json').read_text())['stage'] == 'interrupted'


@pytest.mark.parametrize('failure', ['transport', 'invalid_until_cancel'])
def test_review_error_or_cancellation_ends_before_action(jpeg, tmp_path, monkeypatch, failure):
    c = make(jpeg)
    for n in range(1, 4):
        advance(c, jpeg, n)
    cancel = threading.Event()
    calls = []
    def review_request(*args, **kwargs):
        calls.append(args)
        if failure == 'transport':
            raise TimeoutError('review timeout')
        if len(calls) == 4:
            cancel.set()
        if len(calls) > 4:
            pytest.fail('cancel must bound this intentionally malformed model')
        return response('propose_move')
    monkeypatch.setattr(grounding, 'request', review_request)
    monkeypatch.setattr(client, 'request', lambda *a, **k: pytest.fail('no action call after review failure'))
    exception, match = (RuntimeError, 'progress_review_failed') if failure == 'transport' else (InterruptedError, 'operator_stopped')
    with pytest.raises(exception, match=match):
        c.prepare_decision('fake', tmp_path, cancel)
    assert len(calls) == (1 if failure == 'transport' else 4)
    assert not c.progress_reviews and c.pending is None
    assert not (tmp_path/'repair-request.json').exists()
    assert json.loads((tmp_path/'error.json').read_text())['stage'] == 'progress_review'


def test_review_format_repair_remains_readonly_and_does_not_consume_action_retry(jpeg, tmp_path, monkeypatch):
    c = make(jpeg)
    for n in range(1, 4):
        advance(c, jpeg, n)
    reviews, actions = [], []
    mock_review(monkeypatch, [review_answer(3, before=0, observation_id='foreign')]*3+[review_answer(3)], reviews)
    monkeypatch.setattr(conversation, 'tokenize', lambda *a, **k: 3000)
    body = c.prepare_decision('fake', tmp_path)
    mock_action(monkeypatch, [action(3)], actions)
    client.propose('fake', body, obs(3), SESSION, tmp_path, conversation=c)
    assert len(reviews) == 4 and len(actions) == 1
    assert (tmp_path/'progress_review/repair-request.json').exists()
    assert not (tmp_path/'repair-request.json').exists()
    assert all(r['tools'][0]['function']['name'] == 'review_progress' for r in reviews)


def test_no_motion_cannot_describe_a_completed_hand_command(jpeg, tmp_path):
    c = make(jpeg)
    advance(c, jpeg, 1)
    with pytest.raises(ValueError, match='no_motion_requires_observation_only'):
        c.parse_response(action(1, effect='no_motion'), obs(1), SESSION, tmp_path)
    advance(c, jpeg, 2, name='request_observation')
    p = c.parse_response(action(2, effect='no_motion'), obs(2), SESSION, tmp_path)
    assert p['arguments']['xyz_mm'] == [0, 0, 10]


def test_execution_binding_rejects_changed_observation_and_mutated_command(jpeg, tmp_path, monkeypatch):
    c = make(jpeg)
    mock_action(monkeypatch, [action(0)])
    proposal = client.propose('fake', c.body(), obs(0), SESSION, tmp_path, conversation=c)
    c.check_execution(proposal, 'live-0')
    with pytest.raises(ValueError, match='observation_mismatch'):
        c.check_execution(proposal, 'live-elsewhere')
    changed = copy.deepcopy(proposal)
    changed['arguments']['xyz_mm'][2] = 20
    with pytest.raises(ValueError, match='accepted_progress_proposal_mismatch'):
        c.check_execution(changed, 'live-0')


def test_compaction_retains_progress_ledger_recent_reviews_and_current_review(jpeg, tmp_path, monkeypatch):
    c = make(jpeg)
    for n in range(1, 9):
        advance(c, jpeg, n)
        if n in (1, 6, 8):
            mock_review(monkeypatch, [review_answer(n)])
            c._review_progress('fake', tmp_path/f'review-{n}', None)
    original = copy.deepcopy(c.turns)
    monkeypatch.setattr(conversation, 'request', lambda *a, **k: summary_response())
    c.compact('fake', tmp_path/'compaction')
    assert c.turns == original and c.start == 3
    ledger = json.loads((tmp_path/'compaction/result.json').read_text())['earlier_action_ledger']
    assert all(item['progress']['source'] == 'model_not_human' for item in ledger)
    assert ledger[1]['progress_review']['observation_id'] == 'live-1'
    messages = c.messages()
    reviews = [json.loads(m['content'][0]['text']) for m in messages
               if isinstance(m.get('content'), list) and m['content'][0].get('type') == 'text'
               and 'DIRECT_PROGRESS_REVIEW' in m['content'][0]['text']]
    assert {r['assessment']['observation_id'] for r in reviews} == {'live-6', 'live-8'}
    assert len([m for m in messages if m['role'] == 'assistant']) == 5
    assert 'progress' in json.loads([m for m in messages if m['role'] == 'assistant'][0]['tool_calls'][0]['function']['arguments'])
    assert c.last_review_round == 8 and c.progress_records['live-7']['observation_id'] == 'live-7'


def test_second_prepare_same_observation_does_not_repeat_independent_review(jpeg, tmp_path, monkeypatch):
    c = make(jpeg)
    for n in range(1, 4):
        advance(c, jpeg, n)
    calls = []
    mock_review(monkeypatch, [review_answer(3)], calls)
    monkeypatch.setattr(conversation, 'tokenize', lambda *a, **k: 3000)
    first = c.prepare_decision('fake', tmp_path)
    again = c.prepare_decision('fake', tmp_path)
    assert len(calls) == 1 and first == again
    assert len(c.progress_contexts) == 1


@pytest.mark.parametrize('recovery', ['request_observation', 'zero_move'])
def test_refresh_or_zero_move_cannot_clear_unresolved_direction(jpeg, tmp_path, monkeypatch, recovery):
    c = make(jpeg)
    advance(c, jpeg, 1)
    name = 'request_observation' if recovery == 'request_observation' else 'propose_move'
    reply = action(1, name=name, xyz=(0, 0, 0), effect='unknown',
                   view='unknown', button_horizontal='unknown', button_vertical='unknown')
    # No conflicting relation here: this isolates unresolved progress rather
    # than the separate fresh-view recovery path for model disagreement.
    mock_action(monkeypatch, [action(1, effect='unknown', button_vertical='level'), reply])
    mock_review(monkeypatch, [review_answer(1, wrist_vertical='level')])
    proposal = client.propose('fake', c.body(), obs(1), SESSION, tmp_path/'first', conversation=c)
    result = None if recovery == 'request_observation' else dict(status='completed',
        request=dict(type='move', xyz_mm=[0, 0, 0], command_id='zero'),
        motion=dict(action_completed=True), capture=dict(observation_id='live-2'))
    c.append_result(proposal, result, obs(2), jpeg, jpeg)
    assert c.progress_unresolved == {'Z': 1}
    reviews = []
    mock_review(monkeypatch, [review_answer(2, before=0, wrist_vertical='level')], reviews)
    monkeypatch.setattr(conversation, 'tokenize', lambda *a, **k: 3000)
    body = c.prepare_decision('fake', tmp_path/'next')
    assert len(reviews) == 1 and image_count(reviews[0]['messages']) == 4
    assert c.progress_reviews['live-2']['previous_observation_id'] == 'live-0'
    assert c.progress_unresolved == {'Z': 1}
    trigger = json.loads((tmp_path/'next/progress_review/trigger.json').read_text())
    assert 'unresolved_progress_requires_fresh_review' in trigger['monitor']['reasons']
    # Even a claim that the new frame is "above" cannot resume the rejected +Z.
    calls = []
    effect = 'no_motion' if recovery == 'request_observation' else 'improved'
    bad = action(2, effect=effect)
    good = action(2, name='report_unable', effect=effect, view='unknown',
                  button_vertical='unknown', button_horizontal='unknown', cid='unable')
    mock_action(monkeypatch, [bad]*4+[good], calls)
    proposal = client.propose('fake', body, obs(2), SESSION, tmp_path/'next', conversation=c)
    assert len(calls) == 5 and proposal['name'] == 'report_unable'
    assert c.pending['tool_calls'][0]['id'] == 'unable'
    assert len(reviews) == 1  # Reuse this round's review; never replay a command to collect evidence.


def test_execution_rechecks_review_observation_binding(jpeg, tmp_path, monkeypatch):
    c = make(jpeg)
    for n in range(1, 4):
        advance(c, jpeg, n)
    mock_review(monkeypatch, [review_answer(3)])
    monkeypatch.setattr(conversation, 'tokenize', lambda *a, **k: 3000)
    body = c.prepare_decision('fake', tmp_path)
    mock_action(monkeypatch, [action(3)])
    proposal = client.propose('fake', body, obs(3), SESSION, tmp_path, conversation=c)
    c.progress_reviews['live-3']['previous_observation_id'] = 'foreign'
    with pytest.raises(ValueError, match='progress_review_observation_mismatch'):
        c.check_execution(proposal, 'live-3')

