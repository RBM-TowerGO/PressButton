"""Direct-loop progress evidence, anomaly review and no-motion recovery contracts."""
import copy
import json
import threading

import pytest

from pressb.agent_loop import grounding, progress
from pressb.agent_loop.client import TOOLS
from pressb.agent_loop.conversation import image_count, live_message
from test_conversation import jpeg, obs, response


def values(oid='live-3', **changes):
    result = dict(observation_id=oid, subgoal='align_vertical', view='wrist',
                  button_vertical='above', button_horizontal='left', last_effect='improved')
    result.update(changes)
    return result


def reviewed(oid='live-3', before='live-2', **changes):
    result = dict(observation_id=oid, previous_observation_id=before, comparison_view='wrist',
                  head_vertical='unknown', head_horizontal='unknown', wrist_vertical='above',
                  wrist_horizontal='left', vertical_change='improved', horizontal_change='unchanged',
                  approach_change='unknown', note='The button center is still above the marked contact end.')
    result.update(changes)
    return result


def review_response(**changes):
    return response('review_progress', reviewed(**changes))


def move(vector=(0, 0, 10)):
    return dict(name='propose_move', arguments=dict(xyz_mm=list(vector), reason='rationale'))


def turn(n, vector=(0, 0, 10), **result_changes):
    command = dict(type='move', xyz_mm=list(vector))
    result = dict(status='completed', sends_completed=True, command=command,
                  command_id=f'cmd-{n}', observation_id=f'live-{n}')
    result.update(result_changes)
    return dict(ledger=dict(round=n, before_observation_id=f'live-{n-1}',
        after_observation_id=f'live-{n}', proposal=move(vector), progress=values(f'live-{n-1}'), result=result))


def test_tools_extend_without_changing_action_schema_or_mutating_original():
    old = copy.deepcopy(TOOLS)
    tools = progress.extend_tools(TOOLS)
    assert TOOLS == old
    for new, original in zip(tools, old):
        parameters = new['function']['parameters']
        assert parameters['required'] == original['function']['parameters']['required'] + ['progress']
        assert parameters['properties']['progress']['additionalProperties'] is False
        assert set(parameters['properties']['progress']['properties']) == set(progress.PROGRESS_PROPERTIES)


def test_unknown_is_valid_and_bound_to_current_observation():
    data = progress.validate_progress(values(view='unknown', button_vertical='unknown',
        button_horizontal='unknown', last_effect='unknown'), 'live-3', True)
    assert data['source'] == 'model_not_human' and data['format'] == progress.FORMAT
    with pytest.raises(ValueError, match='observation_mismatch'):
        progress.validate_progress(values(), 'other', True)
    with pytest.raises(ValueError, match='requires_usable_view'):
        progress.validate_progress(values(view='unknown'), 'live-3', True)


@pytest.mark.parametrize('changes,previous', [({'last_effect':'first'}, True),
    ({'last_effect':'improved'}, False), ({'button_vertical':'up'}, True),
    ({'view':'left_camera'}, True), ({'subgoal':'move'}, True), ({'extra':'field'}, True)])
def test_invalid_progress_rejected(changes, previous):
    with pytest.raises(ValueError):
        progress.validate_progress(values(**changes), 'live-3', previous)


def test_three_completed_same_direction_commands_trigger_review_not_failure():
    turns = [turn(n) for n in range(1, 4)]
    result = progress.monitor(turns, 'live-3')
    assert result['review_required'] and result['reasons'] == ['repeated_completed_Z']
    assert result['streaks']['Z'] == dict(count=3, direction=1, total_mm=30.)
    assert result['last_command'] == dict(type='move', xyz_mm=[0, 0, 10])
    assert result['last_move_before_observation_id'] == 'live-2'
    assert result['last_move_round'] == 3
    assert not progress.monitor(turns, 'live-3', since_round=3)['review_required']
    result = progress.monitor(turns+[turn(4)], 'live-4', since_round=3)
    assert result['streaks']['Z']['count'] == 1
    assert result['last_command'] == dict(type='move', xyz_mm=[0, 0, 10])


def test_observe_does_not_count_or_erase_streak_and_unsent_moves_do_not_count():
    turns = [turn(1), turn(2, sends_completed=False),
             dict(ledger=dict(round=3, result=dict(status='observed', observation_id='live-3'))),
             turn(4, status='failed'), turn(5), turn(6)]
    assert progress.monitor(turns, 'live-6')['streaks']['Z']['count'] == 3
    assert not progress.monitor([turn(1, status='rejected')], 'live-1')['review_required']


def test_head_or_gripper_changes_end_comparable_segment():
    for kind, command in [('head', dict(type='head', head_deg=[0, 30])), ('gripper', dict(type='gripper', key='c'))]:
        turns = [turn(1), turn(2), turn(3), turn(4, command=command), turn(5)]
        result = progress.monitor(turns, 'live-5')
        assert not result['review_required'] and result['streaks']['Z']['count'] == 1


def test_different_axis_or_direction_starts_new_streak_and_xyz_is_preserved():
    turns = [turn(1, (0, 10, 10)), turn(2, (0, 10, 10)), turn(3, (0, -10, 0))]
    result = progress.monitor(turns, 'live-3')
    assert result['streaks']['Y'] == dict(count=1, direction=-1, total_mm=-10.)
    assert result['streaks']['Z']['count'] == 0


@pytest.mark.parametrize('effect', ['unchanged', 'worsened', 'crossed', 'unknown'])
def test_adverse_candidate_effect_triggers_before_three_moves(effect):
    result = progress.monitor([turn(1)], 'live-1', values('live-1', last_effect=effect), move())
    assert result['review_required'] and 'reported_effect_'+effect in result['reasons']
    recovery = dict(name='request_observation', arguments=dict(reason='need fresh view'))
    assert not progress.monitor([turn(1)], 'live-1', values('live-1', last_effect=effect), recovery)['review_required']


def test_reached_and_crossed_relationship_override_claimed_improvement():
    for relation, expected in [('level', 'alignment_target_reached'), ('below', 'relation_crossed')]:
        result = progress.monitor([turn(1)], 'live-1', values('live-1', button_vertical=relation), move())
        assert any(expected in reason for reason in result['reasons'])
    # An actual new-axis proposal is not an unsupported Z repeat.
    result = progress.monitor([turn(1)], 'live-1', values('live-1', button_vertical='level'), move((0, -10, 0)))
    assert not result['review_required']


def test_review_input_is_independent_four_images_and_excludes_rationales(jpeg):
    before, current = live_message(obs(2), jpeg, jpeg), live_message(obs(3), jpeg, jpeg)
    before['content'][0]['text'] = json.dumps(dict(observation_id='live-2', reason='SECRET_REASON',
        state='SECRET_STATE', assessment='SECRET_OLD_ASSESSMENT', demo='SECRET_DEMO'))
    command_result = dict(turn(3)['ledger']['result'], reason='SECRET_REASON', state_trace='SECRET_TRACE')
    command_result['command']['reason'] = 'SECRET_REASON'
    body = progress.build_review_request('press down-call', before, current, command_result)
    assert image_count(body['messages']) == 4
    assert 'SECRET' not in json.dumps(body)
    assert 'BEFORE' in json.dumps(body) and 'CURRENT' in json.dumps(body)
    assert body['temperature'] == 0 and body['max_tokens'] == 512
    assert body['chat_template_kwargs'] == dict(enable_thinking=False)
    assert body['tools'][0]['function']['name'] == 'review_progress'
    assert 'WHITE ARROW' in body['messages'][0]['content']
    for reference, needle in [('red_marker', 'RED/DARK-RED'), ('selected_fingertip', 'same selected')]:
        assert needle in progress.build_review_request('press', before, current, command_result, reference)['messages'][0]['content']


def test_review_validation_rejects_bad_ids_and_noncomparable_claims():
    result = progress.validate_review(review_response(), 'live-3', 'live-2')
    assert result['source'] == progress.SOURCE
    assert result['evidence_observation_ids'] == ['live-2', 'live-3']
    with pytest.raises(ValueError, match='observation_mismatch'):
        progress.validate_review(review_response(oid='demo'), 'live-3', 'live-2')
    with pytest.raises(ValueError, match='comparable_view'):
        progress.validate_review(review_response(comparison_view='unknown'), 'live-3', 'live-2')
    with pytest.raises(ValueError, match='identifiable_current'):
        progress.validate_review(review_response(wrist_vertical='unknown'), 'live-3', 'live-2')
    data = reviewed(comparison_view='unknown', vertical_change='unknown', horizontal_change='unknown', approach_change='unknown')
    assert progress.validate_review(response('review_progress', data), 'live-3', 'live-2')['comparison_view'] == 'unknown'


@pytest.mark.parametrize('effect', ['unchanged', 'worsened', 'crossed', 'unknown', 'no_motion'])
def test_review_blocks_unsupported_same_direction_but_never_generates_or_changes_action(effect):
    review = progress.validate_review(review_response(vertical_change=effect), 'live-3', 'live-2')
    candidate = move(); original = copy.deepcopy(candidate)
    last = dict(type='move', xyz_mm=[0, 0, 10])
    assert progress.continuation_errors(review, candidate, last)
    assert candidate == original
    assert not progress.continuation_errors(review, move((0, -10, 0)), last)
    assert not progress.continuation_errors(review, move((0, 0, -10)), last)
    assert not progress.continuation_errors(review, dict(name='report_unable'), last)


def test_improved_allows_repeat_until_level_reached_and_x_has_own_evidence():
    review = progress.validate_review(review_response(), 'live-3', 'live-2')
    last = dict(type='move', xyz_mm=[0, 0, 10])
    assert not progress.continuation_errors(review, move(), last)
    review['wrist_vertical'] = 'level'
    assert progress.continuation_errors(review, move(), last)
    # Same row never authorizes repeated approach, whose evidence is independent.
    assert progress.continuation_errors(review, move((10, 0, 0)), dict(type='move', xyz_mm=[10, 0, 0]))
    review['approach_change'] = 'improved'
    assert not progress.continuation_errors(review, move((10, 0, 0)), dict(type='move', xyz_mm=[10, 0, 0]))


def body(jpeg):
    return progress.build_review_request('press', live_message(obs(2), jpeg, jpeg),
                                         live_message(obs(3), jpeg, jpeg), turn(3)['ledger']['result'])


def test_inspect_repairs_once_and_records_metrics(jpeg, tmp_path, monkeypatch):
    replies = iter([review_response(oid='invented'), review_response()]); calls = []
    monkeypatch.setattr(grounding, 'request', lambda *a, **k: calls.append(k) or next(replies))
    result = progress.inspect('fake', body(jpeg), 'live-3', 'live-2', tmp_path)
    assert result['observation_id'] == 'live-3' and len(calls) == 2
    assert all(row['timeout'] == 120 for row in calls)
    assert (tmp_path/'repair-request.json').exists()
    metrics = json.loads((tmp_path/'metrics.json').read_text())
    assert metrics['image_count'] == 4 and metrics['attempts'] == 2


def test_repeated_invalid_review_continues_until_valid(jpeg, tmp_path, monkeypatch):
    calls = []
    replies = iter([review_response(oid='bad') for _ in range(3)]+[review_response()])
    monkeypatch.setattr(grounding, 'request', lambda *a, **k: calls.append(a[2]) or next(replies))
    assert progress.inspect('fake', body(jpeg), 'live-3', 'live-2', tmp_path)['observation_id'] == 'live-3'
    assert len(calls) == 4 and (tmp_path/'assessment.json').exists()
    assert all(len(payload['messages']) == len(calls[0]['messages'])+1 for payload in calls[1:])
    assert json.loads((tmp_path/'metrics.json').read_text())['validation_failures'] == 3


@pytest.mark.parametrize('when', ['before', 'after'])
def test_cancellation_never_accepts_late_review(jpeg, tmp_path, monkeypatch, when):
    cancel = threading.Event(); calls = []
    if when == 'before':
        cancel.set()
    def late(*a, **kw):
        calls.append(a); cancel.set(); return review_response()
    monkeypatch.setattr(grounding, 'request', late)
    with pytest.raises(InterruptedError):
        progress.inspect('fake', body(jpeg), 'live-3', 'live-2', tmp_path, cancel)
    assert len(calls) == (0 if when == 'before' else 1)
    assert not (tmp_path/'assessment.json').exists()


def test_bad_request_binding_rejected_before_network(jpeg, tmp_path, monkeypatch):
    monkeypatch.setattr(grounding, 'request', lambda *a, **k: pytest.fail('must not call model'))
    with pytest.raises(ValueError, match='request_observation_mismatch'):
        progress.inspect('fake', body(jpeg), 'different', 'live-2', tmp_path)
    modified = body(jpeg); modified['messages'][1]['content'].pop(-2)
    with pytest.raises(ValueError, match='request_observation_mismatch'):
        progress.inspect('fake', modified, 'live-3', 'live-2', tmp_path)


def test_duplicate_command_receipts_count_once_even_with_distinct_observation_ids():
    turns = [turn(1), turn(2, command_id='cmd-1'), turn(3)]
    result = progress.monitor(turns, 'live-3')
    assert result['streaks']['Z']['count'] == 2
    assert not result['review_required']


@pytest.mark.parametrize('mutate', [
    lambda r: r['choices'][0].update(message=None),
    lambda r: r['choices'][0]['message'].update(content={'unexpected': 'object'}),
    lambda r: r['choices'][0]['message'].update(tool_calls=[None]),
    lambda r: r['choices'][0]['message']['tool_calls'][0].update(function=None),
])
def test_malformed_review_structure_uses_single_repair(jpeg, tmp_path, monkeypatch, mutate):
    bad = review_response(); mutate(bad)
    replies = iter([bad, review_response()]); calls = []
    monkeypatch.setattr(grounding, 'request', lambda *a, **kw: calls.append(a) or next(replies))
    assert progress.inspect('fake', body(jpeg), 'live-3', 'live-2', tmp_path)['observation_id'] == 'live-3'
    assert len(calls) == 2 and (tmp_path/'repair-request.json').exists()
