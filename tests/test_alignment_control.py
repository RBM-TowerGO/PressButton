"""Observed image relations and executed axis activity must remain distinct."""
import copy
import json

from pressb.agent_loop import client
from test_conversation import jpeg
from test_grounding import shifted, unknown
from test_short_loop import SESSION, action, make, mock_inspections, commit, result, state


def above(**kw):
    return dict(shifted(), head_vertical='above', **kw)


def prepare_sequence(jpeg, tmp_path, monkeypatch, commands, assessments=None):
    c = make(jpeg)
    mock_inspections(monkeypatch, assessments or [above()] * (len(commands) + 1))
    for n in range(len(commands) + 1):
        c.prepare_decision('fake', tmp_path / str(n))
        if n < len(commands):
            xyz = commands[n]
            p, _ = commit(c, tmp_path / str(n), args=dict(xyz_mm=xyz, reason='Inspect the remaining offset.'))
            c.append_result(p, result(n + 1, xyz), state(n + 1), jpeg, jpeg)
    return c


def test_subject_relations_are_inverted_without_new_evidence_or_command(jpeg, tmp_path, monkeypatch):
    c = prepare_sequence(jpeg, tmp_path, monkeypatch, [])
    control = c.memory['alignment_control']
    assert control['axes']['vertical']['button_relative_to_finger'] == 'above'
    assert control['axes']['vertical']['finger_relative_to_button'] == 'below'
    assert control['axes']['horizontal']['finger_relative_to_button'] == 'left'
    assert control['attention_axis'] == 'vertical'
    assert 'not_new_visual_evidence' in control['source']
    assert c.pending is None and not c.turns
    model_memory = json.loads(c.messages()[2]['content'][0]['text'])
    assert 'alignment_control' not in model_memory
    assert 'alignment_progress' not in model_memory
    assert 'control_plan' in model_memory


def test_repeated_forward_and_lateral_commands_expose_unattended_height(jpeg, tmp_path, monkeypatch):
    c = prepare_sequence(jpeg, tmp_path, monkeypatch, [[30, 0, 0], [0, -10, 0], [10, 0, 0]])
    control = c.memory['alignment_control']
    h, v = control['axes']['horizontal'], control['axes']['vertical']
    assert h['unresolved_observations'] == v['unresolved_observations'] == 4
    assert h['transitions_without_axis_command'] == 1
    assert v['transitions_without_axis_command'] == 3
    assert v['recent_nonzero_axis_commands'] == []
    assert h['recent_nonzero_axis_commands'][0]['delta_mm'] == -10
    assert control['attention_axis'] == 'vertical'
    old = copy.deepcopy(control)
    c.prepare_decision('fake', tmp_path / 'repeat')
    assert c.memory['alignment_control'] == old


def test_axis_activity_is_not_labelled_as_correcting_error(jpeg, tmp_path, monkeypatch):
    c = prepare_sequence(jpeg, tmp_path, monkeypatch, [[0, -10, -10], [0, 10, 0], [0, -10, 0]])
    axes = c.memory['alignment_control']['axes']
    assert axes['vertical']['recent_nonzero_axis_commands'][0]['delta_mm'] == -10
    assert axes['vertical']['unresolved_observations'] == 4
    assert axes['vertical']['transitions_without_axis_command'] == 2
    assert axes['horizontal']['axis_direction_reversals'] == 2
    assert 'not calibrated correction' in c.memory['alignment_control']['limitation']


def test_unknown_relation_breaks_a_streak_and_has_no_attention_axis(jpeg, tmp_path, monkeypatch):
    c = prepare_sequence(jpeg, tmp_path, monkeypatch, [[10, 0, 0], [5, 0, 0]], [above(), unknown(), above()])
    assert c.memory['alignment_control']['axes']['vertical']['unresolved_observations'] == 1
    unknown_control = c._alignment_control(unknown())
    assert unknown_control['attention_axis'] is None
    assert unknown_control['axes']['vertical']['finger_relative_to_button'] == 'unknown'


def test_head_pose_change_resets_comparison(jpeg, tmp_path, monkeypatch):
    c = make(jpeg)
    mock_inspections(monkeypatch, [above(), above()])
    c.prepare_decision('fake', tmp_path / '0')
    p, _ = commit(c, tmp_path / '0', 'propose_head', dict(head_deg=[0, 30], reason='Improve framing.'), 'improve_head_view')
    feedback = dict(status='completed', request=dict(type='head', head_deg=[0, 30], command_id='head'),
                    capture=dict(observation_id='live-1'))
    c.append_result(p, feedback, state(1, head=(0, 30)), jpeg, jpeg)
    c.prepare_decision('fake', tmp_path / '1')
    assert c.memory['alignment_control']['axes']['vertical']['unresolved_observations'] == 1


def test_saved_replay_counts_actual_axes_not_new_suggestion(jpeg, tmp_path, monkeypatch):
    c = make(jpeg)
    mock_inspections(monkeypatch, [above(), above()])
    c.prepare_decision('fake', tmp_path / '0')
    p, _ = commit(c, tmp_path / '0', args=dict(xyz_mm=[0, 0, 12.1234567], reason='Raise the hand.'))
    feedback = result(1, p['arguments']['xyz_mm'], replay=dict(recorded_actual_command=dict(type='move', xyz_mm=[10, 0, 0])))
    feedback['status'] = 'not_executed_replay'
    c.append_result(p, feedback, state(1), jpeg, jpeg)
    c.prepare_decision('fake', tmp_path / '1')
    vertical = c.memory['alignment_control']['axes']['vertical']
    assert vertical['transitions_without_axis_command'] == 1
    assert vertical['recent_nonzero_axis_commands'] == []
    assert p['arguments']['xyz_mm'] == [0, 0, 12.1234567]


def test_attention_is_not_a_forward_prohibition(jpeg, tmp_path, monkeypatch):
    c = prepare_sequence(jpeg, tmp_path, monkeypatch, [[10, 0, 0]])
    assert c.memory['alignment_control']['attention_axis'] == 'vertical'
    p, _ = commit(c, tmp_path, args=dict(xyz_mm=[1.1234567, 0, 0], reason='Forward probe; height remains unresolved.'))
    assert p['arguments']['xyz_mm'] == [1.1234567, 0, 0]


def test_missing_actual_replay_command_is_not_counted_as_no_movement(jpeg, tmp_path, monkeypatch):
    c = prepare_sequence(jpeg, tmp_path, monkeypatch, [[10, 0, 0]])
    c.turns[-1]['result']['replay'] = dict(note='actual command missing')
    control = c._alignment_control(c.assessments[c.current_id])
    assert control['axes']['vertical']['unresolved_observations'] == 1


def test_recorded_wrong_subject_relation_is_repaired_before_accept(jpeg, tmp_path, monkeypatch):
    c = prepare_sequence(jpeg, tmp_path, monkeypatch, [])
    bad = action(c, args=dict(xyz_mm=[0, -10, -10], reason=(
        'HEAD sees the down button at right/below the open right finger; '
        'move right and down to align the right finger with the button face before pressing.')))
    good = action(c, args=dict(xyz_mm=[0, -10, 10], reason=(
        'HEAD shows the button above the right finger; raise the hand to address height.')))
    replies = iter([bad, good]); requests = []
    monkeypatch.setattr(client, 'request', lambda *a, **k: requests.append(a[2]) or next(replies))
    p = client.propose('fake', c.body(), state(0), SESSION, tmp_path / 'action', conversation=c)
    assert len(requests) == 2
    assert p['arguments']['xyz_mm'] == [0, -10, 10]
    assert 'reason_conflicts_with_current_head_vertical:above' in requests[-1]['messages'][-1]['content']
    assert not c.turns  # Accepting a proposal does not execute it.


def test_recorded_negative_visibility_can_request_observation(jpeg, tmp_path, monkeypatch):
    c = prepare_sequence(jpeg, tmp_path, monkeypatch, [], [unknown()])
    reason = '当前头部与右腕画面均未清楚看到目标按钮，需刷新视图以确认按钮位置。'
    p, _ = commit(c, tmp_path, 'request_observation', dict(reason=reason), 'observe')
    assert p['arguments']['reason'] == reason
