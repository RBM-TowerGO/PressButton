"""Archived-frame replay must distinguish proposed from historically executed actions."""
import copy
import json
from pathlib import Path

import pytest

import validate_agent_localized as replay


def test_saved_feedback_is_original_actual_command_not_new_proposal():
    recorded = dict(status='completed', request=dict(command_id='old-command', type='move', xyz_mm=[0, 0, 10]),
                    motion=dict(action_completed=True), capture=dict(observation_id='old-after'))
    original = copy.deepcopy(recorded)
    proposal = dict(name='propose_move', arguments=dict(xyz_mm=[0, -15, 0], reason='new suggestion'))
    result = replay.replay_feedback(recorded, proposal)
    assert recorded == original
    assert result['request'] == original['request'] and result['motion'] == original['motion']
    assert result['replay']['recorded_actual_command'] == dict(type='move', xyz_mm=[0, 0, 10])
    assert result['replay']['proposal_executed'] is False
    assert result['replay']['saved_response_images_not_result_of_new_proposal'] is True
    assert result['capture']['observation_id'] == 'old-after'
    assert not result['replay']['proposed_action_was_terminal']


def test_prepare_only_preserves_source_images_states_and_contiguous_rounds(tmp_path, monkeypatch):
    # Preparation must not enter execution, construct a robot, or open a model tunnel.
    monkeypatch.setattr(replay, 'execute', lambda *args: (_ for _ in ()).throw(AssertionError('no execution')))
    output = tmp_path/'prepared'
    replay.main(['--output', str(output), '--prepare-only'])
    plan = json.loads((output/'plan.json').read_text())
    assert plan['status'] == 'prepared'
    assert plan['format'] == 'pressb-localized-replay-v2'
    assert plan['input_format'] == 'pressb-conversation-v5'
    assert plan['localization_format'] == 'pressb-localization-v3'
    assert plan['execution_policy'] == 'yz_alignment_before_forward' and plan['forward_guard_policy'] == 'wrist_tip_overlap'
    assert plan['robot_commands_sent'] == 0 and plan['new_proposals_executed'] == 0
    assert [s['label'] for s in plan['sequences']] == ['A', 'B']
    assert [[r['number'] for r in s['rounds']] for s in plan['sequences']] == [[6, 7, 8], [5]]
    assert not (output/'report.json').exists()
    assert plan['demo_selection']['mode'] == 'all'
    assert plan['demo_selection']['selected_steps'] == [f'{step:06d}' for step in range(1,12)]
    assert plan['demo_selection']['model_image_count'] == 11
    hashes = json.loads((output/'source_hashes.json').read_text())
    assert all(replay.sha(path) == digest for path, digest in hashes.items())
    for sequence in plan['sequences']:
        source = Path(sequence['source_run'])
        for index, row in enumerate(sequence['rounds']):
            saved = output/sequence['label']/'observations'/row['observation_id']
            for name in ('manifest.json', 'state.json', 'head.jpg', 'right.jpg', 'right_depth_raw.npy'):
                assert (saved/name).read_bytes() == (source/'observations'/row['observation_id']/name).read_bytes()
            if row['after_observation_id']:
                assert row['after_observation_id'] == sequence['rounds'][index+1]['observation_id']


@pytest.mark.parametrize('tail', [None, 3])
def test_prepared_selection_preserves_full_or_historical_tail(tail):
    demo=replay.Demonstration(replay.ROOT/'outputs/keyboard_sessions/20261101_182424_352b5047')
    if tail is not None:
        demo.select_activation_tail(tail)
    plan=dict(demo=str(demo.root),demo_selection=copy.deepcopy(demo.selection))
    restored=replay.load_prepared_demo(plan)
    assert restored.selection==demo.selection
    assert restored.model_summary==demo.model_summary
    plan['demo_selection']['selected_steps']=[]
    with pytest.raises(ValueError,match='prepared_demo_selection_mismatch'):
        replay.load_prepared_demo(plan)


def test_execute_prepared_reuses_demo_archive_before_model_boundary(tmp_path, monkeypatch):
    from types import SimpleNamespace
    output = tmp_path/'prepared'
    replay.prepare(output, replay.ROOT/'outputs/keyboard_sessions/20261101_182424_352b5047')
    reference = (output/'demonstration/reference.json').read_bytes()
    def forbidden_save(*args, **kwargs):
        pytest.fail('prepared demonstration must not be saved into its existing directory again')
    def boundary(*args, **kwargs):
        raise RuntimeError('model-boundary-no-request-sent')
    monkeypatch.setattr(replay.Demonstration, 'save', forbidden_save)
    monkeypatch.setattr(replay, 'tunnel', boundary)
    with pytest.raises(RuntimeError, match='model-boundary-no-request-sent'):
        replay.execute(SimpleNamespace(output=output, model_base=None, thor_host='unused', port=18026))
    assert (output/'demonstration/reference.json').read_bytes() == reference
    report = json.loads((output/'report.json').read_text())
    assert report['source_hashes_unchanged'] and report['model_stage_totals']['requests'] == 0
    assert report['new_proposals_executed'] == 0


def test_overlay_keeps_raw_reversed_or_off_image_coordinates_as_model_estimates(tmp_path):
    from PIL import Image
    source = tmp_path/'source.png'
    Image.new('RGB', (100, 60), 'black').save(source)
    original = source.read_bytes()
    assessment = dict(button_box=[.8, .8, .2, .2], arrow_box=[-.1, -.1, 1.1, 1.1],
                      contact_tip=[1.2, -.2], derived_relation=dict(dx=-.7, dy=.7))
    before = copy.deepcopy(assessment)
    target = tmp_path/'overlay.png'
    replay.overlay(source, target, assessment, 'HEAD')
    assert assessment == before and source.read_bytes() == original
    with Image.open(target) as rendered:
        assert rendered.size == (100, 82)
