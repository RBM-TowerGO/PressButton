"""New advisory localization archives stay readable alongside strict v4 logs."""
import json

import pytest

from pressb.agent_loop import model_localization
from pressb.agent_loop.common import atomic_json
from pressb.agent_loop.demonstration import image_content
from pressb.agent_loop.failure_case import FailureCase
from test_failure_case import case, snapshot


def localized(case, raw=None, raw_output=None, with_action=True, with_assessment=True):
    folder = case/'agent/rounds/0011'
    atomic_json(folder/'decision_context.json', dict(format='pressb-conversation-v5',
        decision_mode='localized', observation_id='obs11', previous_observation_id='obs10',
        stage='localization', localization_format=model_localization.FORMAT,
        localization_revision=1, progress_monitor_format='pressb-localized-progress-v3',
        execution_policy='model_decides_after_progress_review', forward_guard_policy='none', view_phase='model_decides'))
    atomic_json(folder/'localization/request.json', dict(messages=[dict(role='user', content=[
        dict(type='text', text=json.dumps(dict(frame='CURRENT', observation_id='obs11'))),
        *[image_content((case/'agent/observations/obs11'/(view+'.jpg')).read_bytes())
          for view in ('head', 'right')]])]))
    if with_assessment:
        if isinstance(raw, dict):
            raw = dict(coordinate_units='normalized_01', **raw)
        value = model_localization.normalize(raw, 'obs11', dict(head=[64, 36], wrist=[64, 36]),
                                             raw_output=raw_output)
        atomic_json(folder/'localization/assessment.json', value)
        atomic_json(folder/'localization_current.json', value)
    if not with_action:
        (folder/'request.json').unlink()
        (folder/'error.json').unlink()
    return folder


def positions():
    return dict(button_visibility='partial', contact_visibility='unknown',
                button_box=[.2, .2, .4, .4], arrow_box=None, contact_tip=[.65, .6],
                note='Partial face; contact identity uncertain.')


def test_partial_unknown_coordinates_remain_advisory_with_full_action_archive(case):
    folder = localized(case, dict(head=positions(), wrist=positions()))
    atomic_json(folder/'localized_progress.json', dict(format='pressb-localized-progress-v2',
        previous_observation_id='obs10', observation_id='obs11', raw_changes=dict(wrist=[.001, -.003])))
    original = snapshot(case)
    failure = FailureCase(case)
    row = failure.summary['rounds'][-1]
    summary = row['localization_summary']
    assert summary['execution_policy'] == 'model_decides_after_progress_review'
    assert summary['forward_guard_policy'] == 'none'
    assert summary['views']['wrist']['derived_relation']['dx'] == pytest.approx(-.35)
    assert summary['views']['wrist']['button_visibility'] == 'partial'
    assert 'usable_views' not in summary and 'recovery_views' not in summary
    assert summary['progress']['raw_changes']['wrist'] == [.001, -.003]
    assert failure.summary['completed_commands'] == 11
    assert not row['executed']
    assert failure.model_summary['rounds'][-1]['localization_summary']['observation_ref'] == 'F11'
    assert snapshot(case) == original


def test_unparseable_localization_preserves_raw_output_without_retry_or_archive_failure(case):
    raw_output = [{'content': 'The view is uncertain; no readable coordinate object.'}]
    folder = localized(case, raw='not a coordinate object', raw_output=raw_output)
    atomic_json(folder/'localization/response-0.json', dict(response=raw_output))
    atomic_json(folder/'localization/metrics.json', dict(attempts=1, status='completed'))
    original = snapshot(case)
    failure = FailureCase(case)
    row = failure.summary['rounds'][-1]
    assert row['localization_current']['raw_locations'] == 'not a coordinate object'
    assert row['localization_current']['raw_output'] == raw_output
    assert row['localization_summary']['views']['head']['derived_relation']['dx'] is None
    assert row['localization_summary']['parse_notes']
    assert row['localization']['metrics']['attempts'] == 1
    assert 'agent/rounds/0011/localization/response-0.json' in failure.source_hashes
    assert snapshot(case) == original


def test_v3_summary_recomputes_geometry_without_rewriting_original_archive(case):
    folder = localized(case, dict(head=positions(), wrist=positions()))
    value = json.loads((folder/'localization/assessment.json').read_text())
    value['wrist']['derived_relation']['dx'] = 99
    atomic_json(folder/'localization/assessment.json', value)
    atomic_json(folder/'localization_current.json', value)
    original = snapshot(case)
    failure = FailureCase(case)
    row = failure.summary['rounds'][-1]
    assert row['localization_current']['wrist']['derived_relation']['dx'] == 99
    assert row['localization_summary']['views']['wrist']['derived_relation']['dx'] == pytest.approx(-.35)
    assert snapshot(case) == original


def test_v5_location_transport_stop_is_an_unexecuted_round(case):
    folder = localized(case, with_action=False, with_assessment=False)
    atomic_json(folder/'localization/error.json', dict(stage='localization', error='operator_stopped'))
    row = FailureCase(case).summary['rounds'][-1]
    assert row['status'] == 'localization_interrupted'
    assert not row['executed'] and row['proposal'] is None
    assert row['localization_summary']['execution_policy'] == 'model_decides_after_progress_review'


@pytest.mark.parametrize('field,value,error', [
    ('observation_id', 'obs10', 'failure_localization_observation_mismatch'),
    ('source', 'human_confirmed', 'failure_localization_model_provenance_required'),
    ('images', {}, 'failure_localization_image_mismatch'),
])
def test_v5_archive_still_binds_model_evidence_to_exact_observation(case, field, value, error):
    folder = localized(case, dict(head=positions(), wrist=positions()))
    record = json.loads((folder/'localization/assessment.json').read_text())
    record[field] = value
    atomic_json(folder/'localization/assessment.json', record)
    with pytest.raises(ValueError, match=error):
        FailureCase(case)


def test_v3_invalid_coordinates_are_unknown_without_rewriting_archive(case):
    value = positions()
    value.update(button_box=[1.1, -.1, 1.3, .2], contact_tip=['0.7', '0.8'])
    localized(case, dict(head=value, wrist=dict(positions(), contact_tip='unreadable')))
    summary = FailureCase(case).summary['rounds'][-1]['localization_summary']
    assert summary['views']['head']['button_box'] is None
    assert summary['views']['head']['derived_relation']['dx'] is None
    assert summary['views']['wrist']['contact_tip'] is None
    assert summary['parse_notes']


def test_legacy_v2_units_remain_readable_without_guessing_1000_scale(case):
    folder = localized(case, dict(head=positions(), wrist=positions()))
    context = json.loads((folder/'decision_context.json').read_text())
    context.update(localization_format='pressb-localization-v2', progress_monitor_format='pressb-localized-progress-v2', execution_policy='model_decides')
    atomic_json(folder/'decision_context.json', context)
    value = json.loads((folder/'localization/assessment.json').read_text())
    value['format'] = 'pressb-localization-v2'
    value['raw_locations'].pop('coordinate_units')
    value['raw_locations']['wrist']['contact_tip'] = [700, 760]
    atomic_json(folder/'localization/assessment.json', value)
    atomic_json(folder/'localization_current.json', value)
    original = snapshot(case)
    row = FailureCase(case).summary['rounds'][-1]
    assert row['localization_summary']['views']['head']['derived_relation']['dx'] == pytest.approx(-.35)
    assert row['localization_summary']['views']['wrist']['contact_tip'] is None
    assert row['localization_current']['raw_locations']['wrist']['contact_tip'] == [700, 760]
    assert snapshot(case) == original


def test_hypothesis_review_stop_is_unexecuted_and_keeps_evidence(case):
    folder = localized(case, dict(head=positions(), wrist=positions()), with_action=False)
    decision = json.loads((folder/'decision_context.json').read_text())
    decision.update(stage='interrupted', hypothesis_review_format='pressb-hypothesis-review-v1')
    atomic_json(folder/'decision_context.json', decision)
    atomic_json(folder/'hypothesis_review/request.json', {'messages': []})
    atomic_json(folder/'hypothesis_review/trigger.json', dict(observation_id='obs11', baseline_observation_id='obs8', reasons=['repeated_direction_without_progress']))
    atomic_json(folder/'hypothesis_review/error.json', dict(stage='hypothesis_review', error='operator_stopped'))
    atomic_json(folder/'hypothesis_review/metrics.json', dict(observation_id='obs11', attempts=1, elapsed_s=5))
    original = snapshot(case)
    failure = FailureCase(case)
    row = failure.summary['rounds'][-1]
    assert row['status'] == 'hypothesis_review_interrupted'
    assert not row['executed'] and row['proposal'] is None
    assert row['hypothesis_review']['trigger']['baseline_observation_id'] == 'obs8'
    assert row['hypothesis_review']['metrics']['attempts'] == 1
    assert 'agent/rounds/0011/hypothesis_review/request.json' in failure.source_hashes
    assert snapshot(case) == original


def test_hypothesis_review_rejects_other_observation(case):
    folder = localized(case, dict(head=positions(), wrist=positions()))
    atomic_json(folder/'hypothesis_review/trigger.json', dict(observation_id='obs11', baseline_observation_id='obs8'))
    atomic_json(folder/'hypothesis_review/assessment.json', dict(format='pressb-hypothesis-review-v1', source='model_not_human', observation_id='obs10', previous_observation_id='obs8', evidence_observation_ids=['obs8', 'obs10']))
    with pytest.raises(ValueError, match='failure_hypothesis_review_observation_mismatch'):
        FailureCase(case)


def overlap_archive(case, *, failed=False):
    from pressb.agent_loop.spatial_localization import image_metadata
    folder = localized(case, dict(head=positions(), wrist=positions()))
    context = json.loads((folder/'decision_context.json').read_text())
    context.update(forward_guard_policy='wrist_tip_overlap', execution_policy='yz_alignment_before_forward',
                   stage='forward_guard', forward_guard_format='pressb-wrist-tip-overlap-v1')
    atomic_json(folder/'decision_context.json', context)
    picture = image_content((case/'agent/observations/obs11/right.jpg').read_bytes())
    atomic_json(folder/'forward_guard/request.json', dict(messages=[dict(role='user', content=[
        dict(type='text', text=json.dumps(dict(frame='CURRENT', observation_id='obs11'))), picture])]))
    atomic_json(folder/'forward_guard/assessment.json', dict(format='pressb-wrist-tip-overlap-v1',
        source='model_not_human', observation_id='obs11', images={'wrist': image_metadata(picture)},
        button_visibility='full', contact_visibility='full', tip_inside_button='no',
        button_box=[.2,.3,.4,.5], contact_tip=[.6,.4]))
    atomic_json(folder/'forward_guard/decision.json', dict(format='pressb-forward-guard-v1',
        source='program_rule_on_model_judgment', policy='wrist_tip_overlap', observation_id='obs11',
        assessment_observation_id='obs11', allowed=False, basis={'tip_inside_button': 'no'}))
    atomic_json(folder/'forward_guard/candidate-0.json', dict(proposal=dict(name='propose_move', arguments=dict(xyz_mm=[50,10,0])),
        decision=json.loads((folder/'forward_guard/decision.json').read_text())))
    if failed:
        (folder/'error.json').unlink()
        atomic_json(folder/'forward_guard/error.json', dict(stage='forward_guard', error='operator_stopped'))
    return folder


def test_wrist_overlap_archive_retains_current_policy_and_rejection_not_execution(case):
    folder = overlap_archive(case, failed=True)
    original = snapshot(case)
    failure = FailureCase(case)
    row = failure.summary['rounds'][-1]
    assert row['status'] == 'forward_guard_interrupted'
    assert row['executed'] is False and row['proposal'] is None
    assert row['forward_guard']['candidates'][0]['proposal']['arguments']['xyz_mm'] == [50,10,0]
    summary = row['localization_summary']
    assert summary['forward_guard_policy'] == 'wrist_tip_overlap'
    assert summary['execution_policy'] == 'yz_alignment_before_forward'
    assert summary['forward_guard']['decision']['allowed'] is False
    assert failure.summary['completed_commands'] == 11
    assert 'agent/rounds/0011/forward_guard/candidate-0.json' in failure.source_hashes
    assert failure.model_summary['rounds'][-1]['localization_summary']['forward_guard']['decision']['allowed'] is False
    assert snapshot(case) == original


@pytest.mark.parametrize('change,error', [
    ('observation', 'failure_wrist_overlap_observation_mismatch'),
    ('image', 'failure_wrist_overlap_image_mismatch'),
    ('source', 'failure_wrist_overlap_model_provenance_required'),
])
def test_wrist_overlap_archive_rejects_wrong_bindings(case, change, error):
    folder = overlap_archive(case)
    path = folder/'forward_guard/assessment.json'
    record = json.loads(path.read_text())
    if change == 'observation':
        record['observation_id'] = 'obs10'
    elif change == 'image':
        record['images']['wrist']['sha256'] = 'wrong-image'
    else:
        record['source'] = 'human_confirmed'
    atomic_json(path, record)
    with pytest.raises(ValueError, match=error):
        FailureCase(case)


def test_wrist_overlap_rejected_candidate_cannot_reference_another_observation(case):
    folder = overlap_archive(case)
    path = folder/'forward_guard/candidate-0.json'
    record = json.loads(path.read_text())
    record['decision']['observation_id'] = 'obs10'
    atomic_json(path, record)
    with pytest.raises(ValueError, match='failure_wrist_overlap_observation_mismatch'):
        FailureCase(case)
