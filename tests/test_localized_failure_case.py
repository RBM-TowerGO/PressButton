"""V4 failure cases bind positions to explicit observations and keep interruptions unexecuted."""
import copy
import json

import pytest

from pressb.agent_loop import spatial_localization as localization
from pressb.agent_loop.common import atomic_json
from pressb.agent_loop.demonstration import image_content
from pressb.agent_loop.failure_case import FailureCase
from test_failure_case import case, snapshot


def positions():
    return dict(button_visibility='full', contact_visibility='full',
                button_box=[.2, .2, .4, .4], arrow_box=[.6, .5, .7, .6],
                contact_tip=[.65, .6], note='Target and marked contact end are visible.')


def stage(case, directory, oid='obs11', revision=1):
    body = dict(messages=[dict(role='user', content=[
        dict(type='text', text=json.dumps(dict(frame='CURRENT', observation_id=oid))),
        *[image_content((case/'agent/observations'/oid/(view+'.jpg')).read_bytes())
          for view in ('head', 'right')]])])
    atomic_json(directory/'request.json', body)
    data = dict(observation_id=oid, head=positions(), wrist=positions())
    value = localization.validate_raw(data, oid, dict(head=[64, 36], wrist=[64, 36]), revision=revision)
    atomic_json(directory/'assessment.json', value)
    return value


def localized(case, with_assessment=True, with_action=True):
    folder = case/'agent/rounds/0011'
    atomic_json(folder/'decision_context.json', dict(format='pressb-conversation-v4',
        decision_mode='localized', observation_id='obs11', previous_observation_id='obs10',
        stage='localization', localization_format=localization.FORMAT,
        localization_revision=1, progress_monitor_format='pressb-localized-progress-v1', view_phase='ready'))
    value = stage(case, folder/'localization')
    if not with_assessment:
        (folder/'localization/assessment.json').unlink()
    else:
        atomic_json(folder/'localization_current.json', value)
    if not with_action:
        (folder/'request.json').unlink()
        (folder/'error.json').unlink()
    return folder


def test_localization_only_interruption_preserves_all_commands_and_raw_files(case, tmp_path):
    folder = localized(case, with_assessment=False, with_action=False)
    atomic_json(folder/'localization/error.json', dict(stage='localization', error='operator_stopped'))
    atomic_json(folder/'localization/metrics.json', dict(attempts=1, observation_id='obs11', status='interrupted'))
    original = snapshot(case)
    failure = FailureCase(case)
    row = failure.summary['rounds'][-1]
    assert row['status'] == 'localization_interrupted'
    assert not row['executed'] and row['proposal'] is None and row['after'] is None
    assert failure.summary['completed_commands'] == 11
    assert len(failure.summary['rounds']) == 12
    assert failure.model_summary['rounds'][-1]['localization_summary']['source'].endswith('not_human_confirmation')
    assert 'agent/rounds/0011/localization/request.json' in failure.source_hashes
    failure.save(tmp_path/'saved')
    assert snapshot(case) == original


def test_localization_and_program_geometry_remain_auditable_in_failure_context(case):
    localized(case)
    failure = FailureCase(case)
    row = failure.summary['rounds'][-1]
    assert row['localization_current']['revision'] == 1
    assert row['localization']['assessment']['source'] == 'model_not_human'
    compact = failure.model_summary['rounds'][-1]['localization_summary']
    assert compact['observation_ref'] == 'F11'
    assert compact['views']['wrist']['derived_relation']['horizontal'] == 'left'
    assert compact['views']['wrist']['contact_tip'] == [.65, .6]
    assert not row['executed']


def test_review_preserves_initial_coordinates_and_current_effective_revision(case):
    folder = localized(case, with_action=False)
    before = stage(case, folder/'localization_review/before', 'obs10')
    current = stage(case, folder/'localization_review/current', 'obs11', 2)
    review = dict(format='pressb-localization-review-v1', source='model_relocalization_and_program_comparison',
        previous_observation_id='obs10', observation_id='obs11', conflicting_views=['head'], revision=2)
    atomic_json(folder/'localization_review/assessment.json', review)
    atomic_json(folder/'localization_current.json', dict(current, usable_views=['wrist'], recovery_views=['head']))
    context = json.loads((folder/'decision_context.json').read_text())
    context.update(stage='localization_review', localization_revision=2)
    atomic_json(folder/'decision_context.json', context)
    failure = FailureCase(case)
    row = failure.summary['rounds'][-1]
    assert row['status'] == 'localization_review_interrupted'
    assert row['localization']['assessment']['revision'] == 1
    assert row['localization_review_before']['assessment'] == before
    assert row['localization_current']['revision'] == 2
    assert row['localization_current']['usable_views'] == ['wrist']
    assert row['localization_summary']['review']['conflicting_views'] == ['head']
    assert not row['executed']


@pytest.mark.parametrize('path,field,value,error', [
    ('decision_context.json', 'localization_format', 'v99', 'unsupported_failure_localization_format'),
    ('decision_context.json', 'decision_mode', 'direct', 'unsupported_failure_localization_format'),
    ('decision_context.json', 'localization_revision', 2, 'failure_current_localization_version_mismatch'),
    ('localization/assessment.json', 'observation_id', 'obs0', 'failure_localization_observation_mismatch'),
    ('localization/assessment.json', 'source', 'human_confirmed', 'failure_localization_model_provenance_required'),
    ('localization_current.json', 'usable_views', ['imagined_camera'], 'failure_current_localization_version_mismatch'),
])
def test_localization_cannot_change_provenance_version_or_observation(case, path, field, value, error):
    folder = localized(case)
    data = json.loads((folder/path).read_text()); data[field] = value
    atomic_json(folder/path, data)
    with pytest.raises(ValueError, match=error):
        FailureCase(case)


def test_bad_derived_relation_and_wrong_request_frame_are_rejected(case):
    folder = localized(case)
    original = json.loads((folder/'localization/assessment.json').read_text())
    changed = copy.deepcopy(original)
    changed['wrist']['derived_relation']['horizontal'] = 'right'
    atomic_json(folder/'localization/assessment.json', changed)
    with pytest.raises(ValueError, match='failure_localization_derived_relation_mismatch'):
        FailureCase(case)
    atomic_json(folder/'localization/assessment.json', original)
    body = json.loads((folder/'localization/request.json').read_text())
    body['messages'][0]['content'][0]['text'] = json.dumps(dict(frame='CURRENT', observation_id='obs0'))
    atomic_json(folder/'localization/request.json', body)
    with pytest.raises(ValueError, match='failure_localization_request_observation_mismatch'):
        FailureCase(case)


def test_localization_image_bytes_must_match_the_bound_saved_observation(case):
    folder = localized(case)
    body = json.loads((folder/'localization/request.json').read_text())
    body['messages'][0]['content'][1] = image_content((case/'agent/observations/obs0/head.jpg').read_bytes())
    atomic_json(folder/'localization/request.json', body)
    with pytest.raises(ValueError, match='failure_localization_image_mismatch'):
        FailureCase(case)


def test_continuous_history_uses_explicit_decision_observation(case):
    folder = localized(case)
    messages = [dict(role='user', content=[dict(type='text', text=json.dumps(dict(
        context_type='LIVE_OBSERVATION', observation_id=oid)))]) for oid in ('obs1', 'obs10', 'obs11')]
    atomic_json(folder/'request.json', dict(messages=messages))
    assert FailureCase(case).summary['rounds'][-1]['before']['observation_id'] == 'obs11'
    atomic_json(folder/'request.json', dict(messages=list(reversed(messages))))
    with pytest.raises(ValueError, match='failure_decision_observation_mismatch'):
        FailureCase(case)


def evidence(case, folder, name, before, current, judgment='yes'):
    content = []
    for label, oid in (('BEFORE', before), ('CURRENT', current)):
        content += [dict(type='text', text=json.dumps(dict(frame=label, observation_id=oid)))]
        content += [image_content((case/'agent/observations'/oid/(view+'.jpg')).read_bytes())
                    for view in ('head', 'right')]
    atomic_json(folder/name/'request.json', dict(messages=[dict(role='user', content=content)]))
    record = dict(format='pressb-'+name+'-v1', source='model_not_human',
                  previous_observation_id=before, observation_id=current,
                  evidence_observation_ids=[before, current])
    record['light_on' if name == 'activation' else 'withdrawal'] = judgment
    atomic_json(folder/name/'assessment.json', record)
    return record


def test_completion_reviews_can_reference_earlier_live_frames_without_becoming_actions(case):
    folder = localized(case)
    activated = evidence(case, folder, 'activation', 'obs7', 'obs8', judgment='unknown')
    withdrawn = evidence(case, folder, 'withdrawal', 'obs8', 'obs10')
    atomic_json(folder/'error.json', dict(stage='activation_verification', error='operator_stopped'))
    failure = FailureCase(case)
    row = failure.summary['rounds'][-1]
    assert row['before']['observation_id'] == 'obs11'
    assert row['activation'] == activated and row['withdrawal'] == withdrawn
    assert row['status'] == 'activation_verification_interrupted'
    assert not row['executed'] and row['proposal'] is None
    assert failure.summary['completed_commands'] == 11
    compact = failure.model_summary['rounds'][-1]
    assert compact['activation']['source'] == 'model_not_human'
    assert compact['activation']['observation_ref'] == 'F08'
    assert compact['withdrawal']['observation_ref'] == 'F10'


@pytest.mark.parametrize('change,error', [
    ({'source':'human_confirmed'}, 'failure_localized_evidence_model_provenance_required'),
    ({'observation_id':'obs11'}, 'failure_localized_evidence_observation_mismatch'),
    ({'evidence_observation_ids':['obs8','obs7']}, 'failure_localized_evidence_observation_mismatch'),
])
def test_completion_review_provenance_and_pair_cannot_be_relabelled(case, change, error):
    folder = localized(case)
    record = evidence(case, folder, 'activation', 'obs7', 'obs8')
    record.update(change)
    atomic_json(folder/'activation/assessment.json', record)
    with pytest.raises(ValueError, match=error):
        FailureCase(case)


def test_completion_review_requires_live_order_and_same_light_boundary(case):
    folder = localized(case)
    evidence(case, folder, 'activation', 'obs8', 'obs7')
    with pytest.raises(ValueError, match='failure_localized_evidence_requires_ordered_live_observations'):
        FailureCase(case)
    evidence(case, folder, 'activation', 'obs7', 'obs8')
    evidence(case, folder, 'withdrawal', 'obs9', 'obs10')
    with pytest.raises(ValueError, match='failure_localized_withdrawal_must_follow_reviewed_light'):
        FailureCase(case)


@pytest.mark.parametrize('filename', ['proposal.json', 'execution.json', 'result.json'])
def test_localization_only_round_never_counts_as_action(case, filename):
    folder = localized(case, with_assessment=False, with_action=False)
    atomic_json(folder/filename, dict(name='propose_move', arguments=dict(xyz_mm=[1, 0, 0])))
    with pytest.raises(ValueError, match='failure_execution_without_action_request'):
        FailureCase(case)
