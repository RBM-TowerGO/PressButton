"""View acquisition and conflict recovery retain provenance without inventing actions."""
import copy
import json

import pytest

from pressb.agent_loop import view_acquisition as view
from pressb.agent_loop.common import atomic_json
from pressb.agent_loop.failure_case import FailureCase
from test_failure_case import case, snapshot
from test_progress_failure_case import add_progress, assessment as progress_assessment, progress_record
from test_view_acquisition import reply, located


def add_view(case, rd='0011', oid='obs11', phase='get_view', with_assessment=True, with_request=True):
    folder = case/'agent/rounds'/rd
    decision = dict(format='pressb-conversation-v1', decision_mode='direct', observation_id=oid,
                    previous_observation_id='obs10' if oid == 'obs11' else None,
                    stage='view_acquisition', progress_monitor_format='pressb-progress-v1',
                    view_acquisition_format=view.FORMAT, view_phase=phase)
    atomic_json(folder/'decision_context.json', decision)
    if with_request:
        atomic_json(folder/'view_acquisition/request.json', dict(messages=[dict(role='user', content=[
            dict(type='text', text=json.dumps(dict(frame='CURRENT', observation_id=oid))),
            dict(type='image_url', image_url=dict(url='data:image/jpeg;base64,head')),
            dict(type='image_url', image_url=dict(url='data:image/jpeg;base64,wrist'))])]))
    if with_assessment:
        value = view.validate(reply(oid=oid, head=located() if phase == 'ready' else None), oid)
        atomic_json(folder/'view_acquisition/assessment.json', value)
    atomic_json(folder/'view_state.json', dict(format='pressb-view-state-v1',
        source='program_phase_on_model_localization', phase=phase, observation_id=oid,
        assessment_observation_id=oid if with_assessment else None, conflict_observation_id=None,
        reason='head_panel_button_or_contact_not_fully_identified'))
    return folder


def add_conflict(case):
    folder = add_progress(case, with_assessment=True)
    add_view(case, phase='recover_view', with_assessment=False, with_request=False)
    conflict = dict(source='model_disagreement_not_visual_fact', observation_id='obs11',
                    model_claim=progress_record()['candidate_progress'], review_claim=progress_assessment(),
                    conflicting_axes=['vertical'])
    atomic_json(folder/'progress_review/conflict.json', conflict)
    state = json.loads((folder/'view_state.json').read_text())
    state['conflict_observation_id'] = 'obs11'
    atomic_json(folder/'view_state.json', state)
    return folder


def test_view_request_interrupt_before_action_is_unexecuted_and_source_unchanged(case, tmp_path):
    folder = add_view(case, with_assessment=False)
    (folder/'request.json').unlink(); (folder/'error.json').unlink()
    atomic_json(folder/'view_acquisition/metrics.json', dict(tool='inspect_view', observation_id='obs11',
                status='interrupted', attempts=1))
    atomic_json(folder/'view_acquisition/error.json', dict(stage='inspect_view', error='operator_stopped'))
    original = snapshot(case)
    failure = FailureCase(case)
    row = failure.summary['rounds'][-1]
    assert row['status'] == 'view_acquisition_interrupted' and not row['executed']
    assert row['after'] is None and row['proposal'] is None
    assert failure.summary['completed_commands'] == 11 and len(failure.summary['rounds']) == 12
    assert row['view_state']['source'] == 'program_phase_on_model_localization'
    assert row['view_summary']['phase'] == 'get_view'
    assert failure.model_summary['rounds'][-1]['view_summary']['source'].endswith('not_human_confirmation')
    assert 'agent/rounds/0011/view_acquisition/request.json' in failure.source_hashes
    failure.save(tmp_path/'saved')
    assert snapshot(case) == original
    saved = json.loads((tmp_path/'saved/failure_case/reference.json').read_text())
    assert saved['rounds'][-1]['view_metrics']['status'] == 'interrupted'


def test_cancel_before_first_view_network_request_keeps_view_stage(case):
    folder = add_view(case, with_request=False, with_assessment=False)
    (folder/'request.json').unlink()
    row = FailureCase(case).summary['rounds'][-1]
    assert row['status'] == 'view_acquisition_interrupted' and not row['executed']


def test_view_request_error_with_existing_action_request_has_correct_stage(case):
    folder = add_view(case, with_assessment=False)
    atomic_json(folder/'error.json', dict(stage='view_acquisition', error='model_timeout'))
    atomic_json(case/'agent/result.json', dict(status='error', completed_motion_commands=11))
    assert FailureCase(case).summary['rounds'][-1]['status'] == 'view_acquisition_failed'


def test_model_localization_and_conflict_are_never_human_facts(case):
    folder = add_view(case)
    failure = FailureCase(case)
    row = failure.summary['rounds'][-1]
    assert row['view_assessment']['source'] == 'model_not_human'
    assert row['view_assessment']['wrist']['derived_relation']['vertical'] == 'above'
    compact = failure.model_summary['rounds'][-1]
    assert 'view_summary' in compact and 'view_assessment' not in compact
    assert 'button_box' not in json.dumps(compact['view_summary'])
    assert compact['view_summary']['observation_ref'] == 'F11'
    add_conflict(case)
    # This round had no acquisition after entering recovery; retain the previously
    # written assessment only in this fixture, with the same valid observation ID.
    row = FailureCase(case).summary['rounds'][-1]
    assert row['progress_conflict']['source'] == 'model_disagreement_not_visual_fact'
    assert row['view_summary']['conflict']['conflicting_axes'] == ['vertical']
    assert not row['executed']


def test_ready_can_reference_prior_run_localization_without_claiming_current_inspection(case):
    add_view(case, rd='0000', oid='obs0', phase='ready')
    folder = add_view(case, phase='ready', with_request=False, with_assessment=False)
    state = json.loads((folder/'view_state.json').read_text())
    state['assessment_observation_id'] = 'obs0'
    atomic_json(folder/'view_state.json', state)
    row = FailureCase(case).summary['rounds'][-1]
    assert row['view_summary']['assessment_observation_id'] == 'obs0'
    assert row['view_summary']['observation_id'] == 'obs11'
    assert 'localization' not in row['view_summary']


@pytest.mark.parametrize('target,field,value,error', [
    ('decision_context.json', 'view_acquisition_format', 'v99', 'unsupported_failure_view_format'),
    ('decision_context.json', 'decision_mode', 'grounded', 'unsupported_failure_progress_format'),
    ('decision_context.json', 'view_phase', 'approach', 'unsupported_failure_view_format'),
    ('view_acquisition/assessment.json', 'observation_id', 'obs0', 'failure_view_observation_mismatch'),
    ('view_acquisition/assessment.json', 'source', 'human_confirmed', 'failure_view_model_provenance_required'),
    ('view_acquisition/assessment.json', 'usable_head', True, 'failure_view_derived_relation_mismatch'),
    ('view_state.json', 'source', 'human_confirmed', 'failure_view_state_provenance_required'),
    ('view_state.json', 'observation_id', 'obs0', 'failure_view_state_observation_mismatch'),
    ('view_state.json', 'phase', 'ready', 'failure_view_phase_mismatch'),
    ('view_state.json', 'assessment_observation_id', 'obs0', 'failure_view_state_requires_recorded_assessment'),
    ('view_state.json', 'conflict_observation_id', 'obs0', 'failure_view_state_requires_recorded_conflict'),
])
def test_view_provenance_and_observation_references_checked(case, target, field, value, error):
    folder = add_view(case)
    record = json.loads((folder/target).read_text()); record[field] = value
    atomic_json(folder/target, record)
    with pytest.raises(ValueError, match=error):
        FailureCase(case)


def test_view_request_and_derived_relationship_cannot_be_forged(case):
    folder = add_view(case)
    record = json.loads((folder/'view_acquisition/request.json').read_text())
    record['messages'][0]['content'][0]['text'] = json.dumps(dict(frame='CURRENT', observation_id='obs0'))
    atomic_json(folder/'view_acquisition/request.json', record)
    with pytest.raises(ValueError, match='failure_view_request_observation_mismatch'):
        FailureCase(case)
    add_view(case)
    record = json.loads((folder/'view_acquisition/assessment.json').read_text())
    record['wrist']['derived_relation']['vertical'] = 'below'
    atomic_json(folder/'view_acquisition/assessment.json', record)
    with pytest.raises(ValueError, match='failure_view_derived_relation_mismatch'):
        FailureCase(case)


@pytest.mark.parametrize('name', ['proposal.json', 'execution.json', 'result.json'])
def test_view_only_round_cannot_claim_an_executed_action(case, name):
    folder = add_view(case, with_assessment=False)
    (folder/'request.json').unlink()
    atomic_json(folder/name, dict(name='propose_head', arguments=dict(head_deg=[0, 30])))
    with pytest.raises(ValueError, match='failure_execution_without_action_request'):
        FailureCase(case)


@pytest.mark.parametrize('field,value,error', [
    ('source', 'human_confirmed', 'failure_view_conflict_provenance_required'),
    ('observation_id', 'obs0', 'failure_view_conflict_observation_mismatch'),
    ('conflicting_axes', [], 'failure_view_invalid_conflicting_axes'),
    ('conflicting_axes', ['horizontal'], 'failure_view_conflict_requires_disagreement'),
])
def test_conflict_is_a_bound_model_disagreement(case, field, value, error):
    folder = add_conflict(case)
    record = json.loads((folder/'progress_review/conflict.json').read_text()); record[field] = value
    atomic_json(folder/'progress_review/conflict.json', record)
    with pytest.raises(ValueError, match=error):
        FailureCase(case)


def test_conflict_claim_cannot_invent_human_source_or_a_different_review(case):
    folder = add_conflict(case)
    record = json.loads((folder/'progress_review/conflict.json').read_text())
    changed = copy.deepcopy(record); changed['model_claim']['source'] = 'human'
    atomic_json(folder/'progress_review/conflict.json', changed)
    with pytest.raises(ValueError, match='failure_view_conflict_claim_provenance_required'):
        FailureCase(case)
    changed = copy.deepcopy(record); changed['review_claim']['wrist_horizontal'] = 'right'
    atomic_json(folder/'progress_review/conflict.json', changed)
    with pytest.raises(ValueError, match='failure_view_conflict_requires_matching_review'):
        FailureCase(case)
