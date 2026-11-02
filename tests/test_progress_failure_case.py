"""Direct progress-review failures preserve real command boundaries and provenance."""
import copy
import json

import pytest

from pressb.agent_loop.common import atomic_json
from pressb.agent_loop.failure_case import FailureCase
from test_failure_case import case, snapshot


def decision(oid='obs11', previous='obs10', **changes):
    data = dict(format='pressb-conversation-v1', observation_id=oid, previous_observation_id=previous,
                decision_mode='direct', stage='interrupted', progress_monitor_format='pressb-progress-v1')
    data.update(changes)
    return data


def progress_record(oid='obs11'):
    return dict(format='pressb-progress-v1', observation_id=oid,
        candidate_progress=dict(format='pressb-progress-v1', source='model_not_human', observation_id=oid,
            subgoal='align_vertical', view='wrist', button_vertical='above', button_horizontal='left', last_effect='improved'),
        monitor=dict(format='pressb-progress-v1', source='program_execution_events_and_model_judgments',
                     observation_id=oid, reasons=['repeated_completed_Z'], review_required=True))


def assessment(oid='obs11', previous='obs10'):
    return dict(format='pressb-progress-review-v1', source='model_not_human', observation_id=oid,
                previous_observation_id=previous, evidence_observation_ids=[previous, oid],
                comparison_view='wrist', head_vertical='unknown', head_horizontal='unknown',
                wrist_vertical='below', wrist_horizontal='left', vertical_change='crossed',
                horizontal_change='unchanged', approach_change='unknown', note='The button center crossed the marker row.')


def review_request(oid='obs11', previous='obs10', xyz=None):
    return dict(messages=[dict(role='user', content=[
        dict(type='text', text=json.dumps(dict(frame='BEFORE', observation_id=previous))),
        dict(type='text', text=json.dumps(dict(frame='CURRENT', observation_id=oid))),
        dict(type='text', text=json.dumps(dict(command=dict(type='move', xyz_mm=xyz or [10, 0, 0]))))])])


def add_progress(case, rd='0011', oid='obs11', previous='obs10', with_assessment=False):
    folder = case/'agent/rounds'/rd
    atomic_json(folder/'decision_context.json', decision(oid, previous))
    atomic_json(folder/'progress.json', progress_record(oid))
    atomic_json(folder/'progress_review/request.json', review_request(oid, previous))
    if with_assessment:
        atomic_json(folder/'progress_review/assessment.json', assessment(oid, previous))
        atomic_json(folder/'progress_review/decision.json', dict(observation_id=oid,
            source='program_rule_on_model_judgment', allowed=False, requires_new_decision=True,
            reasons=['repeat_Z_unsupported_by_review:crossed']))
    return folder


def copy_observation(case, before, after):
    import shutil
    source = case/'agent/observations'/before
    destination = case/'agent/observations'/after
    shutil.copytree(source, destination)
    manifest = json.loads((destination/'manifest.json').read_text())
    manifest['observation_id'] = after
    atomic_json(destination/'manifest.json', manifest)


def test_direct_review_interrupt_without_action_request_is_not_an_execution(case):
    rd = add_progress(case)
    (rd/'request.json').unlink()
    (rd/'error.json').unlink()
    atomic_json(rd/'progress_review/error.json', dict(stage='review_progress', error='operator_stopped'))
    before = snapshot(case)
    failure = FailureCase(case)
    row = failure.summary['rounds'][-1]
    assert row['status'] == 'progress_review_interrupted'
    assert not row['executed'] and row['proposal'] is None and row['after'] is None
    assert row['before']['observation_id'] == 'obs11'
    assert row['progress']['candidate_progress']['source'] == 'model_not_human'
    assert row['progress_summary']['reported_progress']['subgoal'] == 'align_vertical'
    assert failure.summary['completed_commands'] == 11 and len(failure.summary['rounds']) == 12
    assert snapshot(case) == before


def test_direct_review_request_failure_is_not_an_execution(case):
    rd = add_progress(case)
    (rd/'request.json').unlink()
    atomic_json(case/'agent/result.json', dict(status='error', completed_motion_commands=11))
    row = FailureCase(case).summary['rounds'][-1]
    assert row['status'] == 'progress_review_failed' and not row['executed']


def test_review_assessment_and_program_decision_remain_distinct_sources(case):
    rd = add_progress(case, with_assessment=True)
    before = snapshot(case)
    failure = FailureCase(case)
    row = failure.summary['rounds'][-1]
    assert row['progress_review'] == assessment()
    assert row['progress_review_decision']['source'] == 'program_rule_on_model_judgment'
    assert row['progress_summary']['independent_review']['source'] == 'model_not_human'
    assert row['progress_summary']['program_decision']['allowed'] is False
    assert row['executed'] is False
    assert 'agent/rounds/0011/progress_review/request.json' in failure.source_hashes
    assert snapshot(case) == before


def test_review_pair_can_span_observation_refresh_without_claiming_it_was_a_move(case):
    # Last real movement: obs10 -> obs11. Refresh: obs11 -> obs12.
    copy_observation(case, 'obs11', 'obs12')
    refresh = case/'agent/rounds/0011'
    (refresh/'error.json').unlink()
    atomic_json(refresh/'proposal.json', dict(name='request_observation', arguments=dict(reason='refresh')))
    atomic_json(refresh/'observation_result.json', dict(observation_id='obs12'))
    rd = add_progress(case, rd='0012', oid='obs12', previous='obs10', with_assessment=True)
    # The immediate previous decision is obs11, but comparison BEFORE must remain obs10.
    atomic_json(rd/'decision_context.json', decision('obs12', 'obs11'))
    atomic_json(rd/'error.json', dict(stage='progress_review', error='stopped'))
    row = FailureCase(case).summary['rounds'][-1]
    assert row['status'] == 'progress_review_interrupted'
    assert row['progress_review']['previous_observation_id'] == 'obs10'
    assert row['progress_review']['observation_id'] == 'obs12'
    assert not row['executed']


def test_missing_request_still_requires_explicit_supported_progress_version(case):
    rd = case/'agent/rounds/0011'
    (rd/'request.json').unlink()
    atomic_json(rd/'decision_context.json', decision(progress_monitor_format=None))
    with pytest.raises(ValueError, match='failure_round_missing_action_request'):
        FailureCase(case)
    atomic_json(rd/'decision_context.json', decision(progress_monitor_format='pressb-progress-v99'))
    with pytest.raises(ValueError, match='unsupported_failure_progress_format'):
        FailureCase(case)


@pytest.mark.parametrize('name', ['proposal.json', 'execution.json', 'result.json'])
def test_no_action_request_never_accepts_an_execution_artifact(case, name):
    rd = add_progress(case)
    (rd/'request.json').unlink()
    atomic_json(rd/name, dict(name='propose_move', arguments=dict(xyz_mm=[0, 0, 10], reason='bad archive')))
    with pytest.raises(ValueError, match='failure_execution_without_action_request'):
        FailureCase(case)


@pytest.mark.parametrize('target,field,value,error', [
    ('progress.json', 'observation_id', 'obs0', 'progress_observation_mismatch'),
    ('progress_review/assessment.json', 'observation_id', 'obs0', 'progress_review_observation_mismatch'),
    ('progress_review/assessment.json', 'source', 'human_confirmed', 'progress_review_model_provenance_required'),
    ('progress_review/assessment.json', 'format', 'unknown', 'progress_review_model_provenance_required'),
    ('progress_review/assessment.json', 'evidence_observation_ids', ['obs0', 'obs11'], 'progress_review_observation_mismatch'),
    ('progress_review/decision.json', 'observation_id', 'obs0', 'progress_review_decision_observation_mismatch'),
    ('progress_review/decision.json', 'source', 'human_confirmed', 'progress_review_decision_provenance_required'),
])
def test_progress_evidence_cannot_change_observation_or_provenance(case, target, field, value, error):
    rd = add_progress(case, with_assessment=True)
    record = json.loads((rd/target).read_text()); record[field] = value
    atomic_json(rd/target, record)
    with pytest.raises(ValueError, match='failure_'+error):
        FailureCase(case)


def test_candidate_and_monitor_sources_are_checked(case):
    rd = add_progress(case)
    record = progress_record()
    record['candidate_progress']['source'] = 'human'
    atomic_json(rd/'progress.json', record)
    with pytest.raises(ValueError, match='progress_model_provenance_required'):
        FailureCase(case)
    record = progress_record(); record['monitor']['source'] = 'human'
    atomic_json(rd/'progress.json', record)
    with pytest.raises(ValueError, match='progress_monitor_provenance_required'):
        FailureCase(case)


def test_review_cannot_skip_another_actual_move_or_use_a_foreign_observation(case):
    rd = add_progress(case, previous='obs9', with_assessment=True)
    with pytest.raises(ValueError, match='progress_review_interval_mismatch'):
        FailureCase(case)
    record = assessment(previous='outside_run')
    atomic_json(rd/'progress_review/assessment.json', record)
    with pytest.raises(ValueError, match='failure_case_missing_artifact'):
        FailureCase(case)


def test_review_cannot_cross_head_or_gripper_operation(case):
    rd = add_progress(case, with_assessment=True)
    prior = case/'agent/rounds/0010'
    result = json.loads((prior/'result.json').read_text())
    result['request'].pop('xyz_mm')
    result['request'].update(type='head', head_deg=[0, 30])
    atomic_json(prior/'result.json', result)
    atomic_json(prior/'proposal.json', dict(name='propose_head', arguments=dict(head_deg=[0, 30], reason='look')))
    with pytest.raises(ValueError, match='progress_review_interval_contains_nonmove'):
        FailureCase(case)


def test_review_request_must_describe_the_actual_command(case):
    rd = add_progress(case, with_assessment=True)
    atomic_json(rd/'progress_review/request.json', review_request(xyz=[0, 0, 100]))
    with pytest.raises(ValueError, match='progress_review_request_mismatch'):
        FailureCase(case)


def test_program_decision_requires_independent_review_record(case):
    rd = add_progress(case, with_assessment=True)
    (rd/'progress_review/assessment.json').unlink()
    with pytest.raises(ValueError, match='progress_review_decision_requires_assessment'):
        FailureCase(case)


def test_review_triggered_after_action_request_failure_keeps_its_stage(case):
    rd = add_progress(case)
    atomic_json(rd/'error.json', dict(stage='progress_review', error='model_timeout'))
    atomic_json(case/'agent/result.json', dict(status='error', completed_motion_commands=11))
    row = FailureCase(case).summary['rounds'][-1]
    assert row['status'] == 'progress_review_failed' and not row['executed']


def test_progress_version_cannot_claim_a_different_decision_mode(case):
    rd = add_progress(case)
    atomic_json(rd/'decision_context.json', decision(decision_mode='grounded'))
    with pytest.raises(ValueError, match='unsupported_failure_progress_format'):
        FailureCase(case)
