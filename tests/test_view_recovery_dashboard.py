"""Read-only view acquisition and disagreement remain distinct from movement."""
import threading

import pytest

import s1b_agent_dashboard as dashboard
from pressb.agent_loop.common import atomic_json


def context(stage='view_acquisition', phase='get_view'):
    return dict(format='pressb-conversation-v1', decision_mode='direct',
                observation_id='now', previous_observation_id='before',
                stage=stage, view_phase=phase)


def assessment():
    head = dict(panel_visible='no', button_visibility='not_visible', contact_visibility='not_visible',
                button_box=None, arrow_box=None, contact_tip=None, note='Head is cropped.',
                derived_relation=dict(vertical='unknown', horizontal='unknown', dx=None, dy=None))
    wrist = dict(panel_visible='yes', button_visibility='full', contact_visibility='full',
                 button_box=[.61, .03, .65, .09], arrow_box=[.7, .5, .75, .57],
                 contact_tip=[.73, .54], note='Button is above the contact end.',
                 derived_relation=dict(vertical='above', horizontal='left', dx=-.1, dy=-.48))
    return dict(format='pressb-view-acquisition-v1', source='model_not_human',
                observation_id='now', usable_head=False, head=head, wrist=wrist)


def test_view_stage_has_bound_assessment_and_metrics_before_any_action(tmp_path):
    rd = tmp_path/'agent/rounds/0000'
    atomic_json(rd/'decision_context.json', context())
    atomic_json(rd/'view_state.json', dict(phase='get_view', observation_id='now',
                assessment_observation_id=None, reason='initial_view'))
    view = dashboard.Dashboard(tmp_path, '', threading.Event())
    row = view.state()['rounds'][0]
    assert row['phase'] == '取景检查中' and row['proposal'] is None
    assert row['view_acquisition']['assessment'] is None
    atomic_json(rd/'view_acquisition/assessment.json', assessment())
    atomic_json(rd/'view_acquisition/metrics.json', dict(image_count=2, prompt_tokens=1400,
                completion_tokens=260, elapsed_s=4.2, attempts=1))
    atomic_json(rd/'decision_context.json', context('action_request'))
    atomic_json(rd/'view_state.json', dict(phase='get_view', observation_id='now',
                assessment_observation_id='now', reason='head_not_usable'))
    row = view.state()['rounds'][0]
    assert row['phase'] == '取景决策中'
    assert row['view_acquisition']['assessment'] == assessment()
    assert row['view_acquisition']['metrics']['image_count'] == 2
    assert row['view_state']['assessment_observation_id'] == 'now'
    assert row['proposal'] is None
    atomic_json(rd/'proposal.json', dict(name='propose_head', arguments=dict(head_deg=[0, 30], reason='Improve framing.')))
    assert view.state()['rounds'][0]['phase'] == '已生成建议'


def test_disagreement_changes_phase_and_preserves_both_claims(tmp_path):
    rd = tmp_path/'agent/rounds/0000'
    atomic_json(rd/'decision_context.json', context('action_request', 'recover_view'))
    state = dict(phase='recover_view', observation_id='now', assessment_observation_id='now',
                 reason='model_disagreement')
    conflict = dict(observation_id='now', model_claim=dict(wrist_vertical='above'),
                    review_claim=dict(wrist_vertical='below'), conflicting_axes=['vertical'],
                    source='model_disagreement_not_visual_fact')
    atomic_json(rd/'view_state.json', state)
    atomic_json(rd/'progress_review/conflict.json', conflict)
    atomic_json(rd/'progress_review/decision.json', dict(allowed=False))
    view = dashboard.Dashboard(tmp_path, '', threading.Event())
    row = view.state()['rounds'][0]
    assert row['phase'] == '判断冲突：恢复取景中'
    assert row['progress_review']['conflict'] == conflict
    assert row['view_state'] == state
    atomic_json(rd/'correction.json', dict(status='retrying', attempt=0, next_attempt=1, error='view recovery required'))
    assert view.state()['rounds'][0]['phase'] == '恢复取景决策修正中'
    atomic_json(rd/'decision_context.json', context('view_acquisition', 'recover_view'))
    assert view.state()['rounds'][0]['phase'] == '冲突恢复取景检查中'
    atomic_json(tmp_path/'agent/result.json', dict(status='operator_stopped'))
    assert view.state()['rounds'][0]['phase'] == '已停止'


def test_view_localization_repairs_are_read_by_number_and_can_stop(tmp_path):
    rd = tmp_path/'agent/rounds/0000'
    atomic_json(rd/'decision_context.json', context())
    for attempt in (9, 10):
        atomic_json(rd/'view_acquisition'/f'response-{attempt}.json', dict(response=dict(choices=[attempt])))
        atomic_json(rd/'view_acquisition'/f'validation-{attempt}.json', dict(attempt=attempt, error='invalid box'))
    atomic_json(rd/'view_acquisition/metrics.json', dict(attempts=11))
    view = dashboard.Dashboard(tmp_path, '', threading.Event())
    row = view.state()['rounds'][0]
    assert row['phase'] == '取景定位修正中'
    assert row['view_acquisition']['response']['response']['choices'] == [10]
    assert row['view_acquisition']['correction']['next_attempt'] == 11
    atomic_json(rd/'decision_context.json', context('interrupted'))
    atomic_json(rd/'view_acquisition/error.json', dict(error='operator_stopped'))
    assert view.state()['rounds'][0]['phase'] == '已停止'


@pytest.mark.parametrize('with_round_error', [True, False])
def test_view_failure_is_distinct_from_progress_review_failure(tmp_path, with_round_error):
    rd = tmp_path/'agent/rounds/0000'
    atomic_json(rd/'decision_context.json', context('failed'))
    atomic_json(rd/'view_acquisition/error.json', dict(error='service unavailable'))
    if with_round_error:
        atomic_json(rd/'error.json', dict(stage='view_acquisition', error='service unavailable'))
    row = dashboard.Dashboard(tmp_path, '', threading.Event()).state()['rounds'][0]
    assert row['phase'] == '取景检查失败' and row['proposal'] is None


def test_old_round_has_no_synthetic_view_or_conflict(tmp_path):
    rd = tmp_path/'agent/rounds/0000'
    atomic_json(rd/'proposal.json', dict(name='request_observation', arguments=dict(reason='old')))
    row = dashboard.Dashboard(tmp_path, '', threading.Event()).state()['rounds'][0]
    assert row['phase'] == '请求新观测'
    assert row['view_state'] is None and row['view_acquisition']['assessment'] is None
    assert row['progress_review']['conflict'] is None


def test_view_ui_distinguishes_model_coordinates_and_conflict_from_actions():
    html = dashboard.HTML
    for text in ('renderViewAcquisition(card,row)', '取景阶段与冲突恢复', '当前仅允许调整头部',
                 '取景定位对应观测', '定位是模型判断，非人工确认', '双方判断及冲突观测',
                 '不把复核结论当作事实', '图像上下／左右不是机器人运动方向',
                 '完整取景定位（按钮框、箭头框与物理指端）'):
        assert text in html
    assert 'innerHTML' not in html
