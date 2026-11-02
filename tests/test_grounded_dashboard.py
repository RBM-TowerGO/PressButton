"""Grounded loop dashboard reads model judgments without promoting them to actions."""
import threading

import s1b_agent_dashboard as dashboard
from pressb.agent_loop.common import atomic_json


def test_grounded_phases_and_independent_stage_metrics(tmp_path):
    rd = tmp_path/'agent/rounds/0000'
    decision = dict(format='pressb-conversation-v3', decision_mode='grounded',
                    observation_id='now', previous_observation_id='before',
                    memory_mode='bounded_current_state', stage='perception')
    atomic_json(rd/'decision_context.json', decision)
    view = dashboard.Dashboard(tmp_path, '', threading.Event())
    assert view.state()['rounds'][0]['phase'] == '判读中'
    assessment = dict(format='pressb-grounding-v1', observation_id='now', source='model_not_human',
                      head_visible='yes', wrist_visible='no', contact_finger='head_image_right',
                      head_horizontal='right', head_vertical='level', wrist_relation='not_visible',
                      lamp='off', lamp_view='head')
    atomic_json(rd/'perception/assessment.json', assessment)
    atomic_json(rd/'perception/metrics.json', dict(image_count=2, prompt_tokens=1600,
                completion_tokens=100, elapsed_s=3, attempts=1))
    progress = dict(status='unchanged_or_unresolved', before_observation_id='before',
                    current_observation_id='now', previous_command=dict(type='move', xyz_mm=[5,0,0]),
                    source='categorical_model_comparison_not_metric')
    atomic_json(rd/'alignment_progress.json', progress)
    basis = dict(observation_id='now', validated=True, head_visible='yes', wrist_visible='no',
                 expected_effect='reduce_horizontal_offset')
    atomic_json(rd/'decision_basis.json', basis)
    memory = dict(selected_contact='head_image_right', evidence=dict(light_on=None, withdrawal=None))
    atomic_json(rd/'control_memory.json', memory)
    row = view.state()['rounds'][0]
    assert row['phase'] == '动作决策中'
    assert row['perception']['assessment'] == assessment
    assert row['alignment_progress'] == progress and row['decision_basis'] == basis
    assert row['control_memory'] == memory and row['proposal'] is None

    atomic_json(rd/'decision_context.json', dict(decision, stage='activation_verification'))
    assert view.state()['rounds'][0]['phase'] == '亮灯变化核验中'
    activation = dict(observation_id='now', previous_observation_id='before', light_on='unknown', source='model_not_human')
    atomic_json(rd/'activation/assessment.json', activation)
    atomic_json(rd/'activation/metrics.json', dict(image_count=4, prompt_tokens=3400,
                completion_tokens=12, elapsed_s=1.5, attempts=1))
    atomic_json(rd/'decision_context.json', dict(decision, stage='withdrawal_verification'))
    assert view.state()['rounds'][0]['phase'] == '后撤核验中'
    withdrawal = dict(observation_id='now', previous_observation_id='before',
                      withdrawal='unknown', source='model_not_human')
    atomic_json(rd/'withdrawal/assessment.json', withdrawal)
    atomic_json(rd/'withdrawal/metrics.json', dict(image_count=4, prompt_tokens=3200,
                completion_tokens=15, elapsed_s=2, attempts=1))
    atomic_json(rd/'decision_context.json', dict(decision, stage='action_request'))
    atomic_json(rd/'context_metrics.json', dict(image_count=5, memory_mode='bounded_current_state'))
    atomic_json(rd/'action_metrics.json', dict(prompt_tokens=8000, completion_tokens=80, elapsed_s=5, attempts=2))
    atomic_json(rd/'response-0.json', dict(response=dict(usage=dict(prompt_tokens=7900,completion_tokens=30))))
    atomic_json(rd/'response-1.json', dict(response=dict(usage=dict(prompt_tokens=8000,completion_tokens=50))))
    row = view.state()['rounds'][0]
    assert row['phase'] == '动作决策中' and row['context']['image_count'] == 5
    assert row['withdrawal']['assessment']['withdrawal'] == 'unknown'
    assert row['perception']['metrics']['image_count'] == 2
    assert row['activation']['assessment']['light_on'] == 'unknown'
    assert row['activation']['metrics']['image_count'] == 4
    assert row['activation']['metrics']['elapsed_s'] == 1.5
    assert row['withdrawal']['metrics']['image_count'] == 4
    assert row['action_metrics']['attempts'] == 2 and row['elapsed_s'] == 5
    assert row['usage']['completion_tokens'] == 50  # Original response usage stays per-request.


def test_grounded_failure_and_old_round_compatibility(tmp_path):
    rd = tmp_path/'agent/rounds/0000'
    atomic_json(rd/'decision_context.json', dict(format='pressb-conversation-v3', decision_mode='grounded',
                observation_id='now', stage='interrupted'))
    atomic_json(rd/'perception/error.json', dict(error='operator_stopped'))
    d = dashboard.Dashboard(tmp_path, '', threading.Event())
    assert d.state()['rounds'][0]['phase'] == '已停止'
    old = tmp_path/'agent/rounds/0001'
    atomic_json(old/'proposal.json', dict(name='request_observation', arguments=dict(reason='old run')))
    row = d.state()['rounds'][1]
    assert row['phase'] == '请求新观测'
    assert row['alignment_progress'] is None and row['withdrawal']['assessment'] is None
    assert row['activation']['assessment'] is None
    assert row['forward_guard']['assessment'] is None and row['forward_guard']['decision'] is None
    assert row['control_memory'] is None and row['decision_basis'] is None
    assert row['control_plan'] is None


def test_grounded_ui_has_distinct_model_evidence_and_bounded_memory_labels():
    html = dashboard.HTML
    for text in ('头图左侧手指', '头图右侧手指', '水平 ', '高度 ', '动作后检查（模型判断）',
                 '后撤画面核验（模型判断）', '亮灯变化核验（模型判断）', '不是毫米误差测量', '当前状态短上下文',
                 '运行记忆（含模型判断，非人工确认）', '对应观测 ', '格式／一致性校验'):
        assert text in html
    assert 'innerHTML' not in html and 'cached_tokens' in html


def test_activation_error_stops_before_action_display(tmp_path):
    rd = tmp_path/'agent/rounds/0000'
    atomic_json(rd/'decision_context.json', dict(format='pressb-conversation-v3', decision_mode='grounded',
                observation_id='now', stage='interrupted'))
    atomic_json(rd/'activation/error.json', dict(error='operator_stopped'))
    row = dashboard.Dashboard(tmp_path, '', threading.Event()).state()['rounds'][0]
    assert row['phase'] == '已停止' and row['proposal'] is None
    assert row['activation']['error']['error'] == 'operator_stopped'


def test_forward_guard_running_denied_and_repaired_action_keep_separate_status(tmp_path):
    rd = tmp_path/'agent/rounds/0000'
    decision = dict(format='pressb-conversation-v3', decision_mode='grounded',
                    observation_id='wrist-now', stage='forward_guard')
    atomic_json(rd/'decision_context.json', decision)
    view = dashboard.Dashboard(tmp_path, '', threading.Event())
    assert view.state()['rounds'][0]['phase'] == '前推条件检查中'
    assessment = dict(format='pressb-forward-alignment-v1', observation_id='wrist-now',
                      source='model_not_human', button_visible='yes',
                      two_fingertips_visible='yes', button_between_fingers='no')
    guard = dict(observation_id='wrist-now', policy='wrist_horizontal', allowed=False,
                 reasons=['button_not_between_fingers'])
    atomic_json(rd/'forward_guard/assessment.json', assessment)
    atomic_json(rd/'forward_guard/decision.json', guard)
    atomic_json(rd/'forward_guard/metrics.json', dict(image_count=1, attempts=1,
                prompt_tokens=1100, completion_tokens=45, elapsed_s=5.2))
    atomic_json(rd/'forward_guard/response-0.json', dict(response=dict(choices=['first'])))
    row = view.state()['rounds'][0]
    assert row['phase'] == '前推未放行'
    assert row['forward_guard']['assessment'] == assessment
    assert row['forward_guard']['decision'] == guard
    assert row['forward_guard']['metrics']['image_count'] == 1
    assert row['forward_guard']['metrics']['elapsed_s'] == 5.2
    assert row['forward_guard']['response']['response']['choices'] == ['first']
    assert row['proposal'] is None and row['error'] is None
    # A valid replacement action is a new model choice, not an executed +X move.
    atomic_json(rd/'decision_context.json', dict(decision, stage='action_request'))
    atomic_json(rd/'proposal.json', dict(name='propose_move', arguments=dict(
                xyz_mm=[0,-10,10], reason='Adjust alignment before a future approach.')))
    row = view.state()['rounds'][0]
    assert row['phase'] == '已生成建议'
    assert row['forward_guard']['decision']['allowed'] is False
    atomic_json(rd/'execution.json', dict(decision_observation_id='wrist-now'))
    assert view.state()['rounds'][0]['phase'] == '执行 / 采集新观测中'
    atomic_json(rd/'result.json', dict(status='completed'))
    assert view.state()['rounds'][0]['phase'] == '执行完成'


def test_forward_guard_allows_observation_after_denial_and_handles_stop(tmp_path):
    rd = tmp_path/'agent/rounds/0000'
    atomic_json(rd/'decision_context.json', dict(observation_id='wrist-now', stage='action_request'))
    atomic_json(rd/'forward_guard/decision.json', dict(observation_id='wrist-now', allowed=False))
    atomic_json(rd/'proposal.json', dict(name='request_observation', arguments=dict(reason='Reobserve.')))
    view = dashboard.Dashboard(tmp_path, '', threading.Event())
    assert view.state()['rounds'][0]['phase'] == '请求新观测'
    atomic_json(rd/'decision_context.json', dict(observation_id='wrist-now', stage='interrupted'))
    atomic_json(rd/'forward_guard/error.json', dict(error='operator_stopped'))
    atomic_json(tmp_path/'agent/result.json', dict(status='operator_stopped'))
    row = view.state()['rounds'][0]
    assert row['phase'] == '已停止'
    assert row['forward_guard']['error']['error'] == 'operator_stopped'


def test_forward_guard_failed_call_is_distinct_from_denied_condition(tmp_path):
    rd = tmp_path/'agent/rounds/0000'
    atomic_json(rd/'decision_context.json', dict(observation_id='wrist-now', stage='forward_guard'))
    atomic_json(rd/'forward_guard/error.json', dict(error='invalid_guard_assessment'))
    row = dashboard.Dashboard(tmp_path, '', threading.Event()).state()['rounds'][0]
    assert row['phase'] == '前推检查失败'
    assert row['forward_guard']['decision'] is None


def test_forward_guard_ui_explains_wrist_only_gate_and_exposes_independent_metrics():
    html = dashboard.HTML
    for text in ('前推条件检查（模型判断）', '前推未放行', '正X已放行',
                 '按钮横向位置：', '仅检查腕图横向位于两指之间，不代表高度对齐或已经接触',
                 '前推检查原始输出', '拒绝前推本身不代表会话已停止'):
        assert text in html
    assert "stageMetrics(card,'前推检查',guard.metrics)" in html
    assert 'innerHTML' not in html


def test_localized_geometry_and_advisory_control_plan_are_distinct_from_motion(tmp_path):
    rd = tmp_path/'agent/rounds/0000'
    atomic_json(rd/'decision_context.json', dict(format='pressb-conversation-v3', decision_mode='grounded',
                observation_id='snapshot-now', stage='action_request'))
    geometry = dict(head_button_box=[.5,.3,.6,.4], head_finger_tip=[.3,.35],
        delta_image_norm=[.25,0], error_button_units=[2.5,0],
        magnitude=dict(horizontal='large', vertical='small'))
    assessment = dict(format='pressb-grounding-v2', observation_id='snapshot-now', source='model_not_human',
        head_visible='yes', wrist_visible='unknown', contact_finger='head_image_right',
        head_horizontal='right', head_vertical='level', geometry=geometry)
    atomic_json(rd/'perception/assessment.json', assessment)
    plan = dict(format='pressb-control-plan-v1', observation_id='snapshot-now',
        source='program_advice_from_model_judgments_not_human_confirmation', stage='align',
        focus='horizontal', step_hint_mm=20, geometry_progress=dict(comparable=False,
            current_observation_id='snapshot-now', previous_observation_id=None, reason='no_previous_assessment'),
        head_target_info=dict(last_absolute_target_deg=[0,30], current_measured_head_deg=[0,25]))
    atomic_json(rd/'control_plan.json', plan)
    row = dashboard.Dashboard(tmp_path, '', threading.Event()).state()['rounds'][0]
    assert row['perception']['assessment'] == assessment
    assert row['control_plan'] == plan
    assert row['phase'] == '动作决策中' and row['proposal'] is None


def test_localization_ui_labels_units_model_source_and_snapshot_binding():
    html = dashboard.HTML
    assert "['pressb-grounding-v1','pressb-grounding-v2'].includes(a?.format)" in html
    for text in ('个按钮宽度', '个按钮高度', '按钮中心减所选手指前端',
                 '由模型定位计算，不是毫米校准', '不叠加到实时视频',
                 '控制策略（由模型判断派生的建议）', '策略阶段：', '建议步长：',
                 '建议步长可调整，不是执行限幅', '完整控制策略、前后偏差与头部目标',
                 '当前两次定位不能可靠比较'):
        assert text in html
    assert 'innerHTML' not in html
