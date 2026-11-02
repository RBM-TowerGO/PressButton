"""Progress evidence is displayed separately from executed actions and live video."""
import json
import shutil
import subprocess
import threading

import pytest

import s1b_agent_dashboard as dashboard
from pressb.agent_loop.common import atomic_json


def context(stage='progress_review'):
    return dict(format='pressb-conversation-v1', decision_mode='direct',
                observation_id='now', previous_observation_id='before', stage=stage)


def assessment():
    return dict(format='pressb-progress-review-v1', source='model_not_human',
                observation_id='now', previous_observation_id='before',
                head_vertical='unknown', head_horizontal='unknown',
                wrist_vertical='level', wrist_horizontal='left',
                vertical_change='improved', horizontal_change='unchanged',
                approach_change='unknown', note='Button center is now level with the contact end.')


def progress():
    return dict(format='pressb-progress-v1', observation_id='now',
                candidate_progress=dict(observation_id='now', subgoal='align_vertical',
                    view='wrist', button_vertical='above', button_horizontal='left',
                    last_effect='improved'),
                monitor=dict(review_required=True, reasons=['repeated_positive_Z'],
                    streaks=dict(Z=dict(count=3, delta_mm=30))))


def test_review_running_denial_replacement_and_execution_keep_distinct_states(tmp_path):
    rd = tmp_path/'agent/rounds/0000'
    atomic_json(rd/'decision_context.json', context())
    atomic_json(rd/'progress.json', progress())
    view = dashboard.Dashboard(tmp_path, '', threading.Event())
    row = view.state()['rounds'][0]
    assert row['phase'] == '进展复核中'
    assert row['progress'] == progress()
    assert row['proposal'] is None
    atomic_json(rd/'progress_review/assessment.json', assessment())
    atomic_json(rd/'progress_review/metrics.json', dict(image_count=4, prompt_tokens=3000,
                completion_tokens=80, elapsed_s=3.25, attempts=1))
    atomic_json(rd/'progress_review/response-0.json', dict(response=dict(choices=['first'])))
    atomic_json(rd/'progress_review/response-1.json', dict(response=dict(choices=['repaired'])))
    denied = dict(observation_id='now', allowed=False, blocked_same_direction=True,
                  reasons=['vertical_relation_already_level'])
    atomic_json(rd/'progress_review/decision.json', denied)
    atomic_json(rd/'decision_context.json', context('action_request'))
    row = view.state()['rounds'][0]
    assert row['phase'] == '进展复核：需重新决策'
    assert row['progress_review']['assessment'] == assessment()
    assert row['progress_review']['decision'] == denied
    assert row['progress_review']['metrics']['image_count'] == 4
    assert row['progress_review']['response']['response']['choices'] == ['repaired']
    atomic_json(rd/'proposal.json', dict(name='propose_head', arguments=dict(
                head_deg=[0, 30], reason='Improve the global view.')))
    assert view.state()['rounds'][0]['phase'] == '已生成建议'
    atomic_json(rd/'execution.json', dict(decision_observation_id='now'))
    assert view.state()['rounds'][0]['phase'] == '执行 / 采集新观测中'
    atomic_json(rd/'result.json', dict(status='completed'))
    row = view.state()['rounds'][0]
    assert row['phase'] == '执行完成'
    assert row['progress_review']['decision']['allowed'] is False


@pytest.mark.parametrize('with_round_error', [False, True])
def test_review_failure_is_distinct_and_stop_preserves_evidence(tmp_path, with_round_error):
    rd = tmp_path/'agent/rounds/0000'
    atomic_json(rd/'decision_context.json', context('failed'))
    atomic_json(rd/'progress_review/error.json', dict(error='invalid_review'))
    atomic_json(rd/'progress_review/metrics.json', dict(image_count=4, attempts=2))
    if with_round_error:
        atomic_json(rd/'error.json', dict(stage='progress_review', error='invalid_review'))
    view = dashboard.Dashboard(tmp_path, '', threading.Event())
    assert view.state()['rounds'][0]['phase'] == '进展复核失败'
    atomic_json(rd/'decision_context.json', context('interrupted'))
    atomic_json(tmp_path/'agent/result.json', dict(status='operator_stopped'))
    row = view.state()['rounds'][0]
    assert row['phase'] == '已停止'
    assert row['progress_review']['metrics']['attempts'] == 2
    assert row['progress_review']['error']['error'] == 'invalid_review'
    assert row['proposal'] is None


def test_old_runs_without_progress_and_large_results_remain_readable(tmp_path, monkeypatch):
    rd = tmp_path/'agent/rounds/0000'
    atomic_json(rd/'proposal.json', dict(name='propose_move', arguments=dict(xyz_mm=[0, 0, 10], reason='old')))
    atomic_json(rd/'result.json', dict(motion=dict(state_trace=['not for dashboard'])))
    original = dashboard.read_json
    def checked(path, default=None):
        assert path != rd/'result.json', 'Do not reload per-command state traces on UI polls.'
        return original(path, default)
    monkeypatch.setattr(dashboard, 'read_json', checked)
    row = dashboard.Dashboard(tmp_path, '', threading.Event()).state()['rounds'][0]
    assert row['phase'] == '执行完成'
    assert row['progress'] is None and row['progress_review']['assessment'] is None
    assert row['progress_review']['metrics'] == {}


def test_progress_renderer_displays_provenance_binding_and_metrics_without_html_execution():
    node = shutil.which('node')
    if not node:
        pytest.skip('node is unavailable for the DOM rendering check')
    malicious_note = '<img src=x onerror="throw new Error(123)">'
    row = dict(progress=progress(), decision=context(), progress_review=dict(
        assessment=dict(assessment(), note=malicious_note),
        decision=dict(allowed=False, reasons=['already_level']),
        metrics=dict(prompt_tokens=3000, completion_tokens=80, image_count=4, elapsed_s=3.25, attempts=1)))
    script = dashboard.HTML.split('<script>', 1)[1].split('</script>', 1)[0]
    harness = '''
class Element {
  constructor(){this.children=[];this.textContent='';this.className='';}
  append(...nodes){this.children.push(...nodes);}
}
const elements=new Map();
globalThis.document={createElement:()=>new Element(),getElementById:id=>{
  if(!elements.has(id))elements.set(id,new Element());return elements.get(id);
}};
globalThis.fetch=()=>new Promise(()=>{});
'''
    check = '''
const card=new Element();renderProgress(card,ROW);
function texts(e){return [e.textContent,...e.children.flatMap(texts)];}
process.stdout.write(JSON.stringify(texts(card)));
'''.replace('ROW', json.dumps(row, ensure_ascii=False))
    result = subprocess.run([node, '-e', harness+script+check], text=True,
                            capture_output=True, timeout=10, check=True)
    rendered = '\n'.join(json.loads(result.stdout))
    for expected in ('动作模型自评', '程序仅统计实际完成发送的命令', '次数与累计位移不代表对齐改善',
                     '独立模型判断，非人工确认', 'before → now', '同高', '高度 改善',
                     '本候选同向重复未获支持', '3000', '80', '4 张图', '3.3 秒', malicious_note):
        assert expected in rendered
    assert 'innerHTML' not in script


def test_latest_numeric_response_reads_only_response_ten_not_nine(tmp_path, monkeypatch):
    rd = tmp_path/'agent/rounds/0000'
    for number in (0, 1, 2, 9, 10):
        atomic_json(rd/f'response-{number}.json', dict(response=dict(choices=[number],
                    usage=dict(prompt_tokens=1000+number, completion_tokens=number)), elapsed_s=number))
    atomic_json(rd/'response-backup.json', dict(response=dict(choices=['invalid-name'])))
    atomic_json(rd/'decision_context.json', context('action_request'))
    atomic_json(rd/'action_metrics.json', dict(attempts=11, elapsed_s=55, prompt_tokens=11000, completion_tokens=110))
    atomic_json(rd/'correction.json', dict(attempt=10, next_attempt=11, error='Still not aligned',
                observation_id='now', status='retrying'))
    read = []
    original = dashboard.read_json
    def track(path, default=None):
        if path.parent == rd and path.name.startswith('response-'):
            read.append(path.name)
        return original(path, default)
    monkeypatch.setattr(dashboard, 'read_json', track)
    row = dashboard.Dashboard(tmp_path, '', threading.Event()).state()['rounds'][0]
    assert read == ['response-10.json']
    assert row['response']['choices'] == [10] and row['response_attempt'] == 10
    assert row['usage']['prompt_tokens'] == 1010 and row['elapsed_s'] == 55
    assert row['phase'] == '动作修正中'
    assert row['correction']['next_attempt'] == 11 and row['action_metrics']['attempts'] == 11


@pytest.mark.parametrize('name,stage,phase', [
    ('perception', 'perception', '判读修正中'),
    ('activation', 'activation_verification', '亮灯核验修正中'),
    ('withdrawal', 'withdrawal_verification', '后撤核验修正中'),
    ('forward_guard', 'forward_guard', '前推检查修正中'),
    ('progress_review', 'progress_review', '进展复核修正中'),
])
def test_readonly_validation_retries_are_live_and_accepted_assessment_wins(tmp_path, name, stage, phase):
    rd = tmp_path/'agent/rounds/0000'
    atomic_json(rd/'decision_context.json', context(stage))
    for number in (9, 10):
        atomic_json(rd/name/f'response-{number}.json', dict(response=dict(choices=[number])))
        atomic_json(rd/name/f'validation-{number}.json', dict(error=f'invalid-{number}', attempt=number))
    atomic_json(rd/name/'metrics.json', dict(attempts=11, prompt_tokens=100, completion_tokens=11,
                image_count=4, elapsed_s=12.3))
    view = dashboard.Dashboard(tmp_path, '', threading.Event())
    row = view.state()['rounds'][0]
    assert row['phase'] == phase
    assert row[name]['response']['response']['choices'] == [10]
    assert row[name]['correction']['error'] == 'invalid-10'
    assert row[name]['correction']['next_attempt'] == 11
    assert row[name]['metrics']['attempts'] == 11
    atomic_json(rd/name/'assessment.json', dict(observation_id='now', source='model_not_human'))
    row = view.state()['rounds'][0]
    assert row[name]['correction']['status'] == 'accepted'
    assert row['phase'] == phase.replace('修正中', '完成')
    assert row[name]['correction']['error'] == 'invalid-10'


def test_unlimited_action_correction_can_stop_without_executing_a_candidate(tmp_path):
    rd = tmp_path/'agent/rounds/0000'
    atomic_json(rd/'decision_context.json', context('action_request'))
    atomic_json(rd/'correction.json', dict(attempt=29, next_attempt=30, error='unsupported repetition',
                observation_id='now', status='retrying'))
    atomic_json(rd/'action_metrics.json', dict(attempts=30, prompt_tokens=30000, completion_tokens=3000))
    view = dashboard.Dashboard(tmp_path, '', threading.Event())
    assert view.state()['rounds'][0]['phase'] == '动作修正中'
    atomic_json(rd/'decision_context.json', context('interrupted'))
    atomic_json(rd/'error.json', dict(stage='model_request', error='operator_stopped'))
    atomic_json(tmp_path/'agent/result.json', dict(status='operator_stopped'))
    row = view.state()['rounds'][0]
    assert row['phase'] == '已停止' and row['proposal'] is None
    assert row['correction']['error'] == 'unsupported repetition'
    assert row['action_metrics']['attempts'] == 30


def test_action_proposal_ends_correction_display_but_preserves_last_rejection(tmp_path):
    rd = tmp_path/'agent/rounds/0000'
    atomic_json(rd/'decision_context.json', context('action_request'))
    atomic_json(rd/'correction.json', dict(attempt=10, next_attempt=11, error='old rejection', status='retrying'))
    atomic_json(rd/'proposal.json', dict(name='request_observation', arguments=dict(reason='Reobserve.')))
    row = dashboard.Dashboard(tmp_path, '', threading.Event()).state()['rounds'][0]
    assert row['phase'] == '请求新观测'
    assert row['correction']['status'] == 'accepted'
    assert row['correction']['error'] == 'old rejection'


def test_compaction_correction_has_latest_response_and_cannot_mask_stop(tmp_path):
    rd = tmp_path/'agent/rounds/0000'
    atomic_json(rd/'decision_context.json', context('context_prepare'))
    atomic_json(rd/'context_metrics.json', dict(compression_status='running'))
    for number in (9, 10):
        atomic_json(rd/'compaction'/f'response-{number}.json', dict(choices=[number]))
        atomic_json(rd/'compaction'/f'validation-{number}.json', dict(error='unknown evidence ID', attempt=number))
    atomic_json(rd/'compaction/metrics.json', dict(attempts=11))
    view = dashboard.Dashboard(tmp_path, '', threading.Event())
    row = view.state()['rounds'][0]
    assert row['phase'] == '压缩摘要修正中'
    assert row['compaction']['response']['choices'] == [10]
    assert row['compaction']['correction']['error'] == 'unknown evidence ID'
    atomic_json(tmp_path/'agent/result.json', dict(status='operator_stopped'))
    assert view.state()['rounds'][0]['phase'] == '已停止'


def test_correction_ui_has_unlimited_retry_count_and_latest_reason():
    html = dashboard.HTML
    assert 'renderCorrections(card,row)' in html
    for label in ('动作', '判读', '进展复核', '压缩摘要', '累计请求', '最近拒绝原因',
                  '不设修正次数上限，仍可停止', '修正后已通过'):
        assert label in html
    assert 'innerHTML' not in html

