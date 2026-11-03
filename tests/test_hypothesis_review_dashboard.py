"""Conditional read-only hypothesis review remains distinct from action output."""
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import s1b_agent_dashboard as dashboard
from pressb.agent_loop.common import atomic_json


def setup_round(tmp_path, stage='hypothesis_review'):
    rd = tmp_path/'agent/rounds/0003'
    atomic_json(rd/'decision_context.json', dict(format='pressb-conversation-v5',
        decision_mode='localized', stage=stage, observation_id='now', previous_observation_id='last',
        localization_format='pressb-localization-v3', hypothesis_review_format='pressb-hypothesis-review-v1',
        execution_policy='model_decides_after_progress_review', review_phase='reviewing'))
    atomic_json(rd/'localization/assessment.json', dict(format='pressb-localization-v3', observation_id='now'))
    atomic_json(rd/'hypothesis_review/trigger.json', dict(observation_id='now', baseline_observation_id='start',
        reasons=['repeated_direction_without_progress'], count=3, total_mm=15))
    return rd, dashboard.Dashboard(tmp_path, '', threading.Event())


def test_review_pending_and_result_are_separate_from_action(tmp_path):
    rd, view = setup_round(tmp_path)
    atomic_json(rd/'hypothesis_review/metrics.json', dict(image_count=4, attempts=1, prompt_tokens=9000))
    row = view.state()['rounds'][0]
    assert row['phase'] == '无进展假设复核中'
    assert row['hypothesis_review']['trigger']['baseline_observation_id'] == 'start'
    assert row['hypothesis_review']['metrics']['image_count'] == 4
    assert row['proposal'] is None
    atomic_json(rd/'hypothesis_review/assessment.json', dict(format='pressb-hypothesis-review-v1',
        source='model_not_human', observation_id='now', previous_observation_id='start',
        observed_effect='unchanged', distance_only_hypothesis='unsupported', prediction_check='not_met', next_check='Check lateral alignment.'))
    atomic_json(rd/'hypothesis_review/decision.json', dict(expected_visual_change='Button approaches marked tip.'))
    row = view.state()['rounds'][0]
    assert row['hypothesis_review']['assessment']['source'] == 'model_not_human'
    assert row['hypothesis_review']['assessment']['prediction_check'] == 'not_met'
    assert row['hypothesis_review']['decision']['expected_visual_change']
    assert row['proposal'] is None


def test_review_failure_and_operator_stop_visible(tmp_path):
    rd, view = setup_round(tmp_path)
    atomic_json(rd/'hypothesis_review/error.json', dict(stage='hypothesis_review', error='timeout'))
    assert view.state()['rounds'][0]['phase'] == '假设复核失败'
    atomic_json(tmp_path/'agent/result.json', dict(status='operator_stopped'))
    context = dict(format='pressb-conversation-v5', decision_mode='localized', stage='interrupted')
    atomic_json(rd/'decision_context.json', context)
    assert view.state()['rounds'][0]['phase'] == '已停止'


def test_page_renders_review_and_bound_observations():
    html = dashboard.HTML
    for text in ('renderHypothesisReview(card,row)', 'hypothesis_review', 'baseline_observation_id',
                 'distance_only_hypothesis', 'prediction_check', 'alternative_explanation', '下一步应核对',
                 '无进展后的假设复核（模型判断）', "stageMetrics(card,'假设复核',review.metrics)"):
        assert text in html


def rendering_harness():
    import re
    script = re.search(r'<script>(.*?)</script>', dashboard.HTML, re.S).group(1)
    prefix = script.split('async function stateTick', 1)[0]
    return '''
class Element {
 constructor(tag){this.tag=tag;this.children=[];this.attrs={};this.textContent='';}
 append(...values){this.children.push(...values);}
 setAttribute(k,v){this.attrs[k]=v;}
}
const document={createElement:tag=>new Element(tag),createElementNS:(ns,tag)=>new Element(tag),getElementById:()=>new Element('div')};
'''+prefix+'''
const ui=new Element('div');
renderHypothesisReview(ui,{decision:{stage:'hypothesis_review',observation_id:'current'},
 hypothesis_review:{trigger:{baseline_observation_id:'baseline',observation_id:'current',count:3},
 assessment:{previous_observation_id:'baseline',observation_id:'current',observed_effect:'unchanged',
 distance_only_hypothesis:'unsupported',prediction_check:'not_met',evidence:'target above marker',alternative_explanation:'height mismatch',next_check:'compare both views'},
 metrics:{image_count:4,prompt_tokens:8000,completion_tokens:120,elapsed_s:4.5,attempts:1}}});
const allText=e=>e.textContent+' '+e.children.map(allText).join(' ');
const renderedText=allText(ui);
for(const expected of ['baseline → current','上次预期是否出现：未出现','缺乏支持','height mismatch','compare both views','8000','4 张图','4.5 秒','非人工确认']){
 if(!renderedText.includes(expected))throw Error('missing review field: '+expected);
}
const old=new Element('div');renderHypothesisReview(old,{decision:{}});
if(old.children.length!==0)throw Error('old round has phantom review');
const ordinary=new Element('div');renderHypothesisReview(ordinary,{hypothesis_review:{trigger:{review_required:false},metrics:{}}});
if(ordinary.children.length!==0)throw Error('untriggered round displayed as a model review');
return 'ok';
'''


def test_review_card_runtime_when_node_available():
    import shutil
    import subprocess
    import pytest
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node unavailable; same harness can run in tool V8 runtime')
    result = subprocess.run([node, '-e', 'console.log((function(){'+rendering_harness()+'})())'],
                            capture_output=True, text=True, check=True)
    assert result.stdout.strip() == 'ok'
