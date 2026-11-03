"""Current wrist overlap display and legacy horizontal gate remain distinct."""
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import s1b_agent_dashboard as dashboard
from pressb.agent_loop.common import atomic_json


def test_current_wrist_gate_is_separate_from_candidate_execution(tmp_path):
    rd = tmp_path/'agent/rounds/0000'
    atomic_json(rd/'decision_context.json', dict(format='pressb-conversation-v5', decision_mode='localized',
        stage='forward_guard', observation_id='current', execution_policy='yz_alignment_before_forward',
        forward_guard_policy='wrist_tip_overlap'))
    view = dashboard.Dashboard(tmp_path, '', threading.Event())
    assert view.state()['rounds'][0]['phase'] == '前推条件检查中'
    atomic_json(rd/'forward_guard/assessment.json', dict(format='pressb-wrist-tip-overlap-v1',
        source='model_not_human', observation_id='current', tip_inside_button='no'))
    atomic_json(rd/'forward_guard/decision.json', dict(policy='wrist_tip_overlap', allowed=False,
        observation_id='current', assessment_observation_id='current'))
    atomic_json(rd/'forward_guard/candidate-0.json', dict(name='propose_move', arguments=dict(xyz_mm=[30, 5, 0])))
    atomic_json(rd/'forward_guard/metrics.json', dict(image_count=1, attempts=1, elapsed_s=2))
    row = view.state()['rounds'][0]
    assert row['phase'] == '前推未放行'
    assert row['proposal'] is None
    assert row['forward_guard']['metrics']['image_count'] == 1
    atomic_json(rd/'forward_guard/error.json', dict(stage='forward_guard', error='operator_stopped'))
    atomic_json(rd/'decision_context.json', dict(stage='interrupted'))
    assert view.state()['rounds'][0]['phase'] == '已停止'


def rendering_harness():
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
renderForwardGuard(ui,{decision:{observation_id:'current',forward_guard_policy:'wrist_tip_overlap'},
 forward_guard:{assessment:{format:'pressb-wrist-tip-overlap-v1',observation_id:'current',button_visibility:'full',
 contact_visibility:'full',button_box:[.2,.3,.4,.5],contact_tip:[.3,.4],tip_inside_button:'yes',geometry:{containment:true}},
 decision:{policy:'wrist_tip_overlap',allowed:true,observation_id:'current'},metrics:{image_count:1,elapsed_s:2.5,prompt_tokens:1000}}});
const allText=e=>e.textContent+' '+e.children.map(allText).join(' ');
const renderedText=allText(ui);
for(const expected of ['腕图二维点入框','正X已放行','箭头所指物理指端','横纵均在框内','无正X步长上限','不证明三维接触','current','1 张图']){
 if(!renderedText.includes(expected))throw Error('missing overlap field: '+expected);
}
if(renderedText.includes('两指之间'))throw Error('old horizontal criterion leaked into new policy');
const walk=e=>[e,...e.children.flatMap(walk)];
const images=walk(ui).filter(e=>e.tag==='img');
if(images.length!==1||images[0].src!=='/api/observations/current/right.jpg')throw Error('gate image not bound to current wrist snapshot');
const old=new Element('div');renderForwardGuard(old,{forward_guard:{assessment:{observation_id:'old',button_visible:'yes',
 two_fingertips_visible:'yes',button_between_fingers:'yes'},decision:{allowed:true,policy:'wrist_horizontal'},metrics:{}}});
if(!allText(old).includes('两指之间')||allText(old).includes('无正X步长上限'))throw Error('legacy horizontal meaning changed');
return 'ok';
'''


def test_gate_card_runtime_when_node_available():
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node unavailable; same harness can run in tool V8 runtime')
    result = subprocess.run([node, '-e', 'console.log((function(){'+rendering_harness()+'})())'],
                            capture_output=True, text=True, check=True)
    assert result.stdout.strip() == 'ok'
