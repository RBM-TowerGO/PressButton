"""Bound saved-image overlays and localized-stage archives; no robot connection."""
import json
import re
import shutil
import subprocess
import threading
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

import s1b_agent_dashboard as dashboard
from pressb.agent_loop.common import atomic_json


def context(stage='localization', revision=1):
    return dict(format='pressb-conversation-v4', decision_mode='localized',
                observation_id='current', previous_observation_id='before', stage=stage,
                localization_revision=revision, view_phase='ready')


def assessment(oid='current', revision=1):
    view = dict(button_visibility='full', contact_visibility='full',
                button_box=[.2, .2, .4, .4], arrow_box=[.5, .5, .7, .7], contact_tip=[.6, .6],
                image_size=[640, 360], note='Visible target and marked physical end.',
                derived_relation=dict(horizontal='left', vertical='above', dx=-.3, dy=-.3,
                                      pixel_delta=[-192, -108], error_button_units=[-1.5, -1.5]))
    return dict(format='pressb-localization-v1', source='model_not_human', observation_id=oid,
                revision=revision, usable_views=['head', 'wrist'], head=view, wrist=view)


def test_localization_and_review_are_independent_stages_with_original_versions(tmp_path):
    rd = tmp_path/'agent/rounds/0000'
    atomic_json(rd/'decision_context.json', context())
    view = dashboard.Dashboard(tmp_path, '', threading.Event())
    row = view.state()['rounds'][0]
    assert row['phase'] == '双视角定位中' and row['proposal'] is None
    atomic_json(rd/'localization/assessment.json', assessment())
    atomic_json(rd/'localization/metrics.json', dict(attempts=1, prompt_tokens=2500,
                completion_tokens=220, image_count=2, elapsed_s=4.5))
    atomic_json(rd/'decision_context.json', context('localization_review', 2))
    atomic_json(rd/'localization_review/before/assessment.json', assessment('before'))
    atomic_json(rd/'localization_review/current/assessment.json', assessment(revision=2))
    atomic_json(rd/'localization_review/current/metrics.json', dict(attempts=2, image_count=2))
    atomic_json(rd/'localization_current.json', dict(assessment(revision=2), usable_views=['wrist']))
    atomic_json(rd/'localized_progress.json', dict(observation_id='current', needs_recheck=True))
    row = view.state()['rounds'][0]
    assert row['phase'] == '重新定位与进展复核中'
    assert row['localization']['assessment']['revision'] == 1
    assert row['localization_current']['revision'] == 2
    assert row['localization_review']['before']['assessment']['observation_id'] == 'before'
    assert row['localization_review']['current']['metrics']['attempts'] == 2
    assert row['localized_progress']['needs_recheck']
    assert row['perception']['assessment'] is None
    assert 'data:image/' not in json.dumps(row)


@pytest.mark.parametrize('stage,folder,label', [
    ('localization', 'localization', '定位格式修正中'),
    ('localization_review', 'localization_review/current', '复核定位格式修正中'),
])
def test_localization_repairs_and_stop_are_visible(tmp_path, stage, folder, label):
    rd = tmp_path/'agent/rounds/0000'
    atomic_json(rd/'decision_context.json', context(stage))
    atomic_json(rd/folder/'validation-10.json', dict(attempt=10, error='wrong frame'))
    atomic_json(rd/folder/'response-10.json', dict(response=dict(choices=[])))
    view = dashboard.Dashboard(tmp_path, '', threading.Event())
    assert view.state()['rounds'][0]['phase'] == label
    atomic_json(rd/'decision_context.json', context('interrupted'))
    atomic_json(rd/folder/'error.json', dict(error='operator_stopped'))
    assert view.state()['rounds'][0]['phase'] == '已停止'


def test_review_transport_failure_keeps_metrics_without_any_review_assessment(tmp_path):
    rd = tmp_path/'agent/rounds/0000'
    atomic_json(rd/'decision_context.json', context('failed'))
    atomic_json(rd/'localization/assessment.json', assessment())
    atomic_json(rd/'localization_review/before/metrics.json', dict(attempts=1, elapsed_s=120))
    atomic_json(rd/'localization_review/before/error.json', dict(error='model_timeout'))
    row = dashboard.Dashboard(tmp_path, '', threading.Event()).state()['rounds'][0]
    assert row['phase'] == '定位复核失败'
    assert row['localization_review']['before']['assessment'] is None
    assert row['localization_review']['before']['metrics']['attempts'] == 1
    assert row['localization_review']['before']['error']['error'] == 'model_timeout'


def test_snapshot_endpoint_only_serves_this_runs_bound_images(tmp_path):
    rd = tmp_path/'agent/observations/current'
    rd.mkdir(parents=True)
    (rd/'head.jpg').write_bytes(b'head-jpeg')
    (rd/'right.jpg').write_bytes(b'wrist-jpeg')
    outside = tmp_path/'secret.jpg'
    outside.write_bytes(b'private')
    (rd/'other.jpg').write_bytes(b'not-allowed')
    linked = tmp_path/'agent/observations/linked'
    linked.mkdir()
    (linked/'head.jpg').symlink_to(outside)
    view = dashboard.Dashboard(tmp_path, '', threading.Event())
    server, thread = dashboard.serve(view, 0)
    base = 'http://127.0.0.1:'+str(server.server_port)
    try:
        with urlopen(base+'/api/observations/current/head.jpg') as reply:
            assert reply.read() == b'head-jpeg'
            assert reply.headers['Content-Type'] == 'image/jpeg'
            assert reply.headers['X-Content-Type-Options'] == 'nosniff'
        with urlopen(base+'/api/observations/current/right.jpg') as reply:
            assert reply.read() == b'wrist-jpeg'
        for path in ('current/other.jpg', 'unknown/head.jpg', 'linked/head.jpg',
                     '../secret.jpg', '%2e%2e/head.jpg', 'current%2f..%2f../head.jpg'):
            with pytest.raises(HTTPError) as error:
                urlopen(base+'/api/observations/'+path)
            assert error.value.code == 404
        with pytest.raises(FileNotFoundError):
            view.observation_image('../', 'head')
    finally:
        server.shutdown(); server.server_close(); thread.join(2)


def test_localized_ui_keeps_snapshots_separate_from_live_and_uses_svg():
    html = dashboard.HTML
    for value in ('renderLocalization(card,row)', 'localizationOverlay(', 'createElementNS',
                  '保存快照上的模型定位', '程序计算', '当前有效定位', '定位版本',
                  "svg.setAttribute('preserveAspectRatio','none')", 'vector-effect',
                  'localization_current', '/api/observations/', '图像' ):
        assert value in html
    assert 'innerHTML' not in html
    overlay = html.split('function localizationOverlay(', 1)[1].split('function renderLocalizationAssessment', 1)[0]
    assert "$('head')" not in overlay and "$('right')" not in overlay


def test_svg_coordinates_scale_with_saved_image_and_no_extra_head_gate():
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node unavailable; rendering can also be checked with the tool V8 runtime')
    script = re.search(r'<script>(.*?)</script>', dashboard.HTML, re.S).group(1)
    prefix = script.split('async function stateTick', 1)[0]
    harness = '''
class Element {
 constructor(tag){this.tag=tag;this.children=[];this.attrs={};this.textContent='';}
 append(...values){this.children.push(...values);}
 setAttribute(k,v){this.attrs[k]=v;}
}
const document={createElement:tag=>new Element(tag),createElementNS:(ns,tag)=>new Element(tag),getElementById:()=>new Element('div')};
'''+prefix+'''
const parent=new Element('div');
localizationOverlay(parent,'current','wrist',DATA.wrist);
const frame=parent.children[0],img=frame.children[0],svg=frame.children[1];
if(img.src!='/api/observations/current/right.jpg')throw Error('wrong snapshot');
if(svg.attrs.viewBox!=='0 0 1000 1000'||svg.attrs.preserveAspectRatio!=='none')throw Error('scaling');
const box=svg.children[0];
if(box.attrs.x!=='200'||box.attrs.width!=='200')throw Error('normalized geometry');
const ui=new Element('div');
renderLocalization(ui,{decision:{decision_mode:'localized',view_phase:'ready'},localization:{assessment:DATA,metrics:{}}});
const allText=(e)=>e.textContent+' '+e.children.map(allText).join(' ');
if(!allText(ui).includes('任一视角定位有效'))throw Error('HEAD gate survived');
console.log('ok');
'''
    result = subprocess.run([node, '-e', 'const DATA='+json.dumps(assessment())+';\n'+harness],
                            capture_output=True, text=True, check=True)
    assert result.stdout.strip() == 'ok'


def model_context(stage='action_request'):
    return dict(context(stage), format='pressb-conversation-v5', execution_policy='model_decides',
                forward_guard_policy='none', localization_format='pressb-localization-v2',
                view_phase='model_decides')


def advisory_assessment():
    from pressb.agent_loop import model_localization
    raw = {name: dict(button_visibility='partial', contact_visibility='unknown',
                     button_box=[.2, .2, .4, .4], arrow_box=[.5, .5, .7, .7], contact_tip=[.6, .6],
                     note='Partial button and uncertain contact identity.') for name in ('head', 'wrist')}
    raw['coordinate_units'] = 'normalized_01'
    return model_localization.normalize(raw, 'current', dict(head=[640, 360], wrist=[640, 360]),
        raw_output=[{'content': 'Coordinate output with uncertainty'}], parse_notes=['advisory parsing note'])


def test_v5_parsing_notes_continue_to_action_without_semantic_rejection(tmp_path):
    rd = tmp_path/'agent/rounds/0000'
    atomic_json(rd/'decision_context.json', model_context())
    atomic_json(rd/'localization/assessment.json', advisory_assessment())
    atomic_json(rd/'localization_current.json', advisory_assessment())
    atomic_json(rd/'localized_progress.json', dict(format='pressb-localized-progress-v2',
                observation_id='current', previous_observation_id='before', raw_change=[.3, -.2]))
    row = dashboard.Dashboard(tmp_path, '', threading.Event()).state()['rounds'][0]
    assert row['phase'] == '动作决策中'
    assert row['localization_current']['wrist']['button_visibility'] == 'partial'
    assert row['localization_current']['parse_notes'] == ['advisory parsing note']
    assert row['decision']['forward_guard_policy'] == 'none'
    assert row['localization']['correction'] is None
    assert row['forward_guard']['assessment'] is None
    assert 'data:image/' not in json.dumps(row)


def test_v5_transport_failure_is_distinct_from_a_visual_judgment(tmp_path):
    rd = tmp_path/'agent/rounds/0000'
    atomic_json(rd/'decision_context.json', model_context('failed'))
    atomic_json(rd/'localization/error.json', dict(error='model_timeout'))
    assert dashboard.Dashboard(tmp_path, '', threading.Event()).state()['rounds'][0]['phase'] == '定位请求失败'


def advisory_rendering_harness():
    script = re.search(r'<script>(.*?)</script>', dashboard.HTML, re.S).group(1)
    prefix = script.split('async function stateTick', 1)[0]
    return 'const DATA='+json.dumps(advisory_assessment())+';\n'+'''
class Element {
 constructor(tag){this.tag=tag;this.children=[];this.attrs={};this.textContent='';}
 append(...values){this.children.push(...values);}
 setAttribute(k,v){this.attrs[k]=v;}
}
const document={createElement:tag=>new Element(tag),createElementNS:(ns,tag)=>new Element(tag),getElementById:()=>new Element('div')};
'''+prefix+'''
const ui=new Element('div');
renderLocalization(ui,{decision:{format:'pressb-conversation-v5',decision_mode:'localized',execution_policy:'model_decides'},
 localization:{assessment:DATA,metrics:{attempts:1}}, localized_progress:{raw_change:[0.1,0.2]}});
const allText=(e)=>e.textContent+' '+e.children.map(allText).join(' ');
const text=allText(ui);
for(const expected of ['由模型判断行动','本轮定位（模型判断）','advisory parsing note','Coordinate output with uncertainty','原始差值、状态与命令']){
 if(!text.includes(expected))throw Error('missing v5 text: '+expected);
}
for(const forbidden of ['任一视角定位有效','两视角均无有效定位','前推另需','复核触发记录','当前有效定位']){
 if(text.includes(forbidden))throw Error('legacy semantic gate displayed: '+forbidden);
}
const descendants=e=>[e,...e.children.flatMap(descendants)];
const svgs=descendants(ui).filter(e=>e.tag==='svg');
if(svgs.length!==2)throw Error('missing snapshot overlays');
for(const svg of svgs){
 if(svg.children.filter(e=>e.tag==='line').length!==3)throw Error('partial visibility suppressed coordinate line');
 if(svg.attrs.viewBox!=='0 0 1000 1000')throw Error('wrong overlay scaling');
}
const invalid=new Element('div');
localizationOverlay(invalid,'old','wrist',{button_box:[180,0,390,130],arrow_box:[670,730,790,870],contact_tip:[690,760]},true);
if(invalid.children[0].children[1].children.length!==0)throw Error('invalid legacy coordinates drawn outside SVG');
const scaled=new Element('div');
localizationOverlay(scaled,'current','wrist',{button_box:[.18,0,.39,.13],contact_tip:[.69,.76]},true);
if(scaled.children[0].children[1].children[0].attrs.x!=='180')throw Error('canonical coordinates converted twice');
const old=new Element('div');
renderLocalization(old,{decision:{decision_mode:'localized',view_phase:'get_view'},localization:{assessment:null,metrics:{}}});
if(!allText(old).includes('两视角均无有效定位'))throw Error('v4 archive display changed');
'''


def test_v5_ui_explains_model_authority_and_draws_partial_coordinate_output():
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node unavailable; use advisory_rendering_harness with the tool V8 runtime')
    subprocess.run([node, '-e', advisory_rendering_harness()], capture_output=True, text=True, check=True)
