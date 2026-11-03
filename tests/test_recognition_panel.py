"""Saved Agent recognition previews never borrow live or another round's pixels."""
import json
import re
import shutil
import subprocess

import pytest

import s1b_agent_dashboard as dashboard


def rendering_harness():
    """Return standalone JS for Node or the tool V8 runtime; completion is JSON-safe."""
    script = re.search(r'<script>(.*?)</script>', dashboard.HTML, re.S).group(1)
    prefix = script.split('async function stateTick', 1)[0]
    return r'''
class Element {
 constructor(tag='div'){
  this.tag=tag;this.children=[];this.attrs={};this._text='';this.value='';
  this.checked=false;this.hidden=false;this.style={};this.parentNode=null;
 }
 set textContent(value){this._text=String(value??'');this.children=[];}
 get textContent(){return this._text;}
 append(...values){for(const value of values){this.children.push(value);if(value&&typeof value==='object')value.parentNode=this;}}
 replaceChildren(...values){this._text='';this.children=[];this.append(...values);}
 setAttribute(key,value){this.attrs[key]=String(value);}
 removeAttribute(key){delete this.attrs[key];}
}
const testElements=new Map();
const document={
 createElement:tag=>new Element(tag),
 createElementNS:(namespace,tag)=>new Element(tag),
 getElementById:id=>{
  if(!testElements.has(id))testElements.set(id,new Element());
  return testElements.get(id);
 }
};
const fetch=()=>{throw Error('Recognition rendering must not request models, robots or live cameras');};
const testGet=id=>document.getElementById(id);
testGet('recognitionRound').value='latest';
testGet('recognitionView').value='wrist';
testGet('recognitionBoxes').checked=true;
testGet('head').src='/live-head';testGet('right').src='/live-wrist';
''' + prefix + r'''
const testChecks=[];
const testAssert=(condition,message)=>{if(!condition)throw Error(message);};
const testWalk=node=>[node,...node.children.flatMap(testWalk)];
const testVisibleWalk=node=>node.hidden||node.style?.display==='none'||node.attrs.visibility==='hidden'?[]:[node,...node.children.flatMap(testVisibleWalk)];
const testText=node=>node.textContent+' '+node.children.map(testText).join(' ');
const testPanelText=()=>testText(testGet('recognitionStatus'))+' '+testText(testGet('recognitionImages'));
const testImages=()=>testWalk(testGet('recognitionImages')).filter(node=>node.tag==='img');
const testShapes=()=>testVisibleWalk(testGet('recognitionImages')).filter(node=>['rect','line'].includes(node.tag));
const testBoxes=()=>testShapes().filter(node=>node.tag==='rect');
const testView=(left=.2)=>({
 button_visibility:'full',contact_visibility:'full',button_box:[left,.2,left+.1,.4],
 arrow_box:[.6,.5,.8,.7],contact_tip:[.7,.6],image_size:[640,360],
 coordinate_units:'normalized_01',note:'Saved model localization'
});
const testAssessment=(oid,left=.2)=>({
 format:'pressb-localization-v3',observation_id:oid,revision:1,
 coordinate_units:'normalized_01',head:testView(left),wrist:testView(left)
});
const testRow=(round,oid,left=.2)=>({
 round,phase:'动作决策中',decision:{observation_id:oid,decision_mode:'localized',execution_policy:'model_decides'},
 localization:{assessment:testAssessment(oid,left),metrics:{}}
});
const testExpectImage=(oid,view='wrist')=>{
 const images=testImages();
 testAssert(images.length===1,'Expected one selected recognition image, got '+images.length);
 testAssert(images[0].src==='/api/observations/'+oid+'/'+(view==='wrist'?'right':'head')+'.jpg',
  'Wrong observation or camera: '+images[0].src);
 if(typeof images[0].onload==='function')images[0].onload();
 return images[0];
};
const testExpectFirstBox=x=>{
 for(const image of testImages())if(typeof image.onload==='function')image.onload();
 testAssert(testBoxes().length>0,'Expected a valid model box');
 testAssert(Math.abs(Number(testBoxes()[0].attrs.x)-x)<1e-6,'Wrong model revision or coordinate conversion');
};

// Latest is a round number, not the incoming array's final element.
let testRows=[testRow(3,'obs-three',.3),testRow(1,'obs-one',.1),testRow(2,'obs-two',.2)];
renderRecognition(testRows);
testExpectImage('obs-three');testExpectFirstBox(300);
testChecks.push('latest round uses maximum round number');

testGet('recognitionView').value='head';renderRecognition(testRows);
testExpectImage('obs-three','head');
testGet('recognitionView').value='wrist';renderRecognition(testRows);
testExpectImage('obs-three');
testChecks.push('head and wrist selection uses the same saved observation');

testGet('recognitionRound').value='1';renderRecognition(testRows);
testExpectImage('obs-one');testExpectFirstBox(100);
testRows=[testRows[0],testRow(4,'obs-four',.4),testRows[1],testRows[2]];
renderRecognition(testRows);
testAssert(testGet('recognitionRound').value==='1','Polling reset the selected historical round');
testExpectImage('obs-one');
testChecks.push('history remains selected when newer rounds arrive');

testGet('recognitionBoxes').checked=false;renderRecognition(testRows);
testExpectImage('obs-one');
testAssert(!testWalk(testGet('recognitionImages')).some(node=>node.tag==='svg'),'Hidden boxes still create an overlay');
testGet('recognitionBoxes').checked=true;renderRecognition(testRows);
testExpectImage('obs-one');testExpectFirstBox(100);
testChecks.push('box toggle redraws unchanged rows without changing the snapshot');

testGet('recognitionRound').value='latest';
const testPending={round:10,phase:'双视角定位中',decision:{observation_id:'obs-pending',decision_mode:'localized'},localization:{assessment:null,metrics:{}}};
renderRecognition([testRow(9,'obs-previous'),testPending]);
testExpectImage('obs-pending');
testAssert(testShapes().length===0,'Pending latest round borrowed previous-round markers');
testAssert(/定位|等待|暂无/.test(testPanelText()),'Pending localization has no visible explanation');
testChecks.push('latest pending round shows its own snapshot without previous markers');

const testMismatch=testRow(11,'obs-bound',.1);
testMismatch.localization_current=testAssessment('obs-wrong',.6);
testMismatch.localization_review={current:{assessment:testAssessment('obs-bound',.4)},before:{assessment:testAssessment('obs-before',.5)}};
renderRecognition([testMismatch]);
testExpectImage('obs-bound');testExpectFirstBox(400);
testMismatch.localization_current=testAssessment('obs-bound',.2);
renderRecognition([testMismatch]);testExpectFirstBox(200);
testMismatch.localization_current=testAssessment('obs-wrong',.6);
testMismatch.localization_review.current.assessment=testAssessment('obs-wrong-review',.5);
renderRecognition([testMismatch]);testExpectFirstBox(100);
testMismatch.localization.assessment=testAssessment('obs-wrong-initial',.7);
renderRecognition([testMismatch]);testExpectImage('obs-bound');
testAssert(testShapes().length===0,'Wrong-ID localization was drawn against current pixels');
testChecks.push('current, review and initial priority only accepts matching observation IDs');

const testUnknown=testRow(12,'obs-unknown');
for(const view of ['head','wrist'])Object.assign(testUnknown.localization.assessment[view],{
 button_visibility:'unknown',contact_visibility:'unknown',button_box:null,arrow_box:null,contact_tip:null
});
renderRecognition([testUnknown]);testExpectImage('obs-unknown');
testAssert(testShapes().length===0,'Unknown geometry produced markers');
testAssert(/未知|暂无|未返回|无有效/.test(testPanelText()),'Unknown localization has no visible explanation');
testChecks.push('unknown localization remains an unmarked saved image');

renderRecognition([{round:13,phase:'已完成',localization:{assessment:testAssessment('obs-legacy'),metrics:{}}}]);
testExpectImage('obs-legacy');testExpectFirstBox(200);
testAssert(testPanelText().includes('旧记录'),'Missing decision binding was not labeled as an old record');
testChecks.push('old records use their assessment observation ID with a visible label');

const testLegacyUnits=testRow(14,'obs-legacy-units');
for(const view of ['head','wrist'])Object.assign(testLegacyUnits.localization.assessment[view],{
 button_box:[180,0,390,130],arrow_box:[670,730,790,870],contact_tip:[690,760]
});
renderRecognition([testLegacyUnits]);testExpectImage('obs-legacy-units');
testAssert(testShapes().length===0,'Out-of-range legacy coordinates were drawn or silently rescaled');
testAssert(/无效|单位|范围|未知|无有效/.test(testPanelText()),'Invalid coordinates have no visible explanation');
testChecks.push('out-of-range legacy coordinates are not guessed or rescaled');

renderRecognition([testRow(15,'obs-missing')]);
const testMissingImage=testExpectImage('obs-missing');
testAssert(typeof testMissingImage.onerror==='function','Missing snapshot has no error handler');
testMissingImage.onerror();
testAssert(testShapes().length===0,'Markers remained visible after the underlying snapshot failed');
testAssert(/不可用|失败|缺失|不存在/.test(testPanelText()),'Missing snapshot has no visible explanation');
testAssert(testMissingImage.src==='/api/observations/obs-missing/right.jpg','Missing snapshot fell back to different pixels');
testChecks.push('missing saved images hide overlays and explain the failure');

testAssert(testGet('head').src==='/live-head'&&testGet('right').src==='/live-wrist','Recognition changed the live camera images');
testChecks.push('live camera images remain untouched');
({ok:true,checks:testChecks});
'''


def test_recognition_panel_runtime_when_node_available():
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node unavailable; execute rendering_harness() using the tool V8 runtime')
    program = 'const result=eval(' + json.dumps(rendering_harness()) + ');console.log(JSON.stringify(result));'
    result = subprocess.run([node, '-e', program], capture_output=True, text=True, timeout=10, check=True)
    report = json.loads(result.stdout)
    assert report['ok'] is True
    assert len(report['checks']) == 11
