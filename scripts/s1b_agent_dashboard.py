"""Loopback-only live camera and recorded Agent output dashboard."""
import base64
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import threading
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pressb.agent_loop.common import request


HTML = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>S1B · Agent 实验</title>
<style>
*{box-sizing:border-box}body{margin:0;padding:24px;background:#101720;color:#e7eff7;font:15px system-ui,sans-serif}
header{display:flex;align-items:center;justify-content:space-between;gap:16px;margin-bottom:20px}h1{font-size:24px;margin:0 0 6px}h2{font-size:17px;margin:0 0 12px}p{line-height:1.6}.muted{color:#a9bacc}.status{color:#77d9c3}button{padding:10px 18px;background:#a23838;color:white;border:1px solid #df7777;border-radius:8px;font:inherit;cursor:pointer}button:disabled{opacity:.5;cursor:default}
main{display:grid;grid-template-columns:minmax(0,1.2fr) minmax(340px,1fr);gap:20px}article,.panel{padding:18px;background:#192331;border:1px solid #314154;border-radius:12px;margin-bottom:16px}.screen{background:#080d13;aspect-ratio:16/9;display:grid;place-items:center;overflow:hidden}.screen img{width:100%;height:100%;object-fit:contain}.screen img:not([src]){visibility:hidden}footer{padding-top:10px;color:#a9bacc}.agent{max-height:85vh;overflow:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.6 ui-monospace,monospace;margin:10px 0}summary{cursor:pointer;color:#a9bacc}.round{border-top:1px solid #314154;padding:15px 0}.round:first-child{border-top:0}.reason{white-space:pre-wrap}.badge{color:#77d9c3;margin-left:12px}#error{color:#ffb4a2}#goal{overflow-wrap:anywhere}#terminal{max-height:260px;overflow:auto}.localization-frame{position:relative;line-height:0;background:#080d13;margin:8px 0}.localization-frame img{display:block;width:100%;height:auto}.localization-frame svg{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}.localization-caption{font-size:13px;color:#a9bacc}@media(max-width:950px){main{grid-template-columns:1fr}.agent{max-height:none}body{padding:14px}header{align-items:flex-start}}
.recognition-controls{display:flex;flex-wrap:wrap;gap:12px;align-items:center}.recognition-controls label{display:flex;align-items:center;gap:7px}.recognition-controls select{padding:7px;background:#101720;color:#e7eff7;border:1px solid #52677d;border-radius:6px;font:inherit}#recognitionPanel{border-color:#557b8f}#recognitionStatus{overflow-wrap:anywhere}.recognition-legend{font-size:13px}.recognition-legend span{margin-right:12px}.recognition-empty{color:#ffce85}#recognitionImages .localization-frame{min-height:40px}
</style></head><body>
<header><div><h1>S1B / Agent 闭环实验</h1><div id="status" class="status" aria-live="polite">等待启动</div></div><button id="stop">停止执行</button></header>
<p id="goal" class="muted"></p><p id="error" role="alert"></p>
<main><section><article id="recognitionPanel"><h2>Agent 识别快照</h2>
<div class="recognition-controls"><label>轮次 <select id="recognitionRound"><option value="latest">跟随最新轮次</option></select></label><label>视角 <select id="recognitionView"><option value="wrist">右腕</option><option value="head">头部</option></select></label><label><input id="recognitionBoxes" type="checkbox" checked>显示识别框</label></div>
<p class="recognition-legend"><span style="color:#ffd166">黄框：按钮</span><span style="color:#d697ff">紫框：箭头标记</span><span style="color:#64f4ff">青色十字：物理指端</span></p>
<p id="recognitionStatus" class="localization-caption" aria-live="polite">等待第一轮定位…</p><div id="recognitionImages"></div>
<p class="localization-caption">这是对应轮次推理所用的保存快照及模型估计。下方是实时视频；框不会叠加到更新后的实时画面。</p></article>
<article><h2>头部相机</h2><div class="screen"><img id="head" alt="头部相机画面"></div><footer id="headStatus">等待画面</footer></article>
<article><h2>右腕相机</h2><div class="screen"><img id="right" alt="右腕相机画面"></div><footer id="rightStatus">等待画面</footer></article>
<div class="panel"><div id="depth">固定点深度：—</div><p class="muted">原始画面。深度无效不代表按钮已触发；只有看到目标按钮亮起才能确认触发。</p></div></section>
<section class="panel agent"><h2>Agent 输出</h2><p class="muted">实时视频可能晚于推理快照；每轮判读与动作以记录的观测 ID 为准。任务结束后保留记录。</p><div id="rounds"></div><details><summary>启动与运行日志</summary><pre id="terminal"></pre></details><details><summary>本次结果</summary><pre id="result">尚未结束</pre></details></section></main>
<script>
const $=id=>document.getElementById(id);
const phases={starting:'正在启动机器人服务',waiting_for_service:'等待服务就绪 / 终端接管控制权',running:'Agent 自动闭环运行中',finished:'本次任务已结束',error:'本次任务已停止：发生错误',stopping:'正在停止'};
let rendered='';
const labels={yes:'可见',no:'不可见',full:'完整可见',partial:'部分可见',unknown:'未知',left:'左侧',center:'居中',right:'右侧',above:'上方',level:'同高',below:'下方',not_visible:'不可见',head_image_left:'头图左侧手指',head_image_right:'头图右侧手指',on:'亮',off:'灭',head:'头图',wrist:'腕图'};
const label=value=>labels[value]||value||'未知';
const modelDecides=row=>row.decision?.format==='pressb-conversation-v5'||row.decision?.execution_policy==='model_decides'||['pressb-localization-v2','pressb-localization-v3'].includes((row.localization_current||row.localization?.assessment)?.format);
function jsonBlock(parent,value){const pre=document.createElement('pre');pre.textContent=JSON.stringify(value,null,2);parent.append(pre);}
function detailsBlock(parent,title,value){const detail=document.createElement('details'),summary=document.createElement('summary');summary.textContent=title;detail.append(summary);jsonBlock(detail,value);parent.append(detail);}
function stageMetrics(card,title,metrics){const m=metrics||{},stats=document.createElement('p');stats.className='muted';stats.textContent=title+'输入 '+(m.prompt_tokens??'—')+' / 输出 '+(m.completion_tokens??'—')+' token · '+(m.image_count??'—')+' 张图 · '+(typeof m.elapsed_s==='number'?m.elapsed_s.toFixed(1)+' 秒':'进行中')+' · 请求 '+(m.attempts??'—')+' 次';card.append(stats);}
function renderCorrections(card,row){
 const stages=[['动作',row.correction,row.action_metrics],['定位',row.localization?.correction,row.localization?.metrics],['复核前图定位',row.localization_review?.before?.correction,row.localization_review?.before?.metrics],['复核当前定位',row.localization_review?.current?.correction,row.localization_review?.current?.metrics],['取景定位',row.view_acquisition?.correction,row.view_acquisition?.metrics],['判读',row.perception?.correction,row.perception?.metrics],['亮灯核验',row.activation?.correction,row.activation?.metrics],['后撤核验',row.withdrawal?.correction,row.withdrawal?.metrics],['前推检查',row.forward_guard?.correction,row.forward_guard?.metrics],['进展复核',row.progress_review?.correction,row.progress_review?.metrics],['假设复核',row.hypothesis_review?.correction,row.hypothesis_review?.metrics],['压缩摘要',row.compaction?.correction,row.compaction?.metrics]];
 for(const [name,c,m] of stages){
  if(!c)continue;
  const status=document.createElement('p');status.className='muted';
  const count=Number.isInteger(c.next_attempt)?c.next_attempt:Number.isInteger(c.attempt)?c.attempt+1:Math.max(0,(m?.attempts||1)-1);
  const state=c.status==='accepted'?'修正后已通过':c.status==='retrying'?'修正中':'修正记录';
  status.textContent=name+'：'+state+' · 已拒绝 '+count+' 次 · 累计请求 '+(m?.attempts??'—')+' 次；不设修正次数上限，仍可停止。';card.append(status);
  if(c.error){const reason=document.createElement('p');reason.textContent='最近拒绝原因：'+(typeof c.error==='string'?c.error:JSON.stringify(c.error));card.append(reason);}
  detailsBlock(card,name+'最近修正记录',c);
 }
}
function renderViewAcquisition(card,row){
 if(row.decision?.decision_mode==='localized')return;
 const v=row.view_acquisition,s=row.view_state,conflict=row.progress_review?.conflict;
 if(!s&&!conflict&&!(v&&(v.assessment||v.response||v.error||Object.keys(v.metrics||{}).length))&&!row.decision?.view_phase)return;
 const phases={get_view:'获取有效头部视野',ready:'取景已通过',recover_view:'判断冲突后恢复取景'};
 const phase=s?.phase||row.decision?.view_phase;
 const title=document.createElement('strong');title.textContent='取景阶段与冲突恢复';card.append(title);
 const status=document.createElement('p');status.textContent='当前阶段：'+(phases[phase]||phase||'检查中')+'。'+(['get_view','recover_view'].includes(phase)?'当前仅允许调整头部、重新观察或报告无法继续。':'');card.append(status);
 const a=v?.assessment;
 const scope=document.createElement('p');scope.className='muted';scope.textContent='当前观测 '+(s?.observation_id||row.decision?.observation_id||'未知')+'；取景定位对应观测 '+(a?.observation_id||s?.assessment_observation_id||'未知')+'。定位是模型判断，非人工确认；图像上下／左右不是机器人运动方向或三维接触关系。';card.append(scope);
 if(a){
  for(const view of ['head','wrist']){
   const value=a[view];if(!value)continue;
   const relation=value.derived_relation||{};
   const line=document.createElement('p');line.textContent=label(view)+'：面板 '+label(value.panel_visible)+'；按钮相对箭头所指物理指端：横向 '+label(relation.horizontal)+'，高度 '+label(relation.vertical)+'。';card.append(line);
   if(value.note){const note=document.createElement('p');note.textContent=value.note;card.append(note);}
  }
  if(a.note){const note=document.createElement('p');note.textContent=a.note;card.append(note);}
  detailsBlock(card,'完整取景定位（按钮框、箭头框与物理指端）',a);
 }
 if(s)detailsBlock(card,'取景阶段及恢复原因',s);
 if(conflict){
  const warning=document.createElement('p');warning.textContent='动作模型与独立复核判断冲突：当前关系待确认。先换取景或获取新观测，不把复核结论当作事实，也不自动反向移动。';card.append(warning);
  detailsBlock(card,'双方判断及冲突观测',conflict);
 }
 if(v&&(a||v.response||v.error||Object.keys(v.metrics||{}).length))stageMetrics(card,'取景定位',v.metrics);
 if(v?.error)jsonBlock(card,v.error);
 if(v?.response)detailsBlock(card,'取景定位原始输出',v.response.response||v.response);
}
function renderProgress(card,row){
 const p=row.progress,review=row.progress_review;
 const changes={first:'首轮，无前次动作',no_motion:'期间未执行运动',improved:'改善',unchanged:'未改善',worsened:'恶化',crossed:'已越过原先偏差方向',unknown:'未知'};
 const subgoals={get_view:'获取有效视角',align_vertical:'调整高度',align_horizontal:'调整横向',approach:'接近',press:'按压',withdraw:'撤回',complete:'完成',uncertain:'待确认'};
 if(p){
  const title=document.createElement('strong');title.textContent='进展监测';card.append(title);
  const a=p.candidate_progress;
  if(a){
   const summary=document.createElement('p');summary.textContent='动作模型自评：子目标 '+(subgoals[a.subgoal]||a.subgoal||'未知')+'；依据 '+label(a.view)+'；按钮相对接触指端：横向 '+label(a.button_horizontal)+'，高度 '+label(a.button_vertical)+'；上一动作效果 '+(changes[a.last_effect]||a.last_effect||'未知')+'。';card.append(summary);
  }
  const scope=document.createElement('p');scope.className='muted';scope.textContent='自评是模型判断；程序仅统计实际完成发送的命令，次数与累计位移不代表对齐改善。对应观测 '+(p.observation_id||a?.observation_id||row.decision?.observation_id||'未知')+'。';card.append(scope);
  if(p.monitor){
   const status=document.createElement('p');status.textContent=p.monitor.review_required?'程序检测到复核条件，需核对当前证据。':'本轮未触发额外进展复核。';card.append(status);
   detailsBlock(card,'程序命令统计与复核触发原因',p.monitor);
  }
  detailsBlock(card,'完整进展记录（模型自评与程序统计分开）',p);
 }
 if(review&&(review.assessment||review.decision||review.response||review.error||Object.keys(review.metrics||{}).length||row.decision?.stage==='progress_review')){
  const title=document.createElement('strong');title.textContent='异常进展复核（独立模型判断，非人工确认）';card.append(title);
  const a=review.assessment,d=review.decision;
  const binding=document.createElement('p');binding.className='muted';binding.textContent='复核观测：'+(a?.previous_observation_id||row.decision?.previous_observation_id||'未知')+' → '+(a?.observation_id||row.decision?.observation_id||'未知')+'。仅对应保存快照，不代表实时视频或三维接触已对齐。';card.append(binding);
  if(a){
   const current=document.createElement('p');current.textContent='按钮相对指定指端：头图 横向 '+label(a.head_horizontal)+' / 高度 '+label(a.head_vertical)+'；腕图 横向 '+label(a.wrist_horizontal)+' / 高度 '+label(a.wrist_vertical)+'。';card.append(current);
   const effect=document.createElement('p');effect.textContent='前后变化：横向 '+(changes[a.horizontal_change]||a.horizontal_change||'未知')+'；高度 '+(changes[a.vertical_change]||a.vertical_change||'未知')+'；接近 '+(changes[a.approach_change]||a.approach_change||'未知')+'。';card.append(effect);
   if(a.note){const note=document.createElement('p');note.textContent=a.note;card.append(note);}
   detailsBlock(card,'完整独立进展判读',a);
  }
  if(d){
   const status=document.createElement('p');status.textContent=review.conflict?'双方判断存在冲突，需恢复取景后重新判断。':d.allowed===false||d.blocked_same_direction===true?'本候选同向重复未获支持，需重新决策。':d.allowed===true?'当前候选通过进展检查；这不代表到位、接触或任务成功。':'候选进展检查结果见记录。';card.append(status);
   detailsBlock(card,'候选检查与拒绝原因',d);
  }
  stageMetrics(card,'进展复核',review.metrics);
  if(review.error)jsonBlock(card,review.error);
  if(review.response)detailsBlock(card,'进展复核原始输出',review.response.response||review.response);
 }
}
function localizationOverlay(parent,observationId,view,value,advisory=false,showMarkers=true){
 if(!/^[A-Za-z0-9_-]{1,128}$/.test(observationId||'')||!['head','wrist'].includes(view))return;
 const frame=document.createElement('div');frame.className='localization-frame';
 const img=document.createElement('img');img.loading='lazy';img.alt=label(view)+'保存快照 '+observationId;
 img.src='/api/observations/'+encodeURIComponent(observationId)+'/'+(view==='wrist'?'right':'head')+'.jpg';
 frame.append(img);
 const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');
 svg.setAttribute('viewBox','0 0 1000 1000');svg.setAttribute('preserveAspectRatio','none');svg.setAttribute('aria-hidden','true');
 svg.setAttribute('visibility','hidden');
 img.onload=()=>svg.setAttribute('visibility','visible');
 img.onerror=()=>{svg.setAttribute('visibility','hidden');img.hidden=true;const missing=document.createElement('p');missing.className='recognition-empty';missing.textContent='该观测保存快照不可用，未叠加识别框。';frame.append(missing);};
 const valid=(a,n)=>Array.isArray(a)&&a.length===n&&a.every(v=>typeof v==='number'&&Number.isFinite(v)&&v>=0&&v<=1);
 const shape=(name,attrs,color)=>{const node=document.createElementNS('http://www.w3.org/2000/svg',name);for(const [k,v]of Object.entries(attrs))node.setAttribute(k,String(v));node.setAttribute('stroke',color);node.setAttribute('stroke-width','2');node.setAttribute('vector-effect','non-scaling-stroke');node.setAttribute('fill','none');svg.append(node);};
 for(const [box,color]of [[value.button_box,'#ffd166'],[value.arrow_box,'#d697ff']]){
  if(valid(box,4)&&box[2]>box[0]&&box[3]>box[1])shape('rect',{x:box[0]*1000,y:box[1]*1000,width:(box[2]-box[0])*1000,height:(box[3]-box[1])*1000},color);
 }
 const tip=value.contact_tip;
 if(valid(tip,2)){
  shape('line',{x1:tip[0]*1000-10,y1:tip[1]*1000,x2:tip[0]*1000+10,y2:tip[1]*1000},'#64f4ff');
  shape('line',{x1:tip[0]*1000,y1:tip[1]*1000-10,x2:tip[0]*1000,y2:tip[1]*1000+10},'#64f4ff');
  const b=value.button_box;
  if(valid(b,4)&&(advisory||(value.button_visibility==='full'&&value.contact_visibility==='full')))shape('line',{x1:tip[0]*1000,y1:tip[1]*1000,x2:(b[0]+b[2])*500,y2:(b[1]+b[3])*500},'#64f4ff');
 }
 if(showMarkers)frame.append(svg);parent.append(frame);
}
let recognitionRows=[],recognitionRendered='',recognitionOptions='';
function recognitionBinding(row){
 const candidates=[row.localization_current,row.localization_review?.current?.assessment,row.localization?.assessment].filter(Boolean);
 const bound=row.decision?.observation_id||row.context?.observation_id;
 if(bound)return {id:bound,assessment:candidates.find(a=>a.observation_id===bound),mismatch:candidates.some(a=>a.observation_id!==bound),legacy:false};
 const assessment=candidates.find(a=>/^[A-Za-z0-9_-]{1,128}$/.test(a.observation_id||''));
 return {id:assessment?.observation_id,assessment,mismatch:false,legacy:Boolean(assessment)};
}
function renderRecognition(rows){
 recognitionRows=Array.isArray(rows)?rows:[];
 const sorted=[...recognitionRows].sort((a,b)=>Number(a.round)-Number(b.round));
 const select=$('recognitionRound'),optionKey=JSON.stringify(sorted.map(r=>r.round));
 if(optionKey!==recognitionOptions){
  const selected=select.value||'latest';select.replaceChildren();
  const latest=document.createElement('option');latest.value='latest';latest.textContent='跟随最新轮次';select.append(latest);
  for(const row of [...sorted].reverse()){const option=document.createElement('option');option.value=String(row.round);option.textContent='第 '+row.round+' 轮';select.append(option);}
  select.value=selected==='latest'||sorted.some(r=>String(r.round)===selected)?selected:'latest';recognitionOptions=optionKey;
 }
 const row=select.value==='latest'?sorted[sorted.length-1]:sorted.find(r=>String(r.round)===select.value);
 const view=$('recognitionView').value==='head'?'head':'wrist',showMarkers=$('recognitionBoxes').checked;
 const binding=row?recognitionBinding(row):{},a=binding.assessment;
 const signature=JSON.stringify([row?.round,row?.phase,binding,view,showMarkers]);
 if(signature===recognitionRendered)return;recognitionRendered=signature;
 const root=$('recognitionImages');root.replaceChildren();
 if(!row){$('recognitionStatus').textContent='等待第一轮定位…';return;}
 const base='第 '+row.round+' 轮 · '+label(view)+' · 观测 '+(binding.id||'未知');
 if(!/^[A-Za-z0-9_-]{1,128}$/.test(binding.id||'')){$('recognitionStatus').textContent=base+' · 旧记录未提供可绑定的定位快照';return;}
 const value=a?.[view]||{};
 const valid=(v,n)=>Array.isArray(v)&&v.length===n&&v.every(x=>typeof x==='number'&&Number.isFinite(x)&&x>=0&&x<=1)&&(n!==4||(v[2]>v[0]&&v[3]>v[1]));
 const invalid=[['button_box',4],['arrow_box',4],['contact_tip',2]].some(([key,n])=>value[key]!=null&&!valid(value[key],n))||
  (a?.coordinate_units!=null&&a.coordinate_units!=='normalized_01')||(value.coordinate_units!=null&&value.coordinate_units!=='normalized_01');
 const hasGeometry=valid(value.button_box,4)||valid(value.arrow_box,4)||valid(value.contact_tip,2);
 let status=!a?'本轮定位尚未返回':invalid?'定位坐标单位或范围无效，未绘制标记':!hasGeometry?'本视角无有效定位框':'按钮 '+label(value.button_visibility)+' / 指端 '+label(value.contact_visibility);
 if(binding.mismatch)status+='；其他观测的定位未叠加';
 if(binding.legacy)status+='；旧记录按定位自身观测 ID 展示';
 if(!showMarkers)status+='；识别框已隐藏';
 $('recognitionStatus').textContent=base+' · '+status;
 localizationOverlay(root,binding.id,view,invalid?{}:value,true,showMarkers);
 if(value.note){const note=document.createElement('p');note.className='localization-caption';note.textContent=value.note;root.append(note);}
}
function renderLocalizationAssessment(card,a,title){
 if(!a)return;
 const advisory=['pressb-localization-v2','pressb-localization-v3'].includes(a.format);
 const heading=document.createElement('strong');heading.textContent=title+' · 观测 '+a.observation_id+' · 定位版本 '+(a.revision??'未知');card.append(heading);
 const source=document.createElement('p');source.className='localization-caption';source.textContent='保存快照上的模型定位：黄框按钮、紫框箭头、青色十字物理指端。连线与上下左右由程序计算；不是人工确认、实时视频或三维接触证据。'+(advisory?' 坐标和可见性供模型判断，程序不据此限制动作；部分可见框的中心仅指返回框中心。':'');card.append(source);
 const number=v=>typeof v==='number'&&Number.isFinite(v)?v.toFixed(1):'未知';
 for(const view of ['head','wrist']){
  const v=a[view];if(!v)continue;
  const invalidCoordinates=[v.button_box,v.arrow_box,v.contact_tip].some(x=>x!=null&&(!Array.isArray(x)||x.some(n=>typeof n!=='number'||!Number.isFinite(n)||n<0||n>1)));
  const d=invalidCoordinates?{}:v.derived_relation||{};
  if(invalidCoordinates){const warning=document.createElement('p');warning.textContent=label(view)+'：历史坐标单位或范围无效；未绘制越界标记，历史派生偏差不可信。原始值保留在记录中。';card.append(warning);}
  const text=document.createElement('p');text.textContent=label(view)+'：按钮 '+label(v.button_visibility)+' / 指端 '+label(v.contact_visibility)+'；程序关系 '+label(d.horizontal)+'、'+label(d.vertical)+'；偏差 '+number(d.pixel_delta?.[0])+', '+number(d.pixel_delta?.[1])+' px。';card.append(text);
  localizationOverlay(card,a.observation_id,view,v,advisory);
  if(v.note){const note=document.createElement('p');note.textContent=v.note;card.append(note);}
 }
 if(advisory&&a.parse_notes?.length)detailsBlock(card,'定位解析说明（交由动作模型判断）',a.parse_notes);
 if(advisory&&a.raw_output!=null)detailsBlock(card,'定位原始输出',a.raw_output);
 detailsBlock(card,'原始定位与程序计算结果',a);
}
function renderLocalization(card,row){
 const initial=row.localization,review=row.localization_review;
 if(!(initial?.assessment||initial?.response||initial?.error||Object.keys(initial?.metrics||{}).length||row.decision?.decision_mode==='localized'))return;
 const active=row.localization_current||initial?.assessment;
 const advisory=modelDecides(row);
 const phase=document.createElement('p');phase.textContent=row.decision?.forward_guard_policy==='wrist_tip_overlap'?'先用Y/Z对齐：偏差明显时较大调整，接近后微调。每个正X候选须通过当前腕图指端点入按钮框检查；通过后原XYZ不变，无正X步长上限。':row.decision?.hypothesis_review_format?'由模型选择动作；连续同向且无明确进展时，先复核原假设，再重新决策。程序不改写XYZ或自动撤回。':advisory?'由模型判断行动：定位、可见性和坐标变化仅作为输入，不按阈值禁用动作，不另做前推放行或强制复核。':row.decision?.view_phase==='get_view'?'两视角均无有效定位：先调整头部或重新观察。':'任一视角定位有效即可用于决策；前推另需当前腕图横向检查。';card.append(phase);
 renderLocalizationAssessment(card,active,advisory?'本轮定位（模型判断）':'当前有效定位');
 if(initial)stageMetrics(card,'定位',initial.metrics);
 if(initial?.error)jsonBlock(card,initial.error);
 if(initial?.response)detailsBlock(card,'定位原始模型输出',initial.response.response||initial.response);
 if(initial?.assessment&&active?.revision!==initial.assessment.revision)detailsBlock(card,'初始定位（已被复核版本取代）',initial.assessment);
 if(row.localized_progress){
  const title=document.createElement('strong');title.textContent=advisory?'程序计算的原始坐标变化':'程序计算的投影进展';card.append(title);
  const note=document.createElement('p');note.textContent=advisory?'分别列出头图与腕图的前后坐标差和实际命令。是否可比较、是否改善及下一步行动由模型判断。':'只比较同一视角、可比较观测中的模型定位；相机、夹爪或目标身份变化可使比较未知。';card.append(note);
  detailsBlock(card,advisory?'原始差值、状态与命令':'偏差变化与复核触发记录',row.localized_progress);
 }
 if(review&&(review.assessment||review.error||review.before?.assessment||review.current?.assessment||review.before?.error||review.current?.error||Object.keys(review.before?.metrics||{}).length||Object.keys(review.current?.metrics||{}).length||row.decision?.stage==='localization_review')){
  const block=document.createElement('details'),summary=document.createElement('summary');summary.textContent='异常时重新定位与程序复核';block.append(summary);
  for(const [key,title]of [['before','复核运动前'],['current','复核当前']]){
   const part=review[key];if(!part)continue;renderLocalizationAssessment(block,part.assessment,title);stageMetrics(block,title,part.metrics);if(part.error)jsonBlock(block,part.error);
  }
  if(review.assessment)detailsBlock(block,'复核关系、冲突视角与有效版本',review.assessment);
  if(review.error)jsonBlock(block,review.error);
  card.append(block);
 }
}
function renderPerception(card,row){
 const id=document.createElement('p');id.className='muted';id.textContent='决策观测 '+(row.decision?.observation_id||'旧记录未提供')+' · 模式 '+(row.decision?.decision_mode||'direct');card.append(id);
 renderCorrections(card,row);
 renderViewAcquisition(card,row);
 renderProgress(card,row);
 renderLocalization(card,row);
 const p=row.perception;
 if(p&&(p.assessment||p.response||p.error||Object.keys(p.metrics||{}).length)){
  const title=document.createElement('strong');title.textContent='独立画面判读（模型判断）';card.append(title);
  const a=p.assessment;
  if(['pressb-grounding-v1','pressb-grounding-v2'].includes(a?.format)){
   const summary=document.createElement('p');summary.textContent='头图按钮：'+label(a.head_visible)+'；腕图按钮：'+label(a.wrist_visible)+'。接触手指：'+label(a.contact_finger)+'。按钮相对该手指：水平 '+label(a.head_horizontal)+'，高度 '+label(a.head_vertical)+'。';card.append(summary);
   const lamp=document.createElement('p');lamp.className='muted';lamp.textContent='当前灯态：'+label(a.lamp)+'（'+label(a.lamp_view)+'）；对应观测 '+a.observation_id+'。图像位置不是已标定的三维接触关系。';card.append(lamp);
   if(a.geometry){
    const g=a.geometry,e=g.error_button_units,m=g.magnitude||{},size={small:'小',medium:'中',large:'大',unknown:'未知'};
    const number=value=>typeof value==='number'&&Number.isFinite(value)?value.toFixed(2):'未知';
    const geometry=document.createElement('p');geometry.textContent='定位偏差：横向 '+number(e?.[0])+' 个按钮宽度（'+(size[m.horizontal]||'未知')+'），纵向 '+number(e?.[1])+' 个按钮高度（'+(size[m.vertical]||'未知')+'）。';card.append(geometry);
    const scope=document.createElement('p');scope.className='muted';scope.textContent='按钮中心减所选手指前端，图像向右／向下为正。由模型定位计算，不是毫米校准；仅对应观测 '+a.observation_id+'，不叠加到实时视频。';card.append(scope);
    detailsBlock(card,'按钮框、手指前端与归一化偏差',g);
   }
   detailsBlock(card,'完整当前判读',a);
  }else if(a){jsonBlock(card,a);}
  stageMetrics(card,'判读',p.metrics);
  if(p.error)jsonBlock(card,p.error);
  if(p.response)detailsBlock(card,'判读原始输出',p.response.response||p.response);
 }
 if(row.alignment_progress){
  const progress=row.alignment_progress,description={improved:'偏差类别改善',worsened:'偏差类别恶化',unchanged_or_unresolved:'偏差类别未改善或仍不明确',unknown:'改善情况未知'};
  const text=document.createElement('p');text.textContent='动作后检查（模型判断）：'+(description[progress.status]||progress.status||'未知')+'。仅比较类别，不是毫米误差测量。';card.append(text);
  detailsBlock(card,'前后观测与实际命令',progress);
 }
 if(row.control_plan){
  const plan=row.control_plan,stages={get_view:'获取有效视角',align:'纠正对齐偏差',recheck:'重新核对定位与变化',approach:'评估接近',withdraw:'撤回并取证',complete:'报告完成'},focus={horizontal:'横向',vertical:'纵向'};
  const title=document.createElement('strong');title.textContent='控制策略（由模型判断派生的建议）';card.append(title);
  const summary=document.createElement('p');summary.textContent='策略阶段：'+(stages[plan.stage]||plan.stage||'未知')+'；重点：'+(focus[plan.focus]||plan.focus||'未指定')+'；建议步长：'+(typeof plan.step_hint_mm==='number'?plan.step_hint_mm+' mm':'未指定')+'。';card.append(summary);
  const scope=document.createElement('p');scope.className='muted';scope.textContent='建议步长可调整，不是执行限幅，也不代表已标定像素到毫米；对应观测 '+(plan.observation_id||'未知')+'。';card.append(scope);
  const progress=plan.geometry_progress;
  if(progress){
   const trend={improved:'改善',worsened:'恶化',unchanged:'基本不变'},change=document.createElement('p');
   change.textContent='定位偏差变化（模型估计）：'+(progress.comparable?'横向 '+(trend[progress.axes?.horizontal?.trend]||'未知')+'，纵向 '+(trend[progress.axes?.vertical?.trend]||'未知'):'当前两次定位不能可靠比较')+'。';card.append(change);
  }
  detailsBlock(card,'完整控制策略、前后偏差与头部目标',plan);
 }
 renderForwardGuard(card,row);
 const activation=row.activation;
 if(activation&&(activation.assessment||activation.response||activation.error||Object.keys(activation.metrics||{}).length)){
  const title=document.createElement('strong');title.textContent='亮灯变化核验（模型判断）';card.append(title);
  if(activation.assessment)jsonBlock(card,activation.assessment);
  stageMetrics(card,'亮灯变化核验',activation.metrics);
  if(activation.error)jsonBlock(card,activation.error);
  if(activation.response)detailsBlock(card,'亮灯变化核验原始输出',activation.response.response||activation.response);
 }
 const w=row.withdrawal;
 if(w&&(w.assessment||w.response||w.error||Object.keys(w.metrics||{}).length)){
  const title=document.createElement('strong');title.textContent='后撤画面核验（模型判断）';card.append(title);
  if(w.assessment)jsonBlock(card,w.assessment);
  stageMetrics(card,'后撤核验',w.metrics);
  if(w.error)jsonBlock(card,w.error);
  if(w.response)detailsBlock(card,'后撤核验原始输出',w.response.response||w.response);
 }
 if(row.decision_basis)detailsBlock(card,'动作依据（格式／一致性校验）',row.decision_basis);
 if(row.control_memory)detailsBlock(card,'运行记忆（含模型判断，非人工确认）',row.control_memory);
}
function renderForwardGuard(card,row){
 const guard=row.forward_guard;
 if(guard&&(guard.assessment||guard.decision||guard.response||guard.error||Object.keys(guard.metrics||{}).length)){
  const overlap=guard.decision?.policy==='wrist_tip_overlap'||guard.assessment?.format==='pressb-wrist-tip-overlap-v1'||row.decision?.forward_guard_policy==='wrist_tip_overlap';
  const title=document.createElement('strong');title.textContent=overlap?'前推条件检查（腕图二维点入框）':guard.assessment?.button_box?'前推条件检查（模型定位＋程序横向计算）':'前推条件检查（模型判断）';card.append(title);
  const a=guard.assessment,d=guard.decision;
  const status=document.createElement('p');status.textContent=d?.allowed===true?'正X已放行':d?.allowed===false?'前推未放行':'前推条件检查中';card.append(status);
  if(a&&overlap){
   const relation={yes:'横纵均在框内',no:'至少一轴在框外',unknown:'未知'};
   const summary=document.createElement('p');summary.textContent='腕图按钮：'+label(a.button_visibility||a.button_visible)+'；箭头所指物理指端：'+label(a.contact_visibility||a.contact_tip_visible)+'；指端与按钮框：'+(relation[a.tip_inside_button]||'未知')+'。';card.append(summary);
   localizationOverlay(card,a.observation_id,'wrist',a,true);
   if(a.geometry)detailsBlock(card,'物理指端点与按钮框的横纵计算',a.geometry);
  }else if(a){
   const relation={yes:'位于两指之间',no:'未位于两指之间',unknown:'未知'};
   const summary=document.createElement('p');summary.textContent='腕图按钮：'+label(a.button_visible)+'；两指前端：'+label(a.two_fingertips_visible)+'；按钮横向位置：'+(relation[a.button_between_fingers]||'未知')+'。';card.append(summary);
   if(a.geometry)detailsBlock(card,'双指坐标与横向区间计算',a.geometry);
  }
  const scope=document.createElement('p');scope.className='muted';scope.textContent=(overlap?'只检查当前腕图横纵二维点入框；未知不放行，失败拒绝整条候选。通过后原XYZ不变，无正X步长上限；负X/Y/Z自由。此检查不证明三维接触，不依赖头图或深度。对应观测 ':'仅检查腕图横向位于两指之间，不代表高度对齐或已经接触。对应观测 ')+(d?.observation_id||a?.observation_id||row.decision?.observation_id||'未知')+'。';card.append(scope);
  if(d?.allowed===false&&row.proposal){const note=document.createElement('p');note.className='muted';note.textContent='前推未放行后，当前建议见下方；拒绝前推本身不代表会话已停止。';card.append(note);}
  if(d)detailsBlock(card,'前推放行依据',d);
  if(a)detailsBlock(card,'完整腕图条件判读',a);
  stageMetrics(card,'前推检查',guard.metrics);
  if(guard.error)jsonBlock(card,guard.error);
  if(guard.response)detailsBlock(card,'前推检查原始输出',guard.response.response||guard.response);
 }
}
function renderHypothesisReview(card,row){
 const review=row.hypothesis_review;
 if(!review||!(review.trigger||review.assessment||review.error||Object.keys(review.metrics||{}).length||row.decision?.stage==='hypothesis_review'))return;
 if(review.trigger?.review_required===false&&!review.assessment&&!review.error&&!Object.keys(review.metrics||{}).length&&row.decision?.stage!=='hypothesis_review')return;
 const title=document.createElement('strong');title.textContent='无进展后的假设复核（模型判断）';card.append(title);
 const a=review.assessment,t=review.trigger||{};
 const ids=document.createElement('p');ids.textContent='复核快照：'+(a?.previous_observation_id||t.baseline_observation_id||'未知')+' → '+(a?.observation_id||t.observation_id||row.decision?.observation_id||'未知')+'；不是实时视频判断。';card.append(ids);
 if(a){
  const statuses={improved:'改善',unchanged:'不变',worsened:'恶化',unknown:'未知',supported:'有支持',unsupported:'缺乏支持'};
  const result=document.createElement('p');result.textContent='观察到的效果：'+(statuses[a.observed_effect]||a.observed_effect||'未知')+'；“仅距离不足”假设：'+(statuses[a.distance_only_hypothesis]||a.distance_only_hypothesis||'未知');card.append(result);
  if(a.prediction_check){const p=document.createElement('p');const predictions={not_supplied:'首次复核，没有上次预期',met:'已出现',not_met:'未出现',unknown:'无法确认'};p.textContent='上次预期是否出现：'+(predictions[a.prediction_check]||a.prediction_check);card.append(p);}
  for(const [key,label] of [['evidence','画面证据'],['alternative_explanation','其他解释'],['next_check','下一步应核对']]){if(a[key]){const p=document.createElement('p');p.textContent=label+'：'+a[key];card.append(p);}}
 }
 if(review.trigger)detailsBlock(card,'程序触发原因与已执行命令',review.trigger);
 if(a)detailsBlock(card,'完整复核结果（非人工确认）',a);
 if(review.decision)detailsBlock(card,'动作对复核的响应与预期变化',review.decision);
 stageMetrics(card,'假设复核',review.metrics);
 if(review.error)jsonBlock(card,review.error);
 if(review.response)detailsBlock(card,'假设复核原始输出',review.response.response||review.response);
}
function renderRounds(rows){
 const signature=JSON.stringify(rows);if(signature===rendered)return;rendered=signature;const root=$('rounds');root.replaceChildren();
 if(!rows.length){root.textContent='等待第一轮模型请求…';return;}
 for(const row of [...rows].reverse()){
  const card=document.createElement('div');card.className='round';const title=document.createElement('strong');title.textContent='第 '+row.round+' 轮';const badge=document.createElement('span');badge.className='badge';badge.textContent=row.phase;card.append(title,badge);renderPerception(card,row);renderHypothesisReview(card,row);
  if(row.proposal){const p=row.proposal;const pre=document.createElement('pre');const args={...p.arguments};delete args.reason;delete args.evidence;pre.textContent=p.name+'\n'+JSON.stringify(args,null,2);const reason=document.createElement('p');reason.className='reason';reason.textContent=p.arguments.reason||p.arguments.evidence||'';card.append(pre,reason);}
  if(typeof row.elapsed_s==='number'){const timing=document.createElement('small');timing.className='muted';timing.textContent='动作模型用时 '+row.elapsed_s.toFixed(1)+' 秒';card.append(timing);}
  const stats=document.createElement('p');stats.className='muted';const u=row.usage||{},c=row.context||{};
  stats.textContent='动作输入 '+(row.action_metrics?.prompt_tokens??u.prompt_tokens??c.input_tokens_exact??'—')+' token / 输出 '+(row.action_metrics?.completion_tokens??u.completion_tokens??'—')+' token · '+(c.image_count??'—')+' 张图 · 请求 '+(row.action_metrics?.attempts??'—')+' 次';
  if(c.memory_mode==='bounded_current_state'||row.decision?.memory_mode==='bounded_current_state')stats.textContent+=' · 当前状态短上下文';
  else stats.textContent+=' · 压缩代次 '+(c.compression_generation??0)+(c.compression_status==='completed'?'（本轮已压缩）':'');
  const cached=u.prompt_tokens_details?.cached_tokens;if(typeof cached==='number')stats.textContent+=' · 最后一次动作请求缓存 '+cached+' token';card.append(stats);
  if(row.error)jsonBlock(card,row.error);
  if(row.response)detailsBlock(card,'动作模型原始输出',row.response);
  root.append(card);
 }
}
async function stateTick(){try{const r=await fetch('/api/agent',{cache:'no-store',signal:AbortSignal.timeout(3000)});if(!r.ok)throw Error('HTTP '+r.status);const s=await r.json();$('status').textContent=phases[s.launcher.phase]||s.launcher.phase||'等待启动';$('goal').textContent=s.goal||'';$('error').textContent=s.launcher.error||s.result?.error||'';$('stop').disabled=Boolean(s.read_only)||['finished','error','stopping'].includes(s.launcher.phase);$('stop').textContent=s.read_only?'只读查看':'停止执行';renderRecognition(s.rounds);renderRounds(s.rounds);$('terminal').textContent=s.log;$('result').textContent=s.result?JSON.stringify(s.result,null,2):'尚未结束';}catch(e){$('status').textContent='网页服务连接中断，状态已停止更新';}finally{setTimeout(stateTick,700);}}
async function cameraTick(){try{const r=await fetch('/api/cameras',{cache:'no-store',signal:AbortSignal.timeout(3000)});if(!r.ok)throw Error('HTTP '+r.status);const s=await r.json();for(const v of ['head','right']){const f=s.frames?.[v];if(f?.jpeg)$(v).src='data:image/jpeg;base64,'+f.jpeg;$(v+'Status').textContent=s.archived?'已停止实时更新 · 最后保存的观测':f?(f.stale?'画面已过期':'实时画面')+' · '+Math.round(f.age_ms)+' ms':'等待画面';}const d=s.depth?.median_m;$('depth').textContent='固定点深度：'+(typeof d==='number'?(d*1000).toFixed(1)+' mm':'无有效读数');}catch(e){for(const v of ['head','right'])$(v+'Status').textContent='连接中断 · 显示的画面已过期';$('depth').textContent='固定点深度：—';}finally{setTimeout(cameraTick,400);}}
$('stop').onclick=async()=>{$('stop').disabled=true;try{const r=await fetch('/api/stop',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}',signal:AbortSignal.timeout(3000)});if(!r.ok)throw Error('HTTP '+r.status);$('status').textContent='正在停止';}catch(e){$('error').textContent='停止请求未确认，请在终端按 Ctrl-C。';$('stop').disabled=false;}};
for(const id of ['recognitionRound','recognitionView','recognitionBoxes'])$(id).onchange=()=>renderRecognition(recognitionRows);
stateTick();cameraTick();
</script></body></html>'''


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def latest_numbered_json(directory, prefix):
    """Read only the latest numeric response/validation; IDs above nine stay ordered."""
    candidates = []
    for path in Path(directory).glob(prefix+'-*.json'):
        suffix = path.stem[len(prefix)+1:]
        if suffix.isdigit():
            candidates.append((int(suffix), path))
    if not candidates:
        return None, None
    index, path = max(candidates, key=lambda entry: entry[0])
    return read_json(path), index


def read_correction(directory, completed=False):
    correction = read_json(Path(directory)/'correction.json')
    if not isinstance(correction, dict):
        validation, attempt = latest_numbered_json(directory, 'validation')
        if not isinstance(validation, dict):
            return None
        correction = dict(validation, attempt=validation.get('attempt', attempt),
                          next_attempt=validation.get('next_attempt', attempt+1), status='retrying')
    if completed:
        correction = dict(correction, status='accepted')
    return correction


def read_model_stage(directory, result_name='assessment.json'):
    directory = Path(directory)
    assessment = read_json(directory/result_name)
    response, _ = latest_numbered_json(directory, 'response')
    return dict(assessment=assessment, decision=read_json(directory/'decision.json'),
                metrics=read_json(directory/'metrics.json', {}), error=read_json(directory/'error.json'),
                response=response, correction=read_correction(directory, completed=assessment is not None))


class Dashboard:
    def __init__(self, directory, camera_base, stop, read_only=False):
        self.root = Path(directory)
        self.run = self.root / 'agent'
        self.camera_base = camera_base
        self.stop = stop
        self.read_only = read_only

    def state(self):
        rows = []
        outcome = read_json(self.run/'result.json')
        for folder in sorted((self.run/'rounds').glob('*')):
            if not folder.is_dir() or not folder.name.isdigit():
                continue
            proposal = read_json(folder/'proposal.json')
            # Completed round files are atomic and contain large state traces.
            # Their existence is enough here; never reload those traces on a UI poll.
            completed = (folder/'result.json').exists()
            error = read_json(folder/'error.json')
            response, response_attempt = latest_numbered_json(folder, 'response')
            correction = read_correction(folder, completed=proposal is not None)
            action_metrics = read_json(folder/'action_metrics.json', {})
            context = read_json(folder/'context_metrics.json', {})
            decision = read_json(folder/'decision_context.json', {})
            model_decides = decision.get('format') == 'pressb-conversation-v5' or decision.get('execution_policy') == 'model_decides'
            perception = read_model_stage(folder/'perception')
            activation = read_model_stage(folder/'activation')
            withdrawal = read_model_stage(folder/'withdrawal')
            forward_guard = read_model_stage(folder/'forward_guard')
            progress_review = read_model_stage(folder/'progress_review')
            progress_review['conflict'] = read_json(folder/'progress_review'/'conflict.json')
            hypothesis_review = read_model_stage(folder/'hypothesis_review')
            hypothesis_review['trigger'] = read_json(folder/'hypothesis_review'/'trigger.json')
            view_acquisition = read_model_stage(folder/'view_acquisition')
            view_state = read_json(folder/'view_state.json')
            localization = read_model_stage(folder/'localization')
            localization_review = read_model_stage(folder/'localization_review')
            localization_review['before'] = read_model_stage(folder/'localization_review'/'before')
            localization_review['current'] = read_model_stage(folder/'localization_review'/'current')
            compaction = read_model_stage(folder/'compaction', result_name='result.json')
            phase = '整理上下文中' if not (folder/'request.json').exists() else '请求模型中'
            if decision.get('decision_mode') in ('separated', 'grounded'):
                phase = '判读中' if not perception['assessment'] else '动作决策中'
            elif decision.get('decision_mode') == 'localized':
                phase = '双视角定位中' if not localization['assessment'] else '动作决策中'
            if decision.get('stage') == 'activation_verification':
                phase = '亮灯变化核验中'
            elif decision.get('stage') == 'withdrawal_verification':
                phase = '后撤核验中'
            elif decision.get('stage') == 'forward_guard':
                phase = '前推条件检查中'
            elif decision.get('stage') == 'progress_review':
                phase = '进展复核中'
            elif decision.get('stage') == 'hypothesis_review':
                phase = '无进展假设复核中'
            elif decision.get('stage') == 'view_acquisition':
                phase = '取景检查中'
            elif decision.get('decision_mode') == 'grounded' and decision.get('stage') == 'context_prepare':
                phase = '整理当前状态中'
            if context.get('compression_status') == 'running':
                phase = '压缩历史中'
            if (forward_guard['decision'] or {}).get('allowed') is False:
                phase = '前推未放行'
            if decision.get('stage') != 'progress_review' and (progress_review['decision'] or {}).get('allowed') is False:
                phase = '进展复核：需重新决策'
            view_phase = (view_state or {}).get('phase', decision.get('view_phase'))
            if view_phase in ('get_view', 'recover_view'):
                phase = '取景决策中' if view_phase == 'get_view' else '判断冲突：恢复取景中'
            if decision.get('stage') == 'view_acquisition':
                phase = '冲突恢复取景检查中' if view_phase == 'recover_view' else '取景检查中'
            if decision.get('stage') == 'localization':
                phase = '双视角定位中'
            elif decision.get('stage') == 'localization_review':
                phase = '重新定位与进展复核中'
            active_stage = decision.get('stage')
            if correction and correction.get('status') == 'retrying' and active_stage not in (
                    'perception', 'activation_verification', 'withdrawal_verification', 'forward_guard', 'progress_review', 'view_acquisition', 'localization', 'localization_review', 'hypothesis_review'):
                phase = ('恢复取景决策修正中' if view_phase == 'recover_view' else
                         '取景决策修正中' if view_phase == 'get_view' else '动作修正中')
            for stage_name, stage_data, stage_label in (
                    ('perception', perception, '判读修正中'),
                    ('activation_verification', activation, '亮灯核验修正中'),
                    ('withdrawal_verification', withdrawal, '后撤核验修正中'),
                    ('forward_guard', forward_guard, '前推检查修正中'),
                    ('progress_review', progress_review, '进展复核修正中'),
                    ('hypothesis_review', hypothesis_review, '假设复核格式修正中'),
                    ('localization', localization, '定位格式修正中'),
                    ('view_acquisition', view_acquisition, '取景定位修正中')):
                if (active_stage == stage_name and (stage_data['correction'] or {}).get('status') == 'retrying'):
                    phase = stage_label
                elif (active_stage == stage_name and stage_data['assessment'] is not None
                      and (stage_data['correction'] or {}).get('status') == 'accepted'):
                    phase = stage_label.replace('修正中', '完成')
            if active_stage == 'localization_review' and any(
                    (localization_review[name]['correction'] or {}).get('status') == 'retrying'
                    for name in ('before', 'current')):
                phase = '复核定位格式修正中'
            if (context.get('compression_status') == 'running'
                    and (compaction['correction'] or {}).get('status') == 'retrying'):
                phase = '压缩摘要修正中'
            if proposal:
                phase = {'report_done': '模型报告完成', 'report_unable': '模型无法继续',
                         'request_observation': '请求新观测'}.get(proposal['name'], '已生成建议')
            if (folder/'execution.json').exists():
                phase = '执行 / 采集新观测中'
            if completed:
                phase = '执行完成'
            elif outcome and outcome.get('status') in ('error', 'operator_stopped'):
                phase = '已停止'
            if error:
                phase = ('已停止' if (outcome and outcome.get('status') == 'operator_stopped') or decision.get('stage') == 'interrupted'
                         else ('定位请求失败' if model_decides else '定位失败') if error.get('stage') == 'localization' or localization['error']
                         else '定位复核失败' if error.get('stage') == 'localization_review' or localization_review['error']
                         else '取景检查失败' if error.get('stage') == 'view_acquisition' or view_acquisition['error']
                         else '假设复核失败' if error.get('stage') == 'hypothesis_review' or hypothesis_review['error']
                         else '进展复核失败' if error.get('stage') == 'progress_review' or progress_review['error']
                         else '请求失败')
            elif view_acquisition['error']:
                phase = '已停止' if decision.get('stage') == 'interrupted' else '取景检查失败'
            elif localization['error']:
                phase = '已停止' if decision.get('stage') == 'interrupted' else '定位请求失败' if model_decides else '定位失败'
            elif localization_review['error'] or any(localization_review[name]['error'] for name in ('before', 'current')):
                phase = '已停止' if decision.get('stage') == 'interrupted' else '定位复核失败'
            elif hypothesis_review['error']:
                phase = '已停止' if decision.get('stage') == 'interrupted' else '假设复核失败'
            elif progress_review['error']:
                phase = '已停止' if decision.get('stage') == 'interrupted' else '进展复核失败'
            elif perception['error'] or activation['error'] or withdrawal['error']:
                phase = '已停止' if decision.get('stage') == 'interrupted' else '判读失败'
            elif forward_guard['error']:
                phase = '已停止' if decision.get('stage') == 'interrupted' else '前推检查失败'
            rows.append(dict(round=int(folder.name)+1, phase=phase, proposal=proposal, error=error,
                             response=response.get('response') if response else None,
                             response_attempt=response_attempt, correction=correction, compaction=compaction,
                             elapsed_s=action_metrics.get('elapsed_s', response.get('elapsed_s') if response else None),
                             usage=(response.get('response', {}).get('usage') or {}) if response else {},
                             context=context, decision=decision, perception=perception, action_metrics=action_metrics,
                             activation=activation, withdrawal=withdrawal, forward_guard=forward_guard,
                             view_acquisition=view_acquisition, view_state=view_state,
                             localization=localization, localization_review=localization_review,
                             localization_current=read_json(folder/'localization_current.json'),
                             localized_progress=read_json(folder/'localized_progress.json'),
                             progress=read_json(folder/'progress.json'), progress_review=progress_review,
                             hypothesis_review=hypothesis_review,
                             alignment_progress=read_json(folder/'alignment_progress.json'),
                             control_plan=read_json(folder/'control_plan.json'),
                             control_memory=read_json(folder/'control_memory.json'),
                             decision_basis=read_json(folder/'decision_basis.json')))
        try:
            with (self.root/'terminal.log').open('rb') as stream:
                stream.seek(0, 2)
                stream.seek(max(0, stream.tell()-24000))
                log = stream.read().decode(errors='replace')
        except OSError:
            log = ''
        task = read_json(self.run/'task.json', {})
        launcher = read_json(self.root/'launcher.json', {})
        return dict(launcher=launcher, goal=task.get('goal', launcher.get('goal')), rounds=rows, read_only=self.read_only,
                    result=outcome, log=log)

    def observation_image(self, observation_id, view):
        """Serve only immutable image snapshots belonging to this run, never arbitrary paths."""
        if (not isinstance(observation_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', observation_id)
                or view not in ('head', 'right')):
            raise FileNotFoundError('snapshot_not_found')
        root = (self.run/'observations').resolve()
        if root != self.run.resolve()/'observations':
            raise FileNotFoundError('snapshot_not_found')
        path = (root/observation_id/(view+'.jpg')).resolve()
        if not path.is_relative_to(root) or path.parent != root/observation_id or not path.is_file():
            raise FileNotFoundError('snapshot_not_found')
        return path.read_bytes()

    def cameras(self):
        try:
            snapshot = request(self.camera_base, '/api/snapshot', timeout=1)
            snapshot['archived'] = False
            return snapshot
        except (OSError, ValueError):
            manifests = sorted((self.run/'observations').glob('*/manifest.json'),
                               key=lambda p: p.stat().st_mtime_ns, reverse=True)
            frames = {}
            metadata = {}
            if manifests:
                metadata = read_json(manifests[0], {})
                for view in ('head', 'right'):
                    path = manifests[0].parent / (view+'.jpg')
                    if path.exists():
                        frames[view] = dict(jpeg=base64.b64encode(path.read_bytes()).decode(), stale=True)
            return dict(frames=frames, depth={}, archived=True,
                        observation_id=metadata.get('observation_id'))


def serve(dashboard, port=8766):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, data, html=False):
            body = data.encode() if html else json.dumps(data, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'text/html; charset=utf-8' if html else 'application/json')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            try:
                snapshot = re.fullmatch(r'/api/observations/([A-Za-z0-9_-]{1,128})/(head|right)\.jpg', self.path)
                if self.path == '/':
                    self.reply(200, HTML, html=True)
                elif self.path == '/api/agent':
                    self.reply(200, dashboard.state())
                elif self.path == '/api/cameras':
                    self.reply(200, dashboard.cameras())
                elif snapshot:
                    try:
                        body = dashboard.observation_image(*snapshot.groups())
                    except FileNotFoundError:
                        self.reply(404, dict(error='snapshot_not_found'))
                        return
                    self.send_response(200)
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Cache-Control', 'private, max-age=3600')
                    self.send_header('X-Content-Type-Options', 'nosniff')
                    self.send_header('Content-Length', str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.reply(404, dict(error='not_found'))
            except Exception as exc:
                self.reply(503, dict(error=str(exc)))

        def do_POST(self):
            self.connection.settimeout(2)
            origin = self.headers.get('Origin')
            if origin and urlsplit(origin).netloc != self.headers.get('Host'):
                self.reply(403, dict(error='cross_origin_request'))
                return
            if self.path != '/api/stop':
                self.reply(404, dict(error='not_found'))
                return
            if dashboard.read_only:
                self.reply(403, dict(error='read_only_dashboard'))
                return
            try:
                if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                    raise ValueError('json_required')
                if self.headers.get('Transfer-Encoding') or int(self.headers.get('Content-Length', '0')) != 2:
                    raise ValueError('empty_object_required')
                if self.rfile.read(2) != b'{}':
                    raise ValueError('empty_object_required')
                dashboard.stop.set()
                self.reply(200, dict(status='stop_requested'))
            except (ValueError, OSError) as exc:
                self.reply(400, dict(error=str(exc)))

    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def main(argv=None):
    """View an existing run without restarting its Agent or controlling the robot."""
    import argparse
    parser = argparse.ArgumentParser(description='Read-only dashboard for an existing Agent run.')
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8767)
    parser.add_argument('--camera-base', default='', help='optional existing camera HTTP URL for live images')
    args = parser.parse_args(argv)
    if not (args.run_dir/'agent').is_dir():
        parser.error('--run-dir must contain an agent directory')
    server, thread = serve(Dashboard(args.run_dir, args.camera_base, threading.Event(), read_only=True), args.port)
    print(f'Read-only Agent viewer: http://127.0.0.1:{server.server_port}', flush=True)
    try:
        thread.join()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown(); server.server_close(); thread.join(2)


if __name__ == '__main__':
    main()
