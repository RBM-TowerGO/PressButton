"""Generate an evidence-linked fast-iteration report without inferring missing outcomes."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from run_astribot_online_rl import atomic_json


def read(path,default=None):
 try:return json.loads(Path(path).read_text())
 except (FileNotFoundError,ValueError):return default


def continuity_and_auxiliary(root):
 result={}
 for name in ['shared','auxiliary']:
  entries=[]
  for file in (root/name).glob('**/residuals.npz'):
   report=read(file.parent/'report.json')
   if not report:continue
   with np.load(file,allow_pickle=False) as data:
    x=data['requested'].reshape(-1,8)[:report['control_steps']].astype(np.float64)
   if len(x)<2:continue
   difference=x[1:]-x[:-1]
   boundary=np.arange(1,len(x))%7==0
   rows={}
   for group,mask in [('within_chunk',~boundary),('chunk_boundary',boundary)]:
    d=difference[mask]
    rows[group]=dict(normalized_difference_rms=float(np.sqrt(np.mean(d**2))) if len(d) else None,
      arm_difference_rms_radians=float(np.sqrt(np.mean((d[:,:7]*.01)**2))) if len(d) else None,
      gripper_difference_rms_radians=float(np.sqrt(np.mean((d[:,7:]*.025)**2))) if len(d) else None)
   left,right=x[:-1,:7],x[1:,:7];norms=np.linalg.norm(left,axis=1)*np.linalg.norm(right,axis=1)
   valid=norms>0
   angle=np.rad2deg(np.arccos(np.clip((left[valid]*right[valid]).sum(1)/norms[valid],-1,1)))
   rows.update(mean_arm_direction_angle_degrees=float(angle.mean()) if len(angle) else None,missing_zero_direction_angles=int((~valid).sum()))
   trace=file.parent/'physics_trace.npz'
   if trace.exists():
    with np.load(trace,allow_pickle=False) as data:
     if 'target_contact' in data:
      contact=data['target_contact'].astype(bool);hits=np.flatnonzero(contact)
      rows.update(target_contact_detected=bool(len(hits)),target_contact_signal_lost_after_first=bool((~contact[hits[0]:]).any()) if len(hits) else None)
   entries.append(dict(path=str(file.parent),scene=report['candidate_id'],status=report['status'],**rows))
  result[name]=entries
 errors={}
 for file in (root/'auxiliary').glob('**/auxiliary_labels.jsonl'):
  for line in file.read_text().splitlines():
   row=json.loads(line);label=np.asarray(row['label_mm'])
   if not np.isfinite(label).all() or not (np.abs(label)<=30).all():continue
   phase='postcontact' if row['ever_contact'] else 'precontact'
   domain='development' if 'checkpoints' in file.parts else 'training_online'
   for network in ['actor','critic']:
    key=domain+'_'+phase+'_'+network
    errors.setdefault(key,[]).append(np.abs(np.asarray(row[network+'_prediction_mm'])-label[:2]))
 result['auxiliary_mae_mm']={k:dict(samples=len(v),mae_mm=np.mean(v,axis=0).tolist()) for k,v in errors.items()}
 result['contact_note']='Contact-signal disappearance is a diagnostic tag, never a replacement physical failure rule; later success is retained.'
 return result


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True);a=p.parse_args()
 root=a.campaign.resolve();out=root/'report';out.mkdir(exist_ok=True)
 lines=['# Astribot latent 快速迭代实验记录','',f'产物目录：`{root}`。','',
 '这是刻意偏重历史失败的小样本开发集诊断；随机种子的重复回合不算独立场景。未运行 test-40；val-40 已用于开发，不声称独立泛化。','',
 '## 阶段 0','', '| 模式 | 残差执行种子 | 已完成回合 | 成功 | 异常接触回合 | 成功场景 |', '|---|---|---:|---:|---:|---|']
 metrics={};amplitudes={}
 for mode in ['zero','B_mean','B_sample','unconditional']:
  seeds=[20261024,20261025] if mode in ['B_sample','unconditional'] else [20261024]
  for seed in seeds:
   rows=[read(f) for f in sorted((root/'stage0'/mode/str(seed)/'episodes').glob('*/report.json'))]
   rows=[r for r in rows if r and r['status']!='infrastructure_error']
   success=[r['candidate_id'] for r in rows if r['status']=='success']
   lines.append(f"| {mode} | {seed} | {len(rows)}/12 | {len(success)} | {sum(r['unexpected_contacts']>0 for r in rows)} | {success} |")
   metrics[f'{mode}_{seed}']=dict(completed=len(rows),successes=len(success),success_scenes=success)
  arrays=[]
  for f in (root/'stage0'/mode).glob('*/episodes/*/residuals.npz'):
   with np.load(f,allow_pickle=False) as data:arrays.append(data['executed'].astype(np.float64))
  if arrays:
   x=np.concatenate(arrays);physical=x*np.asarray([.01]*7+[.025])
   amplitudes[mode]=dict(arm_rms_degrees=float(np.rad2deg(np.sqrt(np.mean(physical[:,:7]**2)))),gripper_rms_degrees=float(np.rad2deg(np.sqrt(np.mean(physical[:,7:]**2)))),executed_control_steps=len(x))
 protocol=read(root/'protocol.json',{})
 bank=protocol.get('bank_rms')
 lines+=['','### 实际执行残差与残差库幅度','', '| 来源 | 手臂 RMS（°） | 夹爪 RMS（°） |','|---|---:|---:|']
 if bank:lines.append(f"| 残差库 | {bank['arm']['degrees_rms']:.6g} | {bank['gripper']['degrees_rms']:.6g} |")
 for name,row in amplitudes.items():lines.append(f"| {name} 实际执行 | {row['arm_rms_degrees']:.6g} | {row['gripper_rms_degrees']:.6g} |")
 lines+=['','残差库匹配原 B 训练 replay 下的边际分布；闭环实际幅度以上表为准，不能直接声称噪声强度相同。', '', '## 派生训练','', '| 候选 | 新增经验 | 在线 learner／actor 更新 | 离线更新 | 开发成功 | 异常接触回合 | 状态 |','|---|---:|---:|---:|---:|---:|---|']
 candidates=[]
 for name in ['reg_01','reg_001','reg_001_low_entropy','shared','auxiliary','auxiliary_no_online']:
  directory=root/name;status=read(directory/'status.json')
  if not status:
   lines.append(f'| {name} | 待执行 | 待执行 | 待执行 | 待执行 | 待执行 | 待执行／条件未触发 |');continue
  for cp in sorted((directory/'checkpoints').glob('new_*')):
   s=read(cp/'status.json',{});ev=read(cp/'eval_summary.json')
   result=dict(name=name,checkpoint=str(cp),status=s,evaluation=ev);candidates.append(result)
   successes=ev['successes'] if ev else '待执行'
   contacts=ev['abnormal_contact_episodes'] if ev else '待执行'
   lines.append(f"| {name} | {s.get('new_transitions','待执行')} | {s.get('updates','待执行')}／{s.get('actor_updates','待执行')} | {s.get('offline_updates','待执行')} | {successes}/12 | {contacts} | {'完成' if ev else '评估未完成'} |")
  if not list((directory/'checkpoints').glob('new_*')):
   lines.append(f"| {name} | {status['new_transitions']} | {status.get('updates',0)}／{status.get('actor_updates',0)} | {status['offline_updates']} | 待执行 | 待执行 | {status['state']} |")
 lines+=['','阶段 1 没有同预算 lambda_a=1 续训对照；若有收益，不能全部归因于正则变化。', '', '## 决定与中断证据','']
 for filename in ['supervisor.json','finish_progress.json','stage1_decision.json','stage2_decision.json','stage3_decision.json','final_decision.json']:
  data=read(root/filename)
  if data is not None:
   if filename.endswith('progress.json') or filename=='supervisor.json':text=f"状态 `{data.get('state')}`。"
   elif filename=='final_decision.json':text=f"工程投入门槛：{'通过' if data['passed'] else '未通过／未完成复测'}。{data['note']}"
   else:text=f"已保存决定；是否推进：{data.get('qualifies',data.get('qualified','见原始 JSON'))}。"
   lines.append(f'- [{filename}](../{filename})：{text}')
   if data.get('error'):lines.extend(['','```text',data['error'].strip(),'```',''])
 progress=read(root/'progress.json',{})
 if progress.get('error'):lines.extend(['','```text',progress['error'].strip(),'```',''])
 lines+=['','未完成回合保留为中断证据，不补零，不计为物理失败。仅有 timeout 时，失败原因保持尚不能分类，不猜测未对准或未压够。']
 (out/'report.md').write_text('\n'.join(lines)+'\n')
 atomic_json(out/'metrics.json',dict(stage0=metrics,amplitudes=amplitudes,candidates=candidates))
 atomic_json(out/'continuity_and_auxiliary.json',continuity_and_auxiliary(root))
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 for name in ['reg_01','reg_001','reg_001_low_entropy','shared','auxiliary','auxiliary_no_online']:
  path=root/name/'gradient_diagnostics.jsonl'
  if not path.exists():continue
  rows=[json.loads(l) for l in path.read_text().splitlines()]
  if not rows:continue
  fig,ax=plt.subplots(figsize=(7,4))
  for term in ['value','entropy','regularization']:
   ax.plot([r['updates'] for r in rows],[r['mean_head_gradient_norm'][term] for r in rows],label=term)
  ax.set(xlabel='New learner updates',ylabel='Mean-head gradient norm',yscale='symlog');ax.legend();fig.tight_layout();fig.savefig(out/f'{name}_gradients.png');plt.close(fig)
  points=[]
  initial=read(root/name/'initial_actions.json')
  if initial:points.append((0,initial))
  for cp in sorted((root/name/'checkpoints').glob('new_*')):
   stats=read(cp/'fixed_observation_actions.json')
   if stats:points.append((int(cp.name.split('_')[1]),stats))
  if points:
   fig,ax=plt.subplots(figsize=(7,4))
   for mode in ['B_mean','B_sample']:
    ax.plot([p[0] for p in points],[p[1][mode]['arm']['degrees_rms'] for p in points],marker='o',label=mode)
   ax.set(xlabel='New transitions',ylabel='Arm residual RMS (degrees)',yscale='log');ax.legend();fig.tight_layout();fig.savefig(out/f'{name}_rms.png');plt.close(fig)
 print(out/'report.md')

if __name__=='__main__':main()
