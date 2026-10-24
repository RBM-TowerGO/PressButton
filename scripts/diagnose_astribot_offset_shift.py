"""Audit frozen offset probes under state distribution shift; never refit.

Mean substitution is a diagnostic counterfactual, NOT a replacement benchmark
or evidence that an actual controller should ignore velocity/gripper state.
"""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import joblib

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from analyze_astribot_offset_probe import load, inputs
from pressb.online_rl.offset_probe import metrics, near_mask
from pressb.astribot_collection.contract import JOINT_NAMES
from astribot_closedloop_control import GRIP


def scaled_prediction(model,z,family):
    if family=='ridge': return model.predict(z)
    import torch
    network,mean,scale=model
    with torch.no_grad():return network(torch.tensor(z,dtype=torch.float32)).numpy()*scale+mean


def main(args):
    import torch
    torch.set_num_threads(2)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    args.output.mkdir(parents=True,exist_ok=True)
    controlled,closed=load(args.controlled),load(args.closedloop)
    mask=near_mask(closed['label_mm']) & ~closed['ever_contact']
    y,groups=closed['label_mm'][mask,:2],closed['scene'][mask]
    saved=np.load(args.analysis/'predictions.npz',allow_pickle=False)
    result=dict(purpose='Frozen-input sensitivity only; primary models and predictions remain unchanged',
                near_precontact_samples=len(y),groups=len(np.unique(groups)),models={})
    interventions={'original':[], 'gripper_q_mean':GRIP.tolist(),
                   'qd_mean':list(range(32,64)),
                   'gripper_q_and_qd_mean':GRIP.tolist()+list(range(32,64)),
                   'all_state_mean':list(range(64))}
    for path in sorted(args.analysis.glob('*.joblib')):
        key=path.stem
        if not(key.endswith('_state_action') or key.endswith('_augmented')):continue
        family=key.split('_')[0]
        kind='state_action' if key.endswith('_state_action') else 'augmented'
        scaler,model,_=joblib.load(path)
        x=inputs(closed,kind)[mask]
        if family=='ridge':x=x.astype(np.float64)
        z=scaler.transform(x)
        rows={}
        for name,indices in interventions.items():
            altered=z.copy();altered[:,indices]=0.
            prediction=scaled_prediction(model,altered,family)
            if name=='original':
                np.testing.assert_allclose(prediction,saved[key+'__closedloop_precontact'],rtol=1e-5,atol=1e-4)
            rows[name]=metrics(y,prediction,groups)
        result['models'][key]=rows
        if key=='ridge_20261023_state_action':
            contribution=np.abs(z[:,:,None]*model.coef_.T[None,:,:]).mean((0,2))
            names=['q:'+n for n in JOINT_NAMES]+['qd:'+n for n in JOINT_NAMES]+[f'base:{i}' for i in range(56)]
            result['top_linear_contributions']=[dict(feature=names[i],train_std=float(scaler.scale_[i]),
                closed_max_abs_z=float(np.abs(z[:,i]).max()),mean_abs_term_mm=float(contribution[i]))
                for i in np.argsort(contribution)[-20:][::-1]]
    result['saved_prediction_reproduction_passed']=True
    (args.output/'diagnosis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    fig,axes=plt.subplots(2,2,figsize=(10,8))
    for row,(domain,data,selection) in enumerate([
        ('controlled_test',controlled,controlled['split']==2),
        ('closedloop_precontact',closed,mask)]):
        actual=data['label_mm'][selection,:2]
        for axis,ax in enumerate(axes[row]):
            for key in ['ridge_20261023_latent','mlp_20261023_latent']:
                pred=saved[key+'__'+domain]
                ax.scatter(actual[:,axis],pred[:,axis],s=5,alpha=.25,label=key.split('_')[0])
            lo,hi=actual[:,axis].min(),actual[:,axis].max()
            ax.plot([lo,hi],[lo,hi],color='black',lw=1,label='identity')
            ax.set(title=domain+': '+['horizontal','vertical'][axis],xlabel='Measured offset (mm)',ylabel='Latent-only prediction (mm)')
            ax.legend()
    fig.tight_layout();fig.savefig(args.output/'latent_prediction.png',dpi=180);plt.close(fig)
    lines=['# 冻结探针的输入分布诊断','',
           '仅改变已训练预测器的测试输入：将指定标准化坐标设为零，即训练均值。没有重新拟合模型，没有替换正式结果。',
           '这些结果用于定位失稳输入，不能作为新的正式 A/B 比较，也不建议据此直接删除实际控制器的状态输入。','',
           '| 模型 | 输入替换 | 水平 MAE mm | 垂直 MAE mm |','|---|---|---:|---:|']
    for key,rows in result['models'].items():
        for name,r in rows.items():
            h,v=r['group_macro_mae_mm'];lines.append(f'| {key} | {name} | {h:.3f} | {v:.3f} |')
    lines+=['','所有原始预测均与保存的预测文件数值一致。','', '![latent 单独解码的泛化](latent_prediction.png)','']
    (args.output/'report.md').write_text('\n'.join(lines))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['controlled','closedloop','analysis','output']:p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
