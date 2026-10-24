"""Train small offline probes; reserve whole scenes and freeze OOD evaluation."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import hashlib
from collections import Counter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from pressb.online_rl.offset_probe import metrics, paired_improvement, near_mask


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def load(directory):
    shards = sorted(directory.glob('scene_*/samples.npz'))
    if not shards:
        raise ValueError(f'No sample shards: {directory}')
    chunks = []
    for path in shards:
        with np.load(path, allow_pickle=False) as f:
            chunks.append({k:f[k].copy() for k in f.files})
    data = {k:np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
    for k in ('q','qd','base','latent','label_mm'):
        if not np.isfinite(data[k]).all():
            raise ValueError(f'Nonfinite {k}')
    data['state_action'] = np.concatenate([data['q'], data['qd'], data['base'].reshape(-1,56)], axis=1)
    return data


def inputs(data, kind):
    if kind == 'state_action': return data['state_action']
    if kind == 'latent': return data['latent']
    return np.concatenate([data['state_action'],data['latent']],axis=1)


def fit_ridge(x, y, train, val):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    # Highly correlated 2048-D latents can make float32 normal equations
    # inaccurate at small alpha; retain double precision throughout ridge.
    x = np.asarray(x, dtype=np.float64)
    scaler = StandardScaler().fit(x[train])
    xt, xv = scaler.transform(x[train]), scaler.transform(x[val])
    best = None
    for alpha in (.01, .1, 1., 10., 100., 1000., 10000.):
        model = Ridge(alpha=alpha).fit(xt, y[train])
        score = float(np.abs(model.predict(xv)-y[val]).mean())
        if best is None or score < best[0]: best = (score, model, alpha)
    return scaler, best[1], dict(alpha=best[2],validation_mae_mm=best[0])


def fit_mlp(x, y, train, val, seed):
    import torch
    from sklearn.preprocessing import StandardScaler
    torch.set_num_threads(2)
    scaler = StandardScaler().fit(x[train])
    xt = torch.tensor(scaler.transform(x[train]), dtype=torch.float32)
    xv = torch.tensor(scaler.transform(x[val]), dtype=torch.float32)
    ym = y[train].mean(0)
    ys = np.maximum(y[train].std(0), 1.)
    yt = torch.tensor((y[train]-ym)/ys,dtype=torch.float32)
    best = None
    for decay in (1e-4, 1e-2):
        torch.manual_seed(seed)
        model = torch.nn.Sequential(torch.nn.Linear(x.shape[1],64),torch.nn.ReLU(),
                                    torch.nn.Linear(64,64),torch.nn.ReLU(),torch.nn.Linear(64,2))
        opt = torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=decay)
        local_best, stale = float('inf'), 0
        for epoch in range(300):
            model.train()
            for batch in torch.randperm(len(xt)).split(256):
                loss = torch.nn.functional.mse_loss(model(xt[batch]),yt[batch])
                opt.zero_grad(); loss.backward(); opt.step()
            model.eval()
            with torch.no_grad(): prediction = model(xv).numpy()*ys+ym
            score = float(np.abs(prediction-y[val]).mean())
            if score < local_best - 1e-5:
                local_best, stale = score, 0
                if best is None or score < best[0]:
                    best = (score,deepcopy(model.state_dict()),decay,epoch+1)
            else:
                stale += 1
            if stale >= 25: break
        print(json.dumps(dict(fit='mlp',seed=seed,inputs=x.shape[1],weight_decay=decay,epochs=epoch+1,validation_mae_mm=local_best)),flush=True)
    model.load_state_dict(best[1]);model.eval()
    return scaler, (model,ym,ys), dict(seed=seed,weight_decay=best[2],epoch=best[3],validation_mae_mm=best[0])


def predict(scaler, model, x, family):
    if not len(x): return np.empty((0,2))
    x = scaler.transform(np.asarray(x, dtype=np.float64) if family == 'ridge' else x)
    if family == 'ridge': return model.predict(x)
    import torch
    network, ym, ys = model
    with torch.no_grad(): return network(torch.tensor(x,dtype=torch.float32)).numpy()*ys+ym


def run(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import joblib
    args.output.mkdir(parents=True,exist_ok=True)
    controlled, closed = load(args.controlled), load(args.closedloop)
    if set(controlled['scene']) & set(closed['scene']):
        raise ValueError('Controlled and closedloop scenes overlap')
    for g in np.unique(controlled['scene']):
        if len(np.unique(controlled['split'][controlled['scene']==g])) != 1:
            raise ValueError('Scene leakage')
    masks = [controlled['split']==i for i in range(3)]
    if not all(m.any() for m in masks): raise ValueError('Missing train/val/test')
    near = near_mask(closed['label_mm'])
    domains = {'controlled_test':(controlled,masks[2]),
               'controlled_near_surface':(controlled,masks[2] & (np.abs(controlled['label_mm'][:,2])<=10)),
               'closedloop_near':(closed,near),
               'closedloop_precontact':(closed,near & ~closed['ever_contact']),
               'closedloop_precontact_near_surface':(closed,near & ~closed['ever_contact'] & (np.abs(closed['label_mm'][:,2])<=10)),
               'closedloop_postcontact':(closed,near & closed['ever_contact'])}
    report = dict(protocol=dict(train_only_scaling=True,group_split='scene',
        bootstrap='2000 paired scene resamples, equal scene weights',
        main_decision_rule='Ridge paired mean-axis improvement CI lower >0 AND augmented MAE beats training-mean constant; closedloop needs >=20 precontact scenes',
        target='TCP horizontal/vertical offset mm; not contact point or correction action',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
        quality={},models={},comparisons={})
    if args.audit is not None:
        report['instrumentation_audit'] = json.loads(args.audit.read_text())
    for name,data in [('controlled',controlled),('closedloop',closed)]:
        report['quality'][name]=dict(samples=len(data['q']),scenes=len(np.unique(data['scene'])),
            label_quantiles_mm=np.quantile(data['label_mm'],[0,.05,.5,.95,1],axis=0).tolist(),
            near_samples=int(near_mask(data['label_mm']).sum()),
            near_scenes=len(np.unique(data['scene'][near_mask(data['label_mm'])])))
        directory = args.controlled if name == 'controlled' else args.closedloop
        summaries = [json.loads(p.read_text()) for p in directory.glob('scene_*/summary.json')]
        report['quality'][name]['completed_scene_records'] = len(summaries)
        report['quality'][name]['outcomes'] = dict(Counter(s['status'] for s in summaries))
        if name == 'controlled':
            report['quality'][name]['planned_samples'] = sum(s['planned'] for s in summaries)
            report['quality'][name]['failure_events'] = dict(Counter(f['reason'] for s in summaries for f in s['failures']))
            report['quality'][name]['missing_samples'] = sum(s['planned']-s['valid'] for s in summaries)
            report['quality'][name]['split_scenes'] = [len(np.unique(data['scene'][m])) for m in masks]
            observations = [json.loads(line) for p in directory.glob('scene_*/observations.jsonl') for line in p.read_text().splitlines()]
            if observations:
                actual = np.array([r['label_mm'] for r in observations])
                requested = np.array([r['requested_mm'] for r in observations])
                report['quality'][name]['actual_minus_requested_quantiles_mm'] = np.quantile(actual-requested,[0,.5,1],axis=0).tolist()
                report['quality'][name]['tcp_front_normal_difference_quantiles_mm'] = np.quantile([r['tcp_to_front_normal_mm'] for r in observations],[0,.5,1]).tolist()
    predictions={}
    # Controlled samples are settled poses; real rollouts can have larger qd.
    # Quantify this shift instead of interpreting failed transfer as no information.
    from sklearn.preprocessing import StandardScaler
    shift_scaler = StandardScaler().fit(inputs(controlled,'augmented')[masks[0]])
    shifted = (np.abs(shift_scaler.transform(inputs(closed,'augmented')[near]))
               if near.any() else np.empty((0,2168)))
    report['quality']['closedloop']['standardized_feature_shift'] = {
        name: dict(abs_z_quantiles=np.quantile(shifted[:,start:end],[.5,.95,1]).tolist(),
                   fraction_abs_z_gt5=float(np.mean(shifted[:,start:end]>5)))
        for name,start,end in [('q',0,32),('qd',32,64),('base_action',64,120),('latent',120,2168)]
    } if len(shifted) else {}
    average=controlled['label_mm'][masks[0],:2].mean(0)
    report['constant']={name:metrics(d['label_mm'][m,:2],np.tile(average,(m.sum(),1)),d['scene'][m]) for name,(d,m) in domains.items()}
    for family in ['ridge','mlp']:
        for seed in ([20261023] if family=='ridge' else [20261023,42,43]):
            for kind in ['state_action','latent','augmented']:
                key=f'{family}_{seed}_{kind}'
                x=inputs(controlled,kind)
                if family=='ridge': fitted=fit_ridge(x,controlled['label_mm'][:,:2],masks[0],masks[1])
                else: fitted=fit_mlp(x,controlled['label_mm'][:,:2],masks[0],masks[1],seed)
                scaler,model,config=fitted
                joblib.dump(fitted,args.output/f'{key}.joblib')
                report['models'][key]=dict(config=config,domains={})
                for name,(data,mask) in domains.items():
                    pred=predict(scaler,model,inputs(data,kind)[mask],family)
                    predictions[(key,name)]=pred
                    report['models'][key]['domains'][name]=metrics(data['label_mm'][mask,:2],pred,data['scene'][mask])
                    if kind=='augmented':
                        shuffled=inputs(data,kind)[mask].copy()
                        shuffled[:,120:]=shuffled[np.random.default_rng(seed).permutation(len(shuffled)),120:]
                        shuffle_pred=predict(scaler,model,shuffled,family)
                        report['models'][key]['domains'][name]['latent_shuffled']=metrics(data['label_mm'][mask,:2],shuffle_pred,data['scene'][mask])
                print(json.dumps(dict(completed=key,test=report['models'][key]['domains']['controlled_test'])),flush=True)
                write_json(args.output/'metrics.partial.json',report)
            comparison=f'{family}_{seed}'
            report['comparisons'][comparison]={}
            for name,(data,mask) in domains.items():
                report['comparisons'][comparison][name]=paired_improvement(data['label_mm'][mask,:2],
                    predictions[(f'{comparison}_state_action',name)],predictions[(f'{comparison}_augmented',name)],data['scene'][mask])
    # Freeze all per-row predictions for independent audit.
    np.savez_compressed(args.output/'predictions.npz',**{k+'__'+d:v for (k,d),v in predictions.items()})
    report['quality']['closedloop']['episode_coverage']=report['quality']['closedloop']['near_scenes']/max(1,report['quality']['closedloop']['completed_scene_records'])
    write_json(args.output/'metrics.json',report)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,(name,data) in zip(axes,[('Controlled',controlled),('Closed loop',closed)]):
        mask=np.ones(len(data['q']),bool) if name=='Controlled' else near
        ax.scatter(*data['label_mm'][mask,:2].T,s=3,alpha=.25)
        ax.set(title=name,xlabel='Horizontal offset (mm)',ylabel='Vertical offset (mm)')
        ax.axhline(0,color='gray',lw=.5);ax.axvline(0,color='gray',lw=.5)
    fig.tight_layout();fig.savefig(args.output/'coverage.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,8))
    for row,domain in enumerate(['controlled_test','closedloop_precontact']):
        d,m=domains[domain]
        for axis,ax in enumerate(axes[row]):
            for kind in ['state_action','augmented']:
                pred=predictions[(f'ridge_20261023_{kind}',domain)]
                ax.scatter(d['label_mm'][m,axis],pred[:,axis],s=3,alpha=.3,label=kind)
            ax.set(title=f'{domain}: {"horizontal" if axis==0 else "vertical"}',xlabel='Measured (mm)',ylabel='Predicted (mm)')
            ax.legend()
    fig.tight_layout();fig.savefig(args.output/'prediction.png',dpi=180);plt.close(fig)
    lines=['# latent 偏移诊断实验结果','',
        '预测目标是几何 TCP 相对按钮中心的水平、垂直偏移，不是成功率，也不是关节纠偏动作。',
        '所有标准化、超参数选择和早停仅使用受控训练／验证场景。闭环数据只用于冻结模型的外部测试。','',
        '| 数据 | 样本数 | 场景数 | 近按钮场景数 |','|---|---:|---:|---:|']
    for name,q in report['quality'].items():lines.append(f"| {name} | {q['samples']} | {q['scenes']} | {q['near_scenes']} |")
    lines += ['', '正的改善表示加入 latent 降低误差。下表按场景等权，95% CI 为场景级配对 bootstrap。','',
        '| 模型 | 测试集 | 两轴平均 MAE 改善 mm | 95% CI |','|---|---|---:|---|']
    for model,domains_report in report['comparisons'].items():
        for name,r in domains_report.items():
            if r['groups']:
                lo,hi=r['mean_axes_ci95_mm']
                lines.append(f"| {model} | {name} | {r['mean_axes_improvement_mm']:.3f} | [{lo:.3f}, {hi:.3f}] |")
    r=report['comparisons']['ridge_20261023']
    augmented_domains=report['models']['ridge_20261023_augmented']['domains']
    def beats_constant(domain):
        return (augmented_domains[domain]['n']>0 and
                np.mean(augmented_domains[domain]['group_macro_mae_mm']) <
                np.mean(report['constant'][domain]['group_macro_mae_mm']))
    controlled_good=r['controlled_test']['mean_axes_ci95_mm'][0]>0 and beats_constant('controlled_test')
    closed_r=r['closedloop_precontact']
    closed_good=(closed_r.get('groups',0)>=20 and closed_r.get('mean_axes_ci95_mm',[0])[0]>0
                 and beats_constant('closedloop_precontact'))
    verdict='有稳定增量信息（主线性探针）' if controlled_good and closed_good else ('仅受控条件有效（主线性探针）' if controlled_good else '未获得充分证据（主线性探针）')
    report['decision'] = dict(primary_ridge=verdict, controlled_incremental_evidence=bool(controlled_good),
                              closedloop_incremental_evidence=bool(closed_good),
                              augmented_beats_constant={d:bool(beats_constant(d)) for d in domains})
    write_json(args.output/'metrics.json',report)
    lines += ['',f'主探针判定：**{verdict}**。需结合三个 MLP 种子的方向和误差分布解读，不能将解码结果等同于闭环纠错能力。',
        '上述判定还要求加 latent 的预测误差低于训练均值常数基线，防止把仍然不可用的分布外预测改善误判为可靠信息。',
        '', '限制：受控与闭环样本来自不同场景，但均来自冻结 VLA 原训练场景集合；本实验检验探针的跨场景泛化，未检验 VLA 对全新场景的泛化。',
        '增量预测改善是在本实验预测器容量和训练预算下的结果，不是条件互信息证明：latent 对状态信息的重新编码也可能让有限容量预测器更容易解码。',
        '受控样本是到位后的姿态，闭环样本包含运动状态；metrics.json 记录了各输入块的标准化范围变化。闭环迁移失败不能直接解释为 latent 不含信息。',
        'latent 打乱属于输入扰动诊断，可能造成分布外组合，不能单独作为因果证据。',
        'TCP 在夹爪张开时不是实际接触点；接触落点与指尖几何信息见采集 JSONL。',
        '', '![偏移覆盖](coverage.png)', '', '![预测结果](prediction.png)', '']
    if 'instrumentation_audit' in report:
        a = report['instrumentation_audit']['scenes'][0]
        lines += ['## 记录侵入性验收', '',
                  f"同一冻结观测的开关记录对照：{a['on']['paired_frozen_requests']} 次请求，动作及 latent 逐元素一致，物理时间与 q/qd 不变。",
                  f"独立重置 off/off 逐步一致：{a['independent_reset_deterministic']}；独立重置 off/on 逐步一致：{a['independent_reset_instrumentation_trace_equal']}。",
                  '因此不能宣称独立重置后的逐步轨迹等价；本次只读证据来自同步冻结观测对照。', '']
    (args.output/'report.md').write_text('\n'.join(lines))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--controlled',type=Path,required=True)
    p.add_argument('--closedloop',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--audit',type=Path)
    run(p.parse_args())
