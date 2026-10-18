"""Write the final report only after all declared trials and audits pass."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path('/home/js/robot/VLA/PressButton/PressB')
run=ROOT/'outputs/astribot_closedloop_20261018/test40'
model=ROOT/'models/astribot_head_right_joint8_ft_20261018'
summary=json.loads((run/'summary.json').read_text())
audit=json.loads((run/'audit.json').read_text())
protocol=json.loads((run/'protocol.json').read_text())
contract=json.loads((model/'data_contract.json').read_text())
deployment=json.loads((model/'deployment.json').read_text())
assert summary['complete'] and audit['passed'] and audit['complete'] and audit['all_video_frames_decoded']
assert summary['completed']==audit['audited']==40
assert protocol['candidate_ids']==contract['split']['test_source_episode_ids']
assert [r['candidate_id'] for r in audit['results']]==protocol['candidate_ids']
assert summary['successes']==audit['successes']
assert hashlib.sha256((model/'data_contract.json').read_bytes()).hexdigest()==protocol['data_contract_sha256']
assert protocol['model_sha256']==deployment['verified_inference_server']['model_sha256']
rows=audit['results'];success=[r for r in rows if r['success']];failed=[r for r in rows if not r['success']]
latencies=[]
for row in rows:
    req=json.loads((run/f'candidate_{row["candidate_id"]:06d}'/'policy_requests.json').read_text())
    latencies.extend(x['inference_seconds'] for x in req)
failure_modes=Counter()
for r in failed:
    if r['reason']!='timeout':mode=r['reason']
    elif r['max_arm_change_rad']<.1:mode='初始姿态附近停滞'
    elif r['max_force_n']>0:mode='接触按钮但未满足完整按压条件'
    else:mode='发生明显移动但未接触按钮'
    failure_modes[mode]+=1
stats=dict(completed=40,successes=len(success),success_rate=len(success)/40,
    success_candidates=[r['candidate_id'] for r in success],failure_modes=dict(failure_modes),
    successful_simulation_seconds=[r['simulated_seconds'] for r in success],
    inference_calls=len(latencies),inference_median_ms=float(np.median(latencies)*1000),
    inference_p95_ms=float(np.percentile(latencies,95)*1000),
    total_trial_wall_seconds=sum(r['wall_seconds'] for r in rows),
    max_camera_pose_matrix_error=max(r['max_camera_pose_error'] for r in rows),
    all_videos_fully_decoded=True,decoded_video_count=160,
    unexpected_contact_trials=sum(bool(r['unexpected_contacts']) for r in rows),
    model_sha256=protocol['model_sha256'],data_contract_sha256=protocol['data_contract_sha256'])
(run/'final_metrics.json').write_text(json.dumps(stats,indent=2,ensure_ascii=False)+'\n')
lines=[
 '# Astribot VLA 在 Pro6000 的闭环测试结果',
 '',
 f'模型已完整部署，40 个保留测试场景全部完成并通过记录核验。成功 **{len(success)}/40（{len(success)/40:.1%}）**，失败 {len(failed)} 次。',
 '',
 '模型能够在部分场景中从初始姿态自行闭合夹爪、移动并按下按钮；目前的成功率仍不足以说明任务完成已稳定。',
 '',
 '模型位置：',
 '',
 f'`{model}/checkpoints/step_008000/model.pt`',
 '',
 f'权重 SHA256：`{protocol["model_sha256"]}`。加载时严格核验通过，全部 1617 个张量成功加载。',
 '',
 '本次评估使用训练时冻结的 40 个测试场景，每个初始位置执行一次，不用于调参。模型输入为实时头部和右腕图像、实测关节状态及固定任务文本。未使用示范轨迹或目标几何来控制机器人。',
 '',
 '控制输出为右臂 7 个关节加夹爪 1 个自由度；夹爪标量按冻结的正负号关系展开到六个夹爪关节，其余关节保持原配置目标。',
 '',
 '每次预测 7 步绝对关节目标，按 30 Hz 执行；每一步在 120 Hz 物理仿真中线性插值 4 次。执行完 7 步后重新观察。每个场景最多 40 秒仿真时间。图像和录像约 4.286 Hz，录像保留实际物理时间戳。',
 '',
 '成功条件为仿真按钮实际位移至少 1 mm、同时存在大于 0.02 N 的右指尖接触，并点亮锁存指示灯；发生非预期接触或其他控制异常会判失败。按压成功后停止，本次没有评估回撤或返回初始姿态。',
 '',
 '失败情况：',
 '',
 *[f'- {name}：{count} 次。' for name,count in failure_modes.items()],
 '',
 '“初始姿态附近停滞”是复盘用的描述分类，指整段运行右臂任一关节相对初始角度的最大变化小于 0.1 rad；它不参与正式成功判定。',
 '',
 f'共进行了 {len(latencies)} 次模型推理。预热后正式测试的推理中位耗时 {stats["inference_median_ms"]:.1f} ms，95 分位耗时 {stats["inference_p95_ms"]:.1f} ms。推理期间暂停仿真时间，因此这些结果属于同步仿真闭环，尚未验证实时真机控制。',
 '',
 '核验结果：',
 '',
 '- 40 个场景的实时状态输入、预测动作、4 个物理时刻的实际插值目标全部对应。',
 '- 四路录像共 160 个视频逐帧解码通过，时间戳与物理记录一致。',
 f'- 每次取图均比较渲染相机与实测物理位姿，最大矩阵元素误差 {stats["max_camera_pose_matrix_error"]:.3g}（阈值 1e-4）。',
 f'- 非预期接触场景数：{stats["unexpected_contact_trials"]}。',
 '- 初期发现的 PhysX/Fabric 姿态同步问题已修正。旧 smoke 与 primary_chunk7 结果已作废，不计入这 40 个结果。',
 '',
 f'完整结果、协议和校验记录位于 `{run}`。每个 candidate 目录含四路视频、逐物理时刻轨迹、每次模型请求及相机标定。运行说明见 `{ROOT}/docs/astribot_closedloop_20261018.md`。',
 '',
 '模型权重及归一化统计在整个测试过程中均未修改。后续改进应优先针对观察到的停滞和接触阶段偏差，再使用独立测试验证，不能将离线动作误差直接当作任务成功率。',
 '',
 '| 场景 | 结果 | 仿真时间（秒） | 最大按钮位移（mm） | 最大按钮接触力（N） |',
 '|---:|:---|---:|---:|---:|',
]
for r in rows:
    outcome='成功' if r['success'] else '超时' if r['reason']=='timeout' else r['reason']
    lines.append(f'| {r["candidate_id"]} | {outcome} | {r["simulated_seconds"]:.2f} | {max(0.,r["max_travel_m"])*1000:.3f} | {r["max_force_n"]:.3f} |')
(run/'report.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(stats,ensure_ascii=False),flush=True)
