# 双臂预备姿态随机站位结果

更新：2026-10-14 12:42:04 Asia/Shanghai。正式50/50终结，重试0；标定回合不计入。

范围：原中心(-0.12,0.28)m，ΔX=[-0.30,+0.05]m，ΔY=[-0.30,+0.30]m；PCG64 seed20261014；初始双臂已抬起并要求返回预备姿态。

- 启动成功：34 / 50 = 68.00%，95% Wilson 54.19%–79.24%。
- 完整成功：34 / 50 = 68.00%，95% Wilson 54.19%–79.24%。
- 按压成功：全体 34 / 50 = 68.00%；已运动 34 / 36 = 94.44%。
- IK可达：42 / 50 = 84.00%。
- 初始重叠：6 / 50 = 12.00%；动态碰撞：2 / 36 = 5.56%，动态未知14。

终态：`{"success": 34, "target_ik_failed": 8, "initial_collision": 6, "unexpected_contact": 2}`。并行选择4路，正式调度耗时21.62分钟。48项测试通过，独立批次审计passed，成功回合23项物理检查全部通过。

[完整报告](../../../outputs/vib820_ready_parallel/20261014_seed20261014_n50_ready_dx30_5_dy30_v1/report.md) · [精确成功坐标](../../../outputs/vib820_ready_parallel/20261014_seed20261014_n50_ready_dx30_5_dy30_v1/successful_stations.csv) · [全部站位](../../../outputs/vib820_ready_parallel/20261014_seed20261014_n50_ready_dx30_5_dy30_v1/summary.csv) · [失败明细](../../../outputs/vib820_ready_parallel/20261014_seed20261014_n50_ready_dx30_5_dy30_v1/failure_groups.json) · [独立核验](../../../outputs/vib820_ready_parallel/20261014_seed20261014_n50_ready_dx30_5_dy30_v1/independent_batch_audit.json)

![站位分布](../../../outputs/vib820_ready_parallel/20261014_seed20261014_n50_ready_dx30_5_dy30_v1/station_results.png)

离散成功点不能证明连续区域安全。旧零位30回合保持不变，未混合统计，未commit/push。

## 失败案例视频

[观看双视角合集（约26秒）](../../../outputs/vib820_ready_parallel/20261014_seed20261014_n50_ready_dx30_5_dy30_v1/supplemental_failure_videos_v1/failure_cases.mp4)。包含sample_016夹爪碰面板、sample_017夹爪碰墙面两次独立动态复现（半速，末尾失败定格），以及sample_002 IK失败、sample_010初始重叠的静态现场说明。

两次动态复现的碰撞步号、对象及接触力与原始记录一致；所有视频通过完整解码检查。补录与静态展示均排除正式50回合统计。原速视频、单案例片段和来源见[补录说明](../../../outputs/vib820_ready_parallel/20261014_seed20261014_n50_ready_dx30_5_dy30_v1/supplemental_failure_videos_v1/README.md)。
