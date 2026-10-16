# VIB-820 随机站位验证结果

更新时间：2026-10-13 20:34:13 Asia/Shanghai。状态：正式 30/30 回合完整终结；无重抽样、无正式重试。名义冒烟和视频补录均不计正式分母。

批次：`20261013_seed20261013_n30_pm20cm_v1`；完整结果、全部失败证据及数据定义见 [完整批次报告](../../../outputs/vib820_random_base/20261013_seed20261013_n30_pm20cm_v1/report.md)、[manifest](../../../outputs/vib820_random_base/20261013_seed20261013_n30_pm20cm_v1/manifest.json)、[summary.csv](../../../outputs/vib820_random_base/20261013_seed20261013_n30_pm20cm_v1/summary.csv)、[summary.json](../../../outputs/vib820_random_base/20261013_seed20261013_n30_pm20cm_v1/summary.json)。

采用 NumPy 1.26.0 的 PCG64，seed 20261013；ΔX、ΔY 独立均匀 [-0.20,+0.20] m，名义 (-0.12,0.28)，Z=0.10200000149 m、yaw=0 固定。120 Hz，GPU1，600秒超时；关键求解器、控制速度、按钮物理与碰撞过滤保持冻结。

源码 commit `10885a4b29ce8b4479005efb09cedf76d1f7dc05`，dirty 源码/配置/资产快照、补丁及哈希在 sources/ 和 manifest.source_sha256。manifest SHA-256 `192768dee572652d236180172e1e49b3ccc6e8bbf332884667fd157b1f7e7000`。输出被 Git ignore；本轮未 commit/push。

| 指标 | 精确结果 | 未知 / 覆盖 |
| --- | --- | --- |
| 关键位姿 IK 可达率 | 30 / 30 = 100.00% | 未知 0；检查覆盖 30 / 30 = 100.00% |
| 尝试按压成功率 | 25 / 30 = 83.33% | 已运动 30；按压数据未知 0 |
| 全体样本按压比例 | 25 / 30 = 83.33% | 同一步行程≥1mm、力>0.02N |
| 完整回合成功率 | 25 / 30 = 83.33% | 按压/释放/返回/复位/独立审计全部通过 |
| 初始禁止重叠率 | 0 / 30 = 0.00% | 未知 0；覆盖 30 / 30 = 100.00% |
| 动态禁止碰撞率 | 5 / 30 = 16.67% | 未知 0；覆盖 30 / 30 = 100.00% |

主终态：`{"success": 25, "unexpected_contact": 5}`；禁止接触回调条目 5（不等于独立撞击次数）。所有 5 个碰撞回合均在按压开始前失败，仍计入尝试分母。没有关键位姿IK失败、初始碰撞、路径IK失败、崩溃或超时。

关键位姿可达不证明全路径安全。初始查询采用 PhysX 对全部36个机器人碰撞体逐一 overlap_shape（Mesh凸近似）；完整查询范围和允许对象对保存于逐回合 report.initial_overlap。动态阈值为机器人相关接触估计力 >0.1N，仅允许当前按钮与指定指尖、对向指尖及原资产相邻/机构过滤；没有新增环境白名单。所有已检测动态接触均为机器人自碰撞。

![正式30站位展示图](../../../outputs/vib820_random_base/20261013_seed20261013_n30_pm20cm_v1/station_results_display.png)

| 自碰撞对象 / 阶段 | 正式样本与 ΔXY |
| --- | --- |
| astribot_arm_right_link_7 ↔ astribot_torso_link_3 / move_to_work_pose | sample_002 (+0.0728,-0.1562) m |
| astribot_arm_right_link_3 ↔ astribot_arm_right_link_5 / move_to_work_pose | sample_003 (+0.1089,-0.0465) m；sample_016 (+0.1828,+0.0567) m |
| astribot_gripper_right_Link_L11 ↔ astribot_torso_link_2 / move_to_work_pose | sample_007 (+0.1276,-0.1609) m |
| astribot_arm_right_link_7 ↔ astribot_torso_link_4 / move_to_work_pose | sample_029 (+0.1730,-0.1223) m |

名义冒烟位于 `outputs/vib820_random_smoke_20261013_final`，3534物理步、19/19独立审计通过。正式回合的每份可审计轨迹均完成独立审计；碰撞中止的审计不通过是如实结果。

必要测试 25 passed，另有4个构造中断/恢复/独立重试场景通过；[独立批次审计](../../../outputs/vib820_random_base/20261013_seed20261013_n30_pm20cm_v1/independent_batch_audit.json) 18/18 检查通过，直接核对物理数组、身份/来源、分母和CSV。第20回合结果原件及校验和保存在 [checkpoints/20](../../../outputs/vib820_random_base/20261013_seed20261013_n30_pm20cm_v1/checkpoints/20/checkpoint.json)，当时16成功4自碰撞、10待执行。

- 正式成功 sample_001：[整体快照](../../../outputs/vib820_random_base/20261013_seed20261013_n30_pm20cm_v1/episodes/sample_001/attempts/0001/run/press_0_overview.png)。
- 正式碰撞 sample_002：[失败快照](../../../outputs/vib820_random_base/20261013_seed20261013_n30_pm20cm_v1/episodes/sample_002/attempts/0001/run/failure_overview.png)，其余失败均保留原始结果、日志及部分轨迹。
- 双视角补录：[成功视频](../../../outputs/vib820_random_base/20261013_seed20261013_n30_pm20cm_v1/supplemental_videos/sample_001/demo.mp4)、[碰撞视频](../../../outputs/vib820_random_base/20261013_seed20261013_n30_pm20cm_v1/supplemental_videos/sample_002/demo.mp4)。这是正式结束后的2个独立补录，物理证据与审计见补录目录，全部排除正式30个样本统计。

结论：未满足扩大专家采集建议门槛（尝试成功率不足90%，且存在已知自碰撞）。先改进去工作起始位姿的关节插值路径，考虑安全中间位姿或碰撞感知规划，再保持原物理条件新建批次复验。成功点仅为已观察到的离散证据，不能保证整个±20cm区域安全。本轮为脚本专家验证，不是VLA策略测试或完整训练数据集。

归档：`outputs/vib820_random_base/20261013_seed20261013_n30_pm20cm_v1.tar.gz` 及 SHA-256 文件；完整批次目录永久保留。scene.usda仍引用仓库资产，归档不是独立可移植场景包。
