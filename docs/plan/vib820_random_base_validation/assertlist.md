# 验收清单

本清单已按实际执行证据核验。正式30/30样本完成；数据与代码证据见下列路径及 results.md。构造故障测试与实际实验结果明确区分，历史固定站位不计随机分母。

## A. 范围与可复现性

- [x] A01 正式样本数预先声明为 20–50，默认 30；冒烟回合单独标记。
  证据：manifest.N_planned=30；名义冒烟独立目录，不在samples中。
- [x] A02 ΔX、ΔY 独立均匀采样于 [-0.20,+0.20] m；坐标边界为 X [-0.32,0.08]、Y [0.08,0.48] m。
  证据：manifest.samples及verification/pytest.log；PCG64原始二维uniform一次生成30点，未筛选。
- [x] A03 Z 与 yaw 固定；没有沿用已作废的 ±5 cm 范围。
  证据：独立批次审计config_identity_and_physical_evidence：所有config仅改XY；initial_poses记录同一Z和yaw=0。
- [x] A04 相同 seed/RNG 版本重建相同清单；实际 manifest 在第一回合前保存。
  证据：manifest_precedes_every_attempt检查通过；采样重建测试通过；manifest SHA192768dee572652d236180172e1e49b3ccc6e8bbf332884667fd157b1f7e7000。
- [x] A05 所有站位保留，不因 IK 失败、碰撞或崩溃重新抽样。
  证据：summary.csv含30唯一行，25success/5unexpected_contact，无正式重试；全部首尝试保留。
- [x] A06 批次有 commit、dirty 状态、源文件/配置快照或补丁及哈希。
  证据：manifest.commit/dirty_status/source_sha256与sources/working_tree.patch；原始23项快照含新增脚本、模块、USD/URDF。

## B. 基线与站位

- [x] B01 名义站位重放并通过现有独立审计，不计入正式随机分母。
  证据：outputs/vib820_random_smoke_20261013_final/independent_audit.json，19/19 passed、3534步。
- [x] B02 修改根变换时同步固定关节锚点，记录请求位置及实际初始变换。
  证据：每份report.initial_poses记录requested/actual_root_xyz、fixed_joint_world_anchor及base_link位姿；独立批次核对根/锚点。
- [x] B03 各回合 base link 漂移以本回合初始位姿为参考，坐标系含义明确。
  证据：runner以首个实际base读数为漂移参考；report与独立审计base_drift；batch审计确认初始base帧一致。
- [x] B04 每回合重新初始化关节、按钮、灯光、接触累计与轨迹缓存。
  证据：30次独立新进程；runner新建target/forces/light_states/cartesian_cache/records，并执行world.reset。
- [x] B05 初始禁止重叠检查已实现，查询范围、过滤表及无法查询的情况有记录。
  证据：每回合report.initial_overlap：36/36 collider查询，完整scene_colliders和filtered_body_pairs；方法为PhysX overlap_shape，错误保留unknown。

## C. 执行与证据

- [x] C01 每个站位重新求解关键位姿 IK，保留残差、限位结果与失败阶段。
  证据：30份report.key_pose_ik，各含home/approach_0/touch_0残差、q和限位；90个关键位姿检查通过。
- [x] C02 关键位姿通过与运行中轨迹 IK 失败分别记录，不宣称全路径无碰撞。
  证据：report.key_pose_ik与termination_reason分别记录；runner.cartesian的ValueError分支设置path_ik_failed，未声称连续路径安全。
- [x] C03 仍使用刚体按钮、原夹爪和物理接触触发，没有直接设置成功标志或按钮位移。
  证据：runner/场景源码快照及trajectory.travel/force；冻结按钮物理、原夹爪，未写入按钮位移或成功标志伪造触发。
- [x] C04 有效按压由同一物理步的行程及力共同确认，不只比较两个峰值。
  证据：独立逐回合audit与batch审计从同一步travel>=0.001且force>0.02复算，25确认触发。
- [x] C05 释放、返回、锁存、清灯及现有物理审计条件保留。
  证据：25份成功独立审计19项全部通过；行程上限、释放、返回、灯光及reset条件未放宽。
- [x] C06 禁止碰撞停止当前回合并记录对象、阶段、时间/步号与力估计。
  证据：5份碰撞result.collision_events均含actors/phase/step/simulation_time/force_n；记录后立即抛异常停止，failure_groups.json列明。
- [x] C07 未为通过测试屏蔽手臂与墙、门框、面板等环境碰撞。
  证据：astribot_press.py哈希冻结；报告列出原资产过滤表，新增initial checker未加入手臂/环境白名单。
- [x] C08 120 Hz 轨迹时间连续性可审计；失败时保存已有部分轨迹。
  证据：全部30份trajectory存在；batch逐轨迹检查120Hz时间连续性和physics_steps一致，5次碰撞保存部分轨迹。
- [x] C09 进程启动失败、提前 IK 失败、超时、审计缺失均有明确终态。
  证据：run_vib820_random_base.py启动/600s超时/退出兜底，classify对早期无轨迹写unavailable；构造崩溃/中断测试通过，本次实际无崩溃/超时。
- [x] C10 支持中断恢复，重试单独存放，主统计不重复计算或覆盖首次失败。
  证据：verification/resume_tests.log：3个中断保留场景+独立0002重试存放/不覆盖0001/首尝试汇总测试通过；实际正式重试0。

## D. 统计实现的必要检查

- [x] D01 用包含成功、不可达、初始碰撞、动态碰撞和崩溃的构造记录验证汇总。
  证据：tests/test_vib820_random_base.py中5类构造记录检验计数与分母；25项pytest通过。
- [x] D02 零分母得到 N/A，缺失数据保持 unknown/null，不被填成零碰撞。
  证据：test_no_denominator_does_not_become_zero_collision通过；JSON零分母fraction=null，unknown不转clear。
- [x] D03 CSV 每个计划样本一行；主终态计数之和等于 N_planned。
  证据：batch审计csv_has_every_unique_sample/primary_outcomes_sum_to_planned通过，CSV30行、主终态合计30。
- [x] D04 可达率、尝试按压成功率、全体按压比例、完整成功率均有分子/分母。
  证据：summary.metrics含完整numerator/denominator/fraction；results.md逐项列出。
- [x] D05 动态碰撞率和初始重叠率分别报告，并列出监测/检查覆盖率。
  证据：summary.metrics.initial_overlap/initial_coverage/dynamic_collision/dynamic_coverage分开；正式初始和动态覆盖均30/30。
- [x] D06 “按压后碰撞”仅可计入触发成功，不能计入完整成功。
  证据：test_denominators_and_press_then_collision构造触发后碰撞，只计press_confirmed、不计full_success。
- [x] D07 未执行/中断且无碰撞证据的回合不能计为 clear。
  证据：pending/崩溃构造测试通过；原interim_20记录10 pending且dynamic_unknown=10，无填零。
- [x] D08 失败回合、未知项和重试记录不会从汇总中静默消失。
  证据：30首尝试结果齐全；构造重试保留首失败，CSV仍1唯一行；缺失轨迹/audit原因显式保存。

## E. 实验交付

- [x] E01 完成预定 20–50 个唯一站位样本，或如实说明中断、已完成数与待执行数。
  证据：manifest30、N_terminal30、pending0，所有唯一站位完成。
- [x] E02 第 20 回合中期结果与最终结果可追溯到原始样本清单。
  证据：interim_20.json/csv；checkpoints/20含原件/哈希/第20终结时间；当时16成功4碰撞10pending。
- [x] E03 输出 summary.csv/json、XY 站位结果图、失败分类和批次报告。
  证据：summary.csv/json、station_results_display.png、failure_groups.json、report.md与docs/results.md均已生成。
- [x] E04 提供成功及实际出现失败类型的代表材料，记录无法录制的原因。
  证据：正式sample_001 press_0_overview.png、sample_002 failure_overview.png及5个失败现场；另2独立补录demo.mp4均完整解码通过，不进入正式分母。
- [x] E05 碰撞结论带有过滤规则和力阈值，不把“未检测到”写为绝对无碰撞。
  证据：report.md明确>0.1N、允许对象及凸几何查询范围；5次均自碰撞，未宣称绝对无接触。
- [x] E06 说明是否满足扩大采集的建议门槛；不达标时给出基于失败证据的下一步。
  证据：25/30=83.33%且有5已知碰撞，门槛不通过；先修move_to_work_pose安全路径，再独立新批次复验。
- [x] E07 更新 taskmemory.md，记录代码状态、输出位置、已完成事项与后续入口。
  证据：taskmemory.md已写最终状态、来源、批次/归档路径、指标和后续安全规划入口，无commit/push。

任务是否完成以文档和实验可追溯性为准；质量门槛是否达标是实验结论，不能通过删除失败或改判据获得。

证据根目录：`outputs/vib820_random_base/20261013_seed20261013_n30_pm20cm_v1/`。verification目录保存测试输出；每份episodes/<id>/attempts/0001/run/report.json与trajectory.npz为正式原始证据。构造记录没有混入30个正式样本。独立批次18/18核验通过；父代理另逐30回合独立复核，结论一致（会话证据，无额外文件路径）。
