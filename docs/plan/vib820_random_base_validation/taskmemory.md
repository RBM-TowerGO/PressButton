# 任务记忆

更新时间：2026-10-13（Asia/Shanghai）。当前状态：正式30/30回合已完整执行，完整成功25，动态自碰撞5；独立批次审计通过，结果见 results.md。

## 用户已确认的要求

- 在 `docs/plan` 下新建目录，包含计划、`taskmemory.md`、`assertlist` 等，下一步先做 20–50 个站位验证回合。
- **最新随机范围：底盘前后、左右各 ±20 cm**。此前 ±5 cm 建议已被用户覆盖，不能恢复旧范围。
- 统计可达率、按压成功率和碰撞情况，目的是为后续专家数据采集做验证。
- 用户随后明确授权调用一个 GPT-6.1-sol 按本计划实现、运行并记录结果；本轮无需 commit/push，不进行训练。

默认 30 回合、种子 `20261013`、固定 yaw 和建议质量门槛属于本计划的实施选择，不是用户测量值或已完成事实。

## 仓库与工作区

- 仓库：`/home/js/robot/VLA/PressB`。
- 文档建立前 HEAD：`10885a4b29ce8b4479005efb09cedf76d1f7dc05`，`Add measured VIB-820 elevator scene and S1 button pressing`。
- 文档建立前已有未提交修改：`docs/vib820.md`、`scripts/run_astribot_press.py`。与此前录制演示和 GPU 参数相关；续做时先看 diff，不覆盖、不回退，也不自动提交。
- 当前计划文档会新增到本目录。`outputs/`、`.cache/` 和生成纹理等受 Git ignore 管理。

## 已验证基线

最近记录位于 [outputs/vib820_press_v5_video](../../../outputs/vib820_press_v5_video)，创建计划时已读取其报告核对：

| 项目 | 历史固定站位结果 |
| --- | --- |
| 正式按钮动作 | 1/1 成功，站位 X=-0.12、Y=0.28 m |
| 独立审计 | `independent_audit.json`，19 项通过 |
| 禁止接触 | 现有监测规则下 0 条 |
| 物理采样 | 3534 步，120 Hz，29.45 s |
| 最大按钮位移 | 约 1.060 mm |
| 最大接触力估计 | 约 0.315 N |
| 返回关节误差 | 约 0.00397 rad |
| 底盘漂移 | 报告值 0 m |
| 演示 | `demo.mp4`，30 fps，约 29.47 s，左右为整体/局部视角 |

这些数据只证明已有固定站位案例成功，不能视为随机范围验证结果。较早输出目录不得与此次批次混算。

## 场景与实现入口

| 文件 | 作用 |
| --- | --- |
| [configs/astribot_vib820.json](../../../configs/astribot_vib820.json) | 当前参数与尺寸来源 |
| [scripts/run_astribot_press.py](../../../scripts/run_astribot_press.py) | 机器人执行、IK 调用、接触反馈、轨迹与报告 |
| [scripts/audit_astribot_press.py](../../../scripts/audit_astribot_press.py) | 轨迹独立审计 |
| [src/pressb/astribot_kinematics.py](../../../src/pressb/astribot_kinematics.py) | 有限位的 IK/FK |
| [src/pressb/astribot_press.py](../../../src/pressb/astribot_press.py) | 原夹爪 TCP、闭合目标、机器人碰撞几何与过滤 |
| [src/pressb/vib820_scene.py](../../../src/pressb/vib820_scene.py) | VIB-820、按钮物理及电梯环境 |
| [src/pressb/vib820_materials.py](../../../src/pressb/vib820_materials.py) | 材质与纹理 |
| [docs/vib820.md](../../vib820.md) | 建模来源、最新场景与录制说明 |

面板 400×95 mm，黑窗 89×189 mm，黑窗下沿距地 1.28 m，按钮中心低于黑窗下沿 80 mm，即高 1.20 m；可动帽面直径 30 mm、外圈 37 mm、行程 2 mm。门净宽 1.30 m，门框纵深按墙面到关闭门面的 0.35 m 建模。部分外观、门高和按钮弹簧/力阈值仍是仿真假设，不能称为实测参数。

## 续做必须保留的实现事实

1. 当前是场景真值目标 + IK + 位置驱动的脚本专家，没有运行 VLA 策略，也没有视觉定位或独立力控制器。
2. `--repeats` 只重复同一站位。随机批次需新增调度/采样，不是把 repeats 设为 30。
3. 位姿换算目前是目标世界位置减 base 平移；机器人根变换与固定关节 `localPos0` 都会更新。yaw 随机化需要完整 SE(3)、固定锚点朝向及 FK 适配，不在本轮范围。
4. `home_tip` 是固定世界坐标目标，每个站位重新求 IK；不可照搬名义站位关节解。轨迹缓存不能跨站位沿用。
5. 当前启动前求 home、approach、touch IK；运行时笛卡尔轨迹约每 4 个物理步求一次 IK。已有 reachability 字段不是全路径验证。
6. 物理步进 120 Hz；有效触发要求同一步行程 ≥1 mm 且力 >0.02 N。灯光锁存是触发后的可视反馈。
7. 非预期接触报告有 >0.1 N 阈值及明确过滤规则；相邻连杆过滤与允许指尖接触需保留并审查，不能屏蔽环境碰撞。
8. 早期 IK 错误可能没有轨迹；进程崩溃可能没有最终报告。批次父进程必须兜底，不能静默丢弃失败样本。
9. 当前相机是世界固定 overview/panel，用于排查。未来机器人视觉专家数据需另外设计观测与动作同步。
10. 现有运行时底盘漂移中止阈值与独立审计阈值不同；完整通过以独立审计为准，不放宽现有检查以提高通过率。

## 环境与下一步入口

使用 `source scripts/activate_local.sh` 启用仓库 Python 3.11 / Isaac Sim 环境。GPU 0 曾有训练任务占用，GPU 1 曾用于录制，但每次运行前必须重新检查占用。指定 GPU 1 时沿用 `CUDA_VISIBLE_DEVICES=0,1` 与 `--gpu 1`，不要假定物理编号重映射为 0。

不要给 Isaac Sim 运行注入 `.cache/usd-inspect` 的 PYTHONPATH。输出必须是新目录。相关资产已在本地，当前任务无须重新搜索或下载材质。

原始续做顺序（现已完成）：

1. 阅读本目录计划、指标与验收清单，查看工作区 diff。
2. 实现持久化采样清单、逐回合调度和失败记录；做必要的采样/统计边界验证。
3. 重放名义站位基线，再执行预先声明的随机样本；同步更新本文件。
4. 生成统计和结果报告，填写验收证据。没有数据时保持“未执行”，不要填入估计成功率。

## 续做日志模板

```text
更新时间 / 执行者：
当前阶段：
代码 commit / dirty 源码快照：
批次 ID / 输出路径 / manifest 哈希：
计划数 / 已终结 / 待执行：
本次更改与验证证据：
已发现失败与待解决事项：
下一步命令或入口：
```

## 实施进行中日志（2026-10-13，历史记录）

- 已核对既有修改并保留；未找到适用 AGENTS.md。GPU 0 有其他任务，正式仿真使用 GPU 1。
- 新增 scripts/run_vib820_random_base.py、scripts/summarize_vib820_random_base.py、src/pressb/astribot_validation.py，扩展 runner 关键位姿/实际位姿/初始几何检查及失败快照。IK 求解参数、按钮物理和动态接触阈值未放宽。
- 测试记录：.cache/vib820_random_tests.log，25 passed。
- 冒烟：outputs/vib820_random_smoke_20261013_final；19/19 独立审计通过，3534 物理步，不计随机分母。
- 正式批次：outputs/vib820_random_base/20261013_seed20261013_n30_pm20cm_v1；seed 20261013，30 样本，600 秒/回合，串行新进程。manifest 在首回合之前保存；source 快照含 USD/URDF，正式 runner/物理实现冻结。
- 无回合的 outputs/vib820_random_base/20261013_seed20261013_n30_pm20cm 是开发预览 manifest，不是实验批次，无需计入。
- 恢复入口：source scripts/activate_local.sh；export CUDA_VISIBLE_DEVICES=0,1；python scripts/run_vib820_random_base.py --output outputs/vib820_random_base/20261013_seed20261013_n30_pm20cm_v1 --resume。正式主统计只读 attempts/0001；未完整终态的首尝试保留物理证据及进程状态未知，不覆盖重跑。

## 最终完成记录（2026-10-13 20:34:13 Asia/Shanghai）

- 正式批次 `20261013_seed20261013_n30_pm20cm_v1`，30/30 终结，0 待执行，0 正式重试；未重抽样。manifest SHA256 `192768dee572652d236180172e1e49b3ccc6e8bbf332884667fd157b1f7e7000`。
- IK：30 / 30 = 100.00%，覆盖 30 / 30 = 100.00%；尝试按压：25 / 30 = 83.33%；完整成功：25 / 30 = 83.33%。
- 初始重叠：0 / 30 = 0.00%；动态自碰撞：5 / 30 = 16.67%；初始/动态/IK 未知均0。所有碰撞发生于 move_to_work_pose，按压前失败样本详见 failure_groups.json。
- 原始轨迹、阶段快照、逐回合审计、日志、summary.csv/json、中期checkpoint及展示图保留在批次目录；docs/plan 下 results.md 为可版本记录报告。
- 额外2个独立补录视频在 supplemental_videos/，全部不计正式分子分母。
- 验证：25个pytest测试通过；4个构造恢复/重试场景通过；独立批次审计 18/18 通过。源码/配置/资产原始23项快照哈希仍匹配，后处理工具与验证日志另存哈希。
- 不进入扩大采集：关键位姿可达但部分关节插值路径自碰撞。下一步研究安全中间位姿/碰撞感知规划，维持物理条件另开批次；不扩大碰撞白名单。
- 无commit/push，原docs/vib820.md和--gpu录制修改保留。归档包路径与SHA文件见 results.md；保留仓库资产以解析导出场景引用。
