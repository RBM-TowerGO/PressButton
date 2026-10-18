# Astribot VLA 闭环评估（2026-10-18）

## 模型与部署

项目：`/home/js/robot/VLA/PressButton/PressB`。

最佳检查点位于 `models/astribot_head_right_joint8_ft_20261018/checkpoints/step_008000/model.pt`。
预期 SHA256：`db98b7c5488c666c3e10fa721dc29b25e069c41237bf8cb42b638611863eaaa8`。
这是完整模型权重；继续训练用的优化器状态不在本次推理部署中。

同一模型目录保存动作契约、训练契约、冻结数据划分、训练集归一化统计、处理器和 HPC 运行源码。
`config.hpc.yaml` 保留原配置；`config.yaml` 只重定位两个骨干模型的本地配置目录。
`deployment.json` 保存配置与源码指纹；`inference_environment.txt` 保存推理依赖版本。

推理使用 `.venv-astribot-inference`（基于仿真 Python 创建的独立依赖覆盖环境）。
Isaac Sim 继续使用原 `.conda/envs/pressb`，其依赖未修改。
模型在 GPU 1，仿真在 GPU 0。

## 预先确定的评估方案

- 先在验证集候选场景 1 做 2 秒接口检查，并检查相同输入及随机种子的重复预测。
- 正式运行冻结划分中的全部 40 个测试场景，不根据结果筛选场景。
- 每个场景最多 40 秒仿真时间。从原配置的举臂、张开夹爪姿态开始；不执行专家前导动作。
- 头部 RGB 1280×720、右腕 RGB 640×480，加当前实测 32 维关节状态及固定任务文本输入模型。
- 模型内部按原契约压缩状态、使用训练集统计归一化、缩放图像并输出 7×8 动作。
- 8 维动作是右臂 7 关节与夹爪 1 自由度的绝对角度。展开为 32 个关节，其他关节使用冻结常量。
- 每次执行全部 7 步动作，动作频率 30 Hz。相邻命令之间按 120 Hz 线性插值 4 个物理时刻，然后重新观察与预测。
- 图像与视频在每次策略观察时采样，约 4.286 Hz；视频使用真实物理时间戳，不插值或重复画面。
- 每次取图前显式同步 PhysX 到 Fabric，并独立比较头部、左右腕实际渲染相机位姿与物理刚体位姿。使用独立相机产品以逐路验证。
- 不裁剪预测，不加入专家 IK、目标位置控制、示范动作或示范图像。不合规的动作判失败并停止本次试验。
- 任一物理时刻同时满足按钮行程 ≥1 mm、右指尖接触力 >0.02 N，按原场景逻辑点亮锁存指示灯；没有非预期接触等失败条件才记成功。
- 场景坐标只用于初始化物理世界及评分，不作为模型输入或控制目标。
- 模型计算时物理时间暂停。本实验是同步仿真闭环，不代表真实机器人实时控制能力。记录推理延迟供后续判断。

## 运行

在项目根目录，先运行：

```bash
bash scripts/start_astribot_policy_server.sh
```

服务只监听本机 `127.0.0.1:8794`，启动时验证模型 SHA256 和各项契约并严格加载全部 1617 个张量。
另一个终端执行：

```bash
bash scripts/run_astribot_closedloop.sh smoke outputs/astribot_closedloop_20261018/smoke_final
bash scripts/run_astribot_closedloop.sh test outputs/astribot_closedloop_20261018/test40
```

输出目录必须不存在，以防覆盖既有实验。重复试验应使用新目录。

## 结果文件

正式结果目录的 `protocol.json` 记录模型身份和试验方案，`summary.json` 记录已完成的场景及成功率。
只有 `complete=true` 才表示全部预定场景完成。

每个 `candidate_XXXXXX/` 保存：

- `report.json`：终止原因、接触、按钮行程、灯状态、推理延迟、时长。
- `trajectory.npz`：每个物理时刻的实测状态、实际下发目标、接触力、按钮行程。
- `policy_requests.json`：每次实时观测状态、图像哈希、随机种子和完整预测动作。
- `camera_timing.json`、`calibration.json`：同步时间及已验证的相机投影。
- 头部、右腕、按钮、全景四路视频，以及初始与最终 PNG。

视频包含初始观察与最终快照；相同物理时间的最终快照不重复编码。精确物理时间同时保存在记录文件中。
接口错误单独记为 `infrastructure_error` 并中止批次，不算作模型成功或正常任务失败。

早期 `smoke` 和 `primary_chunk7` 使用的渲染缓存未同步，其结果已经 `INVALIDATED.json` 标记作废。
`smoke_v2` 是相机校验开发期间中止的接口试验；`smoke_v3` 验证了修正后的相机同步。
最终方案以 `smoke_final` 和 `test40` 为准。所有调试均未修改模型权重或训练集归一化统计。

## 完成结果

40/40 场景完成，17 次成功（42.5%）、23 次超时。超时中 16 次发生明显移动但未接触按钮，7 次停滞在初始姿态附近。无非预期接触。160 个视频全部逐帧解码通过，实时状态、动作插值、相机位姿及时间戳核验通过。

最终报告：`outputs/astribot_closedloop_20261018/test40/report.md`；机器可读汇总为同目录 `final_metrics.json` 和 `audit.json`。模型推理服务在测试结束后已停止，显卡资源已释放；再次使用时按上述说明启动。
