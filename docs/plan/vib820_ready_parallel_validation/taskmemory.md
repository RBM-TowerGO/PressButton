# 任务记忆

2026-10-14 Asia/Shanghai：用户授权实现并执行完整计划，无需commit/push。

已实现：不对称XY范围、显式中心及配置、1–4路独立进程调度、GPU槽位、批次文件锁、首次结果保留、超时/中断进程组回收、恢复、资源监测、按实际第20个终态保存中期checkpoint。runner新增startup物理步号/误差，独立audit从轨迹复核，汇总增加启动指标和Wilson区间，绘图读取manifest边界。

已有工作区修改包含双臂姿态接入，全部保留；旧批次不改动。必要测试47项通过，日志 .cache/vib820_ready_validation/pytest.log。最初直接调用环境Python时加载ROS pytest插件失败，随后使用仓库激活脚本及PYTEST_DISABLE_PLUGIN_AUTOLOAD=1正确执行。

标定：outputs/vib820_ready_parallel/20261014_seed20261014_n50_ready_dx30_5_dy30_v1_benchmarks；warmup及w1/w2/w4均独立于正式样本。

正式预定：outputs/vib820_ready_parallel/20261014_seed20261014_n50_ready_dx30_5_dy30_v1；执行前冻结manifest与源文件快照。运行source scripts/activate_local.sh后export CUDA_VISIBLE_DEVICES=0,1。恢复命令使用run_vib820_random_base.py --output <正式目录> --resume，仅继续未开始样本；改变并发可加--workers并写入execution历史，不能修改物理源文件继续同批次。

执行前发现磁盘可用约4GiB，当前完整回合约46MiB，预计50回合加快照约2.4GiB；调度器记录磁盘余量，低于0.5GiB停止派发并保留证据，不删除旧数据。

最新：用户要求停止额外标定，直接估计配置执行。当前正式4路（每卡2进程）、每进程8CPU工作线程，无8线程单独性能测试。原4路标定被用户中止，结果不进入正式分母。正式启动脚本.cache/vib820_ready_validation/run_formal.py，日志formal.log；必要测试48项通过。

## 最终结果

2026-10-14 12:42:04 Asia/Shanghai：正式50/50终结，启动34，完整34，初始重叠6，动态碰撞2；并行4路。48项测试通过；批次独立核验passed。原始证据、成功坐标、失败明细及报告在上述正式目录。未commit/push。

2026-10-14T13:36:34.391496+08:00：按用户要求补录失败案例。supplemental_failure_videos_v1/failure_cases.mp4 为双视角合集；sample_016/017动态复现原碰撞步号、对象和力，sample_002/010为明确标注的原始静态说明。正式summary.json和summary.csv SHA256复核未变；未修改仿真代码。
