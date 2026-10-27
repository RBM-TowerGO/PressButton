# astribot_image_service 对比与日志队列修复

对照本机 `/home/js/robot/VLA/astribot_image_service/server.py` 和 README，分析运行 `live_1793007590172285931`。

| 项目 | 参考服务 | 本项目及此次处理 |
| --- | --- | --- |
| 高频图像 | `ZmqPublisherThread` 每路队列容量 1，覆盖旧帧，后台发送 | ROS 回调解码 RGB，用有界历史缓冲匹配图像与关节时间；暂未改为 ZMQ |
| 图像传输 | 可直接转发压缩图像字节，客户端解码 | 两幅 RGB 随每次 HTTP 请求传输，NPZ 约 3.46 MB；传输成本仍有优化空间 |
| 日志 | 回调更新计数，`log_status()` 周期输出；控制线程不逐条记录所有部件回读 | 原来逐条排队记录状态/回读，超过后台写入能力；现在同类同来源诊断只保留最新值，约 10 Hz 写入及累计计数 |
| 控制线程 | legacy `_command_loop` 在机器人本地按 250 Hz 重复发送最新目标，使用 filter | 独立 250 Hz SDK/filter 线程，模型响应只更新预测队列；目标过期可同拍重选一次 |
| 预测队列空缺 | legacy 播放器没有新动作时不更新 latest target，控制线程继续发送最后目标 | 原观测时间加 7 个 30 Hz 偏移形成约 233 ms horizon；耗尽停止，因此此前 `queue_exhausted` 与参考服务 legacy 行为不同 |
| 动作接口 | 25D 全身目标，兼容 16D 双臂夹爪 | 训练契约固定 10D 右臂、右夹爪、头部，不能直接替换成参考服务的动作接口 |

参考服务还实现 safe/g1 分支，不能用 legacy 行为概括全部模式。这里借鉴的是最新值缓存、独立发送与周期诊断的线程组织，不直接移植动作维度或改变预测时间语义。

## 最新故障的证据

本次 `sent_commands=0`、`sdk_imported=false`、`execution_enabled=false`，故障发生于 SDK 获取控制权之前。机器人保存的 events 中约 1.60 秒内有 2797 条 state、2791 条 accepted_command、74 条 driver_status；sources 还有图像及三个关节状态流。原版本将这些诊断全部放进容量 4096 的同一 FIFO，触发 `log_queue_full`。停止处理再次向故障 logger 写入，导致最终 traceback 被次级异常覆盖。

## 实施的修改

- `AsyncLog` 对正常 sources、state、command_topic、accepted_command、正常 driver_status 使用各来源最新值缓存；每条采样记录附 `telemetry_received_count`。实时 `RobotState` 状态更新及控制检查仍处理收到的每条数据。
- 错误来源记录、驱动错误、控制候选、请求结果、控制 tick、停止事件仍走关键事件队列。真实磁盘错误和关键队列满仍报告失败，不静默丢弃。
- 文件使用 64 KiB 缓冲，后台约 100 ms 刷新，正常关闭排空；摘要 `log_writer` 给出接收、合并、待写数量及错误。采样日志不再是完整逐帧记录。
- SDK 停止先执行，停止日志异常保存到 `stop_log_error`；主异常处理中的停止异常也不覆盖原始异常。

## 验证与限制

140 项本地部署测试通过，包括阻塞磁盘下 30000 条诊断合并、关键事件保留、日志失败时停止及资源释放。

在 S1B 上用本次 events 文件重放 20 次，加入模拟 servo_tick 日志，不加载 SDK、不控制硬件：113340 条事件约 0.358 秒处理完成；113240 条诊断中合并 113180 条，最终最新值槽与关键队列均为空，error 为 null，写入 1098191 字节。此回放验证的是机器人主机上的日志吞吐，不是 ROS+SDK+推理的完整实时验证。

此前 `queue_exhausted`、SDK 频率警告和停止稳定性核验仍不能据此宣布解决。此次未改变预测期限，也未进行真机运动。
