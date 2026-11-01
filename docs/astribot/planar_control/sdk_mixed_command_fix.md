# S1B 混合指令越界故障定位与修复（2026-10-30）

> 本文记录 mixed 接口问题的定位与当时的适配方案。后续对比发现它与原键盘运动路径并不等价；当前 Agent 已改回公开 Cartesian 接口，不再启用此私有适配。当前实现见[接入对比](keyboard_agent_comparison.md)。

## 确认的根因

现场表现为 SDK 连接成功后退出：`index 1 is out of bounds for axis 0 with size 1`。

使用 S1B 当前安装的原生 SDK、绕过机器人初始化并替换所有控制和发布对象后，复现了完全相同的异常。过程不申请控制权，不创建真实发布器，也不发送机器人指令。复现脚本为 `scripts/probe_s1b_mixed_command.py`，结果及 SDK 二进制哈希见 [原生隔离验证记录](mixed_command_probe_20261030.json)。

实际调用链：

```text
AstribotFunction.set_different_type_command (astribot_function.py:833)
  → BaseFunction.cartesian_pose_control (util.py:574)
  → BaseFunction.execute_ik_and_set_position (util.py:528)
  → BaseFunction.calculate_twist_from_poses (util.py:644)
  → tf_transformations.quaternion_inverse / quaternion_conjugate
  → IndexError
```

该版本混合 API 没有公开 `remote_wbc_option` 参数。`use_wbc=False` 不等于禁止远程 WBC：当 WBC 服务存在时，内部 Cartesian 子调用仍可使用默认的 `remote_wbc_option=True`。

混合调用共享同一份目标字典。其中躯干是 4 维关节目标，但远程 WBC 分支把该条目当成 7 维 `[x,y,z,qx,qy,qz,qw]` 位姿。切出 `[3:]` 后只有 1 个值，四元数处理访问第 2 个元素时越界。对前一帧为 7 维、后一帧为 4 维的组合，还会出现 4 与 1 维点积不匹配。

之前仅将躯干从 Cartesian 改为 joints，没有处理这条隐式 WBC 路由，是不完整的修复。更早一次 `torso_drift` 没有保存完整现场数据，不能据此证明其所有物理成因；本次修复明确消除了与预期本地控制不一致的路由。

## 修复方式

在当前 Agent 的 SDK 实例上安装兼容适配，服务退出恢复原方法，不修改机器人 SDK 文件、不停止 WBC 服务：

- 保留每周期一次混合命令，右臂 Cartesian、躯干四关节保持、夹爪闭合量 100。
- 内部 Cartesian 子调用只接收 Cartesian 名称对应的 7 维位姿，不再夹带躯干关节数组和夹爪数值。
- 显式指定 `remote_wbc_option=False` 与 `use_wbc=False`，走 SDK 已有本地 IK 路径。
- 使用前检查原生方法是否具有对应参数；不兼容时在启动发送循环前明确报错。
- 保留原有躯干偏移判定、控制权和停止行为。

适配必须访问 `AstribotInterface` 的私有原生对象，因为当前公开混合 API 未暴露该路由参数。该依赖集中在 `sdk_compat.py`；SDK 升级时应重新运行原生隔离验证。

## 验证与异常记录

隔离验证结果：旧路径复现原异常；修复路径连续三次远程 WBC 调用数为 0，本地 IK 路由调用数为 3；关节分支仍收到原躯干四关节目标和夹爪 100。脚本也验证了当前原生类实例支持该方法适配及恢复。

验证中的 IK 求解、关节接口及发布器为桩对象，所以它证明的是原生路由和异常修复，不代替真机跟踪、停止或接触按压验收。相关 CPU 回归共 86 项通过。

后台 servo 异常现在保留完整 traceback。即使 backend 构造失败、API 尚未就绪，也会停止并清理线程，并把完整启动异常写入机器人端：

- 会话尚未创建：`~/.local/share/pressb/agent_sessions/fault.json`
- 会话已经创建：`~/.local/share/pressb/agent_sessions/<session_id>/fault.json`

重新运行原服务命令会自动部署修复后的代码：

```bash
python scripts/s1b_agent_service.py --host S1B
```
