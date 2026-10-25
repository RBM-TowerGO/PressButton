# S1B 真机—Joint8 模型接口核查（2026-10-25）

结论：右臂名称与 SDK 部件顺序已核对，SDK 原始右臂 7＋夹爪可以提取；**目前不能宣称得到可直接送入当前模型的完整 state8/action8**。夹爪量纲不一致、真机录制指令时间语义与模型标签不同，仍需转换与重新对齐。

检查通过 SSH S1B（hostname `orin`）读取 `/home/astribot/astribot_sdk_aarch64`，实际路径为 `/home/astribot/Downloads/astribot_sdk_aarch64`，SDK commit `2644cdfba47f3dd311d923b172ac6c24909f6fda`。仅查看文件、订阅 ROS；未创建 Astribot 控制客户端、未下发运动。客户端初始化会申请控制权，因此没有用初始化客户端的方式做只读检查。

模型范围为本仓库 `models/astribot_head_right_joint8_ft_20261018/action_schema.json` 和同目录 data_contract/runtime；未证明 S1B 正在运行某个特定权重。

## 25 维名称与顺序

SDK `astribot_base.py` 的 `whole_body_names` 和 S1 `whole_body_dofs=[3,4,7,1,7,1,2]` 给出下表。录制 MCAP 的各部件 `RobotJointController.name` 与部件内名称逐一核对通过；本地 HDF5 的 25 维数值布局、独立夹爪字段与该拼接一致。HDF5 本身没有 names/单位属性，不能仅凭 shape 自动认定任意新文件也采用此顺序。现有归档 MCAP 与本地保留 HDF5 未找到时间重叠，未完成同帧逐值回溯，因此 HDF5 导出器拼接顺序的源码/同帧证明仍是待补证项。

| 0 基列号 | 部件 | 部件内名称（按顺序） | 单位 |
|---|---|---|---|
| 0,1,2 | astribot_chassis | astribot_chassis_x, astribot_chassis_y, astribot_chassis_z_rot | m,m,rad |
| 3,4,5,6 | astribot_torso | astribot_torso_joint_1, _2, _3, _4（均为相同完整前缀） | rad |
| 7,8,9,10,11,12,13 | astribot_arm_left | astribot_arm_left_joint_1 … _7 | rad |
| 14 | astribot_gripper_left | astribot_gripper_joint | 0–100 闭合刻度 |
| 15,16,17,18,19,20,21 | astribot_arm_right | astribot_arm_right_joint_1 … _7 | rad |
| 22 | astribot_gripper_right | astribot_gripper_joint | 0–100 闭合刻度 |
| 23,24 | astribot_head | astribot_head_joint_1, astribot_head_joint_2 | rad |

完整 25 项名字在 `audit.json/qualified_names25`。左右夹爪原始名字相同，必须用部件/话题限定，不能建一个丢失命名空间的 `name→value` 字典。

实时 `/astribot_whole_body/joint_states` 本次实测只有 **20** 个 position，`name=[]`，并非上述 25 维拼接。参见 `live_whole_body.txt`。部件状态消息也可 `name=[]`；应使用经过核验的 topic→部件关节名配置绑定并校验长度，不能退回仿真下标。

## 提取与动作含义

模型顺序为 `astribot_arm_right_joint_1 … joint_7, gripper_angle`。前七维是关节绝对位置（rad）；夹爪是六个仿真指关节按 `[-1,-1,+1,+1,+1,-1]` 取带符号均值的角度（rad）。仿真的 arm_indices `[5,8,11,13,15,17,19]` 和 gripper_indices `[24,25,26,27,30,31]` 仅对应仿真 32 维布局，禁止用于真机 25 维。

`scripts/audit_s1b_contract.py` 用经核对的部件限定名称构建映射提取，不直接套用仿真索引。`sample_sdk_joint8.npz` 含首个排序 episode 的 `state_sdk8`、`command_sdk8`、名字、单位，最后一维仍是 SDK 刻度，**不是模型就绪数据**。模型应取 measured state；command 是监督动作来源，不可冒充观测状态。

SDK `open_effector()` 给目标 0，`close_effector()` 给目标 100；夹爪 YAML 上下限同为 0/100。因此该值不是米、毫米或弧度，语义是闭合程度刻度；不应假设与物理开口宽度或仿真角度线性。实际状态会略超界（本次实时右夹爪约 -0.160），这是反馈值，不改变接口端点定义。

SDK `set_joints_position(names, position, ...)` 传递绝对目标，不自行加到当前状态。MCAP 的 mode=1 指令包含 3×DOF 数值；右臂为 21 项，不能把整个 command 当作 7 个关节位置，应按该模式拆解位置段并核实模式。模型反归一化后的 `action8` 也是绝对目标，不是增量；残差策略另行叠加的 correction 不能与基础动作混淆。不要向真机发送仿真展开后的 `action32/fixed_action32`。

尚未找到 SDK 0–100 与六指关节角的可验证换算或标定曲线。不能猜测 `g/100`、乘固定弧度或直接使用 100。模型归一化也必须使用 checkpoint 的训练 mean/std，不能把 SDK 百分刻度送进其弧度归一化器。右臂关节名称相同也不能替代真机/仿真的机械零位、正方向标定；本次没有运动验证。

## 图像、状态与指令时间

模型契约为 30 Hz，动作 horizon=7：`image(t),state(t) → target(t+1/30), …, target(t+7/30)`。仿真导出源码 `src/pressb/astribot_collection/contract.py:110` 明确取下一采样点的 target；末帧重复前一个动作。并非当前帧指令直接作为 action，也不是下一帧 measured state。

模型图像输入顺序是 head、right wrist 的 RGB 图像；真机候选对应 `head_rgbd`、`right_wrist_rgbd`。真实还有 head_stereo，不能只凭“head”字样将其替代。SDK 解码输出 BGR，需要转 RGB；HDF5 head 的具体来源及相机内外参仍需导出配置佐证。SDK 图像回调保留原消息 header，但不自动将两路图像和关节同步。

对本地 elevator1 的 102 段、23,572 帧计算，各列相对 `time` 的差（毫秒）：

| 字段 | 最小 | 中位 | P95 | 最大 |
|---|---:|---:|---:|---:|
| head rgb_timestamp | 0 | 0 | 0 | 0 |
| state_timestamp | -3.975 | 0.004 | 1.809 | 3.096 |
| command_timestamp | -11.565 | 0.023 | 5.165 | 1793.821 |
| right rgb_timestamp | -30.532 | 1.568 | 14.946 | 24.941 |

采样间隔中位 33.152 ms，但范围 6.029–63.627 ms，不是严格固定 30 Hz。episode 0 第 0 行 command_timestamp 超前 head 1793.821 ms；该行不可直接训练为正常对齐样本。

原始 MCAP 中状态 header 是 epoch 时间（如 1792921211 秒），控制指令 header 却是每部件不同的较小秒值（如右臂约 809 秒）。**禁止直接跨这两类 header 最近邻匹配**。`audit.json/raw_first_messages` 同时记录 header、MCAP log_time、publish_time，保留核查依据；不能将 log_time 当作已证明的曝光时间。HDF5 command_timestamp 已是 epoch，具体重建规则未在已找到的 Python 包装层中定位，不能声称其已与采样/执行时刻等价。

重建训练样本需先证明统一时钟和命令时间语义，保留每部件来源时间，按 head 采样对齐 measured state、右腕图像，并按模型下一采样点目标契约生成标签；异常/过期帧应标记或排除。在线还需记录曝光、状态、推理起止、命令发送时刻，处理推理延迟，避免把旧观测产生的第一个动作无条件当成当前目标。本次无活跃相机 topic，未测在线曝光到执行延迟。

## 复核材料与未完成确认

- `audit.json`：完整命名、提取单位、时间统计、MCAP 原始消息证据；`model_ready=false`。
- `astribot_base.py`、`astribot_client.py`、`astribot_interface.py`：本次 S1B SDK 只读快照。
- `live_topics.txt`：右夹爪实时反馈和 SDK commit；右臂指令订阅超时，没有获得实时指令，指令名称证据来自录制 MCAP。
- `live_whole_body.txt`：20 维实时反馈证据。
- 复跑：`.venv-data-review/bin/python scripts/audit_s1b_contract.py`。

尚需夹爪双向标定、HDF5 导出顺序/时间重建源码或同帧原始记录、head 实际来源及在线同步延迟证据，才能完成可执行真机适配。此次检查确认了不兼容点，并未以猜测转换掩盖缺项。
