# Astribot 数据字段与时序

原始 `trajectory.npz` 为 120 Hz：`q` 实测角、`qd` 实测角速度、`target` 控制目标（均 32 维）；`tip_pose` 齐次末端位姿、`tip` 位置、`base`/`base_quat` 底盘位姿、`travel`/`force`/`light` 按钮物理反馈、`phase`、`epoch`、`simulation_time`。关节角 rad，速度 rad/s，距离 m，力 N。

`camera_timing.json` 每四步一条，包含原生渲染时间戳，`step` 对应原始数组索引。`calibration.json` 保存分辨率、内参、挂载变换、初始世界位姿、父链接和场景平移。固定相机必须保留旧接口最终的方形像素光圈校正；不能直接使用场景构建器的默认 16:9 光圈代替。

`training.npz`：`state`/`action`/`velocity` 为 30 Hz、32 维；`physical_steps` 保存原始索引，`trigger_step` 保存首次物理触发，`terminal_video_frame` 保存原始视频精确截断位置。触发后保持原控制目标至首个同步采样帧（最多额外三物理步），然后才撤回。非末帧 action 为下一采样目标，末帧重复前一 action；训练末帧必须仍在 press 阶段且目标等于物理触发时的目标。

LeRobot `observation.state`、`action` 和 `observation.velocity` 使用下列冻结名称顺序；所有固定关节也保留。七路 `observation.images.<name>` 均为 RGB。任务固定 `Press the down hall-call button.`。

| 索引 | 关节名称 |
|---|---|
| 0 | `astribot_torso_joint_1` |
| 1 | `astribot_torso_joint_2` |
| 2 | `astribot_torso_joint_3` |
| 3 | `astribot_torso_joint_4` |
| 4 | `astribot_arm_left_joint_1` |
| 5 | `astribot_arm_right_joint_1` |
| 6 | `astribot_head_joint_1` |
| 7 | `astribot_arm_left_joint_2` |
| 8 | `astribot_arm_right_joint_2` |
| 9 | `astribot_head_joint_2` |
| 10 | `astribot_arm_left_joint_3` |
| 11 | `astribot_arm_right_joint_3` |
| 12 | `astribot_arm_left_joint_4` |
| 13 | `astribot_arm_right_joint_4` |
| 14 | `astribot_arm_left_joint_5` |
| 15 | `astribot_arm_right_joint_5` |
| 16 | `astribot_arm_left_joint_6` |
| 17 | `astribot_arm_right_joint_6` |
| 18 | `astribot_arm_left_joint_7` |
| 19 | `astribot_arm_right_joint_7` |
| 20 | `astribot_gripper_left_joint_L1` |
| 21 | `astribot_gripper_left_joint_L2` |
| 22 | `astribot_gripper_left_joint_R1` |
| 23 | `astribot_gripper_left_joint_R2` |
| 24 | `astribot_gripper_right_joint_L1` |
| 25 | `astribot_gripper_right_joint_L2` |
| 26 | `astribot_gripper_right_joint_R1` |
| 27 | `astribot_gripper_right_joint_R2` |
| 28 | `astribot_gripper_left_joint_L11` |
| 29 | `astribot_gripper_left_joint_R11` |
| 30 | `astribot_gripper_right_joint_L11` |
| 31 | `astribot_gripper_right_joint_R11` |

| 相机 | 宽×高 |
|---|---|
| head | 1280×720 |
| overview | 960×600 |
| wrist_left | 640×480 |
| wrist_right | 640×480 |
| panel | 640×480 |
| front | 640×480 |
| lobby | 640×480 |

仅有物理成功不等于可提交：完整七路解码、时间戳、首个成功采样帧的面板亮灯、数据形状及末帧动作均通过后才原子提交。训练数据经过成功筛选，失败与额外尝试另存，不能把筛选后的 400 条用作总体成功率的分母。
