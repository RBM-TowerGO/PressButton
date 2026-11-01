# S1B 接口参考

返回[使用指南](README.md)。本文描述当前代码接口；网页服务和 Agent 控制服务是两个独立进程，不共享测距点配置或运动状态。

新增 `/v2` 运动、观测保存与 Thor 调用已实现，启动方式和接口见 [人工确认闭环](agent_run.md)。本页保留原标定 API 与网页接口参考。

## 服务边界

| 服务 | 默认地址 | 能力 |
|---|---|---|
| 相机网页 | `http://127.0.0.1:8765` | 图像、选定点深度、选点与保存；不控制机器人 |
| 标定控制 API | `http://127.0.0.1:6210` | 观察、二维微移、按压、状态、锁定停止 |

两者仅监听机器人 loopback。通过 SSH 隧道访问，直接键盘启动器自动建立网页隧道。控制 API 需要单独启动，部署条件见[标定控制服务](calibrated_control.md)。

## 相机网页 HTTP 接口

| 方法与路径 | 请求体 | 成功返回 |
|---|---|---|
| GET `/` | 无 | HTML 网页 |
| GET `/health` | 无 | `{"service":"s1b-camera-view-v1","pid":12345}`，PID 为实际子进程号 |
| GET `/api/snapshot` | 无 | 当前画面、深度及选择状态 |
| GET `/api/capture` | 无 | 同步快照：双路原始 JPEG、原始深度的 data_base64、参数与选点版本；不可用返回 503 |
| POST `/api/selection` | `{"point_normalized":[0.4,0.6]}` | 更新后的选择状态 |
| POST `/api/selection` | `{"point_normalized":null}` | 恢复默认中心后的选择状态 |
| POST `/api/selection/save` | `{"revision":1}` | 保存后的选择状态 |

成功状态码为 200。POST 需要 `Content-Type: application/json`，请求体为 1–1024 字节，不接受 Transfer-Encoding 或额外字段。仅接受本地主机 Host；有 Origin 时必须与 Host 同源。校验错误为 400，来源拒绝为 403，保存 I/O 错误为 500，返回 `{"error":"说明"}`。未知 GET 路径为 404。响应禁止缓存。

### 选择状态与保存

选点和保存返回相同结构，示例：

```json
{
  "point_normalized": [0.4, 0.6],
  "revision": 1,
  "saved": false,
  "config_path": "/home/astribot/.config/pressb/right_wrist_depth_point.json",
  "error": null
}
```

- `point_normalized`：右腕彩色图归一化位置，左上角 `[0,0]`、右下角 `[1,1]`。每个分量必须是有限数且在 0–1 内；`null` 表示默认中心。
- 像素换算：`x=floor(nx*(width-1)+0.5)`，y 同理。默认中心为 `[width//2,height//2]`。
- `revision`：当前进程内的选择版本，每次选点递增，不写入配置文件。
- `saved`：当前选择是否等于已保存的位置。
- `error`：配置加载错误或 null。损坏配置不会阻止页面启动，回退中心并报告错误。

保存必须提交最新选择响应中的 revision，版本不符返回 400，避免保存已被其他操作改动的位置。选点本身不写磁盘。多个网页连接共享同一个测距点。

以下例子仅改变图像测距点，不驱动机械臂：

```python
import requests

base = 'http://127.0.0.1:8765'
r = requests.post(base + '/api/selection',
                  json={'point_normalized': [0.4, 0.6]}, timeout=2)
r.raise_for_status()
selection = r.json()
r = requests.post(base + '/api/selection/save',
                  json={'revision': selection['revision']}, timeout=2)
r.raise_for_status()
print(r.json())
```

示例客户端需要 `requests`；服务端不依赖它。

### 快照字段

`GET /api/snapshot` 返回对象：

| 字段 | 内容 |
|---|---|
| `frames` | 已收到的 `head` / `right` 画面；尚未收到时对应键不存在 |
| `frames.<camera>.jpeg` | JPEG 的 base64 字符串，不带 data URL 前缀 |
| `frames.<camera>.source_ns` | ROS 图像来源时间戳，ns |
| `frames.<camera>.age_ms` | 自本地接收起经过的时间，ms |
| `frames.<camera>.stale` | 是否超过 1 秒 |
| `depth` | 下面的深度结果；尚未可用时只包含部分字段 |
| `selection` | 当前选择状态 |
| `activation` | 相机服务启用状态说明 |
| `error` | 传感器处理错误或 null |

深度成功时的主要字段：

| 字段 | 类型 / 单位 | 含义 |
|---|---|---|
| `center_px` | `[x,y]` 像素 | **当前选定点**，保留历史字段名，并非始终在中心 |
| `center_m` | 数值或 null，m | 选定单像素的相机光轴 Z 深度 |
| `median_m` | 数值或 null，m | 周围区域有效深度的中值 |
| `valid_pixels` | 整数 | 区域内有效像素数量 |
| `window` | 整数，当前为 5 | 名义区域边长 |
| `window_pixels` | 整数 | 裁剪到图像边界后的实际区域像素数 |
| `scale_m` | m/count | 原始 uint16 深度尺度 |
| `serial` | 字符串 | 出厂参数读取的设备标识 |
| `status` | 字符串 | `ok` 或不可用原因 |
| `depth_source_ns` / `rgb_source_ns` | ns | 配对深度 / 彩色来源时间戳 |
| `skew_ms` | ms | 配对时间差 |

无效、过期或新选点等待计算时，距离为 null；调用者应允许其他字段缺失，不能把 null 当成 0。图像与深度接收时间使用单调时钟判断新鲜度，不应将来源时间戳与单调时钟直接相减。

### 测距点配置文件

默认文件：机器人端 `~/.config/pressb/right_wrist_depth_point.json`。

```json
{
  "schema": "s1b-view-point-v1",
  "camera": "right_wrist_rgbd",
  "point_normalized": [0.4, 0.6]
}
```

`point_normalized: null` 保存默认中心。位置以原始彩色图为基准，浏览器缩放不改变它。配置通过同目录临时文件、fsync 和原子替换写入；重启网页服务加载。它不包含 TCP、手眼外参或控制服务按压距离。

### ROS 与设备输入

| 数据 | 话题 / 来源 | 格式 |
|---|---|---|
| 头部彩色 | `/astribot_camera/head_rgbd/color_compress/compressed` | CompressedImage JPEG |
| 右腕彩色 | `/astribot_camera/right_wrist_rgbd/color_compress/compressed` | CompressedImage JPEG |
| 右腕深度 | `/astribot_camera/right_wrist_rgbd/depth_compress` | CompressedImage raw，little-endian uint16 |
| 启用相机 | `/astribot_camera_control` | SetBool true |
| 设备映射 | `/etc/config/camera_info.json` | 右腕 UVC 序列号 |
| 深度尺度与内外参 | `s1b_depth_metadata.cpp` / librealsense2 | 当前 640×360、30 FPS profile |

设备助手将 UVC 序列号与 SDK 设备通过 USB 物理路径匹配。当前网页对原始深度做彩色对齐；独立标定服务的深度输入契约不同，不能直接把这个 raw 压缩话题填入它的整流 Image 配置。

## Agent 控制 HTTP / Python 接口

Python 类：`pressb.planar_control.api.PlanarTools(base_url='http://127.0.0.1:6210', timeout=5)`。从仓库使用时需安装项目或设置 `PYTHONPATH=src`。

| Python 方法 | HTTP | 精确请求字段 | 成功码 |
|---|---|---|---|
| `observe()` | POST `/observe` | `{}` | 200 |
| `move_in_plane(du_mm, dv_mm, observation_id, command_id)` | POST `/move` | `du_mm`, `dv_mm`, `observation_id`, `command_id` | 202 |
| `press(observation_id, command_id)` | POST `/press` | `observation_id`, `command_id` | 202 |
| `get_state()` | GET `/state` | 无 | 200 |
| `stop()` | POST `/stop` | `{}` | 200 |

请求体使用 JSON 对象，不接受额外字段。POST 最大 8192 字节，不支持 Transfer-Encoding。参数与状态校验错误返回 400 `{"error":"reason"}`，未知路径 404；服务内部异常会请求故障停止并返回 500。Python 包装把 HTTP 错误转换为 `ValueError`，网络异常仍由底层网络库抛出。

### 移动、按压与重试

```json
{
  "du_mm": 2,
  "dv_mm": -1,
  "observation_id": "从 observe 取得的 ID",
  "command_id": "align-001"
}
```

`du_mm` 正方向为机器人右侧，`dv_mm` 正方向为上方，均相对最近完成位置。运动量使用 mm，内部几何与状态距离使用 m。Agent 不传关节角、法向距离、速度或按压力。

移动和按压成功接收时返回：

```json
{"task_id":"任务 UUID","command_id":"align-001","accepted":true}
```

202 只表示已接收，规划仍可能拒绝，必须轮询状态。动作串行；执行中拒绝其他新动作。相同 command_id 和完全相同参数在当前会话返回原 receipt，不会重复运动；同 ID 不同参数返回 `command_id_conflict`。幂等记录不跨进程重启持久化。

HTTP 超时后使用原 observation_id、command_id 和原参数重试，不能为同一次尝试生成新 ID。已受理动作完成或规划拒绝后需要重新观察。普通客户端断连不会取消已接收的有限动作；机器人反馈、控制权或周期异常会触发本地停止。

### 观察返回

| 字段 | 内容 |
|---|---|
| `observation_id` | 当前观测 ID，受新鲜度和动作代次限制 |
| `images.head` / `images.right` | `jpeg_base64`、`source_ns`、`receipt_ns` |
| `tcp` | TCP 的 4×4 齐次变换矩阵，平移单位 m |
| `range` | 测距结果，至少包含 `trusted`；不可用时带 `reason` |
| `state_source_ns` | 机器人状态来源时间戳 |
| `directions` | `du_positive: "robot right"`、`dv_positive: "up"`、`units: "mm"` |
| `state` | 同 `/state` 的状态对象 |

仅 READY 可取得观测，图像必须新鲜且晚于前次动作完成。方向提示是机器人方向，不是像素方向。网页的 `jpeg` 与这里的 `jpeg_base64` 字段名称不同。

### 状态与停止

| 字段 | 含义 |
|---|---|
| `phase` | READY / PLANNING / MOVING / PRESSING / SETTLING / FAULT / STOPPED |
| `fault` | 故障或停止原因，正常为 null |
| `task` | 当前 / 最近任务及目标，尚无任务时 null |
| `result` | 最近任务结果，尚无结果时 null |
| `completed_uv_m` | 最近完成的平面位置 `[u,v]`，m |
| `completed_normal_m` | 最近完成的法向位置，m |
| `measured_uvd_m` | 实测平面坐标 `[u,v,d]`，m |
| `tcp_error_m` | 实测 TCP 与当前目标的位置误差，m |
| `state_receipt_ns` | 状态本地接收单调时钟时间，ns |

`result.status` 为 `completed`、`rejected` 或 `interrupted`；失败时可有 `reason`。`action_completed` 表示动作是否完成。当前 `contact_evidence` 和 `button_trigger_evidence` 均为 null，不能用轨迹完成代替按钮触发证据。

`stop()` 请求本地停止并锁定当前会话，不是可继续的暂停。故障后不自动撤回；需要查明原因并重新启动会话。

常见拒绝原因：`controller_not_ready`、`stale_or_unknown_observation`、`move_too_large`、`command_id_conflict`、`press_calibration_required`。规划或执行中的拒绝 / 故障通过状态结果报告。

### Python 调用例子

以下例子会申请一次真实的 2 mm 平面移动，前提是已启动完成标定的控制服务：

```python
import time
import uuid
from pressb.planar_control.api import PlanarTools

tools = PlanarTools()
obs = tools.observe()
receipt = tools.move_in_plane(2, 0, obs['observation_id'], str(uuid.uuid4()))
while True:
    state = tools.get_state()
    if state['phase'] in ('READY', 'FAULT', 'STOPPED'):
        break
    time.sleep(0.1)
print(receipt, state['result'])
# 检查结果，再重新 observe；若图像尚未更新，等待新帧后重试观察。
# 接触按压另用 tools.press(new_obs['observation_id'], new_command_id)。
```

服务未提供 HTTP 的夹爪开合或固定 100 mm 前伸接口；这些属于直接键盘入口。控制器内部的 air_probe / advance / retract 也不是公开 HTTP 工具。
