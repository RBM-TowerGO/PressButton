# S1 右臂闭合夹爪按键

使用原装右夹爪，闭合后按压现有 24–35 层物理按钮。底盘在启动前选定位置，
运行中由固定关节保持；躯干、头部和左臂保持零位。按钮位置取自已知场景，
本入口不涉及视觉识别、VLA/RL 或真机 SDK。

## 本机验收（2026-10-13）

正式重复验收为 `outputs/astribot_press_acceptance_v6/`：固定场景下 12 个按钮各
10 次，**120/120 成功，零误按、零异常接触**。进程退出码 0，重置恢复通过。
独立审计从 231,527 个物理采样点重新计算，15 项检查全部通过。

- 底盘最大平移漂移：0 m；旋转保持检查通过。
- 躯干、头部和左臂最大关节偏差：0.001476 rad（约 0.085°）。
- 目标按钮最大行程：1.556–1.624 mm；各次按压峰值接触力：0.461–0.482 N。
- URDF 与 PhysX 指尖位置最大差：0.450 mm；重置后最大关节目标误差：0.00309 rad。
- [验收报告](../outputs/astribot_press_acceptance_v6/report.json)、
  [独立审计](../outputs/astribot_press_acceptance_v6/independent_audit.json)、
  [物理轨迹](../outputs/astribot_press_acceptance_v6/trajectory.npz)。验收对应的六份
  源码/配置按报告 SHA-256 归档在同目录 `sources/`。

这是同一站位、面板位置和控制参数的重复验证，不是随机面板位置的泛化测试。
早期 v1–v5 目录包含调试、中断和纹理读取崩溃记录，不作为正式验收结果。

最终录像为独立的 12 次演示：`outputs/astribot_press_demo_v7/`，12/12 成功，
重置及 15 项独立审计通过；24,422 个物理步对应每路 2,036 帧、10 fps、
203.6 秒视频，按 1× 仿真时间播放，包含初始闭合和重置。场景重新加载检查
确认 33 个机器人刚体全部有碰撞覆盖、12 个按钮关节及所有非运行时资产引用完整。

- [全景视频](../outputs/astribot_press_demo_v7/overview.mp4)
- [按钮近景视频](../outputs/astribot_press_demo_v7/panel.mp4)
- [演示物理审计](../outputs/astribot_press_demo_v7/independent_audit.json)、
  [视频帧数审计](../outputs/astribot_press_demo_v7/video_audit.json)、
  [场景检查](../outputs/astribot_press_demo_v7/scene_audit.json)

## 运行

```bash
cd /home/js/robot/VLA/PressB
source scripts/activate_local.sh

# 单按钮调试；输出目录必须不存在。
python scripts/run_astribot_press.py --headless --video --floors 32 \
  --output outputs/astribot_press_single

# 12 个按钮各一次，输出全景和面板近景视频。
python scripts/run_astribot_press.py --headless --video \
  --output outputs/astribot_press_demo

# 固定场景重复验收，不录制视频以减少渲染开销。
python scripts/run_astribot_press.py --headless --repeats 10 \
  --output outputs/astribot_press_acceptance
python scripts/audit_astribot_press.py outputs/astribot_press_acceptance
```

默认配置为 `configs/astribot_press.json`。底盘参考点 XY 为 `[-0.12, 0.28] m`，
比原 S1 预览站位向面板前移 0.13 m、侧移 0.28 m，朝向不变。Z 由资产离地适配
确定，当前为约 0.102 m。右臂在这个站位求解所有按钮的接近点和按压点；
不可达时启动失败，不通过移动底盘完成任务。

不带 `--headless` 时打开仿真窗口。程序完成任务后关闭；这与保持窗口的静态
`preview_astribot_panel.py` 不同。每个输出目录保留配置和控制器源码 SHA-256。

## 控制与接触

- 复用官方夹爪联动符号：L1/L2/R11 为负角，R1/R2/L11 为正角。
  上游示例限制为 0.5 rad，在当前网格中仍有约 55 mm 开口；本任务按原始
  指尖网格和 URDF 限位求出约 1.00507 rad 的近闭合角，设计间隙 0.3 mm。
  PhysX 实测间隙允许 -0.1–1 mm（含数值接触容差），并验证全部联动关节跟踪。
- TCP 由原始指尖网格最前端推导，不新增按压杆。夹爪宽边沿面板水平方向，
  窄边沿竖直方向；TCP 和手腕之间的变换记录在 `report.json`。
- 使用 URDF 七关节有界 IK，以上一个解作为种子保持肘部姿态连续。
  关节转移采用五次平滑插值，接触阶段以 30 Hz 更新笛卡尔轨迹目标，
  120 Hz 物理步进。这里采用有界 IK 轨迹加位置驱动，不需要加载学习模型。
- 默认接近距离 65 mm，最大命令按入量 3.5 mm；目标按钮实测行程达到
  1.5 mm 且指定指尖接触力超过 0.02 N 时提前结束推进。撤回后需小于
  0.8 mm，归位关节误差小于 0.025 rad。灯光仅显示实测行程状态。
  沿用原面板的瞬时亮灯模式：按下时亮，回弹后灭，不模拟真实电梯的楼层请求锁存。
- 运行中监测关节限位、固定关节偏差、夹爪闭合保持、URDF/PhysX TCP 一致性
  及异常接触。碰到邻键、面板或非允许的机器人部位时终止并保存失败记录。

## 碰撞模型边界

只在此任务的场景层中，为上游缺失碰撞几何的链接添加原可视网格的凸包；
不修改下载的资产或原 S1 预览入口。为避免外壳与本身安装结构重叠，对新增
碰撞体沿合并固定链接后的运动学图过滤两跳以内的相邻链接，并过滤同一夹爪
连杆机构的内部接触。两个相对指尖仍保留碰撞，所有环境接触继续有效。

这些凸包是保守的仿真近似，不代表制造商碰撞模型或任意动作下的全身避碰保证。
新增碰撞体和精确过滤对均记录在报告，可独立检查。

## 产物与独立审计

`overview.mp4`、`panel.mp4` 为按仿真时间播放的视频；`press_<floor>_*.png`
保存首次成功瞬间。`trajectory.npz` 包含 120 Hz 实测关节、速度、指尖位置、
按钮行程、指尖接触力和底盘位姿，以及独立的命令目标、阶段、任务和重置 epoch。
`scene.usda` 为带相对资产引用的场景。`report.json` 保存逐次结果和失败原因。

物理读数使用 PhysX 视图；仅在录制画面或导出场景时同步显示变换，以免每个
物理步重建渲染场景。相机预热调用不推进物理时间。共享标签 PNG 内容相同时
不重写，变化时原子替换，避免渲染器读取到半写入文件。建议串行运行演示和验收。

独立审计重新读取数组，核对按下时行程与接触力同时成立、没有误按、按钮回弹、
右臂归位、夹爪保持、其他关节静止及底盘平移和旋转。最后一次重置验证恢复到
闭合夹爪的工作姿态。失败运行仍保留诊断产物，不应作为验收通过记录。
