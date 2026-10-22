# Astribot S1 资产加载与适配

本入口在现有 Python 3.11 / Isaac Sim 5.0 环境中加载 Astribot S1，验证固定底座、
双臂关节控制、物理接触、重置和 RGB-D 渲染。原 PiPER 按钮任务入口保持独立。
本阶段不包含 Astribot 按钮轨迹、七关节 IK 或 VLA/RL 训练适配。
头部/双腕 RGB-D、雷达、IMU、接触和关节读数已另行接入，见 [传感器使用与验收](astribot_sensors.md)。
使用原装闭合右夹爪、固定底盘的独立按键控制入口见 [S1 右臂按键](astribot_press.md)。

## 本机验收（2026-10-13）

已验收输出：`outputs/astribot_asset_validation_v5/`，进程退出码 0。

- [模型预览](../../outputs/astribot_asset_validation_v5/preview.png)
- [重置后预览](../../outputs/astribot_asset_validation_v5/after_reset.png)
- [物理验证报告](../../outputs/astribot_asset_validation_v5/validation.json)
- [导出场景与轨迹独立检查](../../outputs/astribot_asset_validation_v5/independent_audit.json)

35 个源文件通过 SHA-256 校验。生成资产有 42 个网格、33 个刚体、32 个关节；
运行适配后有 17 个实际碰撞形状。以 120 Hz 完成 2,820 个物理步（23.5 秒），
左右臂全部 14 个关节均完成小幅运动，最大目标误差 0.000913 rad；
静止保持最大误差 0.000965 rad，固定底座无测得漂移，归位和重置通过。
落块与机器人底座产生了非零接触冲量，稳定高度约 0.122 m，未穿过底座落到地面。
RGB-D 分辨率为 960×720，导出场景重新打开后没有缺失机器人或地面资产。

`OmniPBR.mdl` 在独立 OpenUSD 检查中显示为未解析材质模块，这是 Isaac Sim
提供的运行时模块；实际 Isaac 渲染已验证。其余资产引用均可解析。
日志为 `logs/verify-astribot-v5.log`。较早目录为调试运行，正式验收以 v5 为准。

## 下载与准备

```bash
cd /home/js/robot/VLA/PressB
source scripts/activate_local.sh
python scripts/fetch_astribot_assets.py

# 使用本机已有的隔离 OpenUSD；无需更改 Isaac 环境依赖。
PYTHONPATH=.cache/usd-inspect python scripts/prepare_astribot_asset.py
python scripts/fetch_astribot_assets.py --check-only
```

资产来自 `Astribot-Dev/astribot_descriptions`，固定提交
`8c651a1abbe2821824a692aa3a3aa4cbf88e4122`，对应仿真仓库提交
`1403f9623efaea2709731c74629f395869849e89`。
`assets/astribot_sources.lock.json` 记录 35 个文件的固定 URL、大小和 SHA-256，
约 502 MB，包含 LFS 实体、URDF、网格、纹理和上游 LICENSE。
下载器不依赖 Git LFS 客户端，不覆盖校验失败的既有文件。

官方二进制 `astribot_s1.usd` 引用了作者机器上的
`/home/software/urdf2usd/.../astribot_whole_body_maniskill.usd`，无法独立加载。
因此准备入口使用官方配置所选的完整 `astribot_s1.usda`，展开实例后导出
`assets/generated/astribot/astribot_s1.usdc`，避免跨引用子树的原型依赖。
原始下载文件不修改；生成文件与源文件校验和记录在同目录 `preparation.json`。
`OmniPBR.mdl` 由 Isaac Sim 自带材质库提供。

## 运行

同时查看机器人和按钮面板：

```bash
source scripts/activate_local.sh
python scripts/preview_astribot_panel.py
```

该命令在桌面会话中打开并保持仿真窗口，默认视角包含 Astribot 全身和 24–35 层
按钮面板。机器人站在地面上，省去 PiPER 桌台，按钮保留弹簧关节与物理碰撞。
这里只展示任务场景，尚未接入自动按压控制。无窗口生成图片可添加 `--headless`。
每次运行自动创建带时间戳的输出目录，也可通过 `--output` 指定新的目录。
站位、面板高度和镜头参数在 `configs/astribot_panel_preview.json` 中。

已验证的同场景预览：
[机器人与面板全景](../../outputs/astribot_panel_preview_v2/overview.png)、
[面板近景](../../outputs/astribot_panel_preview_v2/panel.png)。
`outputs/astribot_panel_preview_v2/scene.usda` 可重新加载；240 个物理步后机器人稳定，
12 个按钮的滑动关节和资产引用检查通过。原场景的 16 项集成测试通过。

下面是独立的资产运动验证入口：

每次使用新的输出目录：

```bash
source scripts/activate_local.sh
python scripts/verify_astribot_asset.py --headless \
  --output outputs/astribot_asset_validation

# 桌面会话：验证后保持窗口打开。
python scripts/verify_astribot_asset.py --hold \
  --output outputs/astribot_asset_gui
```

输出包括 `validation.json`、`preview.png`、`after_reset.png`、对应深度数组、
`joint_trace.npz` 和带相对资产引用的 `scene.usda`。
地面使用本地生成的静态碰撞体，不引用 NVIDIA 在线默认场景。
退出码非零或报告 `status != "passed"` 表示验证失败。

## 适配内容

`src/pressb/astribot_asset.py` 的 `add_astribot(stage)` 在 SimulationApp
启动后调用，返回机器人路径、双臂关节名、腕部链接和碰撞形状信息。

- 保留官方 33 个刚体、32 个旋转关节及可视几何。
- 使用官方固定底座关节，按可视与碰撞包围盒调整离地高度，并同步世界端锚点。
- 将碰撞 API 从分组 Xform 移到实际碰撞形状；夹爪碰撞网格采用凸包。
- 保留官方已有碰撞几何范围，开启自碰撞。上游有部分链接未配置碰撞几何，
  完整列表写入报告 `adapter.bodies_without_collision`；不能据此宣称全身避碰已验收。
- 为躯干、手臂、头部和夹爪设置明确的位置驱动增益，最大力矩采用 URDF 限值。
  这些是本次仿真保持与小幅运动验证用参数，不代表真机控制参数。
- 按名称绑定左右臂各 7 个关节，不依赖 USD 枚举顺序。

验证包含静止保持、左右臂全部七个关节的小幅运动、返回零位、重置、RGB-D、
固定底座漂移检查，以及落块与机器人底座的真实接触报告。
运动指令为关节空间指令，不调用 PiPER 的六关节 IK。

独立按键入口已补充工作姿态、原装夹爪 TCP、七关节 IK 和接触对象映射；
该任务使用已知按钮位置，视觉定位和 VLA/RL 适配仍属于后续阶段。
