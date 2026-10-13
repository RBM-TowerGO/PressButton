# 第三方软件与资产来源

本仓库不打包下载的机器人网格、NVIDIA 场景资产或 Conda 环境。`scripts/fetch_assets.py` 按固定提交和 SHA-256 下载 35 个文件；`assets/sources.lock.json` 保存来源与许可声明。下载器保留既有冲突文件并报错，不覆盖用户修改。

| 来源 | 固定版本 / 提交 | 使用范围与许可说明 |
|---|---|---|
| NVIDIA Isaac Sim | 当前 5090 环境 5.0.0；原采集环境 4.5.0 | 物理仿真与渲染；遵守 NVIDIA 软件许可 |
| Isaac Lab | 当前 v2.2.0；历史 v2.0.2 | 独立下载到 `vendor/IsaacLab`；适用其上游许可 |
| AgileX `robot_lab` | `b868140eeb1459acefef24a865587ec39a5278c3` | PiPER URDF/USD/网格；下载时同时保存上游 Apache-2.0 LICENSE |
| Astribot `astribot_descriptions` | `8c651a1abbe2821824a692aa3a3aa4cbf88e4122` | S1 USD/URDF/网格与纹理；下载时保存该描述仓库的 Apache-2.0 LICENSE，逐文件校验见 `assets/astribot_sources.lock.json` |
| AgileX `piper_isaac_sim` | `8e1f88fdb7afca49c40e9a0c1c01cc588e86f0d2` | 原厂相机支架与 D435；该提交无根 LICENSE，Piper 包声明不能视为完整授权，不对其几何另行推定许可 |
| Intel RealSense description | 上述固定源树中 `_d435.urdf.xacro`、D435 网格和包声明 | 保留源文件中的 Intel 许可声明与归属信息 |
| NVIDIA Isaac 4.5 内容库 | 逐文件 SHA-256 | 桌子、支架及材质；适用 NVIDIA 的资产使用条款 |
| ambientCG / Lennart Demes：Metal010 | 2K JPG，SHA-256 见 `assets/materials/vib820/manifest.json` | CC0-1.0；VIB-820 拉丝钢颜色、粗糙度及 OpenGL 法线的来源，另生成减弱污渍与凹凸的衍生贴图 |
| Poly Haven / Sergej Majboroda：Large Corridor | 2K HDR，SHA-256 同上 | CC0-1.0；VIB-820 室内环境照明与反射参考，不代表实拍电梯厅的 HDR 扫描 |
| NVIDIA 内置 OmniSurface / OmniPBR | 当前 Isaac Sim 5.0 安装版本 | 分层玻璃、亚克力与金属着色器；只引用内置 MDL 模块，不随模型包分发 NVIDIA 着色器源码 |
| ZPRL / ZPRL-PressB | 方法来源 `a34a1cba029702af82c0fed873be9e0e6bd274b3` | `src/pressb/online_rl/learner.py` 的 SAC 适配保留上游 MIT notice；参考仓库独立克隆，不打包其环境或数据 |
| VLA-JEPA PiPER 适配 | 推理运行版本 `9c9dc71c8e199c7428b25d103ab666395832c001` | 独立模型仓库；PressB 提供冻结推理 addon，不随源码发布模型权重或第三方基础模型 |

`prepare_wrist_asset.py` 在隔离的 OpenUSD 24.11 中提取官方支架与相机子树，以兼容 Isaac Sim 4.5。它保留网格点与拓扑，生成文件及来源记录位于 `assets/generated/official_wrist/`，不提交 Git。

本仓库尚未指定自有代码的开源许可证。公开可见不替代许可授权；第三方组件始终遵循各自的许可。

`docs/media/` 仅发布 PressB 仿真生成的精选压缩演示，并记录原始视频哈希与审核范围；
不包含机器人网格或场景资产，不改变上述第三方许可。仓库版本与用途见 `repos.lock.json`。

上游链接：

- [Isaac Lab](https://github.com/isaac-sim/IsaacLab/tree/v2.0.2)
- [AgileX robot_lab](https://github.com/agilexrobotics/robot_lab/tree/b868140eeb1459acefef24a865587ec39a5278c3)
- [AgileX piper_isaac_sim](https://github.com/agilexrobotics/piper_isaac_sim/tree/8e1f88fdb7afca49c40e9a0c1c01cc588e86f0d2)
- [NVIDIA Omniverse 软件许可](https://docs.omniverse.nvidia.com/platform/latest/common/NVIDIA_Omniverse_License_Agreement.html)
- [ambientCG Metal010](https://ambientcg.com/a/Metal010) / [CC0 许可说明](https://docs.ambientcg.com/license/)
- [Poly Haven Large Corridor](https://polyhaven.com/a/large_corridor) / [CC0 许可说明](https://polyhaven.com/license)
