# VIB-820 材质资产

现场材质判断依据：用户照片及 `DJI_20261013165415_0002_D.MP4` 18–21 秒画面。
面板为细拉丝金属；墙面以白色背漆玻璃外观近似；标识以背印亚克力薄板外观近似。
照片不能确定准确的材料牌号、折射率或粗糙度，下面参数均为视觉匹配估计。

| 来源 | 实际使用 | 许可 |
|---|---|---|
| [ambientCG Metal010](https://ambientcg.com/a/Metal010) | 2K Color / Roughness / NormalGL，减弱污渍、微表面起伏后使用 | [CC0-1.0](https://docs.ambientcg.com/license/) |
| [Poly Haven Large Corridor](https://polyhaven.com/a/large_corridor)，Sergej Majboroda | 2K HDR 反射环境，加上根据视频估计的条形灯 | [CC0-1.0](https://polyhaven.com/license) |
| [NVIDIA OmniSurface](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/templates/OmniSurface.html) / OmniPBR | 当前 Isaac Sim 内置材质，玻璃与亚克力的介质反射层 | NVIDIA 软件许可；包中不复制 MDL 源码 |

`source/` 保留原始下载及源贴图。`steel_clean_*` 是由
`scripts/prepare_vib820_materials.py` 确定性生成的衍生贴图，保留原始横向纹理，
在面板 UV 中将横向映射到世界竖直方向，160 mm 对应一个纹理周期。
颜色空间：颜色 sRGB；法线和粗糙度 raw/linear；HDR 为线性高动态范围。
Displacement / AO 未连接，避免把反光金属做成粗糙凹凸表面。

材质分配和参数：`src/pressb/vib820_materials.py`。
背漆玻璃为白色不透明基底 + IOR 1.52、粗糙度 0.028 的透明反射表层；
亚克力为现有彩色印刷 + IOR 1.49、粗糙度 0.075 的表层。
标签厚度估计 1.6 mm，仅视觉，无碰撞。
这些是单层着色器对背印板的近似，不计算透明板侧面折射或毫米级光程。
HDRI 仅用于近似周围明暗，并非现场环境采集。

USD 同时提供 MDL 和 UsdPreviewSurface。Isaac Sim 5.0 的 RTX 材质效果最完整；
其他 USD 查看器使用回退材质时，各向异性和表层反射可能有差异。
生成模型包会复制所有使用的图片为相对路径，无在线贴图依赖。

下载来源、文件 SHA-256 与处理说明见 `manifest.json`。

重做衍生贴图（不需要启动 Isaac Sim）：

```bash
source scripts/activate_local.sh
python scripts/prepare_vib820_materials.py
```
