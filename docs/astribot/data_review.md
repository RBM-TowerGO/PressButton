# Astribot 真机数据审核（桌面界面 / Rerun）

适用于本项目拷回的 Astribot HDF5：三路拼接 JPEG、25 维关节状态／指令、末端力及时间戳。
不驱动机器人。原始数据只读；审核结果与导出副本独立保存。

## 推荐：本地图形审核界面

```bash
scripts/review_astribot.sh gui
```

打开桌面窗口，不依赖浏览器或 Rerun 服务。界面直接读取 HDF5，沿用已有 `review.json` 的审核决定。

- **上一条 / 下一条**：按左侧当前列表切换轨迹；**◀ 帧 / 帧 ▶**：逐帧查看。
- **下一条自动播放**：默认勾选，点击下一条后待图像载入完成自动播放；取消勾选则停在首帧。上一条和直接点击列表仍暂停。
- **播放 / 暂停、倍速、时间滑块**：三路图像按同一采样行同步显示。
- **保留 / 删除 / 恢复 / 待排查**：决定是否纳入导出；删除不会物理删除原文件。
- **将所有待审核项设为保留**：顶部按钮，对整个数据集生效，不受筛选影响；已删除和待排查项不变。跳过结构错误或无效裁剪范围，保留原备注和范围，可一次撤销批量标记。
- **上移 / 下移**：调整导出顺序；**撤销**：撤销本次会话最近的标记、范围或排序修改（最多 30 次）。
- **当前帧设为起点 / 终点**：选择裁剪范围；界面中的起始和结束帧**均包含**。内部 JSON 仍使用 end-exclusive。
- **保存范围 / 备注**：保存输入；切换轨迹或正常关闭时也会保存。裁剪本身不会自动把待审核轨迹标记为保留。
- **导出保留项并重新编号**：选择父目录、输入新的目录名，仅导出 keep，按左侧顺序连续编号。

左侧可以按审核状态筛选或搜索原编号；状态颜色和同步告警帮助定位问题。
窗口显示单个关节列的实测／指令曲线，可选择 0～24 列，点击曲线跳帧。
导出在后台进行并显示进度，完成前不允许修改审核决定或关闭窗口。
审核清单被外部程序改动时会拒绝覆盖；请关闭后重开。不要同时用 CLI 或多个窗口修改同一清单。

界面使用系统 Tk 和已有 Pillow，不增加训练环境依赖。当前工作站已验证可直接打开；
其他机器需在带图形桌面的终端运行，并确保系统 Python 安装了 tkinter。

## 安装与启动

已安装到 `.venv-data-review`，与仿真、训练环境隔离。重装：

```bash
/usr/bin/python3 -m venv .venv-data-review
.venv-data-review/bin/python -m pip install -r requirements-data-review.txt
```

以下命令在项目根目录运行。当前 102 条已扫描，不需要重复 scan。

```bash
# 可选旧流程：依次审核 pending 轨迹（浏览器查看，终端输入决定）
scripts/review_astribot.sh review

# 列出原编号、审核状态、裁剪范围和告警
scripts/review_astribot.sh list

# 查看第 0 条，浏览器打开终端显示的地址（默认 http://localhost:9090）
scripts/review_astribot.sh view 0 --web

# 或在有桌面显示环境的终端打开原生窗口
scripts/review_astribot.sh view 0 --open
```

交互审核会逐条启动浏览器回放服务。在 http://localhost:9090 查看，切换轨迹后刷新网页。
回到终端输入 `k` 保留、`k 20 180` 裁剪保留、`r` 剔除、`h` 待排查、`s` 跳过或 `q` 退出。
每次决定立即保存，再次运行自动跳过已经审核的条目。可用 `review --start-at 40` 从原编号 40 开始。
`review` 与 `view --web` 不要同时运行，以免端口冲突。

浏览器回放服务只绑定本机，Ctrl-C 停止，再运行 view 查看另一条。
远程电脑访问需 SSH 转发 9090 和 9876 两个端口。
不带 `--web` / `--open` 时只生成 `.rrd` 文件，可用 Rerun 打开。

Rerun 显示 head / left / right 三路图像，关节状态与指令曲线、各通道相对主时间偏差、右臂末端六维力数据。
默认按 `seconds` 时间轴回放；在时间面板切到 `frame` 查看零起始帧号。
关节仅标记为 j00～j24，力仅标记为原始列号；尚未核对厂商字段映射，避免误标关节名称或单位。
曲线可以在 Viewer 中选择部分实体显示。图像按 HDF5 同步行回放，原始相机时间差单独绘图，不修复或掩盖重复帧。

## 审核与裁剪

查看时，在另一个终端保存决定：

```bash
scripts/review_astribot.sh mark 0 hold --note '指令时间戳异常，待排查'
scripts/review_astribot.sh mark 1 keep --note '人工确认按键成功'
scripts/review_astribot.sh mark 3 reject --note '未按中'
# 示例：保留原轨迹第 20～179 帧（必须先回放确认）
scripts/review_astribot.sh mark 4 keep --start 20 --end 180 --note '去掉等待段'
# 撤销决定
scripts/review_astribot.sh mark 4 pending
```

以上编号只是操作示例，不代表这些轨迹已经成功或应当剔除。
`start` 包含，`end` 不包含；省略范围会沿用该条已保存的范围。
`pending` 待审核，`keep` 保留，`reject` 剔除，`hold` 暂缓。
自动检查只识别文件和同步问题，不判断任务成功。存在时间偏差告警的数据仍需人工决定，导出不会修复同步。

默认审核清单：`data/astribot_s1b_20261025_review/review.json`。
默认 Rerun 文件：同目录 `rerun/`。修改标记后重新运行 view 可更新 Viewer 中的说明。
不要同时从多个终端修改审核清单。

## 导出连续编号的副本

```bash
scripts/review_astribot.sh export --output data/astribot_s1b_20261025_reviewed_v1
```

只导出 `keep`，按审核清单中保存的顺序（可在 GUI 上下移动；初始为原编号数值排序），生成 `episodes/episode_000000.hdf5` 等。
同时保存 `episode_mapping.json`（新旧编号、源／目标 SHA-256、裁剪范围、验证结果）与 `review_snapshot.json`。
输出目录已存在会拒绝覆盖。源文件 SHA-256 必须与扫描时一致。
裁剪同步处理所有逐帧数组和 JPEG 字节流，保留原始绝对时间戳与 HDF5 属性。
导出后逐帧解码检查；失败时删除此次临时输出，不改原文件。
输出仍然是 Astribot HDF5，不是 LeRobot；关节映射、动作对齐和 LeRobot 转换是后续步骤。

## 另一批数据

```bash
scripts/review_astribot.sh --review data/another_review/review.json scan --source /path/to/hdf5 --full-images
scripts/review_astribot.sh --review data/another_review/review.json view 0 --web
```

默认 scan 解码每条每相机首／中／末帧；`--full-images` 解码所有帧。
已有审核文件不能直接覆盖，以免丢失标记。

官方 Rerun API：https://rerun.io/docs/getting-started
