# 空地协同链路

无人机顶视图 → 0–255 可通行值 → 全局路线 → 子目标 → CurveNav 局部策略。测评在两个室外数据源（Wanderland 实采场景、NavVerse 合成城市）上、Habitat 仿真中进行，机器人为 Scout Mini。

## 链路

| 环节 | 位置 | 说明 |
|---|---|---|
| 全局模型 | `global_planner/`（训练、评测、推理、数据转换） | 正下视图 → 9 类地表概率 → 0–255 可通行值，见 [GLOBAL.md](GLOBAL.md) |
| 仿真预测图 | `evaluation/topview/maps.py` | Wanderland 各场景、NavVerse 各城市（航拍图由 `evaluation/topview/navverse_aerial.py` 渲染）的可通行值图，并与机器人 C-space 对照 |
| 全局路线 | `global_planner/route.py` | 可通行值图（`[x, y]` 网格，NaN = 未观测）上的最小代价路径，每米代价 `255 / 值`；值为 0 不可通行；视线简化，不切角 |
| 子目标 | `global_planner/route.py` 的 `Subgoals` | 任务带 `route` 时，路线切成一串固定子目标：路线的每个拐点（`line_of_sight` 只保留这些，所以相邻子目标之间的直线在地图上不切角，路缘下沉台阶这类深度相机看不出的障碍不会被抄近路）加上把较长的直段均分到不超过 `SUBGOAL_SPACING_M`（6 m）的点；地图怀疑的段（值 < 100）里不放子目标，在其末端放一个，策略自己穿过去。策略一次只拿到当前子目标；机器人进入其 `SUBGOAL_REACHED_M`（2 m）以内，或沿路线的进度越过它，就换下一个。进度是机器人在路线上的投影，只在上次进度之后的一小段里找、不后退，不会跳到恰好经过附近的后段路线。到达判定仍是最终终点 |
| 局部策略 | CurveNav | 不变 |

```bash
# 仿真预测图（环境 curvenav-topview：pcard 的包 + transformers）
python -m kappanav.evaluation.topview.maps wanderland --checkpoint MODEL --output DIR
python -m kappanav.evaluation.topview.maps navverse --checkpoint MODEL --output DIR --scene vcp_amsterdam
```

预测图文件：`value`（`[x, y]`，float，0–255，NaN = 未观测），`origin_xy`，`cell_m`，`unobserved_value`（未观测格子的值，默认 100；仿真世界在数据边缘结束，`maps.py` 写成下限 `VALUE_FLOOR`，否则路线会从空白处抄近路）。

Wanderland 的"顶视图"是激光点云的正射投影（`maps.npz` 的 `dom_rgb`）：每个格子取第 3 高的点作表面，颜色是表面以下 0.25 m 内点的均值，补上 0.3 m 以内的缝，`dom_valid` 标出有效范围。点云是用地面相机着色的，树冠和屋顶顶面往往带天空的颜色；3DGS 也只用地面视角训练，从上往下渲染同样不对。所以 Wanderland 上的顶视图指标要按开阔处 / 遮挡处分开报，开阔处才代表模型对真实航拍的能力。

## 机器人参数

`kappanav/common/physical.py` 并列定义 `DINGO` 与 `SCOUT_MINI`，`KAPPANAV_ROBOT` 在进程启动前选定（默认 dingo，原有测评不变）。所有模块从这里读取所选机器人；测评的 `inputs.json` 记录 `robot`。

- 非 Dingo 的机器人使用独立的 numba 缓存目录：编译代码固化了模块常量，而磁盘缓存只按源码区分。
- `configs/runtime.json` 是 Dingo 的相机与车体约定，`configs/runtime_scout_mini.json` 是 Scout Mini 的；runtime 与所选机器人不一致时 `scene.configure` 报错。
- Isaac 测评场景目前只支持 Dingo，选择其他机器人时报错。

Scout Mini 的取值来源：

| 量 | 值 | 来源 |
|---|---|---|
| 外接圆半径 | 0.4182 m | URDF 碰撞网格（轮胎外角） |
| 接地 / 车顶 | z = −0.1805 / +0.0805 | URDF；Isaac 静止实测 base_link 离地 0.1806 m |
| 越障 / 坡度 | 0.07 m / 30° | 官方规格（空载） |
| 转向中心 | base_link | 四轮滑移转向，前后轮对称 |
| 轮半径 / 轮距 | 0.080 / 0.4546 m | URDF（轮胎中心） |
| 深度相机 | x 0.243、y +0.025、z +0.068，水平 | URDF `gemini_mount_joint`（左 IR = 深度原点） |
| 相机内参 | 640×360，f = 320 px | Gemini 335 深度 90° 水平视场（规格，未标定） |
| 速度响应 | 一阶 0.04 s | Isaac 实测（0.5 m/s 指令，稳态 0.5007 m/s） |
| 转向响应 | 实际 / 指令 = 0.49，与车速无关 | 四轮滑移转向的库仑摩擦平衡：W² / (W² + L²)，W = 0.4546 m，L = 0.464 m |

`evaluation/simulation/calibrate_scout_mini.py` 在 Isaac 中驱动 Scout Mini（官方质量 26 kg、按几何计算的惯量、圆柱轮胎）。它复现了 30° 坡与 70 mm 台阶和速度响应；转向增益 PhysX 只给出 0.01–0.17（宽轮胎线接触的摩擦处理不对），不采用，改用滑移转向的库仑摩擦模型。实车上应以实测为准。

## 室外仿真链路

两个数据源走同一条链路，只有第一步（把原始数据变成几何和地面高度）各自实现：

| 步骤 | Wanderland | NavVerse | 共用代码 |
|---|---|---|---|
| 原始数据 | 激光点云 + 采集轨迹 + 官方任务（`wanderland/download.py`） | 城市 USD + 官方任务（发布包） | — |
| 渲染几何（机器人无关） | 点云重建碰撞网格 `wanderland/collider.py` | 整座城市 `assets.export_geometry` | `assets.write_stage` 注册为 Habitat 场景 |
| 地面高度 | 激光地面 `maps.npz` | 地形区域三角面 `terrain.npz`（水面不算地面） | — |
| 导航网格（按机器人） | 整个场景一次建成 | 整座城市，按 50 m 分块建（块间重叠 2 m），拼成一张 | `navigation.navmesh_settings`（半径、车高、越障、坡度） |
| C-space 栅格 | | | `navigation.navmesh_ground`（贴地的导航网格才算可走）+ `navigation.known_ground`（丢掉没走过的小连通块），输出 `navigation.npz`（含每格地面高度） |
| 任务校验 | `wanderland/prepare.convert_episodes` | `navverse.official_tasks`；另有 `navverse.long_tasks` 采样长距离任务 | 端点吸附 ≤ 0.5 m、同一连通域、栅格最短路；不满足的任务带原因拒绝 |
| 任务表 | `tasks_<robot>.json` | `<city>_<robot>_tasks.json`、`<city>_<robot>_long_tasks.json` | 同一格式（见下） |
| 全局路线（可选） | | | `global_planner/route.py`，`--maps none\|oracle\|<目录>`；真值路线就是任务自带的 C-space 最短路线（`shortest_path`） |
| 闭环测评 | | | `scripts/evaluation/evaluate.sh` → `evaluation/benchmark/run.py`；`evaluation/simulation/realtime.py` 的车体模型与接触 |

```bash
export KAPPANAV_ROBOT=scout_mini CUDA_VISIBLE_DEVICES=7
# Wanderland：几何（一次）+ 该机器人的导航层 + 任务表
bash scripts/evaluation/prepare_wanderland.sh --workers 8
# NavVerse：城市几何（--export，一次）+ 该机器人的整城导航 + 官方任务表 + 1000 条 100 m 以上的长距离任务
bash scripts/evaluation/prepare_navverse.sh --scene vcp_amsterdam --export --workers 24 --long 1000
# 测评：任一任务表，--maps 选全局路线
bash scripts/evaluation/evaluate.sh --benchmark $KAPPANAV_NAVVERSE_SCENES/vcp_amsterdam_scout_mini_long_tasks.json \
  --maps oracle --checkpoint <policy.pt> --output <目录>
```

回合视频：`python -m kappanav.evaluation.benchmark.video <测评目录> [--episodes ID ...] [--heatmap 目录]` → `<测评目录>/videos/<回合>.mp4`（1920×1080，每个决策一帧）。主画面是相机视角的彩色图，按记录的位姿重新渲染（测评只渲染深度）：NavVerse 用城市的带纹理外观资产（Habitat），Wanderland 用数据集自带的 3D 高斯（`evaluation.benchmark.splat`，在 gsplat 环境 `KAPPANAV_GSPLAT_PYTHON` 里运行）；画面上叠加选中的轨迹和当前子目标，左下角是策略输入的深度。右上是车头朝上的局部地图：航拍图叠加顶视图模型的 可通行值热力图、全局路线、固定子目标（当前的高亮）、已走轨迹；右下是整条路线的总览；下方时间轴是到目标距离、速度和接触区间。热力图默认取该测评的 `--maps` 目录，oracle / none 的测评用 `--heatmap` 指定。gsplat 环境 `/shibo_huang/envs/curvenav-gsplat`：叠在 curvenav-pcard 上的 venv，gsplat 1.5.3 从源码编译（nvcc 12.6 生成 compute_90 PTX，由驱动 JIT 到 sm_120）。3D 高斯在数据集坐标系里，与 Habitat 深度在同一位姿下对齐（图像与几何吻合；期望深度因半透明叠加偏短约 15%）。

`--maps`：`none`（直接给终点）、`oracle`（任务自带的 C-space 最短路线，对照上限）、`<目录>`（`<目录>/<scene_id>.npz` 顶视图预测，在预测图上规划）。规划好路线的任务表随结果存为 `<目录>/tasks.json`，`inputs.json` 记录 `maps`。`summary.json` 按数据源、按距离档（任务带 `range_m` 时）和全部分组。

`--policy`：`curvenav`（默认）或基线规划器 `sandplanner`、`iplanner`、`navdp`、`x-navdp`，见 [baselines/README.md](../baselines/README.md)。基线和 CurveNav 用同一个闭环：同样的任务、子目标、相机、决策延迟、MPC、车体与接触，只换"观测 → 轨迹"这一步；每个 worker 一个基线服务，回合开始时重置。

任务表格式：`robot`（必须等于进程的 `KAPPANAV_ROBOT`，否则测评拒绝启动）；`scenes[]`：`scene_id`、`assets`、`asset_scene_id`、`scene_dataset`、`navigation`；`episodes[]`：`episode_id`、`scene_id`、`start=[x, y, yaw]`、`goal=[x, y]`、`shortest_path_m`、`spl_distance_m`、`shortest_path`（C-space 最短路线的折线顶点，含起终点）、`goal_budget_s`、`collision_ends`、`arrival_hold_s`，长距离任务另有 `range_m`；测评时加上 `route`。渲染进程按 `(assets, asset_scene_id)` 复用场景，所以 NavVerse 同一城市的所有任务只加载一次城市。

共同的规则：

- 坐标：所有栅格按 `[x, y]` 索引，规范平面 XY、Z 向上，Habitat = (X, Z, −Y)。Wanderland 规范系 = 数据集的 (x, z, −y)；NavVerse 规范 XY = USD XY。相机高度取自 `navigation.npz` 的地面高度，官方任务里的 Z 不使用（Wanderland 的是手持相机高度，NavVerse 的是运行时对齐后的地面）。
- 测评规则沿用 `docs/EVALUATION.md`：目标 0.5 m 内停稳 4 s 判成功，碰撞不终止回合（车身停在障碍处，可转向、可退出），SPL 用起终点直线距离；唯一的改动是时限取 max(122, 6 × 最短路径 + 30) s。这不是两个数据集各自的官方协议。
- 接触：Dingo 按圆盘判断；Scout Mini 按实际朝向的矩形车身（半长 0.312 m、半宽 0.293 m，`physical.body_half_extent_m`）判断，C-space 外接圆半径 0.418 m 只用于任务生成与路线规划。
- 机器人做不到的官方任务被拒绝并记录原因，不靠放宽越障或坡度凑数；报成功率时分母是通过校验的任务数，要同时写明官方任务总数。

### Wanderland

- 几何层（`scenes/<id>/`，所有机器人共用）：由原始激光点云重建碰撞网格；地面平滑只抹去低于 `TERRACE_M`（5 cm）的重建台阶，真实台阶由各机器人的导航层判定。扫描范围的开放边界（场景外缘、大于补洞尺寸的未观测区）立 1 m 高的围墙：深度相机看得到世界的边界，不会面对一片被 C-space 判为不可通行、却什么都看不见的虚空。
- 草地：高度低于 0.20 m 且偏绿（ExG ≥ 0.10）的点不算障碍；偏绿的地面格子再做 0.25 m 中值平滑，免得草叶噪声在 5 cm 网格上形成超过坡度上限的假坡。草地能否通行只由底盘的越障和坡度决定。
- 导航层（`scenes/<id>/<robot>/`）：采集者走过的地面是已知地面；没走过、面积又小于 20 m² 的可走连通块（车顶、箱子顶）丢弃。
- `maps.npz` 的 `covered` 表示上方 2.2 m 以上有树冠或屋顶，不表示有无数据（有无数据看 `points_per_cell > 0`）。
- Scout Mini：293 个场景、5236 条官方任务，4139 条通过校验（282 个场景）。

### NavVerse

- 渲染用整座城市（阿姆斯特丹 900 m × 900 m、2100 万三角面）。导航是该机器人的整城 C-space（`scenes/<city>/<robot>/navigation.npz`，18001 × 18001 格）。Detour 一次构建最多容纳 65535 个顶点，密集街区约 100 m 见方就超限，所以按 50 m 分块构建，构建失败的块再四分；块与块重叠 2 m（大于机器人半径），拼接处没有缝。
- 一个城市在测评里只是一个场景：所有任务共用城市的渲染场景和整城 C-space（车体接触按格查表，代价与地图大小无关），预加载只做一次。每条任务在整城 C-space 上求最短路，最短路线随任务写入。
- 发布包的道路比人行道低 0.2 m，路缘石是 0.2 m 的竖直台阶（`*_terrain_regions_summary.json` 的 `road_depth`）；官方导航网格按 0.8 m 越障生成，官方路线随意上下路缘。Scout Mini 的越障是 0.07 m，只能从与路面齐平的地方上下，所以直线几十米的官方任务，Scout 的最短路可以长达几百米。
- 水面（运河）在渲染里保留，但不是地面。官方有一部分起点就放在水面上。
- 官方任务（室外 `vcp`，52 个城市共 4096 条）的官方路径长度都不超过 50 m。阿姆斯特丹 80 条中 Scout Mini 通过 42 条；拒绝 38 条：19 条起终点不在同一连通域，19 条端点在水面上。
- 长距离任务（`--long N --edges 100 200 400`）：N 条平均分到各距离档。每条任务有自己的随机流（种子、档、序号），与其他任务和进程数无关：起点在有 0.10 m 余量的可走格子上均匀采样，朝向均匀，终点在同一连通域内、最短路长度落在该档的格子上均匀采样。
- 彩色渲染资产 `<city>.appearance.glb`：城市地面是 64 张烘焙的 4096 px 贴图（每张约 0.1 km，3 cm/纹素），按原分辨率保留（`GROUND_TEXTURE_MAX_PX`）；其他贴图缩到 1024 px（`TEXTURE_MAX_PX`）；贴图一律存为 JPEG（`TEXTURE_JPEG_QUALITY` 90），存成 PNG 时整城资产超过 glTF 二进制 4 GB 的上限，Habitat 无法加载。以前统一缩到 512 px，地面只有 22 cm/纹素，离地 0.25 m 的相机看到的是一片灰。只渲染深度的仿真器不加载纹理（`requires_textures`），CurveNav 测评不因纹理多占显存，深度逐位不变。发布包缺 1 张贴图（Avengers.jpg），该材质用默认灰色。
- 尚未接入：其余城市（同一命令换 `--scene vcp_<city>`）、室内外连通任务与 GRScenes 多楼层任务。
