# KappaNav（κ-Nav）

空地协同导航：无人机的顶视图给出几百米范围内哪里好走，地面机器人（Scout Mini）凭深度相机处理眼前的几何。

```
无人机顶视图 ──► 0–255 可通行值 ──► 全局路线 ──► 子目标 ──► CurveNav ──► MPC ──► 底盘
                 global_planner/     255 / 值 每米   拐点+每 ≤6 m    深度四帧
                 （模型）            route.py        route.Subgoals  16 条候选曲线
```

| 层 | 负责 | 看不到的 |
|---|---|---|
| 空中（顶视图模型） | 几百米范围内哪里是路、草地、障碍，绕行方向 | 树冠和屋檐下面、0.2 m 级的高差（路缘）、小障碍、地图之后才出现的变化 |
| 地面（CurveNav） | 3.6 m 内的几何：净空、台阶、窄处、动态障碍 | 5 m 以外，死胡同和绕行方向 |

测评在两个室外仿真上进行，共用一条链路：Wanderland（激光实采场景，有顶视图）和 NavVerse（合成城市，整城 C-space，100 m 以上的长距离任务）。机器人参数、仿真链路、任务格式与结果口径见 [docs/AIRGROUND.md](docs/AIRGROUND.md)。

```bash
export KAPPANAV_ROBOT=scout_mini
bash scripts/evaluation/prepare_wanderland.sh --workers 8                                   # Wanderland 几何、导航层、任务表
bash scripts/evaluation/prepare_navverse.sh --scene vcp_amsterdam --export --workers 24 --long 1000  # NavVerse 整城导航与任务
bash scripts/evaluation/evaluate.sh --benchmark <任务表> --maps <none|oracle|预测图目录> --checkpoint <policy.pt> --output <目录>
```

## 目录

| 位置 | 内容 |
|---|---|
| `kappanav/global_planner/` | 全局：顶视图 → 9 类地表 → 0–255 可通行值 → 路线与子目标（`route.py`）；训练、评测、推理、数据转换（`prepare/`） |
| `kappanav/local_planner/` | 局部：CurveNav。`model/` 网络与曲线，`features/` 四帧深度与条件特征，`control/` MPC，`inference/` 模型包与在线推理，`data/` 标签与训练集，`training/` 训练，`capture/` 仿真中的专家采集 |
| `kappanav/evaluation/` | 测评：`simulation/` 仿真世界（Habitat、Wanderland、NavVerse）、车体，`benchmark/` 任务、闭环、Isaac、指标、视频，`topview/` 仿真世界的航拍图与可通行值图 |
| `kappanav/common/` | 机器人参数、几何、栅格最短路、深度、文件读写 |
| `baselines/` | 对照用的局部规划器服务（SanD-Planner、iPlanner、NavDP、X-NavDP） |
| `scripts/`、`tests/` | 按 `global_planner`、`local_planner`、`evaluation` 分组 |
| `configs/`、`docs/`、`data/` | 配置、文档、数据目录说明 |

依赖只能自上而下，由 `tests/test_architecture.py` 检查：`common` 不依赖任何部分；`global_planner` 与 `local_planner` 互不依赖，只通过子目标对接；`evaluation` 可以用前三者；反过来只有 `local_planner/capture`（专家采集）使用仿真。`local_planner` 内部按 model → features → control → inference → data → training → capture 分层。

## 局部规划器 CurveNav

CurveNav 是面向差速轮式机器人的点目标导航策略，输入为最近 2 秒内的四帧深度与位姿：

1. 四帧深度按位姿融合为局部几何场（净空、支持与自由空间）；
2. 生成器 G（两步流匹配）输出 16 条三次 B-spline 候选轨迹；
3. 评分器 Q 在几何场判定无碰撞的候选中选出最优轨迹；
4. MPC 跟踪所选轨迹。

训练监督来自特权地图上的专家：专家驾驶时规划的曲线即为 G 的目标，Q 的目标为同一特权地图上的路线效用。

## 流程

```bash
# 数据：训练场景几何 → 任务 → 专家采集 → 训练集，可续跑
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/local_planner/prepare_data.sh

# 训练：outputs/training/<UTC 时间>/checkpoints/epoch_NNN/{model,ema_model}.pt
CUDA_VISIBLE_DEVICES=0,1,2 bash scripts/local_planner/train.sh

# 测评基准：场景资源、导航地图与任务清单（data/benchmark）
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/evaluation/prepare_benchmark.sh

# 测评：同一任务清单，Habitat 与 Isaac 两个仿真入口，结果格式相同
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluation/evaluate.sh --checkpoint data/model/policy.pt
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/evaluation/evaluate_isaac.sh --checkpoint data/model/policy.pt --workers 8

# 同任务成对比较
python -m kappanav.evaluation.benchmark.results outputs/evaluation/<run> outputs/evaluation/<baseline>
```

所有脚本通过 `scripts/common_env.sh` 设置运行环境，用 `CUDA_VISIBLE_DEVICES` 选择 GPU。

## 文档

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)：数据采集、模型、训练与部署
- [docs/EVALUATION.md](docs/EVALUATION.md)：测评基准与规则
- [docs/GLOBAL.md](docs/GLOBAL.md)：全局模型的类别、优先级、数据、训练与评测
- [docs/AIRGROUND.md](docs/AIRGROUND.md)：空地协同链路、Scout Mini、室外仿真（Wanderland、NavVerse）
- [data/README.md](data/README.md)：数据目录
