# 数据目录

`data/` 不进版本库。脚本的默认路径指向以下位置：

| 位置 | 内容 | 产生者 |
|---|---|---|
| `model/policy.pt` | 部署模型 | `scripts/local_planner/train.sh`（EMA 权重） |
| `model/initial.pt` | 初始模型：归一化与训练集一致，采集时用于录制 Q 的固定候选 | `python -m kappanav.local_planner.data.prepare_replay` |
| `scenes/{grs,hssd}/navigation/train/<scene>/` | 训练场景几何：`scene.navmesh`、`navigation.npz`（5 cm C-space）、`scene.json` | `scripts/local_planner/prepare_data.sh` |
| `scenes/tasks.json` | 训练任务目录 | `scripts/local_planner/prepare_data.sh` |
| `dataset/` | 训练集：`captures/<case_id>/`（原始采集）、`shards/<grid_sha>/`（训练分片）、`train/`、`validation/`、`dataset.json` | `scripts/local_planner/prepare_data.sh` |
| `benchmark/pointgoal/` | 官方起终点（每场景 100 对），来源与哈希见 `PROVENANCE.json` | 导入 |
| `benchmark/assets/` | 40 个测评场景的 Habitat 资源 | `scripts/evaluation/prepare_benchmark.sh` |
| `benchmark/navigation/<scene>/` | 测评地图 | `scripts/evaluation/prepare_benchmark.sh` |
| `benchmark/tasks.json` | 测评任务清单（4000 回合） | `scripts/evaluation/prepare_benchmark.sh` |

- 训练分片的 `raw/*.npy` 是 `captures/*/depths.npy` 的硬链接，二者须在同一文件系统。
- `capture.json` 的哈希是训练行的身份，也是重标注目标的随机种子；已完成的采集文件不原地修改。
- 相同输入重跑 `prepare_data.sh` 会复用已完成的场景；输入改变时使用新的 `OUT` 目录。

外部位置在 `scripts/common_env.sh` 中设定，均可由同名环境变量覆盖：

| 变量 | 内容 |
|---|---|
| `KAPPANAV_PYTHON` | 训练、采集与 Habitat 测评的 Python 环境 |
| `KAPPANAV_ISAAC_PYTHON` | Isaac Sim 5.0 测评的 Python 环境 |
| `KAPPANAV_ASSETS` | 源场景与机器人（`training/grscenes-hab`、`evaluation/scenes`、`evaluation/robots/dingo.usd`） |
| `KAPPANAV_HSSD` | HSSD 训练场景 |
| `KAPPANAV_RUNTIME` | 运行库，见下表 |

运行库（`$KAPPANAV_RUNTIME`）：

| 位置 | 内容 |
|---|---|
| `habitat_dingo_0_3_3/`、`navbench_aux/` | Habitat 与辅助 Python 包 |
| `acados/` | MPC 求解器 |
| `navbench_build_sources/` | Isaac Lab（以可编辑方式安装在 Isaac 测评环境中） |
| `apps/` | Isaac Lab 的 Kit 应用配置 |
| `isaac_kit/` | Omniverse Kit |
| `isaac_torch/` | Isaac 测评使用的 torch（与训练环境相同版本） |
