# 测评

测评任务、到达判定与时限沿用 [NavDP](https://github.com/InternRobotics/NavDP) 仓库 `baselines/x-navdp` 的开源点目标测评。

## 基准

- 场景：官方 `scene_split.json` 的 20 个 home 与 20 个 commercial 测评场景。
- 任务：每场景 100 对官方起终点与初始朝向（`data/benchmark/pointgoal/`），共 4000 回合；每次运行完整任务集。
- 端点：`evaluation.benchmark.prepare.benchmark_episodes` 生成两个仿真共用的任务清单 `data/benchmark/tasks.json`，官方原值保存在 `official` 字段。
  - 起点处机器人无法原地转向（轮轴中点到 C-space 边界不足 4.95 cm）时，起点移到最近的、前方机身高度障碍均在图像内的位置（接触距离 ≥ 0.356 m，由相机安装与视场算出）。
  - 终点在 C-space 外时移到最近的可通行格子。
- 场景资源：官方 USD 按单位与坐标轴转换为 Habitat 资源；毯子按官方加载规则缩小到 1%。

## 回合规则

| 项 | 规则 |
|---|---|
| 时限 | 122 s 仿真时间（1220 个 0.1 s 控制步） |
| 成功 | 每个控制步检查：首次在目标 0.5 m 内且世界系速度 \|vx\|+\|vy\|+\|vz\| < 0.25 m/s 时开始计时，40 步（4 s）后判成功，计时期间离开仍计；同一步内先判终止再计时 |
| 终止 | 只有成功与超时；碰撞与停滞不终止回合 |
| SPL | `success × D / max(D, P)`，D 为起终点直线距离，P 为实际路程 |
| 决策延迟 | 计划由观测时刻的状态计算，固定延迟 0.2 s 后发布（`--decision-latency-s`，须为 0.1 s 的整数倍）；首个计划立即发布 |

## 两个仿真入口

两个入口读取同一任务清单，使用同一策略推理、候选选择（`PlanCommitment`）与控制器（`ReferenceMPC`），输出相同格式的结果。Habitat 入口另可用 `--policy` 换成基线规划器（[baselines/README.md](../baselines/README.md)），其余闭环不变。

| | Habitat（`scripts/evaluation/evaluate.sh` → `evaluation.benchmark.run`） | Isaac（`scripts/evaluation/evaluate_isaac.sh` → `evaluation.benchmark.isaac`） |
|---|---|---|
| 渲染 | Habitat 深度传感器 | Isaac Sim RTX 深度相机 |
| 运动 | 轮轴中点单车模型（`local_planner.control.mpc.body_step`），速度一阶响应 0.15 s；实际转速为指令转速乘以随前进速度变化的响应系数（原地 0.55，0.45 m/s 时 0.96，`realtime.yaw_response`，Isaac 实测） | PhysX 刚体，轮速驱动（`evaluation.benchmark.isaac_scene`） |
| 接触 | 以 base_link 为心、半径 0.1676 m 的圆盘对 5 cm C-space；接触时不平移（不滑动、不穿入），只允许不加深穿透的运动；绕轮轴的转向仍执行，障碍沿外法向把 base_link 推开不超过 2 cm（`realtime.CONTACT_PIVOT_PUSH_M`） | PhysX 接触；base_link 接触力 > 8 N 计为接触 |
| 并行 | 每卡 8 个 worker，共享模型与渲染进程（`--workers-per-gpu`、`--servers-per-gpu`、`--renderers-per-gpu`） | 每场景一个进程，`--workers` 个场景并行 |

相机：640×360，焦距 326.4 px，装在 base_link 上下俯 10°；机器人静止时车身上仰 6°，水平车身坐标系中相机实际下俯 4°，Habitat 按此姿态渲染（`physical.rest_camera_pose`）；策略使用 0.1–5 m 深度。推理批大小为 1 时，每个回合逐位可复现。

Isaac 场景（`evaluation.benchmark.isaac_scene`）：PhysX 100 Hz，每 10 步一个控制步并渲染一次；场景 USD 缩放 0.01（毯子缩小到 1 %），其自带的 physics scene 保留并同样以 100 Hz 步进；带碰撞网格集合的物体按合并三角网格碰撞，接触偏移 0.03 m；机器人从起点上方 0.1 m 放下，每回合重置位姿、速度与关节状态。

## 结果

每次运行输出 `summary.json`（`complete`、`groups`、逐回合 `rows`）与逐回合执行记录（计划、轨迹、候选）。缺失或重复回合的运行不标为完整；`--max-steps` 短步检查的结果标为不完整。

```bash
python -m kappanav.evaluation.benchmark.results RUN BASELINE   # 按相同任务 ID 比较成功率与得失回合，标准误按成对回合计算
```

## 无目标探索（可选）

两个入口均提供 `--exploration`，默认关闭。开启后每个回合在同一仿真器中另跑特权探索专家与无目标策略（各用回合的探索时限，从同一起点出发），覆盖面积由各自的深度图认证，报告覆盖面积、碰撞率与相对专家的 E-SPL：

- 设模型新增面积为 A、专家为 E，取共同面积 a = min(A, E)，T、T* 分别为模型和专家首次达到 a 的时间；
- `area_ratio = min(A/E, 1)`；
- `E-SPL = collision_free × area_ratio × T* / max(T, T*)`。

E-SPL 为本项目定义，不属于标准导航成绩。
