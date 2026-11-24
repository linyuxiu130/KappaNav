# 架构

## 1. 数据与入口

| 步骤 | 命令 | 产物 |
|---|---|---|
| 训练数据 | `bash scripts/local_planner/prepare_data.sh` | `data/scenes/`（几何与任务）、`data/dataset/`（训练集） |
| 训练 | `bash scripts/local_planner/train.sh` | `outputs/training/<UTC 时间>/checkpoints/epoch_NNN/{model,ema_model}.pt` |
| 测评基准 | `bash scripts/evaluation/prepare_benchmark.sh` | `data/benchmark/{assets,navigation,tasks.json}` |
| 测评 | `bash scripts/evaluation/evaluate.sh`、`bash scripts/evaluation/evaluate_isaac.sh` | `outputs/evaluation/<UTC 时间>/summary.json` |
| 新表示的初始模型 | `python -m kappanav.local_planner.data.prepare_replay --root data/dataset --bundle NEW.pt` | 归一化与训练集一致的初始模型 |

数据目录见 [data/README.md](../data/README.md)。

## 2. 采集

`prepare_data.sh` 依次执行：场景几何 → 任务 → `capture_tasks.sh` 启动的场景 worker（`local_planner.capture.capture_pipeline`）→ 训练集索引。每个 worker 处理一个完整场景，在内存中完成驾驶、标注与编译，并原子发布该场景的训练分片；中断后重跑同一命令即可续跑。

**场景与任务**（`local_planner.capture.prepare_scenes`、`evaluation.simulation.tasks`）
- 场景：GRScenes 训练划分与 HSSD 场景，与测评场景不重叠；每个来源中按场景划出验证集。
- 几何：Habitat navmesh 按 Dingo 半径 0.1676 m、碰撞高度、可跨越高度与坡度构建，栅格化为 5 cm C-space（格子中心在 navmesh 内即可通行）。
- 任务：每场景 20 个普通任务（起点净空 ≥ 0.10 m，目标测地距离 3–12 m）与 10 个贴墙任务（起点净空 0–0.10 m）。

**专家**（`local_planner.capture.collect_mpc`、`local_planner.data.bspline_teachers`、`common.geometry`）
- 帧间隔 0.1 s，每 2 帧一个锚点。锚点上专家从当前状态规划路线（考虑净空的 A*，截取 5 m，平滑并认证为曲线），MPC 跟随到下一锚点。专家规划的曲线即该锚点的 teacher。
- 路线在 0.4 m 前视处偏离机身朝向 ≥ 60° 时，曲线从路线方向出发，由 MPC 先原地转向。

**标签**（`local_planner.data.prepare_replay`、`local_planner.data.prepare_arrays`）
- 每个锚点：录制目标及其 teacher；每 5 个锚点另加三个随机可达目标（近、中、远）及无目标探索 teacher。
- teacher 须通过与专家相同的安全认证。Q 的专家池为同一地图、同一位姿的全部合格 teacher；G 的 teacher 池为同一采集、同一四帧、同一位姿的全部合格 teacher。
- 观测：当前帧加最近 2 秒内随机 3 帧，深度 126×224，光学 Z 上限 5 m（`local_planner.features.four_frame_history`）。

**可复现性**：`capture.json` 记录采集相关源码与运行库的哈希（`collect_mpc.acquisition_contract`）；相同输入逐位复现。

## 3. 模型

| 模块 | 代码 | 作用 |
|---|---|---|
| 深度编码 | `local_planner/model/encoders/depth.py` | 126×224 深度 → 384 维像素特征 |
| 四帧融合 | `local_planner/features/four_frame_history.py`、`local_planner/features/history_fusion.py` | 按位姿把像素特征融合到空间单元 |
| 几何场（LH） | `local_planner/features/temporal_geometry.py` | 64² 局部场（±3.6 m）：净空、梯度、支持、重叠、自由；以及场景 token |
| 条件 | `local_planner/features/conditions.py` | 场景 token 与帧龄 token；在线推理与训练回放共用 |
| 生成器 G | `local_planner/model/network/policy.py`、`local_planner/model/network/decoder.py` | 7 个控制点增量的条件流；交叉注意力读取场景 token |
| 曲线 | `local_planner/model/trajectory/` | 固定原点 + 7 个控制点的三次 B-spline，3.6 m 跨度，64 个解码点 |
| 评分器 Q | `local_planner/model/network/bspline_evidence_evaluator.py` | 每条候选按弧长分 8 段 token，与场景 token 交叉注意力后输出分数 |

## 4. 训练

**损失**（`local_planner/training/bspline_objective.py`，五项等权相加）

| 项 | 定义 |
|---|---|
| CFM | 自身 teacher 的条件流匹配 |
| TRAJ | 16 条两步候选对同位姿 teacher 池的能量分数 |
| GF | 候选在特权 C-space 中进入障碍一侧的净空不足 |
| LH | 几何场相对同四帧实际观测的距离与自由空间损失 |
| LQ | Q 分数与效用的中心化 smooth-L1；候选为专家池、当前 16 条候选与录制候选 |

**效用**（`data/value_targets.RouteUtilityTeacher`）：`(进展 − ½·(弧长 + 原地转向折算路程)) / 3.6 m − 接触代价 − asinh(净空不足 / 机身半径)`。进展为到目标的测地距离减少量；无目标时为可通行位移。

**优化**（`local_planner/training/bspline_replay.py`）：全局批 2560 行，AdamW（β 0.9/0.95，权重衰减 0.01），峰值学习率 1e-3，1 epoch 线性预热后余弦衰减至 1%，共 8 epoch；参数 EMA 衰减 min(0.9999, (1+n)/(10+n))。每 epoch 保存检查点与恢复状态（`--resume`），并报告验证损失与选择行为指标。

## 5. 部署

- 一步推理（`local_planner.inference.bundle.LocalBSplineBundle.step`）：写入当前帧 → 四帧编码 → 两步生成 16 条候选 → Q 打分 → 几何场在候选 64 个点上的净空。
- 选择（`PlanCommitment`）：在净空 ≥ 0 的候选中取 Q 最高者（全部不安全时在全部候选中取）；与上一计划前 1 m 平均距离 ≤ 0.3 m 的候选加 0.08。
- 执行（`local_planner.control.reference_mpc`）：所选曲线表示在做出该计划的观测时刻的机器人坐标系中（以该时刻位姿为起点），MPC（0.1 s × 30 步，v ∈ [−0.5, 0.5] m/s，ω ∈ [−0.5, 0.5] rad/s，只跟踪位置）每个计划求解一次，发布后按 0.1 s 开环依次执行。

## 6. 机器人模型

Dingo 为差速底盘：运动学是轮轴中点处的单车模型，base_link（位姿与碰撞圆盘中心）位于其后 4.95 cm（`local_planner.control.mpc.body_step`）。碰撞体为以 base_link 为心、半径 0.1676 m 的圆盘；相机与机身参数见 `physical.py`。静止时车身上仰 6°、base_link 高 5.12 cm（Isaac 实测），相机在水平车身坐标系中的姿态由安装位姿推出（`physical.rest_camera_pose`），采集、训练与 Habitat 测评共用 `configs/runtime.json` 中的这一姿态。原地转向时实际转速约为指令的 55%，随前进速度升至 96%（`evaluation.simulation.realtime.yaw_response`）。

## 7. 工程约定

- 神经网络部分使用 BF16，几何融合使用 FP32；推理时候选生成回放 CUDA graph。
- 每个概念在生产代码中只有一个实现；批处理路径对照的逐行参考实现在 `tests/reference_implementations.py`。
- 检查点保存配置与 policy/head 权重，按当前结构严格加载。
- 数据、检查点与结果放在 `data/` 与 `outputs/`。
