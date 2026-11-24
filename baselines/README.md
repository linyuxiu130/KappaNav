# 基线规划器

这里是对照用的局部规划器，每个都作为独立的策略服务运行，由测评器 `kappanav.evaluation.benchmark.baselines` 启动。它们和 CurveNav 用同一套闭环：同样的任务和子目标、同样的相机、0.2 s 决策延迟、同一个 MPC、同样的车体与接触模型。差别只在"观测 → 轨迹"这一步。

```bash
bash scripts/evaluation/evaluate.sh --policy sandplanner --benchmark <任务表> --maps oracle --output <目录>
```

| `--policy` | 目录 | 输入 | 权重（`$KAPPANAV_BASELINE_WEIGHTS/`） | 来源 |
|---|---|---|---|---|
| `sandplanner` | `sandplanner/` | 深度 | `sandplanner/NoMax.pth` | [SanD-Planner](https://github.com/WangJinCheng1998/sandplanner) `415589e`，见 `sandplanner/UPSTREAM.md` |
| `iplanner` | `iplanner/` | 深度 | `iplanner/plannernet.pt` | [NavDP](https://github.com/InternRobotics/NavDP) `878740a` 的 `baselines/iplanner`（iPlanner 的官方权重） |
| `navdp` | `navdp/` | RGB + 深度 | `navdp/navdp_pretrain.ckpt` | NavDP `878740a` 的 `baselines/navdp` |
| `x-navdp` | `x-navdp/` | RGB + 深度 + 机器人状态 | `x-navdp/x-navdp_posttrain.ckpt` | NavDP `878740a` 的 `baselines/x-navdp`，见 `x-navdp/UPSTREAM.md` |

本地对上游代码的改动只涉及推理：统一的原始数组协议（`serving/protocol.py`）、可选 CUDA 设备、去掉可视化与静默回退。模型结构与权重加载保持上游原样。上游许可证随各目录保留；Depth Anything V2 / DINOv2 由 NavDP 与 X-NavDP 共用，在 `serving/vision/`，附其许可证。

协议：`POST /navigator_reset`（内参、停止阈值 -3、批大小 1、种子 1234），每回合开始调用一次；`POST /pointgoal_step`，发送 RGB `uint8 [1,H,W,3]`、深度 `float32 [1,H,W,1]`（米，相机 z，没有回波处为 inf）、水平车体系目标 `goal_x/goal_y`（x 向前，y 向左）、机器人状态 `state_data`；返回 NPZ 中的 `trajectory`，即同一车体系下的轨迹点。只用深度的规划器收到的是全黑 RGB。

运行环境：`$KAPPANAV_BASELINE_PYTHON`（`/shibo_huang/envs/curvenav-baselines`，在 `curvenav-pcard` 之上另装 diffusers 0.33.1、timm 1.0.15、cupy-cuda12x 13.6.0；CuPy 用 torch 自带的 NVRTC 12.8 编译内核，`evaluation.baselines` 启动服务时把它放在库路径最前：更早的 NVRTC 不支持 Blackwell（sm_120））。每次测评的 `inputs.json` 记录所用权重的路径与 SHA-256（`checkpoint`、`checkpoint_sha256`）。
