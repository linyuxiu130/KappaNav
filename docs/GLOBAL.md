# 全局模块：顶视图可通行性

顶视图地表语义模型：无人机俯视图（5–70 m 高度）、DOM 或卫星图，重采样到 0.1–0.2 m/像素，输出 9 类地表概率，并由此得到 0–255 的平滑可通行值和每米路线代价，供空地协同导航的全局路线使用。

## 类别与优先级

| 类别 | 内容 | 可通行值 (0–255) | 每米路线代价 |
|---|---|---|---|
| `road` | 一切铺装：车道、人行道、广场、停车场 | 255 | 1.0 |
| `dirt` | 土路、碎石、裸地 | 190 | 1.3 |
| `grass` | 草地 | 130 | 2.0 |
| `field` | 农田、葡萄园、翻耕地 | 70 | 3.6 |
| `tree` | 树冠（下方地面不可见） | 50 | 5.1 |
| `shrub` | 灌木、绿篱 | 25 | 10.2 |
| `obstacle` | 建筑、墙、栏杆、台阶、岩石 | 0 | 51 |
| `vehicle` | 车辆 | 0 | 51 |
| `water` | 水体 | 0 | 51 |

可通行值 = Σ 类别概率 × 该类的值（`taxonomy.VALUE`），随置信度平滑变化；每米路线代价 = 255 / 值。顶视图看不出路缘、台阶、沟和坡度，由局部策略处理。

## 多数据集监督

每个数据源的原生类别在 `taxonomy.SOURCE_CLASSES` 里映射为上面类别的一个**集合**：能细到一类的给一类，粗类给集合（如 `vegetation` → {灌木, 树冠}），没有信息的背景、未标注、行人给空集（不监督）。单类标签用交叉熵；集合标签只压低集合外的类别，不偏向集合内任何一类（见 `loss.py`）。

数据仓库 `/shibo_huang/data/globe/<source>/`：`meta.json`（传感器、来源、许可、原生类别表）、`manifest.jsonl`（每张图的划分、场景组、存储分辨率）、`images/`、`labels/`（原生类别 ID，从不按新类别改写）。由 `python -m kappanav.global_planner.prepare <source>` 生成。

## 模型

DINOv3-SAT ViT-L/16（四层特征）→ 同网格融合 → 按 log(GSD) 的零初始化缩放/平移 → RGB 细节分支 → 类别掩码上下文读出（9 类） → 局部 4× 上采样到原生像素。

训练裁片模拟真实输入：分辨率在 0.08–0.25 m 间对数均匀抽取（不细于数据源本身），边长 64–512 像素（低空小视场到高空大视场），同一批同一边长、像素总数固定；8 种旋转翻转与曝光、对比度、饱和度、色温、雾、伽马、模糊、噪声、JPEG 增强。低空无人机数据固定占一半裁片，航测与卫星占另一半；组内按 √(训练图数) 加权。

## 评测

每个数据源都参与训练，各留一部分测试（`Source.test_ids`，每次运行相同）：作者的验证 / 测试划分与训练不共享场景时直接采用；否则整段飞行 / 整片区域留出约 15%；场景组太少的数据源沿用作者划分或按图留出，结果里 `scene_disjoint=false`，分数偏乐观。`data.EXCLUDED` 里的数据源（灾区、无标注）不使用。

逐数据源整图评测（0.15 m）。选模型用 `value_error`：单类像素上 0–255 可通行值的类别均衡平均误差。同时报告每类的平均值 `class_value`、可行驶（road/dirt/grass）对阻挡（shrub/obstacle/vehicle/water）的 AUC、危险误判率 `false_passable`、漏判率和 mIoU。

## 使用

环境：`/opt/conda/envs/lerobot`（torch 2.7 cu128，transformers 4.57；p6000 与云节点镜像一致）。

```bash
export PYTHONPATH=src
python -m pytest -q tests/test_global_*.py
python -m kappanav.global_planner train --output OUT --steps 40000 --batch-size 32          # 单卡
torchrun --nproc-per-node 8 -m kappanav.global_planner train --output OUT --steps 40000   # 多卡
python -m kappanav.global_planner evaluate --checkpoint OUT/best.pt --output OUT/evaluation.json
python -m kappanav.global_planner infer map.tif --checkpoint OUT/best.pt --output OUT/map   # GeoTIFF 读分辨率；其他图像加 --gsd
```

`infer` 输出 `semantic.png`、`overlay.png`、`traversability.png`（0–255 灰度）、`traversability-heat.png`、`posterior.npz`，GeoTIFF 输入同时写出对应的 GeoTIFF 图层。

骨干权重：`KAPPANAV_GLOBAL_BACKBONE`（默认 `/shibo_huang/models/dinov3-sat`）。多卡训练（自动续训）：`bash scripts/global_planner/train.sh NAME --steps 20000 --batch-size 32 --workers 16`。当前模型：`/shibo_huang/outputs/globe/value/best.pt`。
