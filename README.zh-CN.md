# Torch 训练模板（中文）

这是一个面向回归 / 多输出任务的 **PyTorch 训练模板**，适合长期复用：包含实验管理、插件注册、断点续训、指标追踪、AMP、学习率调度器与 CI 基础能力。

---

## 快速开始

```bash
pip install -e .
python scripts/train.py --config configs/default.yaml --exp_name demo
```

输出目录：

```text
runs/demo/<run_id>/
```

---

## 常用脚本

- Loss 可视化：`scripts/plot_loss.py`

- 训练：`scripts/train.py`
- 评估：`scripts/eval.py`
- 推理：`scripts/infer.py`

---

## 断点续训（重点）

基础续训（默认从 `last.pth`）：

```bash
python scripts/train.py --config configs/default.yaml --exp_name demo --resume
```

从最佳模型继续训练并设置新的学习率：

```bash
python scripts/train.py --config configs/default.yaml --exp_name demo --resume --resume_best --lr 1e-4
```

### 当前行为说明

- 续训时会恢复：模型权重、优化器状态、调度器状态（默认）、AMP scaler、随机数状态。
- 当你在续训命令里传入 `--lr` 时：
  - 仍会加载 checkpoint 的优化器历史状态（如动量）；
  - 但会把优化器学习率覆盖为你最新提供的 `--lr` / 配置值。

这可实现：**在原始最佳模型基础上继续训练，同时使用新的学习率**。

### 两个续训开关

1) `--resume_new_lr`

显式开启“续训后使用新学习率覆盖旧学习率”：

```bash
python scripts/train.py --config configs/default.yaml --exp_name demo --resume --resume_new_lr
```

2) `--resume_reset_scheduler`

续训时不加载旧调度器状态，按当前配置重建调度器：

```bash
python scripts/train.py --config configs/default.yaml --exp_name demo --resume --resume_reset_scheduler
```

---

## 学习率调度器

支持：

- `step`
- `cosine`
- `onecycle`

示例：

```yaml
scheduler:
  name: cosine
  t_max: 50
  eta_min: 1e-6
```

---


## 训练可视化（Loss 曲线）

本仓库训练会默认写入 `metrics.jsonl`（通用训练在 `runs/<exp>/<run_id>/metrics.jsonl`，forward 训练在 `runs/.../artifacts/metrics.jsonl`）。

可直接用脚本画曲线：

```bash
python scripts/plot_loss.py \
  --run_dir runs/demo/<run_id> \
  --smooth 20 \
  --out_png runs/demo/<run_id>/artifacts/loss_curve.png \
  --out_csv runs/demo/<run_id>/artifacts/loss_long.csv
```

多实验对比：

```bash
python scripts/plot_loss.py \
  --run_dir runs/exp_a/<run_id_a> runs/exp_b/<run_id_b> \
  --smooth 20 \
  --title "Loss Compare"
```

工业建议：
- 训练侧统一输出 JSONL（已支持），并在 CI/定时任务中自动生成 `loss_curve.png` + `loss_long.csv`。
- 线上监控将 `val_loss`、`lr`、`best_score` 接入告警（如 3 个 epoch 不提升触发告警）。
- 评估时固定对比窗口（如最近 5 次 run）并保留图与原始 CSV，保证可追溯。

---

## 项目结构（简版）

```text
configs/
scripts/
src/ttt/
runs/<exp>/<run_id>/
  config.resolved.yaml
  metrics.jsonl
  last.pth
  best.pth
```

---

## 说明

英文完整说明请看 `README.md`。


---

## EM 正向任务：E/H 量级不平衡建议


### 评估与推理（forward 专用）

新增两个专用入口，便于直接看当前模型效果：

```bash
# 评估（输出 val_mse、12 通道 RMSE、E/H 均值 RMSE）
python -m emfm.tasks.forward.eval \
  --config configs/forward/forward_train.yaml \
  --ckpt runs/forward_norm/best.pth

# 推理（单文件或目录；可导出 tensor 或 target_E/H.csv）
python -m emfm.tasks.forward.infer \
  --ckpt runs/forward_norm/best.pth \
  --input <source_H.csv|.npy|.npz|folder> \
  --out <out_path_or_dir> \
  --out_format tensor
```


### 训练入口选择（`scripts/train.py` vs `emfm.tasks.forward.train`）

如果你的目标是 **EM 正向任务（输入 source H，输出 12 通道 E/H 场）**，推荐使用
`python -m emfm.tasks.forward.train`，而不是通用入口 `scripts/train.py`。

原因：

- `scripts/train.py` 走的是通用 `ttt.engine`，训练损失是原始 `mean((pred - y)^2)`，不会对 E/H 的通道量级差做显式处理。
- `emfm.tasks.forward.train` 支持 `--normalize_y`：先估计目标通道 `mean/std`，在归一化空间计算 loss，通常能显著缓解 “E 很准、H 偏差较大” 的问题。

经验建议：

- 若你以 **整体场重建质量**（尤其关注 H 不被 E 的大幅值淹没）为目标，优先选 `emfm.tasks.forward.train --normalize_y`。
- 若只是做通用框架连通性验证/CI 冒烟测试，`scripts/train.py` 更轻量。

当电场通道量级远大于磁场通道时，直接用原始 MSE 往往会更偏向优化 `E`，导致 `H` 拟合不足。

建议优先使用 `src/emfm/tasks/forward/train.py` 的目标归一化（请用 `python -m` 方式执行，避免直接运行文件时相对导入问题）：

```bash
python -m emfm.tasks.forward.train \
  --data_root <data_root> \
  --train_ids <train_ids.txt> \
  --val_ids <val_ids.txt> \
  --run_dir runs/forward_norm \
  --normalize_y --norm_max_batches 128
```

也支持通过 YAML 加载参数（CLI 参数优先级更高）：

```bash
python -m emfm.tasks.forward.train --config configs/forward/forward_train.yaml
```

`configs/forward/forward_train.yaml` 示例：

```yaml
seed: 42
device: auto

data:
  name: em_forward
  data_root: data/em
  train_ids: data/splits/train_ids.txt
  val_ids: data/splits/val_ids.txt

model:
  name: forward_unet_lite
  in_ch: 4
  out_ch: 12

train:
  epochs: 50
  batch_size: 16
  num_workers: 4
  resume: false

optim:
  name: adamw
  lr: 1.0e-3
  weight_decay: 0.0

loss:
  name: weighted_mse
  normalize_y: true
  norm_max_batches: 128
  norm_eps: 1.0e-6

  # 可选兼容旧方案
  auto_channel_weight: false
  auto_weight_max_batches: 64
  e_weight_multiplier: 1.0
  h_weight_multiplier: 1.0

ckpt:
  run_dir: runs/forward_norm
```

参数说明：

- `--normalize_y`：按通道估计训练集 `mean/std`，在归一化空间计算 loss。
- `--balance_eh_loss`：把 loss 拆成 `E(6通道)` 与 `H(6通道)` 两组后再加权平均，避免 E 振幅更大时“天然主导”总损失。
- `--eh_loss_e_weight` / `--eh_loss_h_weight`：E/H 组损失权重；若当前 H 偏差更大，建议从 `1.0/2.0` 或 `1.0/3.0` 开始网格搜索。
- `--norm_max_batches`：统计量估计使用的 batch 数。
- `--norm_eps`：方差/标准差下限，避免数值问题。

推荐组合（优先）：`--normalize_y --balance_eh_loss`。

产物：

- `artifacts/y_norm_stats.json`：保存归一化统计量，便于复现实验。

兼容保留旧方案：`--auto_channel_weight`（含 `--h_weight_multiplier` / `--e_weight_multiplier`）。


### 细节还不够好时的实战调参顺序（推荐）

如果你感觉“整体轮廓对了，但局部细节发糊/偏差明显”，建议按下面顺序做（每次只改 1~2 个因素，方便定位）：

1. **先拉长训练并降低后期学习率**
   - 把 `train.epochs` 从 `50` 提到 `100~200`；
   - 把 `optim.lr` 从 `1e-3` 降到 `3e-4` 或 `1e-4`（尤其在已基本收敛后做微调）。

2. **提高 H 组损失权重（若 H 细节更差）**
   - 在保持 `balance_eh_loss: true` 的前提下，尝试：
     - `eh_loss_e_weight: 1.0`
     - `eh_loss_h_weight: 2.0`（再试到 `3.0`）

3. **提高归一化统计稳定性**
   - `norm_max_batches` 从 `128` 增加到 `256` 或 `512`，让通道统计更稳定。

4. **减小 batch size 以改善泛化细节（视显存）**
   - 如从 `16` 降到 `8`，常能换来更细致的局部拟合（训练噪声会略增，可配合更小 lr）。

5. **固定验证集并对比可视化样本**
   - 每次实验都保存同一批样本的推理图/误差图，避免“指标涨了但细节观感反而退化”。

可直接从这个命令起步（强调细节的常用组合）：

```bash
python -m emfm.tasks.forward.train \
  --config configs/forward/forward_train.yaml \
  --normalize_y \
  --balance_eh_loss \
  --eh_loss_e_weight 1.0 \
  --eh_loss_h_weight 2.0
```


### 50w 数据的 train/val/test 切分模板（推荐）

目标：既保证不同训练阶段结果可比，又保证最终评估稳定。

建议把数据分成三部分：

- `train_pool`：490,000（训练池）
- `val_fixed`：1,000（固定验证集，跨阶段对比用）
- `test_final`：9,000（最终测试集，仅在里程碑模型上评估）

为什么这样分：

- `val_fixed` 固定为 1k，方便横向比较 `1w -> 5w -> 10w -> 20w -> 50w` 各阶段，不会因验证集变化而“看起来涨跌”。
- `test_final` 与验证集隔离，避免长期调参“磨到验证集”导致乐观偏差。

分阶段训练可直接从 `train_pool` 取前 N（或按固定随机种子采样 N）：

- Stage-1: `train_pool` 取 10,000
- Stage-2: `train_pool` 取 50,000
- Stage-3: `train_pool` 取 100,000
- Stage-4: `train_pool` 取 200,000
- Stage-5: `train_pool` 取 490,000

评估建议：

- 每个 stage 都在同一份 `val_fixed` 上打分（主比较指标）。
- 只在关键里程碑（如 Stage-3/5 最优模型）上跑 `test_final`，防止过于频繁查看测试集。

落地细节：

1. 切分前先去重（至少按样本 ID / 参数组合去重）。
2. 若样本存在“场景族”（例如同几何参数的小扰动），按组切分，避免泄漏。
3. 固定随机种子并保存三份 ID 文件：
   - `data/splits/train_pool_ids.txt`
   - `data/splits/val_fixed_ids.txt`
   - `data/splits/test_final_ids.txt`
4. 每次实验额外保存同一批可视化样本，保证“指标与观感”同步对比。
