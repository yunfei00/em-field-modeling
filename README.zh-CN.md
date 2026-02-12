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
