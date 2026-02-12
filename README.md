# Torch Training Template (v0.1.0)

A **production-ready PyTorch training template** for regression and multi-output tasks. It is designed for **long-term reuse** with experiment management, plugin registry, robust resume, metrics, AMP, scheduler, CI, and artifacts.

- 中文文档（Chinese README）: `README.zh-CN.md`

---

## What is this for?

### Good fit

- Regression / multi-output regression
- Forward / inverse modeling
- Physics / EM field / numerical learning
- Small to medium custom models (MLP, CNN, custom nets)

### Not intended

- NLP / LLM training
- Multi-node distributed training (not yet)

---

## Quick Start (30 seconds)

```bash
pip install -e .
python scripts/train.py --config configs/default.yaml --exp_name demo
```

Training outputs will be written to:

```text
runs/demo/<run_id>/
```


## Model Inference (Single / Batch by File/Folder)

Use `scripts/infer.py` to load a trained checkpoint and run inference for:

- one input file (`.npy` / `.npz`)
- or all input files under one folder (`.npy` / `.npz`)

### 1) Infer one file

```bash
python scripts/infer.py \
  --config configs/default.yaml \
  --ckpt runs/demo/<run_id>/best.pth \
  --input data/sample.npy \
  --input_shape 16 \
  --out outputs/sample_pred.npy
```

### 2) Infer all files in one folder

```bash
python scripts/infer.py \
  --config configs/default.yaml \
  --ckpt runs/demo/<run_id>/best.pth \
  --input data/infer_inputs/ \
  --input_shape 16 \
  --out outputs/preds/
```

- when `--input` is a folder, the script scans all `.npy/.npz` files and infers each file
- `--out` should be a folder in that case, and results are saved as `<input_stem>_pred.npy`
- if input is `.npz`, key `x` is preferred; otherwise the first array is used

The script prints input/output tensor shapes and previews the first few prediction rows for each file.

## Project Structure

```text
configs/                 # experiment configs (YAML)
scripts/
  train.py               # training entry
  eval.py                # evaluation entry
src/ttt/
  registry.py            # plugin registry (models / datasets)
  engine.py              # training loop
  metrics.py             # metrics & best selection
  scheduler.py           # LR schedulers
  checkpoint.py          # save/load checkpoints
  rng.py                 # RNG state for reproducible resume
runs/
  <exp>/<run_id>/        # one experiment run
    config.resolved.yaml
    metrics.jsonl
    last.pth
    best.pth
```

## Experiment Management

One experiment run corresponds to one directory:

```text
runs/<exp_name>/<run_id>/
```

Each run contains:

- `config.resolved.yaml` — exact configuration snapshot
- `metrics.jsonl` — structured training / evaluation logs
- `last.pth` — latest checkpoint
- `best.pth` — best checkpoint according to selected metric

## Resume Training

```bash
python scripts/train.py --config configs/default.yaml --exp_name demo --resume
```

Resume from the **best checkpoint** and set a new learning rate:

```bash
python scripts/train.py --config configs/default.yaml --exp_name demo --resume --resume_best --lr 1e-4
```

By default, when `--lr` is provided during resume, the optimizer state is restored from checkpoint **but LR is overwritten** by the latest config/CLI value.

If you want to force this behavior explicitly (even without `--lr`), use:

```bash
python scripts/train.py --config configs/default.yaml --exp_name demo --resume --resume_new_lr
```

If you also want to ignore old scheduler progress and rebuild scheduler state from current config:

```bash
python scripts/train.py --config configs/default.yaml --exp_name demo --resume --resume_reset_scheduler
```

Tip: You can pin a fixed experiment + run id in config to avoid typing `--run_id` each time:

```yaml
ckpt:
  dir: runs
  exp_name: fwd_v1_10w
  run_id: stable
```

Then always launch with:

```bash
python scripts/train.py --config <your_config>.yaml --resume
```

For this EM project, you can also use a short preset line:

```bash
python scripts/train.py --config <your_config>.yaml --line forward
python scripts/train.py --config <your_config>.yaml --line inverse
```

Default mapping:

- `forward` => `--exp_name em_forward --run_id main`
- `inverse` => `--exp_name em_inverse --run_id main`

You can customize these in YAML:

```yaml
ckpt:
  line_presets:
    forward:
      exp_name: em_forward
      run_id: main
    inverse:
      exp_name: em_inverse
      run_id: main
```

CLI priority remains: explicit `--exp_name/--run_id` > `--line` preset > config defaults.

Resume restores:

- model weights
- optimizer (optionally with LR overridden by latest config via `--resume_new_lr` / `--lr`)
- scheduler (can be skipped via `--resume_reset_scheduler`)
- AMP scaler
- RNG state

## Registry (Plugin System)

This template uses a registry-based plugin system. New models or datasets can be added without modifying core training code.

### Add a Model

```python
from ttt.registry import register_model

@register_model("my_model")
def build_my_model(cfg):
    ...
```

Config:

```yaml
model:
  name: my_model
```

### Add a Dataset

```python
from ttt.registry import register_dataset

@register_dataset("my_dataset")
def build_my_dataset(cfg):
    ...
```

Config:

```yaml
data:
  name: my_dataset
```

## Metrics & Best Model Selection

Supported metrics include:

- `rmse_mean`
- `rel_mean`
- per-output-dimension metrics

Config example:

```yaml
metrics:
  track: rmse_mean
  lower_is_better: true
```

The best checkpoint is selected automatically based on this metric.

## AMP (Mixed Precision)

Enable AMP in config:

```yaml
amp:
  enabled: true
```

Or via CLI override:

```bash
python scripts/train.py ... --amp
```

Notes:

- AMP is enabled only on CUDA
- AMP scaler state is checkpointed and restored on resume

## Learning Rate Scheduler

Supported schedulers:

- `step`
- `cosine`
- `onecycle`

Example:

```yaml
scheduler:
  name: cosine
  t_max: 50
```

Scheduler state is saved and restored on resume.

## CI & Artifacts

GitHub Actions automatically runs:

- smoke training
- resume training
- evaluation

The following artifacts are uploaded:

- `config.resolved.yaml`
- `metrics.jsonl`
- `last.pth`
- `best.pth`

> Note: GitHub mobile app may not display artifacts. Use “Open in browser” to download them.

## Helper Utilities

Find the latest run directory (for humans):

```bash
python scripts/find_latest_run.py runs demo
```

This helper is not used by CI.

## Versioning Policy

- `v0.x`: API stable, internal improvements allowed
- `v1.0`: feature-complete long-term base

Current version: `v0.1.0`

## License

MIT License (recommended)
