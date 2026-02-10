# Torch Training Template (v0.1.0)

A **production-ready PyTorch training template** for regression and multi-output tasks.
Designed for **long-term reuse** with experiment management, plugin registry,
robust resume, metrics, AMP, scheduler, CI and artifacts.

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

runs/demo/<run_id>/
Project Structure


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
Experiment Management
One experiment run corresponds to one directory:


runs/<exp_name>/<run_id>/
Each run contains:
config.resolved.yaml — exact configuration snapshot
metrics.jsonl — structured training / evaluation logs
last.pth — latest checkpoint
best.pth — best checkpoint according to selected metric


Resume Training
Bash

python scripts/train.py --config configs/default.yaml --exp_name demo --resume
Resume restores:
model weights
optimizer
scheduler
AMP scaler
RNG state
Registry (Plugin System)
This template uses a registry-based plugin system. New models or datasets can be added without modifying core training code.
Add a Model
Python

from ttt.registry import register_model

@register_model("my_model")
def build_my_model(cfg):
    ...
Config:
Yaml

model:
  name: my_model
Add a Dataset
Python

from ttt.registry import register_dataset

@register_dataset("my_dataset")
def build_my_dataset(cfg):
    ...
Config:
Yaml

data:
  name: my_dataset

Metrics & Best Model Selection
Supported metrics include:
rmse_mean
rel_mean
per-output-dimension metrics
Config example:
Yaml

metrics:
  track: rmse_mean
  lower_is_better: true
The best checkpoint is selected automatically based on this metric.
AMP (Mixed Precision)
Enable AMP in config:
Yaml

amp:
  enabled: true
Or via CLI override:
Bash

python scripts/train.py ... --amp
AMP is enabled only on CUDA
AMP scaler state is checkpointed and restored on resume
Learning Rate Scheduler
Supported schedulers:
step
cosine
onecycle
Example:
Yaml

scheduler:
  name: cosine
  t_max: 50
Scheduler state is saved and restored on resume.
CI & Artifacts
GitHub Actions automatically runs:
smoke training
resume training
evaluation
The following artifacts are uploaded:
config.resolved.yaml
metrics.jsonl
last.pth
best.pth
Note: GitHub mobile app may not display artifacts. Use “Open in browser” to download them.
Helper Utilities
Find the latest run directory (for humans):
Bash

python scripts/find_latest_run.py runs demo
This helper is not used by CI.
Versioning Policy
v0.x: API stable, internal improvements allowed
v1.0: feature-complete long-term base
Current version: v0.1.0
License
MIT License (recommended)