
# torch-training-template

A reusable PyTorch training template:
- config (YAML)
- checkpoint resume
- experiment directory under `runs/`
- minimal dataset/model plug-in structure

## Quickstart

```bash
pip install -r requirements.txt
python scripts/train.py --config configs/default.yaml
python scripts/train.py --config configs/default.yaml --resume
python scripts/eval.py --config configs/default.yaml --ckpt runs/demo/best.pth


## Handy Utilities

Find the latest experiment run directory:

```bash
python scripts/find_latest_run.py runs demo