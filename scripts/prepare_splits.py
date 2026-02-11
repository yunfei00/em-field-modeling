from __future__ import annotations
import argparse, random
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_dir", default="splits/forward_v1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    args = ap.parse_args()

    root = Path(args.data_root)
    ids = sorted([p.stem for p in root.glob("*.npz")])
    random.Random(args.seed).shuffle(ids)

    n = len(ids)
    n_test = int(n * args.test_ratio)
    n_val = int(n * args.val_ratio)
    test_ids = ids[:n_test]
    val_ids = ids[n_test:n_test+n_val]
    train_ids = ids[n_test+n_val:]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out/"train.txt").write_text("\n".join(train_ids), encoding="utf-8")
    (out/"val.txt").write_text("\n".join(val_ids), encoding="utf-8")
    (out/"test.txt").write_text("\n".join(test_ids), encoding="utf-8")

    # fixed_eval: 固定抽 64 条（回归对比用）
    fixed = train_ids[:64] if len(train_ids) >= 64 else train_ids
    (out/"fixed_eval.txt").write_text("\n".join(fixed), encoding="utf-8")

    print(f"total={n} train={len(train_ids)} val={len(val_ids)} test={len(test_ids)} fixed_eval={len(fixed)}")
    print(f"written to {out}")

if __name__ == "__main__":
    main()
