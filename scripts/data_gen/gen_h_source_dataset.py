#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
生成“合理的”磁场近场场源数据（复数 Hx, Hy），用于后续训练。
输出：每个样本一个CSV：x y z Hx_re Hx_im Hy_re Hy_im

网格：
- x, y: [-5, 5] mm, step=1 mm  => 11x11
- z: 固定 1 mm

场模型（启发式但很实用）：
- 叠加 K 个局部源（随机中心、随机尺度、随机幅相、随机方向耦合）
- 每个源贡献：A * exp(-(r^2)/(2*sigma^2)) * exp(-j*(k*r + phi0))
- 再用随机旋转/耦合矩阵生成 Hx, Hy，得到平滑且有结构的复数近场

你可以把它当成“合成数据生成器”，后续可替换为更物理的偶极子/电流片模型。
"""

import os
import math
import argparse
import numpy as np
import pandas as pd


def make_grid(x_min=-5, x_max=5, step=1, z_mm=1.0):
    xs = np.arange(x_min, x_max + step, step, dtype=np.float32)
    ys = np.arange(x_min, x_max + step, step, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys, indexing="xy")  # shape: (Ny, Nx)
    Z = np.full_like(X, float(z_mm), dtype=np.float32)
    return X, Y, Z


def complex_source_field(X, Y, rng, f_hz=5e9, c=299792458.0,
                         k_scale=1.0,
                         n_sources=(2, 6),
                         amp_range=(0.2, 1.0),
                         sigma_range_mm=(0.8, 3.0),
                         center_margin_mm=1.0,
                         phase0_range=(0.0, 2.0 * math.pi),
                         add_global_plane_wave_prob=0.3):
    """
    生成单个样本的 Hx, Hy（复数）。
    X,Y 单位 mm；内部会把距离换算为 m 用于相位项（可控）。
    """
    Ny, Nx = X.shape
    Hx = np.zeros((Ny, Nx), dtype=np.complex64)
    Hy = np.zeros((Ny, Nx), dtype=np.complex64)

    # 波数 k = 2*pi / lambda
    lam = c / f_hz
    k0 = 2.0 * math.pi / lam  # rad/m

    # 选择源个数
    K = rng.integers(n_sources[0], n_sources[1] + 1)

    # 允许源中心不跑到边界（更像真实：热点通常在内部）
    x_min, x_max = float(X.min()), float(X.max())
    y_min, y_max = float(Y.min()), float(Y.max())
    cx_min, cx_max = x_min + center_margin_mm, x_max - center_margin_mm
    cy_min, cy_max = y_min + center_margin_mm, y_max - center_margin_mm

    for _ in range(K):
        cx = rng.uniform(cx_min, cx_max)
        cy = rng.uniform(cy_min, cy_max)
        amp = rng.uniform(amp_range[0], amp_range[1])
        sigma = rng.uniform(sigma_range_mm[0], sigma_range_mm[1])
        phi0 = rng.uniform(phase0_range[0], phase0_range[1])

        # 源方向/耦合：用一个随机旋转角决定这个源主要投到 Hx 还是 Hy
        theta = rng.uniform(0.0, 2.0 * math.pi)
        mix = rng.uniform(0.2, 1.0)  # 交叉耦合强度（越大 Hx/Hy 越相关）

        dx = (X - cx)  # mm
        dy = (Y - cy)  # mm
        r_mm = np.sqrt(dx * dx + dy * dy) + 1e-6  # 避免0
        # 幅度包络：高斯衰减（mm尺度）
        env = np.exp(-(r_mm * r_mm) / (2.0 * sigma * sigma)).astype(np.float32)

        # 相位项：k*r（把 mm->m）
        r_m = (r_mm * 1e-3).astype(np.float32)
        phase = (k_scale * k0 * r_m + phi0).astype(np.float32)
        contrib = amp * env * np.exp(-1j * phase)

        # 生成该源对 Hx/Hy 的贡献：主分量 + 交叉耦合
        # 主方向：cos/sin(theta)
        hx_part = (np.cos(theta) * contrib + mix * np.sin(theta) * contrib).astype(np.complex64)
        hy_part = (np.sin(theta) * contrib - mix * np.cos(theta) * contrib).astype(np.complex64)

        Hx += hx_part
        Hy += hy_part

    # 可选：叠加一个弱的“全局平面波/渐变相位”，让数据更丰富
    if rng.uniform() < add_global_plane_wave_prob:
        # 平面波方向
        ang = rng.uniform(0.0, 2.0 * math.pi)
        kx = np.cos(ang)
        ky = np.sin(ang)
        # 平面波强度
        a = rng.uniform(0.05, 0.25)
        phi = rng.uniform(0.0, 2.0 * math.pi)
        # 相位：k*(kx*x + ky*y)
        # x,y(mm)->m
        phase_pw = (k_scale * k0 * ((kx * X + ky * Y) * 1e-3) + phi).astype(np.float32)
        pw = a * np.exp(-1j * phase_pw)
        Hx += pw.astype(np.complex64) * (0.7 + 0.3j)
        Hy += pw.astype(np.complex64) * (0.4 - 0.2j)

    # 归一化（可选）：让整体幅值落在更稳定范围
    # 你也可以注释掉，让网络自己学尺度
    peak = max(np.abs(Hx).max(), np.abs(Hy).max())
    if peak > 0:
        scale = rng.uniform(0.8, 1.2) / peak
        Hx *= scale
        Hy *= scale

    return Hx, Hy


def save_sample_csv(out_csv, X, Y, Z, Hx, Hy):
    df = pd.DataFrame({
        "x": X.reshape(-1),
        "y": Y.reshape(-1),
        "z": Z.reshape(-1),
        "Hx_re": np.real(Hx).reshape(-1),
        "Hx_im": np.imag(Hx).reshape(-1),
        "Hy_re": np.real(Hy).reshape(-1),
        "Hy_im": np.imag(Hy).reshape(-1),
    })
    df.to_csv(out_csv, index=False, float_format="%.6e")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="h_source_dataset", help="输出目录")
    ap.add_argument("--n_samples", type=int, default=200, help="样本数量")
    ap.add_argument("--seed", type=int, default=1234, help="随机种子")
    ap.add_argument("--freq_ghz", type=float, default=5.0, help="频率 GHz")
    ap.add_argument("--k_scale", type=float, default=1.0, help="相位项缩放（想更慢/更快变化可调）")
    ap.add_argument("--min_sources", type=int, default=2, help="每个样本最少源个数")
    ap.add_argument("--max_sources", type=int, default=6, help="每个样本最多源个数")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    X, Y, Z = make_grid(x_min=-5, x_max=5, step=1, z_mm=1.0)

    f_hz = args.freq_ghz * 1e9

    for i in range(args.n_samples):
        # 每个样本用不同子种子，保证可复现且多样
        sub_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))

        Hx, Hy = complex_source_field(
            X, Y, sub_rng,
            f_hz=f_hz,
            k_scale=args.k_scale,
            n_sources=(args.min_sources, args.max_sources),
            amp_range=(0.2, 1.0),
            sigma_range_mm=(0.8, 3.0),
            center_margin_mm=1.0,
            add_global_plane_wave_prob=0.35,
        )

        out_csv = os.path.join(args.out_dir, f"sample_{i:05d}.csv")
        save_sample_csv(out_csv, X, Y, Z, Hx, Hy)

    # 额外保存一个索引文件，方便训练时读取
    index_path = os.path.join(args.out_dir, "index.txt")
    with open(index_path, "w", encoding="utf-8") as f:
        for i in range(args.n_samples):
            f.write(f"sample_{i:05d}.csv\n")

    print(f"[OK] Generated {args.n_samples} samples in: {args.out_dir}")
    print(f"[OK] Index file: {index_path}")
    print("CSV columns: x y z Hx_re Hx_im Hy_re Hy_im")


if __name__ == "__main__":
    main()
