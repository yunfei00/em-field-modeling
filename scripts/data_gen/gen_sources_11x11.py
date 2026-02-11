#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate 11x11 near-field "source" on z=1mm plane:
columns: x y z hx_re hx_im hy_re hy_im

Grid:
x,y in [-5,5] mm with step 1 -> 11x11
z = 1 mm fixed

Coverage strategy (mixture of patterns):
- Multi-Gaussian blobs (random amplitude & phase)
- Sum of plane waves (random directions, spatial frequencies)
- Vortex / spiral phase pattern (topological charge)
- Sparse hotspots (few pixels with smooth kernel)
- Band-limited random complex field (low-pass in k-space)

Outputs:
- One CSV per sample in output_dir
- Also writes a manifest.json with generation parameters per sample (optional, useful for debugging)
"""

import os
import json
import math
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, List

import numpy as np


# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed: int):
    np.random.seed(seed)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def complex_from_amp_phase(amp: np.ndarray, phase: np.ndarray) -> np.ndarray:
    return amp * (np.cos(phase) + 1j * np.sin(phase))

def normalize_rms(z: np.ndarray, target_rms: float) -> np.ndarray:
    """Scale complex field so that RMS(|z|) == target_rms."""
    rms = np.sqrt(np.mean(np.abs(z) ** 2)) + 1e-12
    return z * (target_rms / rms)

def clip_dynamic_range(z: np.ndarray, max_ratio: float = 1e3) -> np.ndarray:
    """
    Optional: limit extreme peaks. max_ratio=1e3 means max(|z|) <= max_ratio * median(|z|)
    """
    mag = np.abs(z)
    med = np.median(mag) + 1e-12
    cap = max_ratio * med
    scale = np.minimum(1.0, cap / (mag + 1e-12))
    return z * scale


# -----------------------------
# Patterns
# -----------------------------

@dataclass
class SampleMeta:
    seed: int
    mode: str
    params: Dict

def pattern_gaussian_blobs(X: np.ndarray, Y: np.ndarray, rng: np.random.RandomState) -> Tuple[np.ndarray, Dict]:
    """
    Sum of complex Gaussian blobs with random centers, widths, amplitudes, and phases.
    """
    n_blobs = rng.randint(1, 6)  # 1..5
    field = np.zeros_like(X, dtype=np.complex128)

    params = {"n_blobs": int(n_blobs), "blobs": []}
    for _ in range(n_blobs):
        x0 = rng.uniform(-5, 5)
        y0 = rng.uniform(-5, 5)
        # widths in mm: small to moderate
        sx = rng.uniform(0.6, 2.8)
        sy = rng.uniform(0.6, 2.8)

        amp = rng.uniform(0.3, 1.2)
        phase0 = rng.uniform(-np.pi, np.pi)

        g = np.exp(-(((X - x0) ** 2) / (2 * sx ** 2) + ((Y - y0) ** 2) / (2 * sy ** 2)))
        contrib = complex_from_amp_phase(amp * g, phase0 + 0.0 * X)

        field += contrib
        params["blobs"].append({"x0": x0, "y0": y0, "sx": sx, "sy": sy, "amp": amp, "phase0": phase0})

    # Add mild random phase ripple sometimes
    if rng.rand() < 0.5:
        kx = rng.uniform(-0.8, 0.8)
        ky = rng.uniform(-0.8, 0.8)
        ripple = np.exp(1j * (kx * X + ky * Y))
        field *= ripple
        params["ripple"] = {"kx": kx, "ky": ky}

    return field, params

def pattern_plane_waves(X: np.ndarray, Y: np.ndarray, rng: np.random.RandomState) -> Tuple[np.ndarray, Dict]:
    """
    Sum of plane waves: exp(j*(kx x + ky y) + j phi)
    k in rad/mm (spatial frequency).
    """
    n_waves = rng.randint(2, 9)  # 2..8
    field = np.zeros_like(X, dtype=np.complex128)
    params = {"n_waves": int(n_waves), "waves": []}

    for _ in range(n_waves):
        theta = rng.uniform(0, 2*np.pi)
        # spatial frequency magnitude: keep within reasonable band on 11x11 grid
        k = rng.uniform(0.2, 2.2)  # rad/mm
        kx = k * np.cos(theta)
        ky = k * np.sin(theta)
        amp = rng.uniform(0.2, 1.0)
        phi = rng.uniform(-np.pi, np.pi)
        field += amp * np.exp(1j * (kx * X + ky * Y + phi))
        params["waves"].append({"theta": theta, "k": k, "amp": amp, "phi": phi})

    return field, params

def pattern_vortex(X: np.ndarray, Y: np.ndarray, rng: np.random.RandomState) -> Tuple[np.ndarray, Dict]:
    """
    Vortex-like phase: exp(j*m*atan2(y-y0,x-x0)) with Gaussian envelope.
    """
    x0 = rng.uniform(-2.5, 2.5)
    y0 = rng.uniform(-2.5, 2.5)
    m = rng.randint(-3, 4)  # -3..3
    if m == 0:
        m = 1
    sigma = rng.uniform(1.2, 3.8)
    amp0 = rng.uniform(0.5, 1.5)

    R2 = (X - x0)**2 + (Y - y0)**2
    envelope = np.exp(-R2 / (2 * sigma**2))
    angle = np.arctan2(Y - y0, X - x0)

    field = amp0 * envelope * np.exp(1j * (m * angle + rng.uniform(-np.pi, np.pi)))

    params = {"x0": x0, "y0": y0, "m": int(m), "sigma": sigma, "amp0": amp0}
    return field, params

def pattern_sparse_hotspots(X: np.ndarray, Y: np.ndarray, rng: np.random.RandomState) -> Tuple[np.ndarray, Dict]:
    """
    A few hotspot pixels blurred by a small Gaussian kernel to avoid unrealistically sharp impulses.
    """
    H = np.zeros_like(X, dtype=np.complex128)
    n = rng.randint(1, 6)  # 1..5 hotspots
    params = {"n_hotspots": int(n), "hotspots": []}

    # Grid indices
    xs = np.arange(X.shape[1])
    ys = np.arange(X.shape[0])

    for _ in range(n):
        ix = rng.choice(xs)
        iy = rng.choice(ys)
        amp = rng.uniform(0.6, 2.0)
        phi = rng.uniform(-np.pi, np.pi)
        H[iy, ix] += amp * np.exp(1j * phi)
        params["hotspots"].append({"ix": int(ix), "iy": int(iy), "amp": amp, "phi": phi})

    # Blur with Gaussian kernel in spatial domain
    sigma_px = rng.uniform(0.6, 1.3)
    ky, kx = np.mgrid[-5:6, -5:6]
    kernel = np.exp(-(kx**2 + ky**2) / (2 * sigma_px**2))
    kernel /= (np.sum(kernel) + 1e-12)

    # Convolution (small size -> direct)
    out = np.zeros_like(H)
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            acc = 0+0j
            for di in range(-5, 6):
                for dj in range(-5, 6):
                    ii = i + di
                    jj = j + dj
                    if 0 <= ii < H.shape[0] and 0 <= jj < H.shape[1]:
                        acc += H[ii, jj] * kernel[di+5, dj+5]
            out[i, j] = acc

    params["sigma_px"] = sigma_px
    return out, params

def pattern_bandlimited_random(X: np.ndarray, Y: np.ndarray, rng: np.random.RandomState) -> Tuple[np.ndarray, Dict]:
    """
    Generate complex random field with low-pass filtering in k-space (on 11x11).
    """
    ny, nx = X.shape
    # white complex noise
    noise = (rng.randn(ny, nx) + 1j * rng.randn(ny, nx))

    # FFT
    F = np.fft.fftshift(np.fft.fft2(noise))

    # k-space grid (normalized)
    ky = np.linspace(-1, 1, ny)
    kx = np.linspace(-1, 1, nx)
    KX, KY = np.meshgrid(kx, ky)
    KR = np.sqrt(KX**2 + KY**2)

    cutoff = rng.uniform(0.25, 0.65)  # keep low-mid spatial frequencies
    roll = rng.uniform(6, 14)         # steepness
    mask = 1.0 / (1.0 + np.exp((KR - cutoff) * roll))

    F2 = F * mask
    field = np.fft.ifft2(np.fft.ifftshift(F2))

    params = {"cutoff_norm": cutoff, "roll": roll}
    return field, params


# -----------------------------
# Source generator (Hx/Hy)
# -----------------------------

def generate_source_11x11(rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Returns:
      Hx, Hy: complex arrays [11,11]
      params: dict
    """

    # Grid in mm
    x = np.arange(-5, 6, 1, dtype=np.float64)  # 11 points
    y = np.arange(-5, 6, 1, dtype=np.float64)
    X, Y = np.meshgrid(x, y)  # [11,11] (y,x)

    # Choose a "mode" for base scalar complex field A(x,y)
    modes = [
        ("gaussian_blobs", 0.30),
        ("plane_waves",    0.25),
        ("vortex",         0.15),
        ("sparse_hotspots",0.15),
        ("bandlimited",    0.15),
    ]
    names = [m[0] for m in modes]
    probs = np.array([m[1] for m in modes], dtype=np.float64)
    probs /= probs.sum()

    mode = rng.choice(names, p=probs)

    if mode == "gaussian_blobs":
        A, p = pattern_gaussian_blobs(X, Y, rng)
    elif mode == "plane_waves":
        A, p = pattern_plane_waves(X, Y, rng)
    elif mode == "vortex":
        A, p = pattern_vortex(X, Y, rng)
    elif mode == "sparse_hotspots":
        A, p = pattern_sparse_hotspots(X, Y, rng)
    elif mode == "bandlimited":
        A, p = pattern_bandlimited_random(X, Y, rng)
    else:
        raise RuntimeError("Unknown mode")

    # Now map scalar complex A into vector (Hx, Hy) with random polarization mixing.
    # H = R * [A; A*exp(j*delta)] then rotate by angle psi
    psi = rng.uniform(0, 2*np.pi)              # polarization rotation
    delta = rng.uniform(-np.pi, np.pi)         # relative phase between components
    ratio = rng.uniform(0.3, 1.5)              # magnitude ratio between components

    A2 = ratio * A * np.exp(1j * delta)

    # rotation mixing
    Hx =  np.cos(psi) * A + np.sin(psi) * A2
    Hy = -np.sin(psi) * A + np.cos(psi) * A2

    # Add mild independent perturbation sometimes (break symmetry)
    if rng.rand() < 0.35:
        eps_level = rng.uniform(0.02, 0.08)
        Hx = Hx + eps_level * (rng.randn(*Hx.shape) + 1j*rng.randn(*Hx.shape))
        Hy = Hy + eps_level * (rng.randn(*Hy.shape) + 1j*rng.randn(*Hy.shape))

    # Normalize overall energy to a target RMS amplitude (helps training stability)
    target_rms = rng.uniform(0.2, 1.0)
    Hx = normalize_rms(Hx, target_rms)
    Hy = normalize_rms(Hy, target_rms)

    # Optional: clip extreme peaks (avoid rare crazy outliers)
    if rng.rand() < 0.2:
        Hx = clip_dynamic_range(Hx, max_ratio=1e3)
        Hy = clip_dynamic_range(Hy, max_ratio=1e3)

    params = {
        "mode": mode,
        "mode_params": p,
        "psi": float(psi),
        "delta": float(delta),
        "ratio": float(ratio),
        "target_rms": float(target_rms),
    }
    return Hx, Hy, params


def save_source_csv(filepath: str, Hx: np.ndarray, Hy: np.ndarray):
    """
    Save as rows:
    x y z hx_re hx_im hy_re hy_im
    where x,y in [-5..5] mm, z=1 mm.
    """
    x = np.arange(-5, 6, 1, dtype=np.float64)
    y = np.arange(-5, 6, 1, dtype=np.float64)
    z = 1.0

    rows = []
    # consistent ordering: y major then x (match meshgrid default)
    for iy, yy in enumerate(y):
        for ix, xx in enumerate(x):
            hx = Hx[iy, ix]
            hy = Hy[iy, ix]
            rows.append([xx, yy, z, hx.real, hx.imag, hy.real, hy.imag])

    arr = np.array(rows, dtype=np.float64)
    header = "x,y,z,hx_re,hx_im,hy_re,hy_im"
    np.savetxt(filepath, arr, delimiter=",", header=header, comments="")


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="sources_out", help="output directory")
    ap.add_argument("--n", type=int, default=1000, help="number of samples")
    ap.add_argument("--seed", type=int, default=1234, help="global seed")
    ap.add_argument("--write_manifest", action="store_true", help="write manifest.json")
    args = ap.parse_args()

    ensure_dir(args.out)
    set_seed(args.seed)

    manifest: List[Dict] = []

    for i in range(1, args.n + 1):
        # per-sample seed for reproducibility
        s = (args.seed * 1000003 + i * 9176) & 0x7fffffff
        rng = np.random.RandomState(s)

        Hx, Hy, params = generate_source_11x11(rng)

        fn = f"source_{i:06d}.csv"
        fp = os.path.join(args.out, fn)
        save_source_csv(fp, Hx, Hy)

        if args.write_manifest:
            manifest.append({
                "index": i,
                "file": fn,
                "seed": int(s),
                "gen": params,
            })

    if args.write_manifest:
        mf = os.path.join(args.out, "manifest.json")
        with open(mf, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Done. Generated {args.n} samples into: {args.out}")


if __name__ == "__main__":
    main()
