#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np


def sort_key(path: str):
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", Path(path).name)]


def load_points(path: Path, L: float) -> np.ndarray:
    x = np.loadtxt(path)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return (x[:, :2].astype(float) * L) % L


def min_image(delta: np.ndarray, L: float) -> np.ndarray:
    return (delta + 0.5 * L) % L - 0.5 * L


def choose_grid(L: float, R: float, p: dict) -> int:
    dx = R / float(p["grid_pixels_per_radius"])
    n = int(np.ceil(L / dx))
    n = max(n, int(p["grid_min_n"]))
    q = int(p["grid_round_to"])
    if q > 1:
        n = int(np.ceil(n / q) * q)
    return min(n, int(p["grid_max_n"]))


def build_mask(points: np.ndarray, L: float, R: float, n: int) -> np.ndarray:
    dx = L / n
    centers = (np.arange(n, dtype=float) + 0.5) * dx
    rc = int(np.ceil(R / dx)) + 1
    R2 = R * R
    mask = np.zeros((n, n), dtype=bool)
    for px, py in points:
        cx = int(np.floor(px / dx))
        cy = int(np.floor(py / dx))
        ix = (np.arange(cx - rc, cx + rc + 1) % n).astype(int)
        iy = (np.arange(cy - rc, cy + rc + 1) % n).astype(int)
        dxs = min_image(centers[ix] - px, L)
        dys = min_image(centers[iy] - py, L)
        local = dxs[:, None] ** 2 + dys[None, :] ** 2 <= R2
        mask[np.ix_(ix, iy)] |= local
    return mask


def k2_grid(n: int, L: float) -> np.ndarray:
    dx = L / n
    kx = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    ky = 2.0 * np.pi * np.fft.rfftfreq(n, d=dx)
    kxg, kyg = np.meshgrid(kx, ky, indexing="ij")
    return kxg * kxg + kyg * kyg


def curve(mask: np.ndarray, k2: np.ndarray, D: float, t: np.ndarray) -> tuple[np.ndarray, float]:
    phase2 = mask.astype(np.float32)
    phase1 = 1.0 - phase2
    phi2 = float(np.mean(phase2))
    total = float(np.sum(phase2, dtype=np.float64))
    if not (0.0 < phi2 < 1.0):
        raise RuntimeError("invalid decorated area fraction")
    c_hat = np.fft.rfft2(phase2)
    out = np.empty(t.size, dtype=float)
    last = 0.0
    for i, ti in enumerate(t):
        dt = float(ti - last)
        if dt > 0:
            c_hat *= np.exp(-D * k2 * dt)
        ct = np.fft.irfft2(c_hat, s=phase2.shape).real
        spread = float(np.sum(ct * phase1, dtype=np.float64)) / total
        out[i] = (1.0 - phi2) - spread
        last = ti
    return out, phi2


def time_grid(L: float, R: float, p: dict) -> tuple[np.ndarray, np.ndarray, float]:
    D = float(p["diffusion_coefficient"])
    t_trust = (float(p["trust_length_fraction"]) * L) ** 2 / (4.0 * D)
    tau_trust = D * t_trust / (R * R)
    tau_max = float(p["raw_max_factor_over_trust"]) * tau_trust
    tau_min = float(p["tau_raw_min"])
    decades = np.log10(tau_max / tau_min)
    n = int(np.ceil(decades * int(p["n_time_per_decade"])))
    n = int(np.clip(n, int(p["n_time_min"]), int(p["n_time_max"])))
    tau = np.logspace(np.log10(tau_min), np.log10(tau_max), n)
    t = tau * R * R / D
    return t, tau, tau_trust


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    s = cfg["spreadability"]
    input_dir = Path(cfg["input_dir"])
    output_dir = Path(cfg["output_dir"]) / "routeA_pure_directdiffusion_dualfit_protocol"
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(str(input_dir / "config_*_component_0.txt")), key=sort_key)
    files = files[: int(cfg["num_configs"])]
    if not files:
        raise RuntimeError("no configuration files found")

    npts = np.loadtxt(files[0]).reshape(-1, 2).shape[0]
    rho = float(cfg["density"])
    L = np.sqrt(npts / rho)
    R = np.sqrt(float(s["phi2"]) / (np.pi * rho))
    grid_n = choose_grid(L, R, s)
    t, tau, tau_trust = time_grid(L, R, s)
    k2 = k2_grid(grid_n, L)
    D = float(s["diffusion_coefficient"])

    curves = []
    phi2 = []
    for i, fp in enumerate(files, 1):
        print(f"spreadability {i}/{len(files)}")
        pts = load_points(Path(fp), L)
        mask = build_mask(pts, L, R, grid_n)
        y, ph = curve(mask, k2, D, t)
        curves.append(y)
        phi2.append(ph)

    arr = np.asarray(curves, dtype=float)
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mean)

    out = output_dir / "sample_ensemble_spreadability.txt"
    np.savetxt(out, np.column_stack([t, tau, mean, std]), fmt="%.15e", delimiter="\t", header="t\ttau\tspreadability_mean\tspreadability_std")
    print(f"saved {out}")
    print(f"grid_n={grid_n} R={R:.12g} tau_trust={tau_trust:.12g} phi2_mean={np.mean(phi2):.12g}")


if __name__ == "__main__":
    main()
