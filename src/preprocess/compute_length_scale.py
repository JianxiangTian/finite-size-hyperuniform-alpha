#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def sort_key(path: Path):
    m = re.search(r"config_(\d+)_component_0\.txt", path.name)
    return int(m.group(1)) if m else 10**12


def load_points(path: Path, density: float, reduced: bool):
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    pts = np.asarray(data[:, :2], dtype=float)
    n = pts.shape[0]
    L = float(np.sqrt(n / density))
    if reduced:
        pts = pts * L
    return pts % L, L


def mean_nearest_neighbor(points: np.ndarray, L: float) -> float:
    tree = cKDTree(points, boxsize=L)
    dist, _ = tree.query(points, k=2)
    return float(np.mean(dist[:, 1]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    cfg_path = Path(args.config).resolve()
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    input_dir = root / cfg["input_dir"]
    output_dir = root / cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("config_*_component_0.txt"), key=sort_key)
    if not files:
        raise FileNotFoundError(f"no configuration files found in {input_dir}")

    n_cfg = int(cfg.get("num_configs", len(files)))
    files = files[:n_cfg]

    density = float(cfg["density"])
    if cfg.get("coords_are_reduced", True) is not True:
        raise ValueError(
            "Only reduced coordinates in [0, 1)^2 are supported by this release."
        )
    reduced = True

    vals = []
    L_vals = []
    for i, fp in enumerate(files, 1):
        pts, L = load_points(fp, density, reduced)
        vals.append(mean_nearest_neighbor(pts, L))
        L_vals.append(L)
        print(f"a {i}/{len(files)}")

    a = float(np.mean(vals))
    L_avg = float(np.mean(L_vals))

    out = {
        "valid_configs": len(files),
        "density": density,
        "a": a,
        "average_L": L_avg,
    }

    (output_dir / "a_value.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    with open(output_dir / "a_value.txt", "w", encoding="utf-8") as f:
        f.write(f"density = {density}\n")
        f.write(f"valid_configs = {len(files)}\n")
        f.write(f"a = {a:.10f}\n")
        f.write(f"average_L = {L_avg:.10f}\n")

    print(f"a = {a:.10f}")


if __name__ == "__main__":
    main()
