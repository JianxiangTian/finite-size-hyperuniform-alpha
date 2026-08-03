#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(cmd, cwd=ROOT):
    print(" ".join(str(c) for c in cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--skip-a", action="store_true")
    ap.add_argument("--skip-sk", action="store_true")
    ap.add_argument("--skip-nv", action="store_true")
    ap.add_argument("--skip-spreadability", action="store_true")
    args = ap.parse_args()

    config_path = (ROOT / args.config).resolve()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))

    if cfg.get("coords_are_reduced", True) is not True:
        raise ValueError(
            "This release accepts only reduced coordinates in [0, 1)^2. "
            "Set coords_are_reduced to true and convert absolute coordinates "
            "before running the workflow."
        )

    input_dir = ROOT / cfg["input_dir"]
    output_dir = ROOT / cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    if not (input_dir / "config_0_component_0.txt").exists():
        raise FileNotFoundError(f"missing input file: {input_dir / 'config_0_component_0.txt'}")

    auto_length_scale = bool(cfg.get("auto_length_scale_a", True))
    a_json = output_dir / "a_value.json"
    if auto_length_scale and not args.skip_a:
        run([sys.executable, "src/preprocess/compute_length_scale.py", "--config", str(config_path)])
        a_info = json.loads(a_json.read_text(encoding="utf-8"))
        length_scale_a = float(a_info["a"])
    elif auto_length_scale and args.skip_a and a_json.is_file():
        a_info = json.loads(a_json.read_text(encoding="utf-8"))
        length_scale_a = float(a_info["a"])
    else:
        configured_a = cfg.get("length_scale_a")
        if configured_a is None:
            raise ValueError(
                "length_scale_a is unavailable: either compute it without "
                "--skip-a, retain an existing a_value.json, or specify "
                "length_scale_a in config.json"
            )
        length_scale_a = float(configured_a)

    sk_exe = ROOT / "bin" / "compute_sk"
    nv_exe = ROOT / "bin" / "compute_number_variance"
    sk_exe.parent.mkdir(exist_ok=True)

    if not args.skip_sk:
        run(["g++", "-O3", "-std=c++11", "src/sk/compute_sk.cpp", "-o", str(sk_exe)])
        sk = cfg["sk"]
        run([
            str(sk_exe), str(input_dir), str(output_dir), str(cfg["num_configs"]), str(cfg["density"]),
            str(sk["nk"]), str(sk["kbin_factor"]), str(sk["max_bins"]), str(length_scale_a)
        ])

    nv_enabled = bool(cfg.get("number_variance", {}).get("enabled", True))
    if nv_enabled and not args.skip_nv:
        run(["g++", "-O3", "-std=c++11", "src/number_variance/compute_number_variance.cpp", "-o", str(nv_exe)])
        nv = cfg["number_variance"]
        run([
            str(nv_exe), str(input_dir), str(output_dir), str(cfg["num_configs"]), str(cfg["density"]),
            str(nv["rbin_fraction"]), str(nv["num_samples"]), str(nv["num_radii"]), str(nv["seed"]), str(length_scale_a)
        ])

    if not args.skip_spreadability:
        run([sys.executable, "src/spreadability/compute_spreadability.py", "--config", str(config_path)])

    run([sys.executable, "src/analysis/combined_alpha_analysis.py", "--config", str(config_path)])


if __name__ == "__main__":
    main()
