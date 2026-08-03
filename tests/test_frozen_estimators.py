from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bootstrap = load_module(
    "paired_bootstrap_joint_estimators",
    "src/analysis/paired_bootstrap_joint_estimators.py",
)
combined = load_module(
    "combined_alpha_analysis",
    "src/analysis/combined_alpha_analysis.py",
)


class WeightingTests(unittest.TestCase):
    def test_inverse_variance_weight(self):
        weight = bootstrap.inverse_variance_sk_weight(1.0, 4.0)
        self.assertAlmostEqual(float(weight), 0.8)

    def test_inverse_variance_zero_variances(self):
        weight = bootstrap.inverse_variance_sk_weight(0.0, 0.0)
        self.assertAlmostEqual(float(weight), 0.5)

    def test_error_calibrated_weight(self):
        weight = bootstrap.error_calibrated_sk_weight(
            np.array([2.0]),
            np.array([1.0]),
            np.array([1.25]),
        )
        self.assertAlmostEqual(float(weight), 0.25)


class StructureFactorTests(unittest.TestCase):
    def test_shortest_window_has_boundary_sensitivity(self):
        x = np.geomspace(0.5, 2.0, 5)
        y = x**2 * np.array([1.00, 1.05, 0.96, 1.04, 0.98])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "SK_ensemble_ka.txt"
            np.savetxt(path, np.column_stack([x, y]))
            result = combined.analyze_sk(
                path,
                {
                    "sk_min_points": 5,
                    "sk_min_log10_span": 0.35,
                    "sk_eta": 1.0,
                    "sk_right_bound_mode": "all",
                },
            )
        self.assertEqual(result["n_points"], 5)
        self.assertGreater(result["Delta_alpha"], 0.0)


if __name__ == "__main__":
    unittest.main()
