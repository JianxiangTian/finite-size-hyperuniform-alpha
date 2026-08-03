#!/usr/bin/env python3
"""
Paired repetition-level bootstrap for the finite-size HU exponent analysis.

The resampling unit is one independent run.  Within each prescribed
alpha_reference, alpha_SK and alpha_SP are always resampled with the same run
indices.  Number-variance columns, if present, are ignored.

Main outputs
------------
1. paired_bootstrap_summary.csv
   Point estimates, sample SDs, and percentile bootstrap confidence intervals
   for alpha_SK, alpha_SP, the equal-weight joint estimator, d_method, the
   inverse-variance estimator, and the retrospective error-calibrated
   estimator.
2. run_level_with_joint_estimators.csv
   The input rows plus run-level equal-weight, d_method, inverse-variance, and
   retrospective error-calibrated quantities.
3. weighting_summary.csv
   Compact summary of the default, uncertainty-driven, and error-calibrated
   weights.
4. estimator_performance.csv
   Benchmark MAE/RMSE for the two routes and the three combined estimators.
5. weight_scan.csv
   Global fixed-weight sensitivity scan from all-SP to all-SK.
6. paired_bootstrap_replicates.csv.gz
   Optional bootstrap replicates, enabled with --save-replicates.

Definitions
-----------
alpha_joint_equal^(m) = [alpha_SK^(m) + alpha_SP^(m)] / 2
d_method^(m)          = |alpha_SK^(m) - alpha_SP^(m)| / 2

For inverse-variance weighting, the SK weight is

    w_IV = s_SK^(-2) / [s_SK^(-2) + s_SP^(-2)]
         = s_SP^2 / [s_SK^2 + s_SP^2].

The inverse-variance weight is re-estimated inside every bootstrap resample.

The optional error-calibrated weight minimizes the mean squared relative error
over all benchmark alpha_reference values.  Because it uses known target
values, it is a retrospective comparison and not the default estimator.  Its
weight is also re-estimated inside every bootstrap resample.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd


DEFAULT_BOOTSTRAPS = 10_000
DEFAULT_SEED = 20_260_730
DEFAULT_EXPECTED_REPS = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a paired, repetition-level bootstrap of alpha_SK and alpha_SP "
            "for every prescribed alpha."
        )
    )
    parser.add_argument("input_csv", type=Path, help="Run-level input CSV.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paired_bootstrap_results"),
        help="Output directory (default: paired_bootstrap_results).",
    )
    parser.add_argument(
        "--alpha-col",
        default="alpha_reference",
        help="Column containing the prescribed/theoretical alpha.",
    )
    parser.add_argument(
        "--run-col",
        default="run",
        help="Column containing the paired run identifier.",
    )
    parser.add_argument(
        "--sk-col",
        default="alpha_SK",
        help="Column containing the S(k) exponent estimate.",
    )
    parser.add_argument(
        "--sp-col",
        default="alpha_SP",
        help="Column containing the plateau-based spreadability estimate.",
    )
    parser.add_argument(
        "--bootstraps",
        type=int,
        default=DEFAULT_BOOTSTRAPS,
        help=f"Number of bootstrap resamples (default: {DEFAULT_BOOTSTRAPS}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Two-sided confidence level (default: 0.95).",
    )
    parser.add_argument(
        "--expected-reps",
        type=int,
        default=DEFAULT_EXPECTED_REPS,
        help=(
            "Required paired repetitions per alpha; use 0 to disable the "
            f"check (default: {DEFAULT_EXPECTED_REPS})."
        ),
    )
    parser.add_argument(
        "--run-first",
        type=int,
        default=None,
        help="Optional first numeric run index to retain, inclusive.",
    )
    parser.add_argument(
        "--run-last",
        type=int,
        default=None,
        help="Optional last numeric run index to retain, inclusive.",
    )
    parser.add_argument(
        "--save-replicates",
        action="store_true",
        help="Also save all bootstrap replicates as a gzip-compressed CSV.",
    )
    parser.add_argument(
        "--weight-grid-points",
        type=int,
        default=1001,
        help=(
            "Number of equally spaced SK weights in [0, 1] for the fixed-weight "
            "sensitivity scan (default: 1001)."
        ),
    )
    return parser.parse_args()


def run_number(value: object) -> int:
    """Extract a numeric run index from values such as 'run-001' or 1."""
    if pd.isna(value):
        raise ValueError("A run identifier is missing.")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return int(value)
    match = re.search(r"(\d+)$", str(value).strip())
    if match is None:
        raise ValueError(f"Cannot extract a numeric run index from {value!r}.")
    return int(match.group(1))


def sample_sd(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Sample standard deviation (ddof=1)."""
    return np.std(values, axis=axis, ddof=1)


def percentile_interval(
    values: np.ndarray, confidence: float
) -> tuple[float, float]:
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(values, [tail, 1.0 - tail])
    return float(low), float(high)


def inverse_variance_sk_weight(
    sk_variance: np.ndarray | float,
    sp_variance: np.ndarray | float,
) -> np.ndarray:
    """
    Return the normalized inverse-variance weight assigned to alpha_SK.

    The algebraically equivalent variance-ratio form is numerically stable and
    naturally handles one exactly zero variance.  If both are exactly zero,
    equal weighting is used.
    """
    sk_variance = np.asarray(sk_variance, dtype=float)
    sp_variance = np.asarray(sp_variance, dtype=float)
    denominator = sk_variance + sp_variance
    return np.divide(
        sp_variance,
        denominator,
        out=np.full(np.broadcast_shapes(sk_variance.shape, sp_variance.shape), 0.5),
        where=denominator > 0.0,
    )


def error_calibrated_sk_weight(
    sk_means: np.ndarray,
    sp_means: np.ndarray,
    targets: np.ndarray,
    axis: int | None = None,
) -> np.ndarray:
    """
    Minimize mean squared relative benchmark error over one weight in [0, 1].

    The prediction is sp + w * (sk - sp).  With
        a = (sk - sp) / target
        b = (sp - target) / target,
    the unconstrained least-squares minimizer is -sum(a*b) / sum(a*a).
    """
    sk_means = np.asarray(sk_means, dtype=float)
    sp_means = np.asarray(sp_means, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if np.any(targets == 0.0):
        raise ValueError(
            "Relative-error calibration is undefined when alpha_reference is zero."
        )
    a = (sk_means - sp_means) / targets
    b = (sp_means - targets) / targets
    numerator = -np.sum(a * b, axis=axis)
    denominator = np.sum(a * a, axis=axis)
    weight = np.divide(
        numerator,
        denominator,
        out=np.full_like(np.asarray(numerator, dtype=float), 0.5),
        where=denominator > 0.0,
    )
    return np.clip(weight, 0.0, 1.0)


def add_basic_summary(
    row: dict[str, float | int],
    prefix: str,
    observed: np.ndarray,
    boot_means: np.ndarray,
    target: float,
    confidence: float,
) -> None:
    """Add mean, SD, SE, bias, absolute error, and bootstrap CI fields."""
    mean = float(np.mean(observed))
    sd = float(sample_sd(observed))
    ci_low, ci_high = percentile_interval(boot_means, confidence)
    row[f"{prefix}_mean"] = mean
    row[f"{prefix}_sd"] = sd
    row[f"{prefix}_se"] = sd / np.sqrt(observed.size)
    row[f"{prefix}_bias"] = mean - target
    row[f"{prefix}_abs_error"] = abs(mean - target)
    row[f"{prefix}_boot_ci_low"] = ci_low
    row[f"{prefix}_boot_ci_high"] = ci_high
    row[f"{prefix}_boot_ci_width"] = ci_high - ci_low
    row[f"{prefix}_theory_in_ci"] = bool(ci_low <= target <= ci_high)


def prepare_input(args: argparse.Namespace) -> pd.DataFrame:
    if not args.input_csv.is_file():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")
    data = pd.read_csv(args.input_csv)
    required = [args.alpha_col, args.run_col, args.sk_col, args.sp_col]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(
            "Missing required column(s): "
            + ", ".join(missing)
            + f". Available columns: {list(data.columns)}"
        )

    data = data.copy()
    data["_run_number"] = data[args.run_col].map(run_number)
    if args.run_first is not None:
        data = data.loc[data["_run_number"] >= args.run_first].copy()
    if args.run_last is not None:
        data = data.loc[data["_run_number"] <= args.run_last].copy()
    if data.empty:
        raise ValueError("No rows remain after applying the run filters.")

    for column in (args.alpha_col, args.sk_col, args.sp_col):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    finite = np.isfinite(
        data[[args.alpha_col, args.sk_col, args.sp_col]].to_numpy(dtype=float)
    ).all(axis=1)
    if not finite.all():
        bad = data.loc[~finite, required]
        raise ValueError(
            "Non-finite alpha_reference/alpha_SK/alpha_SP values found:\n"
            + bad.to_string(index=False)
        )

    duplicate = data.duplicated([args.alpha_col, args.run_col], keep=False)
    if duplicate.any():
        bad = data.loc[duplicate, [args.alpha_col, args.run_col]]
        raise ValueError(
            "Duplicate alpha/run pairs found:\n" + bad.to_string(index=False)
        )

    counts = data.groupby(args.alpha_col, sort=True).size()
    if args.expected_reps > 0 and not (counts == args.expected_reps).all():
        raise ValueError(
            f"Expected {args.expected_reps} paired repetitions for every alpha, "
            f"but found:\n{counts.to_string()}"
        )
    if (counts < 2).any():
        raise ValueError("At least two paired repetitions are required per alpha.")

    return data.sort_values(
        [args.alpha_col, "_run_number"], kind="mergesort"
    ).reset_index(drop=True)


def main() -> int:
    args = parse_args()
    if args.bootstraps < 1:
        raise ValueError("--bootstraps must be at least 1.")
    if not 0.0 < args.confidence < 1.0:
        raise ValueError("--confidence must lie strictly between 0 and 1.")
    if args.weight_grid_points < 2:
        raise ValueError("--weight-grid-points must be at least 2.")
    if (
        args.run_first is not None
        and args.run_last is not None
        and args.run_first > args.run_last
    ):
        raise ValueError("--run-first cannot exceed --run-last.")

    data = prepare_input(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    alpha_values = np.array(
        sorted(data[args.alpha_col].unique()), dtype=float
    )
    n_alpha = alpha_values.size
    b_count = args.bootstraps

    # Store route-mean replicates for the global error-calibrated comparison.
    sk_boot_by_alpha = np.empty((b_count, n_alpha), dtype=float)
    sp_boot_by_alpha = np.empty((b_count, n_alpha), dtype=float)
    per_alpha: list[dict[str, object]] = []

    for alpha_index, alpha_reference in enumerate(alpha_values):
        group = data.loc[
            data[args.alpha_col] == alpha_reference
        ].copy()
        sk = group[args.sk_col].to_numpy(dtype=float)
        sp = group[args.sp_col].to_numpy(dtype=float)
        n_rep = sk.size

        # One index matrix is used for both routes: this is the paired bootstrap.
        indices = rng.integers(0, n_rep, size=(b_count, n_rep))
        sk_resampled = sk[indices]
        sp_resampled = sp[indices]
        sk_boot = sk_resampled.mean(axis=1)
        sp_boot = sp_resampled.mean(axis=1)
        sk_boot_by_alpha[:, alpha_index] = sk_boot
        sp_boot_by_alpha[:, alpha_index] = sp_boot

        equal_run = 0.5 * (sk + sp)
        d_method_run = 0.5 * np.abs(sk - sp)
        equal_boot = 0.5 * (sk_boot + sp_boot)
        d_method_boot = d_method_run[indices].mean(axis=1)

        sk_var = float(np.var(sk, ddof=1))
        sp_var = float(np.var(sp, ddof=1))
        iv_weight = float(inverse_variance_sk_weight(sk_var, sp_var))
        iv_point = iv_weight * float(np.mean(sk)) + (
            1.0 - iv_weight
        ) * float(np.mean(sp))

        # Re-estimate both variances and the normalized weight in every resample.
        sk_var_boot = np.var(sk_resampled, axis=1, ddof=1)
        sp_var_boot = np.var(sp_resampled, axis=1, ddof=1)
        iv_weight_boot = inverse_variance_sk_weight(
            sk_var_boot, sp_var_boot
        )
        iv_boot = iv_weight_boot * sk_boot + (
            1.0 - iv_weight_boot
        ) * sp_boot

        per_alpha.append(
            {
                "alpha_reference": float(alpha_reference),
                "group_index": group.index.to_numpy(),
                "sk": sk,
                "sp": sp,
                "equal_run": equal_run,
                "d_method_run": d_method_run,
                "sk_boot": sk_boot,
                "sp_boot": sp_boot,
                "equal_boot": equal_boot,
                "d_method_boot": d_method_boot,
                "iv_weight": iv_weight,
                "iv_weight_boot": iv_weight_boot,
                "iv_point": iv_point,
                "iv_boot": iv_boot,
            }
        )

    # Retrospective, benchmark-error-calibrated weight.  Axis 1 is the set of
    # prescribed alpha values; each row is one global bootstrap realization.
    sk_means = np.array([np.mean(item["sk"]) for item in per_alpha])
    sp_means = np.array([np.mean(item["sp"]) for item in per_alpha])
    error_weight = float(
        error_calibrated_sk_weight(sk_means, sp_means, alpha_values)
    )
    error_weight_boot = error_calibrated_sk_weight(
        sk_boot_by_alpha,
        sp_boot_by_alpha,
        alpha_values[np.newaxis, :],
        axis=1,
    )
    error_weight_ci = percentile_interval(error_weight_boot, args.confidence)

    summary_rows: list[dict[str, float | int | bool]] = []
    replicate_frames: list[pd.DataFrame] = []
    augmented = data.copy()

    for alpha_index, item in enumerate(per_alpha):
        target = float(item["alpha_reference"])
        sk = np.asarray(item["sk"], dtype=float)
        sp = np.asarray(item["sp"], dtype=float)
        equal_run = np.asarray(item["equal_run"], dtype=float)
        d_method_run = np.asarray(item["d_method_run"], dtype=float)
        sk_boot = np.asarray(item["sk_boot"], dtype=float)
        sp_boot = np.asarray(item["sp_boot"], dtype=float)
        equal_boot = np.asarray(item["equal_boot"], dtype=float)
        d_method_boot = np.asarray(item["d_method_boot"], dtype=float)
        iv_weight = float(item["iv_weight"])
        iv_weight_boot = np.asarray(item["iv_weight_boot"], dtype=float)
        iv_point = float(item["iv_point"])
        iv_boot = np.asarray(item["iv_boot"], dtype=float)

        error_point = (
            error_weight * float(np.mean(sk))
            + (1.0 - error_weight) * float(np.mean(sp))
        )
        error_boot = (
            error_weight_boot * sk_boot
            + (1.0 - error_weight_boot) * sp_boot
        )

        row: dict[str, float | int | bool] = {
            "alpha_reference": target,
            "n_rep": int(sk.size),
            "n_bootstrap": int(b_count),
            "bootstrap_seed": int(args.seed),
            "confidence_level": float(args.confidence),
        }
        add_basic_summary(
            row, "alpha_SK", sk, sk_boot, target, args.confidence
        )
        add_basic_summary(
            row, "alpha_SP", sp, sp_boot, target, args.confidence
        )
        add_basic_summary(
            row,
            "alpha_joint_equal",
            equal_run,
            equal_boot,
            target,
            args.confidence,
        )

        d_ci_low, d_ci_high = percentile_interval(
            d_method_boot, args.confidence
        )
        row["d_method_mean"] = float(np.mean(d_method_run))
        row["d_method_sd"] = float(sample_sd(d_method_run))
        row["d_method_se"] = float(
            sample_sd(d_method_run) / np.sqrt(d_method_run.size)
        )
        row["d_method_boot_ci_low"] = d_ci_low
        row["d_method_boot_ci_high"] = d_ci_high
        row["d_method_boot_ci_width"] = d_ci_high - d_ci_low
        row["d_method_from_route_means"] = (
            0.5 * abs(float(np.mean(sk)) - float(np.mean(sp)))
        )
        row["SK_SP_pearson_r"] = float(np.corrcoef(sk, sp)[0, 1])

        iv_ci_low, iv_ci_high = percentile_interval(
            iv_boot, args.confidence
        )
        iv_w_ci_low, iv_w_ci_high = percentile_interval(
            iv_weight_boot, args.confidence
        )
        row["w_IV_SK"] = iv_weight
        row["w_IV_SP"] = 1.0 - iv_weight
        row["w_IV_SK_boot_ci_low"] = iv_w_ci_low
        row["w_IV_SK_boot_ci_high"] = iv_w_ci_high
        row["alpha_joint_IV"] = iv_point
        row["alpha_joint_IV_bias"] = iv_point - target
        row["alpha_joint_IV_abs_error"] = abs(iv_point - target)
        row["alpha_joint_IV_boot_ci_low"] = iv_ci_low
        row["alpha_joint_IV_boot_ci_high"] = iv_ci_high
        row["alpha_joint_IV_boot_ci_width"] = iv_ci_high - iv_ci_low
        row["alpha_joint_IV_theory_in_ci"] = bool(
            iv_ci_low <= target <= iv_ci_high
        )

        error_ci_low, error_ci_high = percentile_interval(
            error_boot, args.confidence
        )
        row["w_error_calibrated_SK"] = error_weight
        row["w_error_calibrated_SP"] = 1.0 - error_weight
        row["w_error_calibrated_SK_boot_ci_low"] = error_weight_ci[0]
        row["w_error_calibrated_SK_boot_ci_high"] = error_weight_ci[1]
        row["alpha_joint_error_calibrated"] = error_point
        row["alpha_joint_error_calibrated_bias"] = error_point - target
        row["alpha_joint_error_calibrated_abs_error"] = abs(
            error_point - target
        )
        row["alpha_joint_error_calibrated_boot_ci_low"] = error_ci_low
        row["alpha_joint_error_calibrated_boot_ci_high"] = error_ci_high
        row["alpha_joint_error_calibrated_boot_ci_width"] = (
            error_ci_high - error_ci_low
        )
        row["alpha_joint_error_calibrated_theory_in_ci"] = bool(
            error_ci_low <= target <= error_ci_high
        )
        summary_rows.append(row)

        group_index = np.asarray(item["group_index"], dtype=int)
        augmented.loc[group_index, "alpha_joint_equal"] = equal_run
        augmented.loc[group_index, "d_method"] = d_method_run
        augmented.loc[group_index, "w_IV_SK_group"] = iv_weight
        augmented.loc[group_index, "alpha_joint_IV_run"] = (
            iv_weight * sk + (1.0 - iv_weight) * sp
        )
        augmented.loc[
            group_index, "w_error_calibrated_SK_global"
        ] = error_weight
        augmented.loc[
            group_index, "alpha_joint_error_calibrated_run"
        ] = error_weight * sk + (1.0 - error_weight) * sp

        if args.save_replicates:
            replicate_frames.append(
                pd.DataFrame(
                    {
                        "alpha_reference": target,
                        "bootstrap_id": np.arange(1, b_count + 1),
                        "alpha_SK_mean_boot": sk_boot,
                        "alpha_SP_mean_boot": sp_boot,
                        "alpha_joint_equal_boot": equal_boot,
                        "d_method_mean_boot": d_method_boot,
                        "w_IV_SK_boot": iv_weight_boot,
                        "alpha_joint_IV_boot": iv_boot,
                        "w_error_calibrated_SK_boot": error_weight_boot,
                        "alpha_joint_error_calibrated_boot": error_boot,
                    }
                )
            )

    summary = pd.DataFrame(summary_rows).sort_values(
        "alpha_reference", kind="mergesort"
    )
    # Do not expose the temporary numeric run column in the saved run-level file.
    augmented = augmented.drop(columns=["_run_number"])

    summary_path = args.output_dir / "paired_bootstrap_summary.csv"
    run_path = args.output_dir / "run_level_with_joint_estimators.csv"
    weight_path = args.output_dir / "weighting_summary.csv"
    summary.to_csv(summary_path, index=False, float_format="%.10g")
    augmented.to_csv(run_path, index=False, float_format="%.10g")

    iv_weight_ci_rows = summary[
        [
            "alpha_reference",
            "w_IV_SK",
            "w_IV_SP",
            "w_IV_SK_boot_ci_low",
            "w_IV_SK_boot_ci_high",
        ]
    ].copy()
    iv_weight_ci_rows.insert(0, "weighting_scheme", "inverse_variance")
    error_weight_row = pd.DataFrame(
        [
            {
                "weighting_scheme": "retrospective_error_calibrated",
                "alpha_reference": np.nan,
                "w_IV_SK": error_weight,
                "w_IV_SP": 1.0 - error_weight,
                "w_IV_SK_boot_ci_low": error_weight_ci[0],
                "w_IV_SK_boot_ci_high": error_weight_ci[1],
            }
        ]
    )
    weights = pd.concat(
        [iv_weight_ci_rows, error_weight_row], ignore_index=True
    ).rename(
        columns={
            "w_IV_SK": "w_SK",
            "w_IV_SP": "w_SP",
            "w_IV_SK_boot_ci_low": "w_SK_boot_ci_low",
            "w_IV_SK_boot_ci_high": "w_SK_boot_ci_high",
        }
    )
    equal_weight_row = pd.DataFrame(
        [
            {
                "weighting_scheme": "equal_default",
                "alpha_reference": np.nan,
                "w_SK": 0.5,
                "w_SP": 0.5,
                "w_SK_boot_ci_low": np.nan,
                "w_SK_boot_ci_high": np.nan,
            }
        ]
    )
    weights = pd.concat([equal_weight_row, weights], ignore_index=True)
    weights.to_csv(weight_path, index=False, float_format="%.10g")

    targets = summary["alpha_reference"].to_numpy(dtype=float)
    performance_specs = [
        (
            "SK_route",
            "route_specific",
            summary["alpha_SK_mean"].to_numpy(dtype=float),
        ),
        (
            "SP_plateau_route",
            "route_specific",
            summary["alpha_SP_mean"].to_numpy(dtype=float),
        ),
        (
            "equal_weight",
            "default",
            summary["alpha_joint_equal_mean"].to_numpy(dtype=float),
        ),
        (
            "inverse_variance",
            "uncertainty_sensitivity",
            summary["alpha_joint_IV"].to_numpy(dtype=float),
        ),
        (
            "retrospective_error_calibrated",
            "benchmark_only",
            summary["alpha_joint_error_calibrated"].to_numpy(dtype=float),
        ),
    ]
    performance_rows = []
    for estimator, role, estimates in performance_specs:
        error = estimates - targets
        relative_error = error / targets
        performance_rows.append(
            {
                "estimator": estimator,
                "role": role,
                "MAE": float(np.mean(np.abs(error))),
                "RMSE": float(np.sqrt(np.mean(error**2))),
                "mean_absolute_relative_error": float(
                    np.mean(np.abs(relative_error))
                ),
                "root_mean_squared_relative_error": float(
                    np.sqrt(np.mean(relative_error**2))
                ),
            }
        )
    performance_path = args.output_dir / "estimator_performance.csv"
    pd.DataFrame(performance_rows).to_csv(
        performance_path,
        index=False,
        float_format="%.10g",
    )

    weight_grid = np.linspace(0.0, 1.0, args.weight_grid_points)
    grid_estimates = (
        weight_grid[:, np.newaxis] * sk_means[np.newaxis, :]
        + (1.0 - weight_grid[:, np.newaxis]) * sp_means[np.newaxis, :]
    )
    grid_error = grid_estimates - alpha_values[np.newaxis, :]
    grid_relative_error = grid_error / alpha_values[np.newaxis, :]
    nearest_calibrated = int(np.argmin(np.abs(weight_grid - error_weight)))
    weight_scan = pd.DataFrame(
        {
            "w_SK": weight_grid,
            "w_SP": 1.0 - weight_grid,
            "MAE": np.mean(np.abs(grid_error), axis=1),
            "RMSE": np.sqrt(np.mean(grid_error**2, axis=1)),
            "mean_absolute_relative_error": np.mean(
                np.abs(grid_relative_error), axis=1
            ),
            "root_mean_squared_relative_error": np.sqrt(
                np.mean(grid_relative_error**2, axis=1)
            ),
            "is_equal_weight": np.isclose(weight_grid, 0.5),
            "is_nearest_error_calibrated": (
                np.arange(args.weight_grid_points) == nearest_calibrated
            ),
        }
    )
    weight_scan_path = args.output_dir / "weight_scan.csv"
    weight_scan.to_csv(weight_scan_path, index=False, float_format="%.10g")

    replicate_path: Path | None = None
    if args.save_replicates:
        replicate_path = (
            args.output_dir / "paired_bootstrap_replicates.csv.gz"
        )
        pd.concat(replicate_frames, ignore_index=True).to_csv(
            replicate_path,
            index=False,
            float_format="%.10g",
            compression="gzip",
        )

    print(f"Input rows: {len(data)}")
    print(f"Alpha groups: {n_alpha}")
    print(f"Paired repetitions per group: {data.groupby(args.alpha_col).size().tolist()}")
    print(f"Bootstrap resamples: {b_count}")
    print(f"Seed: {args.seed}")
    print(f"Retrospective error-calibrated weights: SK={error_weight:.6f}, SP={1.0-error_weight:.6f}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {run_path}")
    print(f"Wrote: {weight_path}")
    print(f"Wrote: {performance_path}")
    print(f"Wrote: {weight_scan_path}")
    if replicate_path is not None:
        print(f"Wrote: {replicate_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
