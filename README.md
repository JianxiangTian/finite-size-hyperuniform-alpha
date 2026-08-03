# Finite-size effective exponent extraction for hyperuniform systems

This repository provides the analysis code associated with the manuscript

**Extracting effective scaling exponents in finite-size hyperuniform systems**

The workflow extracts finite-size effective scaling information from three
complementary observables:

1. the static structure factor `S(k)`;
2. the number variance `sigma_N^2(R)`;
3. the spreadability excess `S(infinity) - S(t)`.

The default joint empirical estimator combines only the `S(k)` and
plateau-window spreadability estimates with equal weights:

```text
alpha_joint = 0.5 * alpha_SK + 0.5 * alpha_SP
d_method    = 0.5 * abs(alpha_SK - alpha_SP)
```

Number variance is retained as an independent real-space Class-like reference.
It is not used in the joint estimator. The traditional global spreadability fit
is retained as a comparison baseline and is also not used in the joint
estimator.

The reported exponents are finite-size effective estimates under the specified
analysis protocol. They should not be interpreted as exact thermodynamic-limit
asymptotic exponents.

## Frozen analysis hierarchy

| Result | Role |
| --- | --- |
| Equal-weight `S(k)` + plateau SP | Default joint estimator |
| Inverse-variance weighting | Uncertainty-driven sensitivity analysis |
| Benchmark-error calibration | Retrospective benchmark-only comparison |
| Number variance | Independent real-space Class-like reference |
| Traditional global SP fit | Fitting-protocol baseline |

The exact definitions and revision regression targets are recorded in
[`METHODS_FREEZE.md`](METHODS_FREEZE.md).

## Repository structure

```text
finite_size_hyperuniform_alpha/
├── README.md
├── METHODS_FREEZE.md
├── RELEASE_MANIFEST.sha256
├── VERSION
├── config.json
├── requirements.txt
├── run_all.py
├── data/
│   ├── N200/
│   │   └── config_*_component_0.txt
│   └── revision_run_level_estimates.csv
├── src/
│   ├── preprocess/
│   │   └── compute_length_scale.py
│   ├── sk/
│   │   └── compute_sk.cpp
│   ├── number_variance/
│   │   └── compute_number_variance.cpp
│   ├── spreadability/
│   │   └── compute_spreadability.py
│   └── analysis/
│       ├── combined_alpha_analysis.py
│       └── paired_bootstrap_joint_estimators.py
└── tests/
    └── test_frozen_estimators.py
```

## Included data

`data/N200/` contains a representative configuration dataset:

```text
dimension        = 2
Np               = 200 particles per configuration
Nc               = 100 configurations
density          = 1.0
alpha_reference  = 3.0
box              = square periodic box
coordinates      = reduced coordinates in [0, 1)^2
```

Each configuration is named `config_i_component_0.txt` and contains two
coordinate columns.

All input configuration files must contain reduced coordinates in
`[0, 1)^2`. Absolute-coordinate input is not supported by this frozen release.
Convert absolute coordinates to reduced coordinates before running the
workflow. The `coords_are_reduced` field in `config.json` must remain `true`.

`data/revision_run_level_estimates.csv` contains the compact route-level
results needed to reproduce the weighting and paired-bootstrap calculations:

```text
alpha_reference, run, alpha_SK, alpha_SP
```

The repository provides the analysis workflow rather than the
configuration-generation workflow. The complete benchmark configuration
collection is not included because of its size; additional configurations are
available from the corresponding author upon reasonable request.

## Requirements

Python 3.10 or newer is required.

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

A C++ compiler is required for the structure-factor and number-variance
calculations. The workflow uses:

```bash
g++ -O3 -std=c++11
```

## Run the representative workflow

### Quick verification

The recommended quick verification skips the computationally intensive
number-variance calculation:

```bash
python run_all.py --config config.json --skip-nv
```

This still reproduces the two numerical inputs to the joint estimator: the
`S(k)` and plateau-window spreadability estimates.

### Full three-route workflow

From the repository root:

```bash
python run_all.py --config config.json
```

This computes:

1. the ensemble-averaged nearest-neighbor length scale `a`;
2. `S(k)`;
3. number variance, when enabled;
4. the spreadability excess;
5. the final route-level analysis.

The main outputs are:

```text
results/N200/combined_alpha_result.txt
results/N200/combined_alpha_result.json
```

The joint result in both files always uses equal-weight `S(k)` and plateau SP.
NV cannot enter the combined estimate.

The number-variance calculation is computationally dominant and may require
several minutes or longer, depending on the hardware.

Other completed calculations may also be skipped:

```bash
python run_all.py --config config.json --skip-a
python run_all.py --config config.json --skip-sk
python run_all.py --config config.json --skip-spreadability
```

If `--skip-a` is used without an existing `a_value.json`, set
`auto_length_scale_a` to `false` and provide `length_scale_a` in `config.json`.

### Expected representative result

For the bundled dataset with `alpha_reference = 3.0`, the full workflow should
give approximately:

```text
alpha_SK:                 3.119006
number_variance_class:    Class I-like
mean_p_eff:               0.944478
alpha_NV:                 nan
alpha_SP_plateau:         2.985769
alpha_joint_equal:        3.052388
d_method:                 0.066618
epsilon_joint_percent:    1.7463
```

Small last-digit differences may occur across platforms and compiler versions.
Number variance is reported only as an independent Class-like reference and
never enters `alpha_joint_equal`.

## Reproduce the weighting and bootstrap analysis

Run:

```bash
python src/analysis/paired_bootstrap_joint_estimators.py \
  data/revision_run_level_estimates.csv \
  --output-dir results/repeated_runs \
  --bootstraps 10000 \
  --seed 20260730 \
  --expected-reps 20 \
  --run-first 1 \
  --run-last 20 \
  --save-replicates
```

The script preserves the `S(k)`-SP pairing within each independent run. For the
inverse-variance estimator, the variances and weights are recalculated inside
every bootstrap resample.

Outputs:

```text
paired_bootstrap_summary.csv
run_level_with_joint_estimators.csv
weighting_summary.csv
estimator_performance.csv
weight_scan.csv
paired_bootstrap_replicates.csv.gz
```

For the frozen revision dataset, the command reproduces:

```text
default equal weights:                 w_SK = 0.500000
                                       w_SP = 0.500000
retrospective error-calibrated weight: w_SK = 0.092728
                                       w_SP = 0.907272
```

The principal frozen regression values are:

```text
SK_route MAE / RMSE:                 0.08544671 / 0.10452717
SP_plateau_route MAE / RMSE:         0.02068429 / 0.03005389
equal_weight MAE / RMSE:             0.03549250 / 0.03997623
inverse_variance MAE / RMSE:         0.04223877 / 0.04912441
retrospective calibrated w_SK:       0.092728
95% bootstrap CI for w_SK:           [0.081018, 0.107048]
```

The calibrated weight uses the known benchmark targets and must not be used as
the default estimator for an unknown system.

## Configuration

The route and analysis parameters are controlled by `config.json`.
`alpha_reference` is used only for benchmark evaluation. Set it to `null` for
a system without a prescribed target exponent.

Number variance can be disabled without changing the joint estimator:

```json
{
  "number_variance": {
    "enabled": false
  }
}
```

## Tests

Run the lightweight estimator tests with:

```bash
python -m unittest discover -s tests -v
```

Verify the integrity of the frozen release files with:

```bash
sha256sum -c RELEASE_MANIFEST.sha256
```
