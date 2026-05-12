# Finite-size effective exponent extraction for hyperuniform systems

This repository provides the analysis code associated with the manuscript

**Extracting effective scaling exponents in finite-size hyperuniform systems**

The code implements a finite-size protocol for extracting effective scaling exponents from hyperuniform point configurations. The workflow combines three complementary routes:

1. static structure factor `S(k)`,
2. number variance `sigma_N^2(R)`,
3. spreadability excess `S(infinity) - S(t)`.

The route-specific outputs are then combined into a joint empirical estimate. The reported exponent should be interpreted as a finite-size effective estimate under the specified analysis protocol.

## Repository structure

```text
finite_size_hyperuniform_alpha/
├── README.md
├── config.json
├── requirements.txt
├── run_all.py
├── data/
│   └── N200/
│       └── config_*_component_0.txt
└── src/
    ├── preprocess/
    │   └── compute_length_scale.py
    ├── sk/
    │   └── compute_sk.cpp
    ├── number_variance/
    │   └── compute_number_variance.cpp
    ├── spreadability/
    │   └── compute_spreadability.py
    └── analysis/
        └── combined_alpha_analysis.py
```

## Included example dataset

The folder

```text
data/N200/
```

contains a representative example dataset for demonstrating the complete analysis pipeline.

Dataset information:

```text
dimension        = 2
Np               = 200 particles per configuration
Nc               = 100 configurations
density          = 1.0
alpha_reference  = 3.0
box              = square periodic box
box length       = sqrt(Np / density)
coordinates      = reduced coordinates in [0, 1)^2
```

Each configuration file is named as

```text
config_i_component_0.txt
```

where `i` is the configuration index. Each file contains two columns corresponding to the particle coordinates.

This repository provides the analysis workflow rather than the configuration-generation workflow. Additional benchmark configurations used in the manuscript are available from the corresponding author upon reasonable request.

## Requirements

Python packages:

```text
numpy
scipy
```

A C++ compiler is required for the structure-factor and number-variance calculations. The C++ codes are compiled with

```bash
g++ -O3 -std=c++11
```

Install the Python requirements with

```bash
pip install -r requirements.txt
```

## Run the workflow

From the repository root directory, run

```bash
python run_all.py --config config.json
```

This command performs the following steps:

```text
1. compute the ensemble-averaged nearest-neighbor length scale a
2. compute the static structure factor S(k)
3. compute the number variance sigma_N^2(R)
4. compute the spreadability excess S(infinity) - S(t)
5. perform the combined route-level analysis
```

The main output file is

```text
results/N200/combined_alpha_result.txt
```

## Output files

The workflow generates route-specific results in

```text
results/N200/
```

Representative output files include

```text
results/N200/a_value.txt
results/N200/a_value.json
results/N200/SK_ensemble.txt
results/N200/SK_ensemble_ka.txt
results/N200/num_var_ensemble.txt
results/N200/num_var_ensemble_R_over_a.txt
results/N200/routeA_pure_directdiffusion_dualfit_protocol/sample_ensemble_spreadability.txt
results/N200/combined_alpha_result.txt
```

The file

```text
combined_alpha_result.txt
```

reports the route-specific estimates and the joint result, including quantities such as

```text
alpha_SK
alpha_SP_plateau
alpha_NV
alpha_joint
u_joint
Delta_alpha_joint
epsilon_joint
```

depending on the diagnosed route availability.

## Configuration file

The main parameters are controlled by

```text
config.json
```

Important fields include

```json
{
  "case_name": "N200",
  "input_dir": "data/N200",
  "output_dir": "results/N200",
  "density": 1.0,
  "num_configs": 100,
  "alpha_reference": 3.0,
  "coords_are_reduced": true,
  "auto_length_scale_a": true
}
```

Here, `alpha_reference` is used only for benchmark evaluation, such as computing the deviation from the prescribed reference exponent. For systems without a prescribed reference exponent, this field can be set to `null`.

The current example assumes reduced coordinates in `[0, 1)^2`.

## Skipping completed steps

If some route-level observables have already been generated, the corresponding steps can be skipped:

```bash
python run_all.py --config config.json --skip-sk
python run_all.py --config config.json --skip-nv
python run_all.py --config config.json --skip-spreadability
```

If the length scale `a` has already been computed and `length_scale_a` is manually specified in `config.json`, use

```bash
python run_all.py --config config.json --skip-a
```

Multiple skip options can be combined.

## Notes

The output exponent is protocol-dependent and finite-size dependent. It should not be interpreted as a strict thermodynamic-limit asymptotic exponent.

The included dataset is a representative example for demonstrating the three route-specific analyses and the joint empirical estimator.
