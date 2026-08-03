# Frozen methods and weighting specification

This file records the analysis definitions used for the revised manuscript.
Changes to these definitions require a new method version and regeneration of
the affected numerical results.

## 1. Structure-factor estimate

The `S(k)` route uses the ensemble-averaged `S(k)` curve expressed against
`ka`.

1. The admissible low-`k` branch ends before the earliest detected local peak
   or transition toward the `S(k) ~ O(1)` plateau.
2. Every contiguous candidate contains at least five points and spans at least
   `0.35` in `log10(ka)`.
3. For every candidate, including a five-point candidate, boundary sensitivity
   is

   ```text
   Delta_alpha = max(
       abs(alpha - alpha_without_left_endpoint),
       abs(alpha - alpha_without_right_endpoint)
   )
   ```

4. The selected window minimizes

   ```text
   Q_SK = RMSE_log / median(RMSE_log)
        + eta_SK * Delta_alpha / median(Delta_alpha)
   ```

   with `eta_SK = 1`.

## 2. Number-variance reference

Number variance supplies a real-space Class-like diagnosis from the selected
effective-slope plateau.

```text
mean_p_eff <= 1.05  -> Class I-like
1.05 < mean_p_eff < 1.15 -> Class II-like
mean_p_eff >= 1.15  -> Class III-like
```

For a Class III-like result only, `alpha_NV = 2 - mean_p_eff` may be reported
as a route-specific reference. Number variance is excluded from all joint and
weighted estimators for every target exponent.

## 3. Spreadability estimate

The primary SP result is extracted from the ensemble-averaged spreadability
excess curve.

1. Use positive finite data within the trusted time range.
2. Compute a 17-point local effective-slope sequence.
3. Candidate plateau windows start at `tau >= 1`, contain at least 25 points,
   and span at least `0.25` decades.
4. Select the window minimizing

   ```text
   Q_t = RMSE_log / median(RMSE_log)
       + eta_t * SD(alpha_eff) / median(SD(alpha_eff))
   ```

   with `eta_t = 1`.
5. Convert the fitted slope `m` in two dimensions through

   ```text
   alpha_SP = -2 * m - 2
   ```

The traditional global fit is evaluated only as a method-comparison baseline.

## 4. Default joint estimator

For independent repetition `m`:

```text
alpha_joint_equal^(m)
    = 0.5 * alpha_SK^(m) + 0.5 * alpha_SP^(m)

d_method^(m)
    = 0.5 * abs(alpha_SK^(m) - alpha_SP^(m))
```

This equal-weight estimator is the only default joint estimate. It is
target-independent and remains applicable when the underlying exponent is
unknown.

## 5. Repetition statistics and paired bootstrap

For each prescribed benchmark exponent, use the 20 independent repetitions as
the statistical units. Report the mean, the sample standard deviation across
repetitions, and a paired-bootstrap confidence interval for the mean.

For each of 10,000 bootstrap resamples:

1. draw 20 run indices with replacement;
2. use the same indices for `alpha_SK` and `alpha_SP`;
3. calculate the route means and derived estimator;
4. use the 2.5th and 97.5th percentiles for the 95% interval.

The frozen random seed is `20260730`.

## 6. Weighting sensitivity analyses

### Equal weighting

```text
w_SK, w_SP = 0.5, 0.5
```

Role: default estimator.

### Inverse-variance weighting

For each prescribed benchmark exponent:

```text
w_IV_SK = s_SP^2 / (s_SK^2 + s_SP^2)
w_IV_SP = 1 - w_IV_SK
```

The variances are sample variances across the 20 repetitions. The weight is
re-estimated inside every bootstrap resample.

Role: uncertainty-driven sensitivity analysis. Repetition-level variance
measures precision and does not account for finite-size systematic bias.

Frozen revision weights:

| `alpha_reference` | `w_IV_SK` | `w_IV_SP` | Bootstrap 95% CI for `w_IV_SK` |
| ---: | ---: | ---: | ---: |
| 0.3 | 0.9356485125 | 0.0643514875 | [0.8663679644, 0.9728561058] |
| 0.5 | 0.8009132260 | 0.1990867740 | [0.6902554842, 0.8726932665] |
| 0.7 | 0.8649523658 | 0.1350476342 | [0.7129944721, 0.9432303052] |
| 1.0 | 0.9392252796 | 0.0607747204 | [0.8410491086, 0.9736581325] |
| 1.5 | 0.9790365664 | 0.0209634336 | [0.9600161577, 0.9916172108] |
| 2.0 | 0.9998723429 | 0.0001276571 | [0.9990876557, 0.9999578351] |
| 2.5 | 0.0087806553 | 0.9912193447 | [0.0045790167, 0.0175528573] |
| 3.0 | 0.0258843389 | 0.9741156611 | [0.0132094691, 0.0936267989] |
| 4.0 | 0.5299995385 | 0.4700004615 | [0.2905048112, 0.7427534547] |

### Retrospective benchmark-error calibration

Choose one global `w_SK` in `[0, 1]` that minimizes the mean squared relative
benchmark error over the nine prescribed exponents:

```text
mean_j [
  (
    w_SK * mean(alpha_SK_j)
    + (1 - w_SK) * mean(alpha_SP_j)
    - alpha_reference_j
  ) / alpha_reference_j
]^2
```

Frozen revision result:

```text
w_SK = 0.09272803626
w_SP = 0.90727196374
bootstrap 95% CI for w_SK = [0.08101751095, 0.10704753100]
```

Role: retrospective benchmark-only comparison. It is not a deployable default
weight because it uses known target exponents.

## 7. Revision regression targets

The compact run-level dataset must reproduce:

| Estimator | MAE | RMSE |
| --- | ---: | ---: |
| `S(k)` route | 0.08544671 | 0.10452717 |
| Plateau SP route | 0.02068429 | 0.03005389 |
| Equal-weight default | 0.03549250 | 0.03997623 |
| Inverse variance | 0.04223877 | 0.04912441 |
| Retrospective calibration | 0.01338973 | 0.01949477 |

These targets are regression checks for the frozen revision data, not claims
of estimator unbiasedness.
