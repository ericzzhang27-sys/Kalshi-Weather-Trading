# NGBoost Distribution Choice Notes

## What Day 13 Suggested

Day 13 Normal `ngboost_normal_v0` had validation NLL 1.3240 and test NLL 1.6855. Validation coverage was 50/80/90% = 0.534/0.830/0.904; test coverage was 0.431/0.726/0.837. Standardized residual std moved from 1.053 on validation to 1.298 on test, with test residual mean -0.257.

The practical read is mixed: validation Normal was slightly wide, while the later test period was undercovered and left too little mass in the tails. Distribution choice is therefore selected on validation probability metrics, with test-period diagnostics treated as robustness context.

## Candidates Tested

- `ngboost_normal_v1`: explicit Normal NGBoost baseline, same chronological split, features, seed, and hyperparameters as the prior Normal path.
- `ngboost_student_t_attempt`: NGBoost `T` distribution for signed forecast errors.
- `ngboost_laplace_attempt`: NGBoost `Laplace` signed-error candidate, saved to `models/ngboost_heavytail_attempt.pkl`.
- Normal post-hoc sigma inflation variants at 1.05, 1.10, 1.15, and 1.20, with the mean unchanged.

Positive-only NGBoost distributions such as Gamma and LogNormal were excluded because raw `forecast_error` can be negative.

## Heavy-Tail Support

Student-t was attempted but was not usable in this environment.
Laplace trained successfully as a signed heavier-tailed candidate.

- ngboost_student_t_attempt: failed or unstable: ValueError: Input y contains NaN.
- ngboost_laplace_attempt: trained successfully with stable prediction parameters.

The mitigation is to use a supported signed heavy-tailed distribution with stable CDF values (`Laplace`) and to keep Normal sigma-inflation variants as conservative calibration checks rather than forcing a fragile custom Student-t implementation.

## Validation Result

Selected model: `ngboost_laplace_attempt`.

| model | dist | sigma x | val NLL | bucket log loss | cov80 | cov90 | PIT note | selected |
|---|---|---:|---:|---:|---:|---:|---|---|
| ngboost_laplace_attempt | laplace | 1.00 | 1.1424 | 0.9029 | 0.837 | 0.921 | hump-shaped / too wide with high-PIT skew | yes |
| ngboost_normal_sigma_x105 | normal | 1.05 | 1.3211 | 0.9034 | 0.846 | 0.916 | hump-shaped / too wide |  |
| ngboost_normal_sigma_x110 | normal | 1.10 | 1.3229 | 0.9022 | 0.861 | 0.923 | hump-shaped / too wide |  |
| ngboost_normal_v1 | normal | 1.00 | 1.3240 | 0.9073 | 0.830 | 0.904 | hump-shaped / too wide |  |
| ngboost_normal_sigma_x115 | normal | 1.15 | 1.3283 | 0.9031 | 0.873 | 0.933 | hump-shaped / too wide |  |
| ngboost_normal_sigma_x120 | normal | 1.20 | 1.3365 | 0.9059 | 0.888 | 0.938 | hump-shaped / too wide |  |

The selected row had the best validation balance across continuous NLL, final-temperature bucket interval log loss, interval coverage, and PIT behavior. The Normal baseline remains the clean reference: validation NLL 1.3240, bucket log loss 0.9073, 80% coverage 0.830.

## Bucket-CDF Compatibility

Every successful candidate was evaluated through CDF differences using the same final-temperature bucket conversion. The selected model is compatible with downstream bucket probability generation: probabilities were finite, nonnegative within tolerance, and row-normalized by construction over the open-ended bucket set.

## Limitations

- This comparison still uses one chronological validation year. A walk-forward validation pass would be a stronger guard against the 2025-2026 undercoverage seen on Day 13.
- Sigma inflation can improve tail coverage but may degrade validation log score when the validation period is already slightly wide.
- Student-t introduces an additional degrees-of-freedom parameter; even when its CDF is stable, it should be monitored for very low df values and over-wide intervals.
- No residual clipping was used for final training or evaluation. Extreme residuals remain in the validation metrics.

## Reproducibility

- Dataset: `data\processed\modeling_rows_v1.csv`
- Feature spec: `outputs\day8_features\feature_columns.json`
- Split: train through 2023-12-31, validation through 2024-12-31
- Feature count: 39
