# Outputs Directory

This folder contains generated artifacts from the weather probability modeling workflow: data audits, feature reports, model predictions, bucket probabilities, calibration diagnostics, figures, and notebook presentation visuals.

Most files in this directory are reproducible outputs rather than hand-written source code. The main source code lives in `src/`, and the workflow entry points live in `scripts/`.

## Output Groups

| Folder | Contents |
|---|---|
| `data_audit/` | Data inventory and verification report from the raw/processed data audit workflow. |
| `demos/` | Small demo CSVs used by early notebooks and probability-pricing examples. |
| `day7_targets/` | Forecast-error target summaries from supervised target construction. |
| `day8_features/` | Feature specs, missingness reports, leakage checks, integrity reports, and modeling-row previews. |
| `day9_empirical_baseline/` | Empirical forecast-error baseline predictions and report. |
| `day10_distribution_pricing/` | Bucket-boundary conversion examples and interval-probability demos. |
| `day17_feature_lists/` | Candidate feature lists used for ablations and final-safe feature selection. |
| `figures/` | Generated plots and presentation images. |
| `visuals/` | Reserved visual-artifact folder. |

## Core Current Artifacts

| File | Purpose |
|---|---|
| `final_feature_list.json` | Selected model feature list used by the current NGBoost training path. |
| `ngboost_distribution_params_v0.csv` | One distribution prediction per validation/test timestamp, including `mu`, `sigma`, split, forecast high, actual high, and realized forecast error. |
| `ngboost_nll_v0.json` | Training metadata, split summary, feature list, NLL metrics, and preprocessing notes. |
| `ngboost_bucket_probs_v0.csv` | Long-format bucket probabilities derived from the model CDF. |
| `ngboost_bucket_probs_validation_v0.csv` | Validation split subset of bucket probabilities. |
| `ngboost_bucket_probs_test_v0.csv` | Test split subset of bucket probabilities. |
| `ngboost_bucket_prob_validation.md` | Probability coherence report for bucket probabilities. |
| `ngboost_evaluation_report.csv` | Summary evaluation metrics for NGBoost and empirical baseline comparisons. |
| `coverage_report.csv` | Interval coverage by split and nominal coverage level. |
| `standardized_residual_summary.csv` | Residual diagnostics by split. |
| `bucket_brier_scores.csv` | Bucket-level Brier scores and calibration gaps. |
| `calibration_tables.csv` | Reliability tables used for calibration curves. |
| `coverage_by_group.csv` | Grouped coverage diagnostics. |

## Calibration Outputs

| File | Purpose |
|---|---|
| `ngboost_calibration_report.csv` | Raw vs sigma-scaled calibration metrics. |
| `ngboost_interval_coverage_raw.csv` | Raw interval coverage before calibration adjustments. |
| `ngboost_interval_coverage_before_after.csv` | Coverage comparison before and after calibration. |
| `ngboost_sigma_scaling_alpha_search.csv` | Validation-only sigma-scaling search results. |
| `ngboost_bucket_probabilities_calibrated.csv` | Bucket probabilities recomputed with calibrated sigma scaling. |
| `ngboost_cdf_reliability_table.csv` | CDF reliability by threshold. |
| `ngboost_bucket_reliability_table.csv` | Bucket reliability tables by method and split. |
| `ngboost_coverage_by_hour.csv` | Coverage grouped by prediction hour. |
| `ngboost_coverage_by_season.csv` | Coverage grouped by season. |
| `ngboost_coverage_by_horizon.csv` | Coverage grouped by forecast horizon. |

## Model Selection and Comparison Outputs

| File | Purpose |
|---|---|
| `ngboost_distribution_comparison.csv` | Distribution candidate comparison across validation/test metrics. |
| `ngboost_distribution_candidate_params.csv` | Candidate model prediction parameters. |
| `ngboost_distribution_group_coverage.csv` | Grouped coverage diagnostics for distribution candidates. |
| `distribution_choice_notes.md` | Notes explaining distribution choice and caveats. |
| `ngboost_current36_distribution_summary.csv` | Summary of current 36-feature distribution candidates. |
| `ngboost_current36_distribution_scale_grid.csv` | Scale-grid comparison for current 36-feature candidates. |
| `ngboost_pruned_candidate_comparison.csv` | Comparison table for pruned candidate feature/model variants. |
| `ngboost_hyperparameter_search.csv` | NGBoost tuning search output. |
| `best_ngboost_v2_notes.md` | Notes on selected v2 NGBoost candidate. |
| `best_model_notes.md` | Notes on broader model selection. |
| `model_selection_metrics.md` | Model selection metric summary. |

## Feature Audit and Ablation Outputs

| File | Purpose |
|---|---|
| `feature_audit.md` | Feature audit notes. |
| `feature_candidate_backlog.md` | Candidate feature backlog. |
| `feature_leakage_review.md` | Feature leakage review notes. |
| `ngboost_feature_ablation.csv` | Feature group ablation metrics. |
| `ngboost_feature_ablation_brier_by_bucket.csv` | Bucket-level Brier diagnostics for ablation runs. |
| `ngboost_feature_ablation_metadata.json` | Metadata for ablation runs. |
| `ngboost_single_feature_ablation.csv` | Single-feature ablation metrics. |
| `ngboost_single_feature_ablation_metadata.json` | Metadata for single-feature ablation runs. |
| `day17_final_safe_validation_metrics.json` | Final-safe feature validation metrics. |
| `day17_final_safe_brier_by_bucket.csv` | Final-safe bucket Brier metrics. |
| `day17_safe_reduced_with_time_validation_metrics.json` | Reduced-with-time feature validation metrics. |

## Figures

The `figures/` folder contains generated PNG and Mermaid artifacts, including:

| File | Purpose |
|---|---|
| `pit_histogram.png` | PIT histogram for distribution calibration. |
| `coverage_by_hour.png` | Coverage diagnostic by prediction hour. |
| `coverage_by_season.png` | Coverage diagnostic by season. |
| `coverage_before_after.png` | Calibration before/after coverage comparison. |
| `sigma_scaling_validation_nll.png` | Validation NLL across sigma-scaling alphas. |
| `calibration_bucket_market_bucket_*.png` | Reliability curves for selected market bucket positions. |
| `ngboost_distribution_coverage_validation.png` | Candidate distribution interval coverage comparison. |
| `ngboost_distribution_standardized_residuals_validation.png` | Candidate standardized residual histograms. |
| `project_architecture.mmd` | Mermaid source for the project architecture diagram. |
| `project_architecture.png` | Rendered architecture diagram for the presentation. |
| `notebook_intraday_feature_path.png` | Intraday temperature path and feature explanation visual. |
| `notebook_bucket_density.png` | Probability density split into bucket intervals. |
| `notebook_bucket_probabilities.png` | Bar chart of bucket probabilities for one prediction row. |
| `notebook_ngboost_trees_information_matrix.png` | NGBoost trees plus Fisher information matrix explanation visual. |

## Regeneration Commands

Run these commands from the repository root.

Build features and leakage reports:

```bash
python scripts/build_features.py
```

Train the configured NGBoost model:

```bash
python -m src.train_ngboost
```

Generate bucket probabilities:

```bash
python -m src.distribution_pricing
```

Evaluate probability quality:

```bash
python scripts/evaluate_ngboost.py
```

Run calibration diagnostics:

```bash
python scripts/calibrate_ngboost.py
```

Open or rerun the presentation notebook:

```bash
jupyter notebook notebooks/project_walkthrough.ipynb
```

## Notes

- Outputs are timestamped or versioned by workflow stage in some places, but not all files include a run ID.
- Some artifacts have historical names, such as `ngboost_normal_v0`, even when the current configured distribution is Laplace.
- Do not treat any single output table as a live-trading signal without timestamp-correct market prices, executable bid/ask data, fees, slippage assumptions, and settlement-rule validation.
