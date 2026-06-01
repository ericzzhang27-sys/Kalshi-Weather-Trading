# Kalshi Weather Trading

This project is focused on building a robust NGBoost/DGBM probability engine for weather bucket markets.

Current direction:

- Primary model: NGBoost / DGBM on `forecast_error`.
- Required comparison: empirical historical forecast-error distribution.
- Bucket probabilities come from the model-implied CDF.
- CDF classifiers, multiclass bucket models, quantile models, and alternative DGBM implementations are optional future extensions only.
- Probability quality, calibration, and interval coverage matter more than point accuracy.

The project is still a probability-signal research layer. Trading logic, execution, sizing, and profitability claims come later.

## Documentation Map

- Project spec: `docs/project/PROJECT_SPEC.md`
- Project context and handoff notes: `docs/project/CONTEXT.md`
- Long-form build spec: `docs/specs/weather_probability_modeling_codex_build_spec_dgbm_ngboost.md`
- Video presentation materials: `docs/presentation/`
