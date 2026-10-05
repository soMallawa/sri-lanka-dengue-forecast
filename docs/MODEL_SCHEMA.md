# Model-Ready Feature Schema

The source of truth for released model schemas is `model_manifest_trusted.json`.

`h1__cases_only__ridge__final` requires epidemiological lag/rolling features, seasonality features, `dengue_cases`, and `district_id`.

The `h2`, `h3`, and `h4` full-weather LightGBM models require the same epidemiological and seasonality features plus rainfall, temperature, and humidity aggregates/lags/rolling features.

The public inference CLI validates:

- Exact feature column set and order.
- Known `district_id` category values.
- Numeric coercion for all non-categorical features.
- SHA-256 hashes for `model.joblib` and `metadata.json`.
- Explicit pickle trust acknowledgement.
