# Official conference tables

These CSVs are the numbers used in the conference paper.

- `table_main_means.csv` — seed-mean PE for the seven conference systems
- `table_bagged.csv` — true bagging (average predictions, then score)
- `table_grasp_storm_bz.csv` — GRASP zero-shot vs fine-tune
- `all_seed_results_full.csv` — per-seed rows (conference models only)
- `ensemble_summary.json` — validation-selected α* diagnostic

Do not treat Access-only probes (alternative gates, wider delay) as part of this repo.

### Noise & Matched-Capacity Checks
- `tf_ffn256_summary.csv` — FFN-matched Transformer and residual-head control
- `noise_robustness.csv` — Gaussian input-noise and channel-dropout probe
