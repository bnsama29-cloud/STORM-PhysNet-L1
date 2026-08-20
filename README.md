# STORM-PhysNet (conference)

Official code for the IEEE conference paper:

**STORM-PhysNet: A Multi-Horizon Transformer for Geostationary Relativistic Electron Flux Forecasting with Physics-Inspired Components and Cross-Satellite Transfer**

Horizons: **1 h / 6 h / 12 h** on hourly GOES–OMNI. Transfer target: GSAT-19 GRASP (Indian longitude ≈ 82°E).

This repository contains only the **seven conference systems**: LSTM, default Transformer, architecture-matched Transformer, STORM-Bz, No-Delay, No-Gate, No-Physics. Extra files under `src/model/` (e.g. `analogy_gates.py`) are unused leftover code and are **not** used for any conference table or figure.

## Repository map

```
STORM-PhysNet-L1/
├── configs/          Training hyperparameters used for the official run
├── datasets/         GOES, OMNI, and GRASP files used by the preprocessor
├── src/              Python package (data, model, training, metrics)
├── notebooks/        One reproduction notebook
├── checkpoints/      Official 15-seed weights for the seven systems
├── results/          Official CSVs / JSON behind the paper tables
├── figures/          Paper figures only (5 files)
├── requirements.txt  Install this
└── LICENSE           MIT (code). Follow NOAA / NASA / ISSDC for data.
```

### `configs/`
| File | What it is |
|------|------------|
| `config.yaml` | Official STORM run: horizons `[1, 6, 12]`, 72 h lookback, 12× storm sampler, chronological split |
| `config_transformer_baseline.yaml` | Default Transformer (`d_model=64`, 3 layers) |

### `datasets/`
| Folder | What a reviewer should look at |
|--------|--------------------------------|
| `datasets/goes/` | GOES-15 EPEAD \(E>2\) MeV (1-min CDF → hourly in `cdf_reader.py`) |
| `datasets/omni/` | OMNI / L1 solar-wind and geomagnetic indices |
| `datasets/grasp/` | GSAT-19 GRASP monthly files for the transfer experiment |

Do not use any synthetic generator. Training reads these three folders only.

### `src/`
| Path | Role |
|------|------|
| `src/data/cdf_reader.py` | GOES / OMNI readers |
| `src/data/preprocessor.py` | Join, clean, chronological 70/15/15, scale on train only |
| `src/data/dataloader.py` | Windows, horizons `[1, 6, 12]`, 12× storm sampler |
| `src/model/storm_physnet.py` | STORM encoder + delay + \(B_z\) gate + residual heads |
| `src/model/propagation_delay.py` | Learnable L1–Earth delay in \([0.5, 1.5]\) h |
| `src/model/bz_gate.py` | Default \(B_z\) gate |
| `src/model/forecasting_heads.py` | Multi-horizon residual heads |
| `src/model/baselines.py` | LSTM and Transformer |
| `src/training/trainer.py` | `Trainer.fit` — official training loop |
| `src/training/physics_loss.py` | Composite objective (short-horizon physics scale) |
| `src/training/transfer_learning.py` | GRASP freeze: delay + gate + encoder; train heads |
| `src/evaluation/metrics.py` | \(\mathrm{PE_{clim}}\) and \(\mathrm{PE_{pers}}\) |

Unused in conference tables: `analogy_gates.py`, `spectral_head.py`, `cross_modal_attention.py`, `itransformer_encoder.py`, `ssm_encoder.py`, `magnetopause_geometry.py`.

### `checkpoints/`
One folder per conference system, seeds 42–56:

`lstm/` · `transformer/` · `transformer_matched/` · `storm_bz/` · `storm_no_delay/` · `storm_no_gate/` · `storm_no_physics/`

GRASP fine-tune weights were produced externally; frozen PE is in `results/table_grasp_storm_bz.csv`.

### `results/`
| File | Paper use |
|------|-----------|
| `table_main_means.csv` | Seed-mean PE (Table I) |
| `table_bagged.csv` | True bagging (average 15 predictions, then score) |
| `table_grasp_storm_bz.csv` | GRASP zero-shot vs fine-tune |
| `table_parameter_counts.csv` | Parameter table |
| `all_seed_results_full.csv` | Per-seed rows |
| `table_means_bootstrap_ci.csv` | Bootstrap CIs |
| `ensemble_summary.json` | Validation-selected \(\alpha^*\) diagnostic |
| `tf_ffn256_summary.csv` | FFN-matched / residual-head probe (Limitations) |
| `noise_robustness.csv` | Input-noise probe (Fig. 3) |

### `figures/`
| File | Paper figure |
|------|----------------|
| `fig_system_architecture.png` | Fig. 1 |
| `fig_horizon_pe.png` | Fig. 2 |
| `noise_robustness.png` | Fig. 3 |
| `fig_case_study_timeseries.png` | Fig. 4 |
| `fig_grasp_domain_gap.png` | Fig. 5 |

### `notebooks/`
`STORM_PhysNet_Master.ipynb` — environment, data load, PE from CSVs, optional retrain (`DEMO_MODE`). Headline numbers come from `results/*.csv`, not from a fresh 15-seed run in the notebook.

## Key results (fifteen seeds)

| System | PE_1h | PE_6h | PE_12h | PE_st,6h | PE_pers,1h |
|--------|-------|-------|--------|----------|------------|
| LSTM | 0.955 | 0.881 | 0.840 | 0.788 | −1.19 |
| Transformer (default) | 0.978 | 0.895 | 0.845 | 0.797 | −0.08 |
| STORM-Bz | 0.986 | 0.900 | 0.854 | 0.812 | +0.31 |
| No-Delay | 0.987 | 0.902 | 0.855 | 0.809 | +0.36 |
| No-Physics | 0.986 | 0.899 | 0.850 | 0.807 | +0.31 |
| No-Gate | 0.986 | 0.900 | 0.856 | 0.819 | +0.32 |
| Transformer matched | 0.980 | 0.895 | 0.845 | 0.809 | −0.01 |
| STORM-Bz bagged | **0.987** | **0.910** | **0.870** | 0.836 | — |
| TF matched bagged | 0.984 | 0.908 | 0.861 | 0.831 | — |

GRASP: 6 h 0.740 → 0.841; 12 h 0.567 → 0.762.

Primary control is the architecture-matched Transformer (`d_model=128`, 2 layers, 4 heads). It is **not** FFN-width matched (`d_ff` 256 vs PyTorch 2048). Module ablations show the 1 h gain is a training-package effect.

## Reproduce

```bash
pip install -r requirements.txt
# open notebooks/STORM_PhysNet_Master.ipynb
# DEMO_MODE = False only if you retrain all 15 seeds
```

Official chronological cut used for tables: 63 394 / 7 924 / 15 850 hours. Test PE is never used for model selection.

## Citation

```bibtex
@inproceedings{samarth2026storm,
  title={STORM-PhysNet: A Multi-Horizon Transformer for Geostationary Relativistic Electron Flux Forecasting with Physics-Inspired Components and Cross-Satellite Transfer},
  author={Samarth BN},
  year={2026},
  note={IEEE conference, under review}
}
```

## License

MIT for code. Follow NOAA NCEI / NASA OMNIWeb / ISSDC terms for GOES, OMNI, and GRASP.
