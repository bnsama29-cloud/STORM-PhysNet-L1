# STORM-PhysNet (conference)

Official code for the **IEEE conference** paper:

**STORM-PhysNet: A Multi-Horizon Transformer for Geostationary Relativistic Electron Flux Forecasting with Physics-Inspired Components and Cross-Satellite Transfer**

Horizons are **1 h / 6 h / 12 h** on hourly GOES–OMNI. Transfer target: GSAT-19 GRASP (Indian longitude).

This repository contains only the conference systems (LSTM, default Transformer, architecture-matched Transformer, STORM-Bz, and the three module ablations). It does not include Access-only probes.

## Key results (fifteen seeds)

| System | PE_1h | PE_6h | PE_12h | PE_st,6h | PE_pers,1h |
|--------|-------|-------|--------|----------|------------|
| LSTM | 0.955 | 0.881 | 0.840 | 0.788 | −1.19 |
| Transformer (default) | 0.978 | 0.895 | 0.845 | 0.797 | −0.08 |
| STORM-Bz | **0.986** | 0.900 | 0.854 | 0.812 | **+0.31** |
| No-Delay | 0.987 | 0.902 | 0.855 | 0.809 | +0.36 |
| No-Physics | 0.986 | 0.899 | 0.850 | 0.807 | +0.31 |
| No-Gate | 0.986 | 0.900 | 0.856 | 0.819 | +0.32 |
| Transformer matched | 0.980 | 0.895 | 0.845 | 0.809 | −0.01 |
| STORM-Bz bagged | **0.987** | **0.910** | **0.870** | 0.836 | — |
| TF matched bagged | 0.984 | 0.908 | 0.861 | 0.831 | — |

- Primary control: architecture-matched Transformer (`d_model=128`, 2 layers, 4 heads).
- Default Transformer (`d_model=64`, 3 layers) is **not** capacity-matched.
- Short-horizon gain is a **training-package** effect; removing delay, gate, or physics loss at inference does not erase it.
- GRASP fine-tune: 6 h PE 0.740 → 0.841; 12 h 0.567 → 0.762 (fifteen seeds).

Official CSVs are in `results/`. Checkpoints for seeds 42–56 are in `checkpoints/<model>/`.

## Reproduce

```bash
pip install -r requirements.txt
# then run notebooks/STORM_PhysNet_Master.ipynb
# set DEMO_MODE = False only if you want a full 15-seed retrain
```

Test PE is computed once after training and is not used for model selection. Headline tables load from `results/*.csv`.

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

MIT for code. Follow NOAA / NASA / ISSDC terms for GOES, OMNI, and GRASP data.
