# STORM-PhysNet

Code, data pipeline, predictions and paper for:

**STORM-PhysNet: A Multi-Horizon Transformer for Geostationary Relativistic Electron Flux Forecasting and Cross-Satellite Transfer**
Samarth B N, Samrudh S Malali, Dhyan M, Sanjana H V (RV College of Engineering). IEEE AGERS 2026, under review.

STORM-PhysNet forecasts GOES-15 >2 MeV electron log-flux at geostationary orbit 1, 6 and 12 h ahead from a 72 h history of flux and OMNI solar-wind/geomagnetic drivers. It adds three components to a Transformer encoder: a learnable propagation delay, a Bz-conditioned gate and residual-over-persistence heads. The study also tests transfer to the GRASP sensor on India's GSAT-19 (about 82°E).

> **Version note (October 2026).** All current numbers come from a leakage-free re-run in [`rerun_2026_10/`](rerun_2026_10/). The original tables in `results/`, `figures/` and `checkpoints/` (15 seeds, gap-filled data, one test period) are kept for the record. They are **superseded** and should not be cited. See [What changed](#what-changed-in-the-re-run).

## Headline results (re-run)

Prediction efficiency against climatology (PE). Seed means over 10 seeds, averaged over three test periods (2012–13, 2015–16, 2018–20; 34,554 test windows).

| Model | Params | PE 1 h | PE 6 h | PE 12 h | PE vs persistence, 1 h |
|---|---:|---:|---:|---:|---:|
| Persistence | – | 0.968 | 0.581 | 0.261 | 0.00 |
| LSTM | 55 k | 0.937 | 0.838 | 0.792 | −0.97 |
| Transformer, architecture-matched | 1.19 M | 0.969 | 0.867 | 0.813 | 0.04 |
| Transformer + residual head | 268 k | 0.976 | **0.872** | **0.815** | 0.25 |
| Transformer, flux-only | 1.19 M | 0.968 | 0.834 | 0.753 | 0.01 |
| **STORM-PhysNet** | 344 k | 0.977 | 0.865 | 0.803 | 0.29 |
| STORM-PhysNet, no delay | 344 k | **0.978** | 0.869 | 0.806 | **0.32** |
| STORM-PhysNet, no gate | 344 k | 0.977 | 0.867 | 0.805 | 0.28 |

What the numbers say:
- **1 h.** STORM-PhysNet beats the matched Transformer in every test period (+0.008, paired p = 3.5e-18 over 30 seed–period pairs). The gain comes from the residual-over-persistence head.
- **6 h and 12 h.** The two models are tied at 6 h (p = 0.14), and the matched Transformer is ahead at 12 h (−0.010). Solar-wind inputs add +0.034 and +0.060 PE at 6 and 12 h.
- **Delay and gate.** They do not improve accuracy on clean data. They do coincide with much better robustness. With unit-variance noise on the drivers, 1 h PE is 0.970 for STORM-PhysNet versus 0.892 for the matched Transformer.
- **GRASP (GSAT-19).** Zero-shot, STORM-PhysNet scores 0.961 / 0.827 / 0.738 at 1 / 6 / 12 h against 0.887 / 0.751 / 0.644 for the matched Transformer. After full fine-tuning the two are equal: 0.964 / 0.861 / 0.802 versus 0.964 / 0.857 / 0.794.

Full tables, paired tests, bootstrap intervals, storm-time errors, event skill, noise and GRASP results are in [`rerun_2026_10/results/`](rerun_2026_10/results/). Start with `PAPER_NUMBERS.md`.

## Repository layout

| Path | Contents | Status |
|---|---|---|
| `rerun_2026_10/kit/` | Leakage-free data pipeline (`storm_fixed/`), training (`train_block.py`), GRASP transfer (`grasp_transfer.py`), report builder (`combine_report.py`), Kaggle notebooks | **current** |
| `rerun_2026_10/predictions/` | Every test prediction: 210 GOES runs (3 periods × 7 models × 10 seeds) plus 10 GRASP seeds, with run logs and split metadata | **current** |
| `rerun_2026_10/results/` | All tables (`.csv` and `.tex`), figures, `numbers.json`, `PAPER_NUMBERS.md` with automatic claim checks | **current** |
| `rerun_2026_10/paper/` | Revised manuscript (`.tex` and `.pdf`), its figures, EDAS abstract | **current** |
| `src/` | Model code: STORM-PhysNet, Transformer, LSTM, trainer and losses. The re-run kit imports it unchanged. | current |
| `configs/` | Training configuration used by both versions | current |
| `datasets/` | GOES-15, OMNI and GRASP input files | current |
| `docs/audit/` | Audit that found the leakage, plus a verification script | reference |
| `docs/literature/` | Annotated literature collection, with corrections to earlier citations | reference |
| `results/`, `figures/`, `checkpoints/`, `notebooks/` | Original submission (15 seeds, gap-filled data) | superseded |

## What changed in the re-run

An audit of the original pipeline ([`docs/audit/AUDIT_FINDINGS.md`](docs/audit/AUDIT_FINDINGS.md)) found four leakage paths. All are fixed in `rerun_2026_10/kit/storm_fixed/data.py`.

1. **Gap filling from the future.** About 27–32% of GOES hours are missing. They were back-filled from later observations. Now inputs are only forward-filled, an observed-flux mask is an input channel, and a window is scored only if its last input hour and all three target hours are real observations.
2. **Centred despiking.** A centred Hampel filter looked at future values. It has been removed.
3. **Kp look-ahead.** OMNI repeats each 3-hourly Kp on all three hours, which exposes up to 2 h of future. Now each hour sees only the last completed Kp interval. Kp is also rescaled (OMNI stores Kp × 10).
4. **Windows across split boundaries.** Every row is now labelled train, validation or test. A window is kept only if all of its 84 rows share one label, which leaves an 84 h buffer at each boundary.

The evaluation design changed too: three disjoint test periods instead of one, a shared validation period, ten seeds, paired tests, bootstrap intervals, and three new controls (residual-head Transformer, flux-only Transformer, four GRASP regimes).

On the same 2018–20 test period, 12 h PE dropped from 0.854 (old pipeline) to 0.804, which shows how much the leaks inflated skill.

## Reproduce

Tables and figures from the saved predictions (CPU, about 1 minute):

```bash
pip install -r requirements.txt
cd rerun_2026_10/predictions
for z in *.zip; do unzip -q "$z" -d "${z%.zip}"; done
python ../kit/combine_report.py --inputs preds_blockA_and_GRASP preds_blockB preds_blockC --out ../results_check
```

Retraining from scratch takes about 25 GPU-hours on a T4. Use the notebooks in `rerun_2026_10/kit/notebooks/` and follow `rerun_2026_10/kit/README.md`. They are written for three free Kaggle accounts in parallel, plus a CPU-only combine step.

The kit was run against this repository at commit `72008c75`.

## Data

- GOES-15 EPEAD >2 MeV electrons: NOAA NCEI, https://www.ngdc.noaa.gov/stp/satellite/goes/
- OMNI hourly solar wind and indices: NASA OMNIWeb, https://omniweb.gsfc.nasa.gov/
- GSAT-19 GRASP: ISSDC PRADAN, https://pradan1.issdc.gov.in/grasp/

Follow each provider's terms of use.

## Citation

```bibtex
@inproceedings{samarth2026stormphysnet,
  title     = {STORM-PhysNet: A Multi-Horizon Transformer for Geostationary Relativistic Electron Flux Forecasting and Cross-Satellite Transfer},
  author    = {Samarth B N and Samrudh S Malali and Dhyan M and Sanjana H V},
  booktitle = {IEEE Asia-Pacific Conference on Geoscience, Electronics and Remote Sensing Technology (AGERS)},
  year      = {2026},
  note      = {Under review}
}
```

## License

MIT for code (see `LICENSE`).
