# Leakage-free re-run (October 2026)

Everything behind the revised STORM-PhysNet paper (v2): code, every prediction, every table and figure, and the manuscript itself. Each number in `paper/` can be regenerated from `predictions/` with `kit/combine_report.py`.

## Design

| | Test period | Train windows | Test windows | Gap-free test | Storm windows (Dst ≤ −50 nT at t+6 h) | Min Dst |
|---|---|---:|---:|---:|---:|---:|
| Block A | Sep 2018 – Mar 2020 | 38,909 | 7,536 | 4,283 | 18 | −69 nT |
| Block B | Jan 2015 – Dec 2016 | 32,296 | 14,021 | 8,452 | 565 | −198 nT |
| Block C | Jan 2012 – Dec 2013 | 33,413 | 12,997 | 7,189 | 354 | −139 nT |

- **Validation:** shared by all three blocks, Mar 2017 – Sep 2018 (9,264 windows). Each block trains on all remaining hours.
- **Models:** seven families, each with seeds 42–51, for 210 GOES runs in total (about 23 T4 GPU-hours).
- **GRASP (GSAT-19):** Jul 2017 – Aug 2018. There are three chronological folds:

  | Fold | Test period | Test windows |
  |---|---|---:|
  | G1 | Feb – Mar 2018 | 1,316 |
  | G2 | Apr – May 2018 | 1,381 |
  | G3 | Jun – Aug 2018, including the 25–26 Aug storm | 2,109 |

  Each fold runs four regimes for STORM-PhysNet and for the matched Transformer: zero-shot, heads-only fine-tuning, full fine-tuning, and training on GRASP only. Every regime uses all ten seeds.

| Family (code name) | Description | Params |
|---|---|---:|
| `storm_bz` | STORM-PhysNet: delay + Bz gate + residual heads | 343,599 |
| `storm_no_delay` | STORM-PhysNet without the delay | 343,599 |
| `storm_no_gate` | STORM-PhysNet without the gate | 343,599 |
| `transformer_matched` | d_model 128, 2 layers, 4 heads, d_ff 2048, no residual | 1,188,867 |
| `tf_ffn_res` | d_model 128, 2 layers, 4 heads, d_ff 256, persistence residual | 267,779 |
| `tf_flux_only` | matched Transformer with driver channels zeroed | 1,188,867 |
| `lstm` | LSTM baseline | 54,979 |

## Folder contents

```
kit/
  storm_fixed/data.py        leakage-free pipeline: causal Kp, no back-fill, observed mask, row-labelled splits, blocks
  storm_fixed/models_extra.py residual-head and flux-only Transformer controls
  train_block.py             trains all families x seeds for one block; resumable; saves predictions + noise probes
  grasp_transfer.py          GOES-15 -> GRASP transfer, 3 folds x 4 regimes x 2 models x 10 seeds
  combine_report.py          builds every table, figure, numbers.json and PAPER_NUMBERS.md from predictions
  make_notebooks.py          regenerates the Kaggle notebooks
  notebooks/                 1-3: one Kaggle account per block (A + GRASP, B, C); 4: combine (CPU)
  README.md                  step-by-step Kaggle instructions
predictions/
  preds_blockA_and_GRASP.zip out_A/ (70 runs, meta_A.npz, runlog_A.jsonl) + out_G/ (10 GRASP seeds, grasp_meta.npz, log)
  preds_blockB.zip           out_B/ (70 runs, meta_B.npz, runlog_B.jsonl)
  preds_blockC.zip           out_C/ (70 runs, meta_C.npz, runlog_C.jsonl)
results/
  PAPER_NUMBERS.md           start here: claim checks, per-block and averaged tables, paired tests, gate/delay stats
  numbers.json               every number as JSON
  tables/                    T0 splits, T1 main (per block, bagged, block average), T2 gap-free, T3 paired tests,
                             T4 event skill, T5 storm vs quiet RMSE, T6 bootstrap CIs, T7 noise, T8 GRASP, T9 compute
  figures/                   horizon PE, per-block deltas, noise, gate/delay, case study (June 2015), GRASP
paper/
  STORM_PhysNet_AGERS2026_v2.tex / .pdf   revised manuscript (7 pages, 20 references)
  figures/                   the figures used in the manuscript
  EDAS_abstract_v2.txt       abstract for the submission system
```

Each `preds_<block>_<family>_s<seed>.npz` holds `pred` (test windows × 3 horizons, log10 flux), `pred_noise{0.5,1,2}` for the noise-probed families, and `tau` and `gate` for `storm_bz`. The matching `meta_<block>.npz` holds the targets, persistence, the gap-free flag, storm flags, Dst at the targets, timestamps and the scaler.

## Regenerate the results

```bash
cd predictions && for z in *.zip; do unzip -q "$z" -d "${z%.zip}"; done
python ../kit/combine_report.py --inputs preds_blockA_and_GRASP preds_blockB preds_blockC --out ../results_check
diff -r ../results/tables ../results_check/tables && echo "tables identical"
```

## Retrain

Follow `kit/README.md`. You need three Kaggle accounts with T4 GPUs, about 9 h each, plus a CPU-only combine step. The kit clones this repository at commit `72008c75` for the model code, so later edits to `src/` cannot change the results.

## Known limitations

- Kp and Dst are definitive values. Real-time products differ.
- Blocks B and C train on data from both before and after their test period. This is blocked cross-validation, not forward chaining.
- Only 937 test windows fall in storm time.
- GRASP covers 14 months near solar minimum. It is not cross-calibrated to GOES and runs without driver inputs.
- p-values are not corrected for multiple comparisons.
