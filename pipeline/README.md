# Pipeline: training, transfer and reporting

This folder trains every model, runs the GRASP transfer experiment, and builds every table, figure and number in `results/` and `paper/` from the saved predictions.

## Files

| File | Purpose |
|---|---|
| `storm_pipeline/data.py` | Causal data pipeline. Missing GOES hours are never filled from later data; inputs are forward-filled and an observed-flux mask is an input channel; Kp is causal (last completed 3-h interval) and rescaled from OMNI's ×10; rows are labelled per split, so no window crosses a boundary; scalers use training rows only. Also defines the three test blocks and the shared validation period. |
| `storm_pipeline/models_extra.py` | Control models: Transformer + residual head, flux-only Transformer |
| `train_block.py` | Trains every model family and seed for one block (A, B or C), saving test predictions, noise-probe predictions and the delay/gate values. Resumable. |
| `grasp_transfer.py` | GOES-15 → GSAT-19 GRASP transfer: 3 folds × 4 regimes (zero-shot, heads-only fine-tune, full fine-tune, GRASP-only) × 2 models × 10 seeds |
| `combine_report.py` | Builds all tables (`.csv` / `.tex`), figures, `numbers.json` and `PAPER_NUMBERS.md` from the predictions |
| `make_notebooks.py` | Regenerates the Kaggle notebooks in `notebooks/` |

Model architectures, the trainer and the loss come from `src/`. Settings are in `configs/config.yaml`: 40 epochs, early-stopping patience 12, learning rate 3e-4, storm oversampling ×12.

## Rebuild the results from the saved predictions (CPU, about 1 min)

```bash
cd predictions && for z in *.zip; do unzip -q "$z" -d "${z%.zip}"; done
python ../pipeline/combine_report.py --inputs preds_blockA_and_GRASP preds_blockB preds_blockC --out ../results_check
```

## Retrain everything (about 25 T4 GPU-hours)

The notebooks split the work across three free Kaggle accounts, plus a CPU-only combine step.

1. Upload this `pipeline/` folder as a private Kaggle dataset (for example `storm-kit`).
2. Import one notebook per account:
   - `1_account1_blockA_GRASP.ipynb`: block A, then GRASP (GRASP needs the block-A checkpoints)
   - `2_account2_blockB.ipynb`: block B
   - `3_account3_blockC.ipynb`: block C
3. In each notebook, add the dataset as an Input. Set Accelerator to GPU T4 and Internet to ON. Then run **Save Version → Save & Run All (Commit)**; each one takes about 9 h.
4. Download `results_A_GRASP.zip`, `results_B.zip` and `results_C.zip` from the Output tabs.
5. Run `4_combine_report.ipynb` with the three zips as inputs (no GPU needed). Alternatively, run `combine_report.py` locally as shown above.

The notebooks clone this repository at a pinned commit for `src/` and `datasets/`, so the code version is fixed.

**If a session hits Kaggle's 12 h limit:** add the previous version's output as an Input and run again. Finished runs are skipped. Seeds run in order 42 to 51 across all families, so a partial run still has balanced seeds.
