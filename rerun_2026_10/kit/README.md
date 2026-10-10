# STORM-PhysNet – fixed re-run kit (3 Kaggle accounts)

What this does: retrains every model on a corrected data pipeline and a fixed split, then builds every table,
figure and number for the paper from the saved predictions, so paper = code = results.

## What is fixed vs. the current repo/paper
- No back-filled / interpolated GOES gaps. Missing hours are forward-filled on the *input* side only, with an explicit
  "observed" mask channel. Targets are real observations only; a window counts only if its last input hour and its
  t+1/t+6/t+12 targets are observed.
- Kp is /10 (OMNI stores x10) and causal (no leak of the current 3-h interval); storm_onset capped at 720 h (no 9999 sentinel);
  no centred Hampel filter.
- Row-labelled splits: a window is used only if all its 84 rows belong to one split (automatic buffers).
- Three test blocks (A 2018-09→2020-03, B 2015-16, C 2012-13) share one validation period (2017-03→2018-09) and each
  trains on the remaining data of its own. Scaler fit on training rows only.
- Models/training recipe = the repo's (Trainer.train_single, config.yaml: 40 epochs, patience 12, lr 3e-4, storm weight 12).
  Physics terms in the loss still act on standardised channels (unchanged, to stay the same model as in the paper).
- New controls with code in this kit: transformer+residual head (tf_ffn_res), flux-only transformer (tf_flux_only).
- GRASP: one script, same recipe for STORM-Bz and matched Transformer; zero-shot / heads-only / full fine-tune / scratch;
  three chronological folds.
All numbers WILL differ from the current paper. Tables/figures/text must be taken from the report.

## Steps
1. On any account: Kaggle -> Datasets -> New Dataset -> upload `storm_kaggle_kit.zip` contents (the folder with
   train_block.py etc.), name it `storm-kit`. Make it private; add the other two accounts as collaborators, or repeat
   the upload on each account.
2. Import the notebook for each account (File -> Import Notebook):
   - Account 1: `1_account1_blockA_GRASP.ipynb`  (block A, then GRASP; GRASP needs the A checkpoints)
   - Account 2: `2_account2_blockB.ipynb`
   - Account 3: `3_account3_blockC.ipynb`
3. In each notebook: Add Input -> your `storm-kit` dataset; Settings: Accelerator = GPU T4 (not P100 – recent PyTorch
   dropped its architecture), Internet = ON. Then **Save Version -> Save & Run All (Commit)**; it runs headless.
4. When finished: Output tab -> download `results_*.zip` (A+GRASP, B, C).
5. Combine account: upload the three zips as one dataset, add `storm-kit` and that dataset as inputs, run
   `4_combine_report.ipynb` (no GPU needed). Download `paper_report.zip`:
   `PAPER_NUMBERS.md` (start here – it has automatic ✅/❌ checks of every claim), `tables/*.tex|csv`, `figures/*.png`, `numbers.json`.

## If a session hits the 12 h limit
Finished runs are saved one by one. Add the previous version's output as an Input (it contains `out_A` etc.) and run
again: finished runs are skipped. Seeds run in order 42..51 for all families, so stopping early still leaves balanced seeds
(the combine step uses only seeds present for every family of a block).

## Cost
Per block: 7 families x 10 seeds = 70 trainings (+ inference with noise). GRASP: 10 seeds x 3 folds x 2 models x 3 trained variants.
Measured on CPU only (smoke test); GPU time is an estimate: roughly 4-10 min per training -> about 5-12 h per block.
Total about 20-35 GPU-h across the three accounts, i.e. well inside 3 x 30 h/week. To cut cost, use `--seeds 42-46` (5 seeds)
in the notebook run cell.

## Where the code comes from
- From GitHub (cloned by the notebook, pinned to commit 72008c75): the STORM-PhysNet-L1 repo's `datasets/` (GOES, OMNI, GRASP),
  `src/` (Trainer, STORM-PhysNet, baselines, physics loss, CDF readers) and `configs/config.yaml`.
  Pinning means later pushes to GitHub do not change the rerun.
- From the `storm-kit` Kaggle dataset (not on GitHub): everything in this folder – the fixed pipeline and the scripts.
  After the rerun, push this folder to the repo (e.g. `rerun/`) so the paper's code link reproduces the new numbers.

## Note on GRASP "heads-only"
STORM-Bz's heads are small MLPs (63,357 trainable weights); the matched Transformer's head is one linear layer (387).
So heads-only is not equal capacity; compare the two models on the `full` fine-tune variant, which updates every weight in both.

## Files
storm_fixed/data.py (pipeline), storm_fixed/models_extra.py, train_block.py, grasp_transfer.py, combine_report.py,
make_notebooks.py (regenerates notebooks), notebooks/*.ipynb.
