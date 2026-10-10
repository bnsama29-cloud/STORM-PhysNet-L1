# STORM-PhysNet audit — what was run and what it found
Run on CPU (no GPU needed) from the released L2 repo: raw GOES/OMNI/GRASP data + all 105 GOES checkpoints (7 model families x 15 seeds).

## How to re-run (Colab or Kaggle, CPU is enough; GPU makes it faster)
    !git clone https://github.com/bnsama29-cloud/STORM-PhysNet-L2
    !pip -q install torch numpy pandas scipy pyyaml cdflib scikit-learn tensorboard tqdm
    # upload verify_storm_physnet.py next to the repo, then:
    !python verify_storm_physnet.py --repo STORM-PhysNet-L2 --out verification --quick   # ~2 min
    !python verify_storm_physnet.py --repo STORM-PhysNet-L2 --out verification --noise    # full, ~25 min CPU

## Confirmed (reproduces exactly)
- All 105 checkpoints x 3 horizons re-score to the CSV with max difference 0.000000 -- but only on the default 70/15/15 split.
- Table I means, bagged rows, parameter counts (343,391 / 844,803 / 1,188,739), GRASP t / p / Cohen's d (5.60, 7.01), ensemble alpha counts.

## Errors found (paper or repo)
1. SPLIT. Results use 70/15/15 = 61,017 / 13,075 / 13,076 rows (60,933 / 12,991 / 12,992 windows). The paper says the 63,394/7,924/15,850 cut was used and 70/15/15 was not -- false. The repo notebook builds the 63,394 cut, which does NOT reproduce the CSVs (1,050 vs 932 storm windows; every PE shifts by +0.001 to +0.01).
2. IMPUTED DATA. NOAA quality flag / flux<=0 -> NaN: 26.7% of train, 17.5% of val, 28.7% of test hours. Short gaps interpolated, long gaps back/forward-filled (back-fill copies LATER values). 65% of test windows contain >=1 imputed hour in the lookback. Not disclosed.
3. HEADLINE ORDERING. On windows with observed targets (no imputed hour in lookback/target), STORM-Bz minus matched Transformer: 1 h +0.006..+0.007 (15/15 seeds, p~1e-4..1e-5) holds; 6 h -0.003..-0.005 and 12 h +0.001..-0.005 (not significant / reversed). Bagged on gap-free windows: STORM 0.870/0.819 vs matched 0.878/0.826 vs default TF 0.881/0.826.
4. "No-Delay gap within seed variation" is wrong: paired p ~ 7e-8 at 1 h (0.0009, tiny but consistent). Removing the delay slightly IMPROVES 1 h.
5. DELAY "remainder spread across the interior" is wrong: 73-99% of test windows sit at tau >= 1.49 h depending on seed (min tau 0.85-1.30 h). The delay is effectively a constant at the upper bound.
6. GATE p = 1.2e-14 not reproducible: seed-averaged storm/quiet AUC 0.89, p < 1e-300 (windows overlap, so nominal p is overstated; report AUC and the small mean shift 0.460 vs 0.440 (15-seed average)).
7. STRONG-STORM BIN (PE6h > 0.70 at Dst <= -100) cannot be computed on the test block: it has 59 storm hours, min Dst -73 nT, zero hours <= -100. The claim can only come from the full sample, which includes training data (in-sample). With target Dst <= -50 (n=59 windows) STORM-Bz PE6h = 0.57-0.61.
8. GRASP "118/1781 test hours" are WINDOWS. Test tail = 1,865 h -> 1,781 windows -> 118 flagged; actual storm hours = 47, all one event (25-26 Aug 2018).
9. GRASP: the matched Transformer beats STORM-Bz after fine-tuning at 6 h (0.878 vs 0.841) and 12 h (0.831 vs 0.762); STORM leads only at 1 h (0.968 vs 0.966). The paper quotes 0.878 but never states it exceeds STORM's headline.
10. FFN-matched + persistence-residual Transformer (tf_ffn256_res) ties or beats STORM-Bz at every horizon at sigma=0 and under input noise (6 h at sigma=1: 0.852 vs 0.848). The noise advantage over "matched TF" comes from the residual heads, not the physics modules.
11. Test period is 7 Sep 2018 - 4 Mar 2020 (declining phase to minimum): 22 short Dst<=-50 episodes, none strong.
12. Repo: figures stale (L1/L2 fig_horizon_pe contradicts Table I); notebook split wrong; requirements omit tensorboard/tqdm (imported by trainer).

## Not reproducible here
- GRASP fine-tuned numbers (0.841 / 0.762 etc.): no GRASP checkpoints or script in the repo (the paper says the run was external).
- Noise probe: same direction (STORM 0.982/0.842, matched 0.931/0.816 at sigma=1 on the full test set) but not the exact CSV numbers (probe used a different fixed subset).
13. Repo: the FFN-matched Transformer (tf_ffn256) and residual-head control (tf_ffn256_res) checkpoints exist (276,867 params each) but no code in src/ builds them; they cannot be re-scored from the released code. Their only numbers are in the noise-probe CSVs (a different fixed evaluation subset).
