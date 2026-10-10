#!/usr/bin/env python3
"""
verify_storm_physnet.py  --  independent audit of the STORM-PhysNet paper claims.

Run from anywhere:
    python verify_storm_physnet.py --repo /path/to/STORM-PhysNet-L2 --out ./verification
    python verify_storm_physnet.py --repo ... --quick        # 2 models x 3 seeds, ~2 min on CPU

Needs: torch numpy pandas scipy pyyaml cdflib scikit-learn tensorboard tqdm  (all in the repo's requirements.txt;
tensorboard + tqdm are imported by src/training/trainer.py).
Everything it needs (raw data, 15-seed checkpoints, result CSVs) is inside the repo, no downloads.

What it checks (each section prints PASS / FAIL / INFO and writes a CSV):
  1. Which train/val/test cut reproduces results/all_seed_results_full.csv (paper cut 63394/7924/15850 vs default 70/15/15)
  2. Re-scores every released checkpoint and reports max |re-run - CSV|
  3. Share of GOES hours that are imputed (NOAA quality flag -> NaN -> interpolate/bfill/ffill) and PE on gap-free windows
  4. Paired seed-level tests (STORM-Bz vs matched Transformer, ablations vs STORM-Bz) on three window subsets
  5. Bagged (15-seed average) PE on the same subsets
  6. Gate storm-vs-quiet Mann-Whitney / AUC and learned-delay saturation
  7. Storm content of the test block (hours, events, min Dst, strong-storm bin)
  8. GRASP test-window counts (118/1781)
"""
import argparse, os, sys, json, time
from copy import deepcopy
from pathlib import Path
import numpy as np, pandas as pd, torch, yaml
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default=".")
ap.add_argument("--out", default="verification")
ap.add_argument("--quick", action="store_true", help="storm_bz + transformer_matched, seeds 42-44 only")
ap.add_argument("--noise", action="store_true", help="also run the sigma=1 input-noise probe (slower)")
ap.add_argument("--batch", type=int, default=512)
args = ap.parse_args()

REPO = Path(args.repo).resolve(); OUT = Path(args.out).resolve(); OUT.mkdir(parents=True, exist_ok=True)
os.chdir(REPO); sys.path.insert(0, str(REPO))
from src.data.cdf_reader import read_goes_directory, read_wind_directory, read_grasp_directory
from src.data.preprocessor import Preprocessor
from src.data.dataloader import make_dataloaders
from src.training.trainer import Trainer

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEV.type == "cpu": torch.set_num_threads(os.cpu_count() or 2)
base = yaml.safe_load(open(REPO / "configs/config.yaml"))
SPECS = {
    "lstm": dict(model_type="lstm", ablation="none", match=False),
    "transformer": dict(model_type="transformer", ablation="none", match=False),
    "transformer_matched": dict(model_type="transformer", ablation="none", match=True),
    "storm_bz": dict(model_type="storm_physnet", ablation="none", match=False),
    "storm_no_delay": dict(model_type="storm_physnet", ablation="no_delay", match=False),
    "storm_no_gate": dict(model_type="storm_physnet", ablation="no_bz_gate", match=False),
    "storm_no_physics": dict(model_type="storm_physnet", ablation="no_physics", match=False),
}
MODELS = ["storm_bz", "transformer_matched"] if args.quick else list(SPECS)
SEEDS = list(range(42, 45)) if args.quick else list(range(42, 57))
HZ = {1: 0, 6: 1, 12: 2}
LOG = []
def say(tag, msg):
    line = f"[{tag}] {msg}"; print(line, flush=True); LOG.append(line)
def pe(a, b): return 1 - np.mean((a - b) ** 2) / np.var(a)

def cfg_for(s):
    c = deepcopy(base); c["model_type"] = s["model_type"]; c["ablation"] = s["ablation"]
    c["match_storm_capacity"] = bool(s["match"]); c.setdefault("model", {})
    c["model"]["gate_type"] = "bz"; c["model"]["use_spectral_head"] = False
    c["training"]["checkpoint_dir"] = str(OUT / "_tmp_ckpt"); c["training"]["log_dir"] = str(OUT / "_tmp_logs")
    return c

@torch.no_grad()
def predict(model, loader, aux=False, sigma=0.0):
    model.eval(); g = torch.Generator().manual_seed(0); Y, P, PP, T, G = [], [], [], [], []
    for b in loader:
        x = b["x_sw"]
        if sigma > 0: x = x + sigma * torch.randn(x.shape, generator=g)
        x, xf, yp = x.to(DEV), b["x_flux"].to(DEV), b["y_persist"].to(DEV)
        try: o = model(x, xf, yp)
        except TypeError: o = model(x, xf)
        p = o["flux_pred"] if isinstance(o, dict) else o
        Y.append(b["y_flux"].numpy()); P.append(p.cpu().numpy()); PP.append(b["y_persist"].numpy())
        if aux and isinstance(o, dict):
            T.append(o["tau"].reshape(len(x), -1).mean(1).cpu().numpy()); G.append(o["gate_values"].reshape(len(x), -1).mean(1).cpu().numpy())
    r = [np.concatenate(a) for a in (Y, P, PP)]
    return r + ([np.concatenate(T), np.concatenate(G)] if aux and T else [])

# ----------------------------------------------------------------------------------------------- data
goes = read_goes_directory(str(REPO / "datasets/goes")); omni = read_wind_directory(str(REPO / "datasets/omni"))
raw = goes.join(omni, how="inner"); n = len(raw)

def build(pre):
    tr, va, te = pre.fit_transform(raw)
    L = make_dataloaders(tr, va, te, seq_len=72, batch_size=args.batch, storm_weight=12.0)
    return tr, va, te, L

# 1. which cut reproduces the CSV ---------------------------------------------------------------------
say("1", "Building the two candidate cuts")
cut_paper = build(Preprocessor(train_frac=63394 / n, val_frac=7924 / n))
cut_def = build(Preprocessor())  # repo default 70/15/15
rows = []
for lab, (tr, va, te, L) in [("paper-text cut 63394/7924/15850", cut_paper), ("default 70/15/15", cut_def)]:
    ds = L[2].dataset
    rows.append(dict(cut=lab, train_rows=len(tr), val_rows=len(va), test_rows=len(te), test_windows=len(ds),
                     test_storm_windows=int(ds.window_storm_flags.sum()),
                     test_start=str(te.index.min()), test_end=str(te.index.max())))
cuts = pd.DataFrame(rows); cuts.to_csv(OUT / "01_cuts.csv", index=False); print(cuts.to_string(index=False))
csv = pd.read_csv(REPO / "results/all_seed_results_full.csv")
n_csv = int(csv.n_storm.iloc[0])
say("1", f"CSV n_storm = {n_csv}; default 70/15/15 gives {int(cuts.test_storm_windows[1])}; paper-text cut gives {int(cuts.test_storm_windows[0])}")

tr, va, te, (trl, val, tel) = cut_def  # all further analysis uses the cut that matches the CSV
ds = tel.dataset; N = len(ds); sf = ds.window_storm_flags > 0
n_sw = int(next(iter(tel))["x_sw"].shape[-1])

# 2. re-score checkpoints -----------------------------------------------------------------------------
say("2", f"Re-scoring {len(MODELS)} model families x {len(SEEDS)} seeds on {DEV}")
preds, taus, gates, noisy = {}, {}, {}, {}
y = yp = None; t0 = time.time()
for name in MODELS:
    for s in SEEDS:
        pts = sorted((REPO / "checkpoints" / name / f"seed_{s}").glob("*_best.pt"))
        if not pts: say("2", f"missing {name} seed {s}"); continue
        m = Trainer(cfg_for(SPECS[name])).build_model(n_sw)
        m.load_state_dict(torch.load(pts[0], map_location="cpu", weights_only=True), strict=True); m.to(DEV)
        r = predict(m, tel, aux=(name == "storm_bz")); y, preds[(name, s)], yp = r[0], r[1], r[2]
        if name == "storm_bz" and len(r) > 3: taus[s], gates[s] = r[3], r[4]
        if args.noise and name in ("storm_bz", "transformer_matched"): noisy[(name, s)] = predict(m, tel, sigma=1.0)[1]
    say("2", f"{name} done ({time.time() - t0:.0f}s)")
mx = 0.0
for (name, s), p in preds.items():
    c = csv[(csv.name == name) & (csv.seed == s)].iloc[0]
    for h, i in HZ.items(): mx = max(mx, abs(pe(y[:, i], p[:, i]) - c[f"PE_{h}h"]))
say("2", f"{'PASS' if mx < 1e-4 else 'FAIL'}: max |re-run - results CSV| on default 70/15/15 = {mx:.6f}")

# 3. imputed hours --------------------------------------------------------------------------------------
na_all = goes["flux_gt2mev"].isna().values
ntr, nva = len(tr), len(va)
frac = pd.DataFrame([dict(split=k, rows=len(v), nan_rows=int(v.sum()), nan_frac=float(v.mean()))
                     for k, v in [("train", na_all[:ntr]), ("val", na_all[ntr:ntr + nva]), ("test", na_all[ntr + nva:])]])
frac.to_csv(OUT / "03_imputed_fraction.csv", index=False); print(frac.to_string(index=False))
na = na_all[ntr + nva:]; j = np.arange(N)
lb_any = np.array([na[a:a + 72].any() for a in j]); last = na[j + 71]; tg = {h: na[j + 72 + h - 1] for h in HZ}
say("3", f"INFO: {na.mean():.1%} of test hours are imputed; {lb_any.mean():.1%} of test windows have >=1 imputed hour in the 72 h lookback")
def mask(kind, h):
    if kind == "all": return np.ones(N, bool)
    if kind == "obs": return ~tg[h] & ~last          # observed target + observed last lookback hour
    return ~tg[h] & ~lb_any                          # strict: nothing imputed in lookback or target
KINDS = ["all", "obs", "strict"]
rows = []
for kind in KINDS:
    for name in MODELS:
        r = dict(subset=kind, model=name)
        for h, i in HZ.items():
            m = mask(kind, h); r[f"n_{h}h"] = int(m.sum())
            r[f"PE_{h}h"] = float(np.mean([pe(y[m, i], preds[(name, s)][m, i]) for s in SEEDS if (name, s) in preds]))
        rows.append(r)
    r = dict(subset=kind, model="persistence")
    for h, i in HZ.items():
        m = mask(kind, h); r[f"n_{h}h"] = int(m.sum()); r[f"PE_{h}h"] = float(pe(y[m, i], yp[m, i]))
    rows.append(r)
T = pd.DataFrame(rows); T.to_csv(OUT / "03_pe_by_subset.csv", index=False)
for k in KINDS: print(f"\n== subset: {k}"); print(T[T.subset == k].round(3).to_string(index=False))

# 4. paired tests ---------------------------------------------------------------------------------------
rows = []
def paired(a, b, kind, h):
    i = HZ[h]; m = mask(kind, h)
    A = np.array([pe(y[m, i], preds[(a, s)][m, i]) for s in SEEDS]); B = np.array([pe(y[m, i], preds[(b, s)][m, i]) for s in SEEDS])
    t = stats.ttest_rel(A, B)
    return dict(a=a, b=b, subset=kind, horizon=f"{h}h", mean_diff=float((A - B).mean()), t=float(t.statistic), p_t=float(t.pvalue), a_wins=f"{int((A > B).sum())}/{len(SEEDS)}")
if "storm_bz" in MODELS and "transformer_matched" in MODELS:
    for kind in KINDS:
        for h in HZ: rows.append(paired("storm_bz", "transformer_matched", kind, h))
for ab in ["storm_no_delay", "storm_no_gate", "storm_no_physics"]:
    if ab in MODELS:
        for kind in KINDS:
            for h in HZ: rows.append(paired(ab, "storm_bz", kind, h))
if rows:
    P = pd.DataFrame(rows); P.to_csv(OUT / "04_paired_tests.csv", index=False); print("\n", P.round(4).to_string(index=False))

# 5. bagging --------------------------------------------------------------------------------------------
rows = []
for kind in KINDS:
    for name in MODELS:
        r = dict(subset=kind, model=name + "_bagged")
        for h, i in HZ.items():
            m = mask(kind, h); Pm = np.mean([preds[(name, s)][:, i] for s in SEEDS if (name, s) in preds], axis=0); r[f"PE_{h}h"] = float(pe(y[m, i], Pm[m]))
        rows.append(r)
B = pd.DataFrame(rows); B.to_csv(OUT / "05_bagged.csv", index=False); print("\n", B.round(3).to_string(index=False))

# 6. gate and delay -------------------------------------------------------------------------------------
if gates:
    g = np.mean([gates[s] for s in SEEDS], axis=0); u = stats.mannwhitneyu(g[sf], g[~sf])
    n1, n2 = int(sf.sum()), int((~sf).sum()); z = (u.statistic - n1 * n2 / 2) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    auc = u.statistic / (n1 * n2); lp = (stats.norm.logsf(abs(z)) + np.log(2)) / np.log(10)
    say("6", f"INFO: gate (seed-averaged) storm mean {g[sf].mean():.3f} vs quiet {g[~sf].mean():.3f}; AUC {auc:.3f}; log10 p ~ {lp:.0f} (paper states p = 1.2e-14)")
    sat = [float((taus[s] >= 1.49).mean()) for s in SEEDS]; tmin = [float(taus[s].min()) for s in SEEDS]
    say("6", f"INFO: share of test windows with tau >= 1.49 h: {min(sat):.1%}..{max(sat):.1%} across seeds; min tau per seed {min(tmin):.2f}..{max(tmin):.2f} h")

# 7. storm content of the test block --------------------------------------------------------------------
dst = omni["dst"].reindex(te.index).values; st = (dst <= -50).astype(int)
ev = int(((st[1:] == 1) & (st[:-1] == 0)).sum() + st[0])
dst6 = dst[np.arange(N) + 72 + 6 - 1]
say("7", f"INFO: test block {te.index.min().date()} .. {te.index.max().date()}: {int(st.sum())} hours with Dst<=-50 in {ev} separate runs; min Dst {np.nanmin(dst):.0f} nT; hours with Dst<=-100: {int((dst <= -100).sum())}")
if "storm_bz" in MODELS:
    for lab, mm in [("target Dst<=-50", dst6 <= -50), ("target Dst<=-100", dst6 <= -100)]:
        v = np.mean([pe(y[mm, 1], preds[("storm_bz", s)][mm, 1]) for s in SEEDS]) if mm.sum() > 2 else float("nan")
        say("7", f"INFO: STORM-Bz PE6h on windows with {lab}: n={int(mm.sum())}, PE={v:.3f}")

# 8. GRASP counts ---------------------------------------------------------------------------------------
gr = read_grasp_directory(str(REPO / "datasets/grasp")); gh = gr.resample("1h").mean().dropna(how="all").join(omni, how="inner")
gs = (gh["dst"] <= -50).astype(int).values
for L in range(1790, 1900):
    N_ = L - 72 - 12
    if N_ == 1781:
        sd = gs[-L:]; fl = int(sum(sd[i:i + 72].max() for i in range(N_)))
        say("8", f"INFO: GRASP window {gh.index.min().date()} .. {gh.index.max().date()} ({len(gh)} h). Test tail of {L} h -> 1781 windows, {fl} flagged ({fl / N_:.1%}); "
                 f"storm HOURS in that tail = {int(sd.sum())} ({sd.sum() / L:.1%}). The paper's '118/1781 test hours' are windows, not hours.")

(OUT / "verification_log.txt").write_text("\n".join(LOG))
json.dump(dict(max_abs_diff_vs_csv=mx, device=str(DEV), models=MODELS, seeds=SEEDS), open(OUT / "summary.json", "w"), indent=1)
import shutil; shutil.rmtree(OUT / "_tmp_ckpt", ignore_errors=True); shutil.rmtree(OUT / "_tmp_logs", ignore_errors=True)
print("\nDone. Outputs in", OUT)
