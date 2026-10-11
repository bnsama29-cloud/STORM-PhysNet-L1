#!/usr/bin/env python
"""
Tables and figures for the driver-robustness probes and the extended GRASP transfer comparison.

  python combine_probes.py --inputs probe_A probe_B probe_C probe_G --out probe_report [--main out_A out_B out_C]

Reads the outputs of train_block.py --probes (preds_<B>_<family>_s<seed>.npz with pred, pred_noise*, pred_out*, pred_lag*)
and of grasp_transfer.py --families ... (grasp_preds_s<seed>.npz, grasp_meta.npz).
--main (optional): the original training runs; their clean scores are listed next to the probe runs' clean scores.
Conventions as in combine_report.py: PE_clim = 1 - MSE/Var(y) per block and horizon, seed means averaged over blocks,
paired t-tests over seed-block pairs, Holm correction within each table.
"""
import argparse, glob, json, os, re
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--inputs", nargs="+", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--main", nargs="*", default=[])
args = ap.parse_args()
OUT = Path(args.out); (OUT / "tables").mkdir(parents=True, exist_ok=True); (OUT / "figures").mkdir(exist_ok=True)
H = [1, 6, 12]
NAMES = {"storm_bz": "STORM-PhysNet", "storm_no_delay": "STORM, no delay", "storm_no_gate": "STORM, no gate",
         "tf_ffn_res": "TF + residual", "transformer_matched": "TF matched"}
COL = {"storm_bz": "#1b9e77", "storm_no_delay": "#66a61e", "storm_no_gate": "#e7298a", "tf_ffn_res": "#7570b3",
       "transformer_matched": "#d95f02"}
PROBES = [("pred", "clean"), ("pred_noise0.5", "noise 0.5"), ("pred_noise1", "noise 1"), ("pred_noise2", "noise 2"),
          ("pred_out6", "drivers missing 6 h"), ("pred_out24", "drivers missing 24 h"), ("pred_out72", "no drivers"),
          ("pred_lag3", "Kp/Dst 3 h late")]
NUM = {}
plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": .3, "figure.dpi": 150, "savefig.dpi": 300,
                     "font.family": "serif"})


def load_blocks(dirs):
    metas, P = {}, {}
    for d in dirs:
        for f in glob.glob(str(Path(d) / "**/*.npz"), recursive=True):
            n = os.path.basename(f)
            if ".tmp." in n: continue
            m = re.match(r"meta_([ABC])\.npz$", n)
            if m: metas[m.group(1)] = dict(np.load(f, allow_pickle=True))
            m = re.match(r"preds_([ABC])_(.+)_s(\d+)\.npz$", n)
            if m: P.setdefault(m.group(1), {}).setdefault(m.group(2), {})[int(m.group(3))] = dict(np.load(f))
    return metas, P


def pe_clim(y, p): return 1 - ((y - p) ** 2).mean(0) / y.var(0)


def paired(a, b):
    d = np.asarray(a) - np.asarray(b); n = len(d)
    if n < 2 or d.std(ddof=1) == 0: return dict(diff=float(d.mean()), p=np.nan, n=n)
    return dict(diff=float(d.mean()), p=float(stats.ttest_1samp(d, 0.0).pvalue), n=n)


def holm(p):
    p = np.asarray(p, float); out = np.full_like(p, np.nan); ok = ~np.isnan(p); q = p[ok]; m = len(q)
    order = np.argsort(q); adj = np.empty(m); run = 0.0
    for r, i in enumerate(order): run = max(run, min(1.0, (m - r) * q[i])); adj[i] = run
    out[ok] = adj; return out


def save(df, name, caption=""):
    df.to_csv(OUT / "tables" / f"{name}.csv", index=False)
    (OUT / "tables" / f"{name}.tex").write_text(f"% {caption}\n" + df.to_latex(index=False, escape=True, na_rep="--"))
    print(f"\n== {name}: {caption}\n" + df.to_string(index=False))


# ------------------------------------------------------------------------------------------------ GOES probes
metas, P = load_blocks(args.inputs)
BLOCKS = [b for b in "ABC" if b in metas and b in P]
if BLOCKS:
    FAMS = [f for f in NAMES if all(f in P[b] for b in BLOCKS)]
    for b in BLOCKS:                                                   # seeds present for every family (paired tests)
        common = set.intersection(*[set(P[b][f]) for f in FAMS])
        for f in FAMS: P[b][f] = {k: v for k, v in P[b][f].items() if k in common}
    NUM["seeds"] = {b: sorted(P[b][FAMS[0]]) for b in BLOCKS}; NUM["blocks"] = BLOCKS
    print("blocks", BLOCKS, "families", FAMS, "seeds", NUM["seeds"])
    # S[(fam, key)] -> (n_pairs, 3) PE per (block, seed) pair, ordered block-major
    S = {}
    for f in FAMS:
        for key, _ in PROBES:
            if all(key in P[b][f][s] for b in BLOCKS for s in P[b][f]):
                S[(f, key)] = np.concatenate([np.stack([pe_clim(metas[b]["y"], P[b][f][s][key]) for s in sorted(P[b][f])])
                                              for b in BLOCKS])
    # block-averaged seed means (same convention as the paper's tables)
    def bavg(f, key):
        out, i = [], 0
        for b in BLOCKS:
            n = len(P[b][f]); out.append(S[(f, key)][i:i + n].mean(0)); i += n
        return np.mean(out, 0)

    rows = []
    for key, lab in PROBES:
        for f in FAMS:
            if (f, key) in S:
                v = bavg(f, key); rows.append(dict(Probe=lab, Model=NAMES[f], **{f"PE {h}h": round(float(v[i]), 3) for i, h in enumerate(H)}))
                NUM[f"{f}|{lab}"] = v.tolist()
    save(pd.DataFrame(rows), "P1_probes", "PE_clim under each driver probe, seed means averaged over blocks")

    rows = []
    for key, lab in PROBES:
        for f in FAMS:
            if f == "storm_bz" or (f, key) not in S or ("storm_bz", key) not in S: continue
            for i, h in enumerate(H):
                r = paired(S[("storm_bz", key)][:, i], S[(f, key)][:, i])
                rows.append(dict(Probe=lab, Comparison=f"STORM-PhysNet - {NAMES[f]}", Horizon=f"{h}h", dPE=r["diff"], p=r["p"], n=r["n"]))
    t = pd.DataFrame(rows); t["p_holm"] = holm(t["p"].values)
    for c in ["dPE"]: t[c] = t[c].map(lambda x: f"{x:+.4f}")
    for c in ["p", "p_holm"]: t[c] = t[c].map(lambda x: f"{x:.2e}")
    save(t, "P2_probe_paired", "Paired differences (seed-block pairs); Holm-adjusted over the whole table")
    NUM["paired"] = t.to_dict("records")

    # degradation relative to clean (how much each model loses), the quantity a robustness claim is about
    rows = []
    for key, lab in PROBES[1:]:
        for f in FAMS:
            if (f, key) in S:
                d = S[(f, key)] - S[(f, "pred")]
                rows.append(dict(Probe=lab, Model=NAMES[f], **{f"loss {h}h": round(float(-d[:, i].mean()), 3) for i, h in enumerate(H)}))
    save(pd.DataFrame(rows), "P3_degradation", "Drop in PE_clim from clean inputs (positive = worse), block-averaged")

    if args.main:                                                      # clean scores of the original runs, side by side
        m2, P2 = load_blocks(args.main); rows = []
        for f in FAMS:
            if all(f in P2.get(b, {}) for b in BLOCKS):
                orig = np.mean([np.mean([pe_clim(m2[b]["y"], P2[b][f][s]["pred"]) for s in P2[b][f]], 0) for b in BLOCKS], 0)
                new = bavg(f, "pred")
                rows.append(dict(Model=NAMES[f], **{f"original {h}h": round(float(orig[i]), 3) for i, h in enumerate(H)},
                                 **{f"probe runs {h}h": round(float(new[i]), 3) for i, h in enumerate(H)}))
        if rows: save(pd.DataFrame(rows), "P0_clean_agreement", "Clean PE_clim: original runs vs the probe runs")

    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=False)
    xl = [lab for key, lab in PROBES]
    for i, h in enumerate(H):
        for f in FAMS:
            ys = [bavg(f, key)[i] if (f, key) in S else np.nan for key, _ in PROBES]
            ax[i].plot(range(len(xl)), ys, "o-", ms=3, lw=1.2, color=COL[f], label=NAMES[f])
        ax[i].set_xticks(range(len(xl))); ax[i].set_xticklabels(xl, rotation=55, ha="right", fontsize=6.5)
        ax[i].set_title(f"{h} h")
    ax[0].set_ylabel(r"$PE_{clim}$"); ax[0].legend(fontsize=6.5); fig.tight_layout()
    fig.savefig(OUT / "figures/fig_probes.png"); plt.close(fig)

# ------------------------------------------------------------------------------------------------ GRASP
G = {}
for d in args.inputs:
    for f in glob.glob(str(Path(d) / "**/grasp_preds_s*.npz"), recursive=True):
        if ".tmp." not in f: G[int(re.search(r"_s(\d+)\.npz", f).group(1))] = dict(np.load(f))
gm = [f for d in args.inputs for f in glob.glob(str(Path(d) / "**/grasp_meta.npz"), recursive=True)]
if G and gm:
    GM = dict(np.load(gm[0])); seeds = sorted(G); folds = sorted({k.split("_")[0] for k in G[seeds[0]]})
    fams = [f for f in NAMES if any(k.startswith(f"{folds[0]}_{f}_") for k in G[seeds[0]])]
    regs = ["zeroshot", "heads", "full", "scratch"]; GS = {}; rows = []
    pers = np.mean([pe_clim(GM[f"{fk}_y"], np.repeat(GM[f"{fk}_persist"][:, None], 3, 1)) for fk in folds], 0)
    rows.append(dict(Model="Persistence", Regime="--", **{f"PE {h}h": round(float(pers[i]), 3) for i, h in enumerate(H)}))
    for f in fams:
        for v in regs:
            if all(f"{fk}_{f}_{v}" in G[s] for fk in folds for s in seeds):
                GS[(f, v)] = np.mean([np.stack([pe_clim(GM[f"{fk}_y"], G[s][f"{fk}_{f}_{v}"]) for s in seeds]) for fk in folds], 0)
                m = GS[(f, v)].mean(0); NUM[f"grasp|{f}|{v}"] = m.tolist()
                rows.append(dict(Model=NAMES[f], Regime=v, **{f"PE {h}h": round(float(m[i]), 3) for i, h in enumerate(H)}))
    save(pd.DataFrame(rows), "P4_grasp", f"GRASP PE_clim, mean over folds {folds}, {len(seeds)} seeds")
    rows = []
    for v in regs:
        for f in fams:
            if f == "storm_bz" or (f, v) not in GS or ("storm_bz", v) not in GS: continue
            for i, h in enumerate(H):
                r = paired(GS[("storm_bz", v)][:, i], GS[(f, v)][:, i])
                rows.append(dict(Regime=v, Comparison=f"STORM-PhysNet - {NAMES[f]}", Horizon=f"{h}h", dPE=r["diff"], p=r["p"], n=r["n"]))
    if rows:
        t = pd.DataFrame(rows); t["p_holm"] = holm(t["p"].values)
        t["dPE"] = t["dPE"].map(lambda x: f"{x:+.4f}")
        for c in ["p", "p_holm"]: t[c] = t[c].map(lambda x: f"{x:.2e}")
        save(t, "P5_grasp_paired", "Paired by seed on fold-averaged PE_clim; Holm-adjusted over the whole table")
        NUM["grasp_paired"] = t.to_dict("records")

json.dump(NUM, open(OUT / "probe_numbers.json", "w"), indent=1, default=float)
print("\nwrote", OUT)
