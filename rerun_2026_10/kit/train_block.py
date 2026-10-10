#!/usr/bin/env python
"""
Train all model families for ONE block (A, B or C) with the fixed pipeline.  Resumable: finished runs are skipped.

  python train_block.py --repo /kaggle/working/STORM-PhysNet-L1 --block A --out /kaggle/working/out_A

Outputs in --out:
  meta_<B>.npz                       targets, persistence, flags, times, scaler for the block's test windows
  preds_<B>_<family>_s<seed>.npz     test predictions (N,3) (+ noisy-input predictions, tau, gate where relevant)
  ckpt_<B>/<family>_s<seed>_best.pt  checkpoints (block A only; needed by grasp_transfer.py)
  runlog_<B>.jsonl                   one line per run: epochs, best val MSE, seconds
"""
import argparse, contextlib, io, json, os, sys, time, copy
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True, help="path to a clone of STORM-PhysNet-L1 (needs datasets/ + src/)")
ap.add_argument("--block", required=True, choices=["A", "B", "C"])
ap.add_argument("--out", required=True)
ap.add_argument("--families", default="lstm,transformer_matched,tf_ffn_res,tf_flux_only,storm_bz,storm_no_delay,storm_no_gate")
ap.add_argument("--seeds", default="42-51", help="e.g. 42-51 or 42,43,44")
ap.add_argument("--epochs", type=int, default=None, help="override config epochs (smoke test only)")
ap.add_argument("--max_train_windows", type=int, default=None, help="smoke test only: subsample train windows")
ap.add_argument("--noise_sigmas", default="0.5,1,2")
ap.add_argument("--keep_ckpt", default="A", help="blocks whose checkpoints are kept")
args = ap.parse_args()

import numpy as np, pandas as pd, torch, yaml
from storm_fixed.data import (build_features, Block, make_loaders, load_raw, SW_COLS, HSTEPS, BLOCKS, VAL)
from storm_fixed.models_extra import FFNResTransformer, FluxOnly

REPO = Path(args.repo).resolve()
OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)
CK = OUT / f"ckpt_{args.block}"; CK.mkdir(exist_ok=True)
sys.path.insert(0, str(REPO))
from src.training.trainer import Trainer

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if args.seeds.count("-") == 1 and "," not in args.seeds:
    a, b = map(int, args.seeds.split("-")); SEEDS = list(range(a, b + 1))
else:
    SEEDS = [int(s) for s in args.seeds.split(",")]
FAMILIES = args.families.split(",")
SIGMAS = [float(s) for s in args.noise_sigmas.split(",") if s]
NOISE_FAMILIES = {"storm_bz", "transformer_matched", "tf_ffn_res"}

SPEC = {
    "lstm": dict(model_type="lstm", ablation="none", match=False),
    "transformer_matched": dict(model_type="transformer", ablation="none", match=True),
    "tf_ffn_res": dict(model_type="transformer", ablation="none", match=True),     # model built separately
    "tf_flux_only": dict(model_type="transformer", ablation="none", match=True),   # model built separately
    "storm_bz": dict(model_type="storm_physnet", ablation="none", match=False),
    "storm_no_delay": dict(model_type="storm_physnet", ablation="no_delay", match=False),
    "storm_no_gate": dict(model_type="storm_physnet", ablation="no_bz_gate", match=False),
}
for f in FAMILIES:
    assert f in SPEC, f"unknown family {f}"

base_cfg = yaml.safe_load(open(REPO / "configs/config.yaml"))
if args.epochs: base_cfg["training"]["epochs"] = args.epochs


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ----------------------------------------------------------------------------------------------- data
log(f"device={DEV}  block={args.block}  families={FAMILIES}  seeds={SEEDS[0]}..{SEEDS[-1]}")
goes, omni = load_raw(str(REPO))
feat = build_features(goes, omni)
blk = Block(feat, args.block)
st = blk.stats(); log(f"rows/windows: {json.dumps(st)}")
test_set = blk.split(2)
meta_path = OUT / f"meta_{args.block}.npz"
if not meta_path.exists():
    y = np.stack([blk.flux_tg[test_set.starts + 72 + h - 1].numpy() for h in HSTEPS], 1)
    pers = blk.flux_tg[test_set.starts + 71].numpy()
    np.savez(meta_path, y=y, persist=pers, strict=test_set.strict, storm_in=test_set.window_storm_flags > 0,
             dst_target=test_set.dst_target, t_origin=test_set.t_origin.astype("datetime64[ns]").astype(np.int64),
             scaler_mu=blk.scaler[0], scaler_sd=blk.scaler[1], stats=json.dumps(st),
             block_range=json.dumps(BLOCKS[args.block]), val_range=json.dumps(VAL))
    log(f"wrote {meta_path.name}: {len(y)} test windows, {int(test_set.strict.sum())} strict, {int((test_set.dst_target[:, 1] <= -50).sum())} storm@6h")


def make_model(family, trainer, n_sw):
    if family == "tf_ffn_res":
        return FFNResTransformer(n_sw).to(DEV)
    if family == "tf_flux_only":
        return FluxOnly(trainer.build_model(n_sw)).to(DEV)
    return trainer.build_model(n_sw)


@torch.no_grad()
def predict(model, loader, sigma=0.0, aux=False):
    model.eval(); g = torch.Generator().manual_seed(0); P, T, G = [], [], []
    for b in loader:
        x = b["x_sw"]
        if sigma > 0:
            noise = sigma * torch.randn(x.shape, generator=g); noise[..., -1] = 0.0   # never perturb the observed-mask channel
            x = x + noise
        x, xf, yp = x.to(DEV), b["x_flux"].to(DEV), b["y_persist"].to(DEV)
        o = model(x_sw=x, x_flux=xf, y_persist=yp)
        P.append(o["flux_pred"].cpu().numpy())
        if aux and "tau" in o:
            T.append(o["tau"].reshape(len(x), -1).mean(1).cpu().numpy()); G.append(o["gate_values"].reshape(len(x), -1).mean(1).cpu().numpy())
    out = {"pred": np.concatenate(P).astype(np.float32)}
    if aux and T: out["tau"] = np.concatenate(T).astype(np.float32); out["gate"] = np.concatenate(G).astype(np.float32)
    return out


runlog = OUT / f"runlog_{args.block}.jsonl"
for seed in SEEDS:                               # seed-major order so that partial results are balanced across families
    for fam in FAMILIES:
        pf = OUT / f"preds_{args.block}_{fam}_s{seed}.npz"
        if pf.exists():
            continue
        t0 = time.time()
        torch.manual_seed(seed); np.random.seed(seed)
        cfg = copy.deepcopy(base_cfg); s = SPEC[fam]
        cfg["model_type"], cfg["ablation"], cfg["match_storm_capacity"] = s["model_type"], s["ablation"], s["match"]
        cfg["training"]["checkpoint_dir"] = str(CK); cfg["training"]["log_dir"] = str(OUT / "tb_tmp")
        trainer = Trainer(cfg)
        tr_loader, va_loader, te_loader, tr_ds, va_ds, te_ds = make_loaders(blk, batch_size=cfg["training"]["batch_size"],
                                                                           storm_weight=cfg["training"]["storm_weight"], seed=seed)
        if args.max_train_windows:
            tr_ds.starts = tr_ds.starts[: args.max_train_windows]; tr_ds.window_storm_flags = tr_ds.window_storm_flags[: args.max_train_windows]
            tr_ds.strict = tr_ds.strict[: args.max_train_windows]
            tr_loader, va_loader, te_loader, *_ = make_loaders(blk, 64, 12.0, seed)  # rebuilt below with truncated train set
            from torch.utils.data import DataLoader, WeightedRandomSampler
            w = torch.from_numpy(np.where(tr_ds.window_storm_flags > 0, 12.0, 1.0).astype(np.float32))
            tr_loader = DataLoader(tr_ds, batch_size=64, sampler=WeightedRandomSampler(w, len(tr_ds), True), drop_last=True)
        model = make_model(fam, trainer, len(SW_COLS))
        name = f"{fam}_s{seed}"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            model = trainer.train_single(model, tr_loader, va_loader, model_name=name)
        txt = buf.getvalue()
        ep = txt.count("Epoch ") ; vals = [float(l.split("Val MSE:")[1].split("|")[0]) for l in txt.splitlines() if "Val MSE:" in l]
        trainer.writer.close()
        res = predict(model, te_loader, aux=(fam == "storm_bz"))
        save = {"pred": res["pred"]}
        if "tau" in res: save["tau"], save["gate"] = res["tau"], res["gate"]
        if fam in NOISE_FAMILIES:
            for sg in SIGMAS: save[f"pred_noise{sg:g}"] = predict(model, te_loader, sigma=sg)["pred"]
        np.savez(pf, **save)
        if args.block not in args.keep_ckpt.split(","):
            (CK / f"{name}_best.pt").unlink(missing_ok=True)
        rec = dict(block=args.block, family=fam, seed=seed, epochs=ep, best_val_mse=min(vals) if vals else None,
                   seconds=round(time.time() - t0, 1), n_params=int(sum(p.numel() for p in model.parameters())))
        with open(runlog, "a") as fh: fh.write(json.dumps(rec) + "\n")
        log(f"done {fam} s{seed}: {rec['epochs']} epochs, val_mse={rec['best_val_mse']}, {rec['seconds']}s")
        del model, trainer
        if DEV.type == "cuda": torch.cuda.empty_cache()

import shutil; shutil.rmtree(OUT / "tb_tmp", ignore_errors=True)
log("BLOCK COMPLETE")
