# STORM-PhysNet journal extension: cross-satellite transfer (GOES-16, GOES-18, GSAT-19 GRASP), 1, 6, 12 and 24 h.
# RUN THIS IN ACCOUNT 1 (block A), AFTER journal24_account1_blockA.py says ALL 90 RUNS DONE: it needs those checkpoints.
# HOW TO USE: Kaggle -> Create -> New Notebook -> paste this whole file into the first cell.
#   Add Input -> Notebook Output Files -> pick the block-A notebook (the one that said ALL 90 RUNS DONE).
#   Settings: Accelerator = GPU T4 x2, Internet = ON. Then Save Version -> Save & Run All (Commit).
# Expected time: about 4 hours (3 satellites, 6 folds, 3 models, 4 variants, 10 seeds).
# The last log line says "ALL 10 SEEDS DONE" (download journal_transfer.zip and send it) or "NOT FINISHED".
# TO CONTINUE: Add Input -> Notebook Output Files -> also add THIS notebook (keep the block-A input) -> Save & Run All.
SESSION_HOURS = 10

import os, subprocess, shutil, glob, time, zipfile
T0 = time.time()
def sh(c): r = subprocess.run(c, shell=True, capture_output=True, text=True); print((r.stdout + r.stderr)[-1500:]); return r
sh("pip -q install cdflib tensorboard tqdm tabulate 2>&1 | tail -1")
REPO = "/kaggle/working/STORM-PhysNet-L1"
if not os.path.exists(REPO):
    sh("cd /kaggle/working && git clone -q https://github.com/bnsama29-cloud/STORM-PhysNet-L1")
sh(f"cd {REPO} && git fetch -q origin && git checkout -q 1a986fffea351a708b09568a50c5a9b14808249b")
assert sh(f"cd {REPO} && git rev-parse HEAD").stdout.strip() == "1a986fffea351a708b09568a50c5a9b14808249b", "wrong code version"
PIPE = REPO + "/pipeline"
import torch
NGPU = torch.cuda.device_count(); print("GPUs:", NGPU)
assert NGPU >= 1, "No GPU: set Accelerator = GPU T4 x2 in the notebook settings"
SEED_SPLITS = ["42-46", "47-51"] if NGPU >= 2 else ["42-51"]

# ---- the block-A checkpoints and meta file from journal24_account1_blockA.py ----
metas = [m for m in glob.glob("/kaggle/input/**/j24_A/meta_A.npz", recursive=True)]
assert metas, "Block-A results not found: Add Input -> Notebook Output Files -> the block-A notebook"
SRC = os.path.dirname(metas[0]); CK = SRC + "/ckpt_A"
need = [f"{f}_s{s}_best.pt" for f in ("storm_bz", "transformer_matched", "tf_ffn_res") for s in range(42, 52)]
miss = [n for n in need if not os.path.exists(f"{CK}/{n}")]
assert not miss, f"{len(miss)} block-A checkpoints missing (e.g. {miss[:3]}): finish the block-A notebook first"
print("using checkpoints from", CK)

OUT = "/kaggle/working/tj"; os.makedirs(OUT, exist_ok=True)
for src in glob.glob("/kaggle/input/**/tj", recursive=True):
    shutil.copytree(src, OUT, dirs_exist_ok=True); print("resumed from", src)
for f in glob.glob(OUT + "/*.tmp.*"): os.remove(f)
print(len(glob.glob(OUT + "/transfer_preds_s*.npz")), "of 10 seeds already done")

def run_parallel(cmd_for_split, tag):
    procs = []
    for i, seeds in enumerate(SEED_SPLITS):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i))
        log = open(f"/kaggle/working/{tag}_gpu{i}.log", "a")
        procs.append(subprocess.Popen(cmd_for_split(seeds), shell=True, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=PIPE))
        time.sleep(20)
    while any(p.poll() is None for p in procs):
        time.sleep(600)
        for i in range(len(procs)):
            lines = open(f"/kaggle/working/{tag}_gpu{i}.log").read().splitlines()
            print(time.strftime("%H:%M"), f"gpu{i}:", lines[-1][:160] if lines else "")
    codes = [p.returncode for p in procs]; print("exit codes", codes)
    for i in range(len(procs)):
        print(f"--- last lines gpu{i}"); print("\n".join(open(f"/kaggle/working/{tag}_gpu{i}.log").read().splitlines()[-12:]))
    assert all(c == 0 for c in codes), "a process failed: read the log lines above"

left_h = max(0.0, SESSION_HOURS - (time.time() - T0) / 3600 - 0.9)   # 0.9 h margin: one seed takes up to ~50 min
print(f"training budget this session: {left_h:.1f} h")
run_parallel(lambda seeds: f"python -u transfer_journal.py --repo {REPO} --ckpt_dir {CK} --meta {SRC}/meta_A.npz --out {OUT} "
                           f"--seeds {seeds} --horizons 1,6,12,24 --stop_after_hours {left_h:.2f}", "transfer")
n = len(glob.glob(OUT + "/transfer_preds_s*.npz"))

name = "/kaggle/working/journal_transfer.zip"
with zipfile.ZipFile(name, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in glob.glob(OUT + "/*"):
        if os.path.isfile(f) and ".tmp." not in f: zf.write(f, os.path.relpath(f, "/kaggle/working"))
    for f in glob.glob("/kaggle/working/transfer_gpu*.log"): zf.write(f, "logs/" + os.path.basename(f))
print(round(os.path.getsize(name) / 1e6, 1), "MB ->", name, f"| session time {(time.time() - T0) / 3600:.1f} h")
print("ALL 10 SEEDS DONE: send", name) if n >= 10 else print(f"NOT FINISHED: {n} of 10 seeds. Add this notebook's output as an Input (keep the block-A input) and Save & Run All again.")
