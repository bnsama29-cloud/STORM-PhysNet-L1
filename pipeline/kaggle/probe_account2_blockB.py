# Kaggle script for account B: driver-robustness probes + extended comparisons.
# HOW TO USE: Kaggle -> Create -> New Notebook -> paste this whole file into the first cell
# (no zip or dataset upload needed). Settings: Accelerator = GPU T4 x2, Internet = ON.
# Then Save Version -> Save & Run All (Commit). Download the zip from the Output tab when done.

# ---- 1. setup: clone the repository at a fixed commit (Internet must be ON) ----
import os, subprocess, shutil, glob, sys, time, zipfile
def sh(c): r = subprocess.run(c, shell=True, capture_output=True, text=True); print((r.stdout + r.stderr)[-1500:])
sh("pip -q install cdflib tensorboard tqdm tabulate 2>&1 | tail -1")
REPO = "/kaggle/working/STORM-PhysNet-L1"
if not os.path.exists(REPO):
    sh("cd /kaggle/working && git clone -q https://github.com/bnsama29-cloud/STORM-PhysNet-L1")
sh(f"cd {REPO} && git fetch -q origin && git checkout -q 037bda0171d0fa62ccf579acfca0a01e6b82159b && git rev-parse HEAD")
PIPE = REPO + "/pipeline"
import torch
NGPU = torch.cuda.device_count(); print("GPUs:", NGPU, [torch.cuda.get_device_name(i) for i in range(NGPU)])
assert NGPU >= 1, "No GPU: set Accelerator = GPU T4 x2 in the notebook settings"
SEED_SPLITS = ["42-46", "47-51"] if NGPU >= 2 else ["42-51"]

def run_parallel(cmd_for_split, tag):
    """One process per GPU; each gets a different seed range. Logs: /kaggle/working/<tag>_gpu<i>.log"""
    procs = []
    for i, seeds in enumerate(SEED_SPLITS):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i))
        log = open(f"/kaggle/working/{tag}_gpu{i}.log", "a")
        procs.append(subprocess.Popen(cmd_for_split(seeds), shell=True, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=PIPE))
        time.sleep(20)                       # let the first process write the shared metadata file
    while any(p.poll() is None for p in procs):
        time.sleep(600)
        for i in range(len(procs)):
            lines = open(f"/kaggle/working/{tag}_gpu{i}.log").read().splitlines()
            print(time.strftime("%H:%M"), f"gpu{i}:", lines[-1][:160] if lines else "")
    codes = [p.returncode for p in procs]; print("exit codes", codes)
    for i in range(len(procs)):
        print(f"--- last lines gpu{i}"); print("\n".join(open(f"/kaggle/working/{tag}_gpu{i}.log").read().splitlines()[-8:]))
    assert all(c == 0 for c in codes), "a process failed: read the log lines above"

# ---- 2. resume: if a previous run of THIS notebook is attached as an Input, finished runs are copied back ----
for d in ["probe_B"]:
    os.makedirs(f"/kaggle/working/{d}", exist_ok=True)
    for src in glob.glob(f"/kaggle/input/**/{d}", recursive=True):
        shutil.copytree(src, f"/kaggle/working/{d}", dirs_exist_ok=True); print("resumed", d, "from", src)
    print(d, len(glob.glob(f"/kaggle/working/{d}/*preds_*.npz")), "runs already done")

# ---- 3. train block B: 5 model families x 10 seeds, with all driver probes (finished runs are skipped) ----
OUT = "/kaggle/working/probe_B"
run_parallel(lambda seeds: f"python -u train_block.py --repo {REPO} --block B --out {OUT} "
                           f"--families storm_bz,transformer_matched,tf_ffn_res,storm_no_delay,storm_no_gate --noise_families storm_bz,transformer_matched,tf_ffn_res,storm_no_delay,storm_no_gate --probes --seeds {seeds} --keep_ckpt none",
             "train_B")
print(len(glob.glob(OUT + "/preds_*.npz")), "of 50 prediction files present")

# ---- zip the results (download this file from the Output tab and send it) ----
name = "/kaggle/working/probe_B.zip"
with zipfile.ZipFile(name, "w", zipfile.ZIP_DEFLATED) as zf:
    for d in "probe_B".split():
        for f in glob.glob(f"/kaggle/working/{d}/**/*", recursive=True):
            if os.path.isfile(f) and not f.endswith(".pt") and ".tmp." not in f: zf.write(f, os.path.relpath(f, "/kaggle/working"))
    for f in glob.glob("/kaggle/working/*.log"): zf.write(f, "logs/" + os.path.basename(f))
print(round(os.path.getsize(name) / 1e6, 1), "MB ->", name)
