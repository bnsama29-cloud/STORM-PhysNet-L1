"""Writes the three Kaggle notebooks for the driver-robustness probes and the extended GRASP comparison.

  python make_probe_notebooks.py <commit-sha>

Each notebook clones this repository at <commit-sha> (code + data), so no Kaggle dataset upload is needed.
Accelerator "GPU T4 x2": the ten seeds are split over the two GPUs (one process each). Works on one GPU too.
"""
import json, sys

SHA = sys.argv[1]
FAMS = "storm_bz,transformer_matched,tf_ffn_res,storm_no_delay,storm_no_gate"


def nb(cells):
    return {"cells": [{"cell_type": "code" if k == "code" else "markdown", "metadata": {},
                       "source": src.strip("\n").splitlines(True),
                       **({"outputs": [], "execution_count": None} if k == "code" else {})} for k, src in cells],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


SETUP = f'''
# ---- 1. setup: clone the repository at a fixed commit (Internet must be ON) ----
import os, subprocess, shutil, glob, sys, time, zipfile
def sh(c): r = subprocess.run(c, shell=True, capture_output=True, text=True); print((r.stdout + r.stderr)[-1500:])
sh("pip -q install cdflib tensorboard tqdm tabulate 2>&1 | tail -1")
REPO = "/kaggle/working/STORM-PhysNet-L1"
if not os.path.exists(REPO):
    sh("cd /kaggle/working && git clone -q https://github.com/bnsama29-cloud/STORM-PhysNet-L1")
sh(f"cd {{REPO}} && git fetch -q origin && git checkout -q {SHA} && git rev-parse HEAD")
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
        log = open(f"/kaggle/working/{{tag}}_gpu{{i}}.log", "a")
        procs.append(subprocess.Popen(cmd_for_split(seeds), shell=True, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=PIPE))
        time.sleep(20)                       # let the first process write the shared metadata file
    while any(p.poll() is None for p in procs):
        time.sleep(600)
        for i in range(len(procs)):
            lines = open(f"/kaggle/working/{{tag}}_gpu{{i}}.log").read().splitlines()
            print(time.strftime("%H:%M"), f"gpu{{i}}:", lines[-1][:160] if lines else "")
    codes = [p.returncode for p in procs]; print("exit codes", codes)
    for i in range(len(procs)):
        print(f"--- last lines gpu{{i}}"); print("\\n".join(open(f"/kaggle/working/{{tag}}_gpu{{i}}.log").read().splitlines()[-8:]))
    assert all(c == 0 for c in codes), "a process failed: read the log lines above"
'''

RESUME = '''
# ---- 2. resume: if a previous run of THIS notebook is attached as an Input, finished runs are copied back ----
for d in [{dirs}]:
    os.makedirs(f"/kaggle/working/{{d}}", exist_ok=True)
    for src in glob.glob(f"/kaggle/input/**/{{d}}", recursive=True):
        shutil.copytree(src, f"/kaggle/working/{{d}}", dirs_exist_ok=True); print("resumed", d, "from", src)
    print(d, len(glob.glob(f"/kaggle/working/{{d}}/*preds_*.npz")), "runs already done")
'''


def train_cell(B, keep):
    return f'''
# ---- 3. train block {B}: 5 model families x 10 seeds, with all driver probes (finished runs are skipped) ----
OUT = "/kaggle/working/probe_{B}"
run_parallel(lambda seeds: f"python -u train_block.py --repo {{REPO}} --block {B} --out {{OUT}} "
                           f"--families {FAMS} --noise_families {FAMS} --probes --seeds {{seeds}} --keep_ckpt {keep}",
             "train_{B}")
print(len(glob.glob(OUT + "/preds_*.npz")), "of 50 prediction files present")
'''


ZIP = '''
# ---- zip the results (download this file from the Output tab and send it) ----
name = "/kaggle/working/{name}.zip"
with zipfile.ZipFile(name, "w", zipfile.ZIP_DEFLATED) as zf:
    for d in "{dirs}".split():
        for f in glob.glob(f"/kaggle/working/{{d}}/**/*", recursive=True):
            if os.path.isfile(f) and not f.endswith(".pt") and ".tmp." not in f: zf.write(f, os.path.relpath(f, "/kaggle/working"))
    for f in glob.glob("/kaggle/working/*.log"): zf.write(f, "logs/" + os.path.basename(f))
print(round(os.path.getsize(name) / 1e6, 1), "MB ->", name)
'''

GRASP = f'''
# ---- 4. GRASP transfer for the five families (uses the block-A checkpoints trained above) ----
OUTG = "/kaggle/working/probe_G"
run_parallel(lambda seeds: f"python -u grasp_transfer.py --repo {{REPO}} --ckpt_dir {{OUT}}/ckpt_A --meta {{OUT}}/meta_A.npz "
                           f"--out {{OUTG}} --families {FAMS} --seeds {{seeds}}", "grasp")
print(len(glob.glob(OUTG + "/grasp_preds_s*.npz")), "of 10 GRASP seed files present")
'''

HOWTO = ("\n\nSettings (right sidebar): **Accelerator = GPU T4 x2**, **Internet = ON**. Then **Save Version -> Save & Run All "
         "(Commit)** so it keeps running with the browser closed. When it finishes, download **{zipname}** from the "
         "Output tab.\n\nIf it stops early (12 h limit), open the notebook again, attach the previous version's output "
         "as an Input (Add Input -> Your Work -> this notebook), and Save & Run All again: finished runs are skipped.")

n1 = nb([("md", "# Account 1: block A (test 2018-09 to 2020-03) + GRASP transfer\nAbout 7 hours." + HOWTO.format(zipname="probe_A_GRASP.zip")),
         ("code", SETUP), ("code", RESUME.format(dirs='"probe_A", "probe_G"')), ("code", train_cell("A", "A")),
         ("code", GRASP), ("code", ZIP.format(name="probe_A_GRASP", dirs="probe_A probe_G"))])
n2 = nb([("md", "# Account 2: block B (test 2015-2016)\nAbout 4-5 hours." + HOWTO.format(zipname="probe_B.zip")),
         ("code", SETUP), ("code", RESUME.format(dirs='"probe_B"')), ("code", train_cell("B", "none")),
         ("code", ZIP.format(name="probe_B", dirs="probe_B"))])
n3 = nb([("md", "# Account 3: block C (test 2012-2013)\nAbout 4-5 hours." + HOWTO.format(zipname="probe_C.zip")),
         ("code", SETUP), ("code", RESUME.format(dirs='"probe_C"')), ("code", train_cell("C", "none")),
         ("code", ZIP.format(name="probe_C", dirs="probe_C"))])
for name, n in [("5_probe_account1_blockA_GRASP", n1), ("6_probe_account2_blockB", n2), ("7_probe_account3_blockC", n3)]:
    json.dump(n, open(f"notebooks/{name}.ipynb", "w"), indent=1)
print("wrote 3 notebooks pinned to", SHA)
