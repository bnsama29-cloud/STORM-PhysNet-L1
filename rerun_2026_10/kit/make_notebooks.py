import json
def nb(cells):
    return {"cells":[{"cell_type":"code" if k=="code" else "markdown","metadata":{},"source":src.strip("\n").splitlines(True),
                      **({"outputs":[],"execution_count":None} if k=="code" else {})} for k,src in cells],
            "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"}},
            "nbformat":4,"nbformat_minor":5}

SETUP = '''
# ---- 1. setup (needs Internet ON in the notebook settings) ----
import os, subprocess, shutil, glob, sys
def sh(c): print(subprocess.run(c, shell=True, capture_output=True, text=True).stdout[-1500:])
sh("pip -q install cdflib tensorboard tqdm tabulate 2>&1 | tail -1")
if not os.path.exists("/kaggle/working/STORM-PhysNet-L1"):
    # pinned to the exact repo version the kit was tested against, so a later push to GitHub cannot change the results
    sh("cd /kaggle/working && git clone https://github.com/bnsama29-cloud/STORM-PhysNet-L1 && cd STORM-PhysNet-L1 && git checkout 72008c75e30c10212d2fba5b05739af82fdc097a")
sh("cd /kaggle/working/STORM-PhysNet-L1 && git rev-parse HEAD")
# the kit = the Kaggle dataset you created from storm_kaggle_kit.zip (folder name may differ: adjust KIT if needed)
cand = sorted(glob.glob("/kaggle/input/**/train_block.py", recursive=True), key=len)
assert cand, "kit not found: add your storm-kit dataset as an Input (right sidebar -> Add Input)"
KIT = os.path.dirname(cand[0]); print("KIT =", KIT)
shutil.copytree(KIT, "/kaggle/working/kit", dirs_exist_ok=True)
REPO = "/kaggle/working/STORM-PhysNet-L1"; KITW = "/kaggle/working/kit"
import torch; print("GPU:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
'''
RESUME = '''
# ---- 2. resume support: if you attach the previous run's output as an Input, its finished runs are copied back ----
OUT = "/kaggle/working/out_{B}"; os.makedirs(OUT, exist_ok=True)
for d in glob.glob("/kaggle/input/**/out_{B}", recursive=True):
    shutil.copytree(d, OUT, dirs_exist_ok=True); print("resumed from", d)
print(len(glob.glob(OUT + "/preds_*.npz")), "runs already done")
'''
def run_cell(B, extra=""):
    return f'''
# ---- 3. train block {B} (10 seeds x 7 model families). Safe to re-run: finished runs are skipped. ----
r = subprocess.run(f"cd {{KITW}} && python -u train_block.py --repo {{REPO}} --block {B} --out {{OUT}} {extra}", shell=True)
print("exit code", r.returncode)
'''
ZIP = '''
# ---- zip the results for the combine step ----
import zipfile
with zipfile.ZipFile("/kaggle/working/{name}.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    for d in "{dirs}".split():
        for f in glob.glob(f"/kaggle/working/{{d}}/**/*", recursive=True):
            if os.path.isfile(f): zf.write(f, os.path.relpath(f, "/kaggle/working"))
assert os.path.getsize("/kaggle/working/{name}.zip") > 1000, "zip is empty - check the out_ folders"
print(os.path.getsize("/kaggle/working/{name}.zip")/1e6, "MB ->", "/kaggle/working/{name}.zip")
'''
MD = lambda t: ("md", t)

n1 = nb([MD("# Account 1 - block A (test 2018-09 to 2020-03) + GRASP transfer\nSettings: Accelerator = GPU (T4 or P100), Internet = ON. Use **Save Version -> Save & Run All (Commit)** so it keeps running with the browser closed. Output: `results_A_GRASP.zip`."),
         ("code",SETUP), ("code",RESUME.format(B="A")), ("code",run_cell("A")),
         MD("### GRASP transfer (uses the block-A checkpoints trained above)"),
         ("code",'''
OUTG = "/kaggle/working/out_G"; os.makedirs(OUTG, exist_ok=True)
for d in glob.glob("/kaggle/input/**/out_G", recursive=True): shutil.copytree(d, OUTG, dirs_exist_ok=True)
r = subprocess.run(f"cd {KITW} && python -u grasp_transfer.py --repo {REPO} --ckpt_dir {OUT}/ckpt_A --meta {OUT}/meta_A.npz --out {OUTG}", shell=True)
print("exit code", r.returncode)
'''),
         ("code",'''
# checkpoints are only needed here; drop everything except block-A storm_bz / transformer_matched ones from the zip to keep it small
for f in glob.glob(OUT + "/ckpt_A/*.pt"):
    if not any(k in os.path.basename(f) for k in ("storm_bz_", "transformer_matched_")): os.remove(f)
'''+ZIP.format(name="results_A_GRASP", dirs="out_A out_G"))])

def nb_block(B, label):
    return nb([MD(f"# Account - block {B} ({label})\nSettings: Accelerator = GPU, Internet = ON. **Save Version -> Save & Run All (Commit)**. Output: `results_{B}.zip`."),
               ("code",SETUP), ("code",RESUME.format(B=B)), ("code",run_cell(B)),
               ("code",ZIP.format(name=f"results_{B}", dirs=f"out_{B}").replace('"results_%s"'%B,'"results_%s"'%B)
                 .replace("make_archive", "make_archive") )])

n2 = nb_block("B","test 2015-2016"); n3 = nb_block("C","test 2012-2013")
# drop ckpts from B/C zips (train_block already deletes them unless --keep_ckpt)
nc = nb([MD("# Combine account - tables, figures, numbers for the paper\nUpload results_A_GRASP.zip, results_B.zip, results_C.zip as ONE Kaggle dataset (or three), add it as Input. GPU not needed (Accelerator = None), Internet ON."),
         ("code",'''
import os, glob, zipfile, subprocess, shutil
subprocess.run("pip -q install tabulate 2>&1 | tail -1", shell=True)
cand = sorted(glob.glob("/kaggle/input/**/train_block.py", recursive=True), key=len)
KIT = os.path.dirname(cand[0]) if cand else None
os.makedirs("/kaggle/working/all", exist_ok=True)
for z in glob.glob("/kaggle/input/**/results_*.zip", recursive=True):
    print("unzip", z); zipfile.ZipFile(z).extractall("/kaggle/working/all")
print(sorted(os.listdir("/kaggle/working/all")))
'''),
         ("code",'''
r = subprocess.run(f"python {KIT}/combine_report.py --inputs /kaggle/working/all --out /kaggle/working/report", shell=True, capture_output=True, text=True)
print(r.stdout[-3000:], r.stderr[-3000:])
shutil.make_archive("/kaggle/working/paper_report", "zip", "/kaggle/working", "report")
print(open("/kaggle/working/report/PAPER_NUMBERS.md").read()[:6000])
''')])
for name,n in [("1_account1_blockA_GRASP",n1),("2_account2_blockB",n2),("3_account3_blockC",n3),("4_combine_report",nc)]:
    json.dump(n, open(f"notebooks/{name}.ipynb","w"), indent=1)
