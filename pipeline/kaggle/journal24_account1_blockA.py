# STORM-PhysNet journal extension, account 1 of 3: block A with 1, 6, 12 and 24 h horizons.
# HOW TO USE: Kaggle -> Create -> New Notebook -> paste this whole file into the first cell.
# Settings (right sidebar): Accelerator = GPU T4 x2, Internet = ON (needs a phone-verified account).
# Then Save Version -> Save & Run All (Commit). All 90 runs need about 6 hours.
# Each session stops starting new runs after SESSION_HOURS (below) and always writes journal24_A.zip.
# The last lines of the log say either "ALL 90 RUNS DONE" or "NOT FINISHED: x of 90".
# TO CONTINUE ANOTHER DAY: open the notebook -> Add Input -> Notebook Output Files -> pick this notebook (its latest
# version) -> Save Version -> Save & Run All again. Finished runs are skipped; nothing is repeated.
# Send journal24_A.zip only when the log says ALL 90 RUNS DONE.
SESSION_HOURS = 10          # training time per session; keep <= 10.5 (Kaggle stops a session at 12 h)

# ---- 1. setup: clone the repository at a fixed commit (Internet must be ON) ----
import os, subprocess, shutil, glob, time, zipfile
T0 = time.time()
def sh(c): r = subprocess.run(c, shell=True, capture_output=True, text=True); print((r.stdout + r.stderr)[-1500:]); return r
sh("pip -q install cdflib tensorboard tqdm tabulate 2>&1 | tail -1")
REPO = "/kaggle/working/STORM-PhysNet-L1"
if not os.path.exists(REPO):
    sh("cd /kaggle/working && git clone -q https://github.com/bnsama29-cloud/STORM-PhysNet-L1")
sh(f"cd {REPO} && git fetch -q origin && git checkout -q f51f68b4ca9845de8de5b95fd48fa79bb143b210 && git rev-parse HEAD")
assert sh(f"cd {REPO} && git rev-parse HEAD").stdout.strip() == "f51f68b4ca9845de8de5b95fd48fa79bb143b210", "wrong code version"
PIPE = REPO + "/pipeline"
import torch
NGPU = torch.cuda.device_count(); print("GPUs:", NGPU, [torch.cuda.get_device_name(i) for i in range(NGPU)])
assert NGPU >= 1, "No GPU: set Accelerator = GPU T4 x2 in the notebook settings"
SEED_SPLITS = ["42-46", "47-51"] if NGPU >= 2 else ["42-51"]
B = "A"; OUT = f"/kaggle/working/j24_{B}"; os.makedirs(OUT, exist_ok=True)

# ---- 1b. inspect the NOAA GOES-16/18 electron files (a few MB; never stops the training if it fails) ----
def goesr_probe():
    import re, urllib.request, json
    P = "/kaggle/working/goesr_probe"; os.makedirs(P, exist_ok=True); rep = {}
    def get(url, t=60): return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=t).read()
    def links(url): return [h for h in re.findall(r'href="([^"?#]+)"', get(url).decode("utf-8", "ignore")) if not h.startswith(("/", "http", ".."))]
    for sat in ("goes16", "goes18"):
        base = f"https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/{sat}/l2/data/mpsh-l2-avg5m_science/"
        try:
            years = sorted(h for h in links(base) if h.strip("/").isdigit()); rep[sat] = {"years": years}
            url = base
            for _ in range(4):                                     # walk year/month folders down to the first .nc file
                ls = links(url); nc = sorted(h for h in ls if h.endswith(".nc"))
                if nc: url = url + nc[0]; break
                sub = sorted(h for h in ls if h.endswith("/"))
                url = url + (sub[len(sub) // 2] if url == base else sub[0])
            rep[sat]["file"] = url
            fn = f"{P}/{sat}_{os.path.basename(url)}"; open(fn, "wb").write(get(url, 120))
            import h5py
            with h5py.File(fn, "r") as h:
                v = {}
                h.visititems(lambda k, o: v.__setitem__(k, dict(shape=list(o.shape), dtype=str(o.dtype),
                    attrs={a: str(o.attrs[a])[:200] for a in o.attrs}) if hasattr(o, "shape") else None))
                rep[sat]["variables"] = v; rep[sat]["global_attrs"] = {a: str(h.attrs[a])[:300] for a in h.attrs}
        except Exception as e:
            rep.setdefault(sat, {})["error"] = repr(e)[:500]
    try:
        open(f"{P}/MPS-HI_L2_ReadMe.pdf", "wb").write(get("https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes16/l2/docs/GOES-R_SEISS_L2_MPS-HI.ReadMe.pdf", 120))
    except Exception as e:
        rep["readme_error"] = repr(e)[:300]
    json.dump(rep, open(f"{P}/goesr_probe.json", "w"), indent=1)
    print("GOES-R probe:", {k: (list(v)[:4] if isinstance(v, dict) else v) for k, v in rep.items()})
try:
    goesr_probe()
except Exception as e:
    print("GOES-R probe skipped:", e)

# ---- 2. resume finished runs from a previous version of this notebook, if attached as an Input ----
for src in glob.glob(f"/kaggle/input/**/j24_{B}", recursive=True):
    shutil.copytree(src, OUT, dirs_exist_ok=True); print("resumed from", src)
for f in glob.glob(OUT + "/*.tmp.*"): os.remove(f)
print(len(glob.glob(OUT + "/preds_*.npz")), "of 90 runs already done")

# ---- 3. train: 9 families x 10 seeds, one process per GPU ----
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
        print(f"--- last lines gpu{i}"); print("\n".join(open(f"/kaggle/working/{tag}_gpu{i}.log").read().splitlines()[-8:]))
    assert all(c == 0 for c in codes), "a process failed: read the log lines above"

left_h = max(0.0, SESSION_HOURS - (time.time() - T0) / 3600 - 0.4)   # 0.4 h margin: the last run started may take ~20 min
print(f"training budget this session: {left_h:.1f} h")
run_parallel(lambda seeds: f"python -u train_block.py --repo {REPO} --block {B} --out {OUT} --horizons 1,6,12,24 "
                           f"--families lstm,transformer_matched,tf_ffn_res,tf_flux_only,storm_bz,storm_no_delay,storm_no_gate,storm_fixed_ma,tf_res_fixed_ma --noise_families storm_bz,transformer_matched,tf_ffn_res,storm_fixed_ma,tf_res_fixed_ma --probes --seeds {seeds} --keep_ckpt A "
                           f"--stop_after_hours {left_h:.2f}",
             f"train_{B}")
n = len(glob.glob(OUT + "/preds_*.npz"))

# ---- 4. zip the results (download journal24_A.zip from the Output tab) ----
name = f"/kaggle/working/journal24_{B}.zip"
with zipfile.ZipFile(name, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in glob.glob(OUT + "/**/*", recursive=True):
        if os.path.isfile(f) and not f.endswith(".pt") and ".tmp." not in f: zf.write(f, os.path.relpath(f, "/kaggle/working"))
    for f in glob.glob("/kaggle/working/*.log") + glob.glob("/kaggle/working/goesr_probe/*"):
        if os.path.isfile(f) and not f.endswith(".nc"): zf.write(f, "logs/" + os.path.basename(f))
print(round(os.path.getsize(name) / 1e6, 1), "MB ->", name)
print("Checkpoints kept in", OUT + f"/ckpt_{B}:", len(glob.glob(OUT + f"/ckpt_{B}/*.pt")), "files")
print(f"session time {(time.time() - T0) / 3600:.1f} h")
print("ALL 90 RUNS DONE: send", name) if n >= 90 else print(f"NOT FINISHED: {n} of 90 runs. Add this notebook's output as an Input and Save & Run All again to continue.")
