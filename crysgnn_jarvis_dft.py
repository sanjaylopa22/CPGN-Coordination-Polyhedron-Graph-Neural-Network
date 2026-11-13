#!/usr/bin/env python3
"""
CrysGNN - JARVIS-DFT baseline on the CPGN protocol          python crysgnn_jarvis_dft.py
=====================================================================================
Run it exactly like cpgn_jarvis_dft.py: no arguments needed. Edit the CONFIG block
below (or pass the same options on the command line) and run

        python crysgnn_jarvis_dft.py

Put it in the folder where cpgn_jarvis_dft.py was run. If cpgn_jarvis_test.csv is
there, CrysGNN is scored on exactly CPGN's test structures and a side-by-side table
is printed at the end. Everything is resumable: stop it any time and run it again.

What "CrysGNN" is here
----------------------
The OFFICIAL kdmsit/CrysGNN downstream recipe (distilled_baslines/cgcnn_distilled):
a CGCNN student trained on the labelled property with knowledge distilled from the
authors' pretrained CrysGNN teacher (model/crysgnn_state_checkpoint_v1.pth.tar).
Nothing is re-implemented and no pretraining is done here. Official JARVIS settings
are used: Adam, lr 0.003, 1000 epochs, batch 128 (their run_prop.sh); only the data
split is replaced by yours (explicit sizes instead of their 80/10/10 ratios).

What one run does  (stages: check -> setup -> export -> train -> collect)
-----------------------------------------------------------------------
 check    verifies the python environment (torch, pymatgen, scikit-learn, ...)
 setup    clones the official code (code only, no multi-GB data folders), applies the
          validation-selection patch, and converts the shipped teacher checkpoint into
          the file the official script expects (model/pretrain_model.pth)
 export   loads JARVIS dft_3d, reproduces the CPGN split, writes CIFs and the label
          files for every property
 train    per property: builds CrysGNN's preprocessed pickles from your split (on
          demand, deleted afterwards to save disk), runs the official training per seed
 collect  validates every prediction file against the split, writes
          crysgnn_jarvis_test.csv / crysgnn_jarvis_summary.csv / crysgnn_vs_cpgn.csv, plots

Same data and split as cpgn_jarvis_dft.py (checked against its source by test)
-----------------------------------------------------------------------------
  data    jarvis.db.figshare.data("dft_3d")          -> 75,993 entries (jarvis-tools 2026.4.2)
  split   random.Random(42).shuffle(range(len)); train = first min(37711, len-10000),
          then 5,000 val and 5,000 test (47,711 used; the other 28,282 are never used)
  labels  None / "na" / NaN / inf / unparsable = missing, as in extract_targets()

CrysGNN is trained per property on the labelled structures of each split.

Differences from the official code, and why (each is intentional and printed/patched)
------------------------------------------------------------------------------------
  * the official script keeps the epoch with the best TEST error and never uses its
    validation set for selection; it is patched to select on VALIDATION error and then
    write the test predictions of that model (same protocol as cpgn_jarvis_dft.py)
  * the released teacher class declares an unused 5th forward() argument `cuda_flag` that the released
    train.py never passes (TypeError on the first batch); it is made optional (no numerical change)
  * the official script needs model/pretrain_model.pth (a whole pickled model) that is
    not in the repository; it is rebuilt from the shipped checkpoint: CrystalAE(92, 41,
    atom_fea_len=64, n_conv=5) matches every tensor of the checkpoint exactly
  * torch >= 2.6 refuses to unpickle whole models by default -> TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
  * seeds: the official code sets none; the launcher seeds python, numpy and torch
  * unchanged official behaviour you should know: the target normaliser is fitted on 2,000
    random structures of the whole dataset (test included); no early stopping is used
    (the --patience option exists but is never read)

Dataset version warning: the split depends on len(dft_3d). jarvis-tools 2022.12.11 ...
2026.4.2 serve the 75,993-entry file used in your paper; 2022.9.16 serves 55,722 and
2026.6.12 a ~94k file. The script refuses any other length (--allow-other-total).

Disk: pickles of the largest properties are ~2 GB each; they are deleted after each
property unless --keep-pkl is given.

Dependencies (same venv as CPGN is fine): torch, jarvis-tools, pymatgen (with spglib),
scikit-learn, scipy, joblib, matplotlib, plus networkx, pandas, pytz and tqdm (imported by
CrysGNN's own code). git is needed to clone the code. Missing packages are listed by the
check stage.
"""
import argparse
import csv
import json
import math
import os
import pickle
import random
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import warnings

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen")   # CIF rounding / missing electronegativity

# ============================================================================
# CONFIG - edit here (all can also be given on the command line)
# ============================================================================
WORK_DIR = "crysgnn_jarvis"       # exports, runs and results are written here
CRYSGNN_DIR = None                # None: ./CrysGNN (code-only clone made automatically if missing)
TRAIN_PYTHON = None               # None: the python running this script
DATASET_JSON = None               # None: load with jarvis-tools like cpgn_jarvis_dft.py
PROPS = "main8"                   # main8 (paper Table 5) | all | e.g. "Ef,Eg_opt,Kv"
SEEDS = [42]                      # e.g. [42, 43, 44] for mean +/- std
EPOCHS = 200                     # CrysGNN's official JARVIS setting
LR = 0.003                        # CrysGNN's official JARVIS setting
BATCH_SIZE = None                 # None: official default (128)
ALIGN_TO = "cpgn_jarvis_test.csv"  # used when this file exists
KEEP_PKL = False                  # keep the preprocessed pickles after each property
JOBS = None                       # featurisation processes (None: all CPUs)

# ---- identical to cpgn_jarvis_dft.py ---------------------------------------
N_TRAIN, N_VAL, N_TEST, SEED = 37711, 5000, 5000, 42
EXPECTED_TOTAL = 75993          # len(dft_3d) of jdft_3d-12-12-2022.json

PRIMARY = ("formation_energy_peratom", "Ef", "eV/atom")
AUX = [                          # same order and keys as JARVIS_PROPS there
    ("optb88vdw_bandgap", "Eg_opt", "eV"),   ("mbj_bandgap", "Eg_mbj", "eV"),
    ("slme", "SLME", "%"),                   ("spillage", "Spill", "-"),
    ("epsilon_avg", "eps_x", "-"),           ("epsilon_ionic", "eps_xi", "-"),
    ("bulk_modulus_kv", "Kv", "GPa"),        ("shear_modulus_gv", "Gv", "GPa"),
    ("exfoliation_en", "Exfol", "meV/at"),   ("efg", "EFG", "V/A^2"),
    ("max_ir_mode", "eij", "C/m^2"),         ("min_ir_mode", "dij", "pm/V"),
    ("ehull", "ehull", "eV/at"),             ("n_Seebeck", "See_n", "uV/K"),
    ("p_Seebeck", "See_p", "uV/K"),          ("n_powerfactor", "PF_n", "uW/mK^2"),
    ("p_powerfactor", "PF_p", "uW/mK^2"),    ("meff_me", "me", "m_e"),
    ("meff_mh", "mh", "m_e"),
]
ALL_PROPS = [PRIMARY] + AUX
KEY = {name: key for key, name, _ in ALL_PROPS}
MAIN8 = ["Ef", "Eg_opt", "Eg_mbj", "SLME", "Spill", "Kv", "Gv", "ehull"]   # paper Table 5
PAPER_MAD = {"Ef": 0.86, "Eg_opt": 0.99, "Eg_mbj": 1.79, "SLME": 10.93,
             "Spill": 0.52, "Kv": 52.80, "Gv": 27.16, "ehull": 1.14}      # paper Table 5
PRED_NAME = "crysgnn_target_prediction_results_test_set.csv"   # CrysGNN: <model>_<target>_...
SPLITS = ("train", "val", "test")


# ============================================================================
# labels, structures, split  (replicas of cpgn_jarvis_dft.py logic)
# ============================================================================
def _safe_float(val, default=0.0):
    if val is None:
        return default
    try:
        v = float(val)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except (TypeError, ValueError):
        return default


def aux_value(entry, key):
    """Validity rule of extract_targets() for auxiliary properties; None = missing."""
    raw = entry.get(key, None)
    if raw == "na" or raw is None:
        return None
    v = _safe_float(raw)
    if v == 0.0 and raw not in (0, 0.0):
        return None
    return v


def ef_value(entry):
    """Formation energy, or None when missing. NOTE: cpgn_jarvis_dft.py instead
    turns a missing value into 0.0 and trains/tests on it (see cpgn_ef_value)."""
    raw = entry.get("formation_energy_peratom", entry.get("formation_energy_per_atom", None))
    if raw is None or raw == "na":
        return None
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(v) or math.isinf(v)) else v


def cpgn_ef_value(entry):
    return _safe_float(entry.get("formation_energy_peratom",
                                 entry.get("formation_energy_per_atom", None)))


def prop_value(entry, name):
    return ef_value(entry) if name == "Ef" else aux_value(entry, KEY[name])


def parse_structure(entry):
    """jarvis_to_structure() of cpgn_jarvis_dft.py."""
    from pymatgen.core import Lattice, Structure
    atoms = entry.get("atoms", {})
    if not atoms:
        raise ValueError(f"No atoms dict: {entry.get('jid', 'unknown')}")
    lattice = Lattice(np.array(atoms["lattice_mat"], dtype=float))
    coords = np.array(atoms["coords"], dtype=float)
    return Structure(lattice, atoms["elements"], coords,
                     coords_are_cartesian=bool(atoms.get("cartesian", False)))


def reproduce_split(n_total, n_train=N_TRAIN, n_val=N_VAL, n_test=N_TEST, seed=SEED):
    """main() of cpgn_jarvis_dft.py, verbatim."""
    indices = list(range(n_total))
    random.Random(seed).shuffle(indices)
    n_tr = min(n_train, n_total - n_val - n_test)
    return {"train": indices[:n_tr],
            "val": indices[n_tr:n_tr + n_val],
            "test": indices[n_tr + n_val:n_tr + n_val + n_test]}


def load_dft3d(path=None, expect_total=EXPECTED_TOTAL, allow_other=False):
    if path:
        with open(path) as f:
            data = json.load(f)
        src = path
    else:
        try:
            from jarvis.db.figshare import data as jdata
        except ImportError:
            sys.exit("[ERROR] jarvis-tools is not installed here. Use --dataset-json, or "
                     "pip install jarvis-tools in an environment where it matches your split.")
        data = jdata("dft_3d")
        src = 'jarvis.db.figshare.data("dft_3d")'
    try:
        import importlib.metadata as md
        ver = md.version("jarvis-tools")
    except Exception:
        ver = "unknown"
    print(f"Dataset: {len(data):,} entries from {src}   (jarvis-tools {ver})")
    if len(data) != expect_total and not allow_other:
        sys.exit(
            f"[ERROR] expected {expect_total:,} entries (jdft_3d-12-12-2022.json) but found "
            f"{len(data):,}. The split is a shuffle over range(len(dft_3d)), so a different "
            "dataset version gives different train/val/test structures.\n"
            "  jarvis-tools 2022.12.11 ... 2026.4.2 -> 75,993 entries (your split)\n"
            "  jarvis-tools 2022.9.16               -> 55,722 entries\n"
            "  jarvis-tools 2026.6.12               -> newer ~94k file\n"
            "Pass --allow-other-total to continue anyway (the split will NOT match cpgn_jarvis_dft.py).")
    return data


# ============================================================================
# export
# ============================================================================
def prepare(args, props, work):
    data = load_dft3d(args.dataset_json, args.expect_total, args.allow_other_total)
    split = reproduce_split(len(data), args.n_train, args.n_val, args.n_test, args.split_seed)
    if args.smoke:
        split = {k: v[:args.smoke] for k, v in split.items()}
    print("Split  : " + "  ".join(f"{k} {len(v):,}" for k, v in split.items())
          + f"   (shuffle seed {args.split_seed})"
          + (f"   [unused entries: {len(data) - sum(len(v) for v in split.values()):,}]"
             if not args.smoke else "   [SMOKE subset]"))

    records, failed = [], []
    ef_zero = {s: 0 for s in SPLITS}
    t0 = time.time()
    for sp in SPLITS:
        for n, i in enumerate(split[sp]):
            e = data[i]
            jid = e.get("jid", f"idx{i}")
            if ef_value(e) is None:
                ef_zero[sp] += 1
            try:
                st = parse_structure(e)
            except Exception as ex:
                failed.append((sp, jid, str(ex)))
                continue
            records.append({"jid": jid, "split": sp, "structure": st,
                            "values": {nm: prop_value(e, nm) for nm in props}})
            if (n + 1) % 10000 == 0:
                print(f"  parsed {sp}: {n + 1:,}  ({time.time() - t0:.0f}s)")
    if len({r["jid"] for r in records}) != len(records):
        sys.exit("[ERROR] duplicate jids in the split; file names would collide.")
    del data

    counts = {nm: {sp: sum(1 for r in records if r["split"] == sp and r["values"][nm] is not None)
                   for sp in SPLITS} for nm in props}
    manifest = {
        "meta": {"model": "CrysGNN", "total_after_parse": len(records), "parse_failures": len(failed),
                 "ef_missing_in_split_(cpgn_script_uses_0.0)": ef_zero, "split_seed": args.split_seed,
                 "sizes_requested": [args.n_train, args.n_val, args.n_test], "smoke": args.smoke},
        "props": props,
        "counts": counts,
        "split_ids": {sp: [r["jid"] for r in records if r["split"] == sp] for sp in SPLITS},
        "test_order": [r["jid"] for r in records if r["split"] == "test"],
        "test_truth": {nm: {r["jid"]: r["values"][nm] for r in records
                            if r["split"] == "test" and r["values"][nm] is not None} for nm in props},
    }
    work.mkdir(parents=True, exist_ok=True)
    with open(work / "manifest.json", "w") as f:
        json.dump(manifest, f)
    for f_ in failed[:10]:
        print("  [parse failure]", f_)
    if any(ef_zero.values()):
        print(f"  [warning] {sum(ef_zero.values())} structures in the split have NO formation energy; "
              "cpgn_jarvis_dft.py silently uses Ef = 0.0 for them, CrysGNN excludes them.")
    print("\n  Labelled structures per property (train / val / test):")
    for nm in props:
        c = counts[nm]
        print(f"    {nm:<8} {c['train']:>7,} {c['val']:>7,} {c['test']:>7,}")
    return records, manifest


# ============================================================================
# check + setup: environment, official CrysGNN code, patch, teacher model
# ============================================================================
REQUIRED = [("torch", "torch"), ("pymatgen", "pymatgen"), ("spglib", "spglib"), ("sklearn", "scikit-learn"),
            ("scipy", "scipy"), ("joblib", "joblib"), ("jarvis", "jarvis-tools"), ("numpy", "numpy"),
            ("matplotlib", "matplotlib"), ("networkx", "networkx"), ("pandas", "pandas"),
            ("pytz", "pytz"), ("tqdm", "tqdm")]       # the last four are imported by CrysGNN's train.py / data.py
BUILD_REQUIRED = []
TOOLS = ("git",)
PROBE = r'''
import importlib, json, sys
res = {}
for m in sys.argv[1:]:
    try:
        mod = importlib.import_module(m)
        res[m] = str(getattr(mod, "__version__", "ok"))
    except Exception:
        res[m] = None
try:
    import torch
    res["_cuda"] = bool(torch.cuda.is_available())
    res["_gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
except Exception:
    pass
print(json.dumps(res))
'''
CRYSGNN_GIT = "https://github.com/kdmsit/CrysGNN.git"
SPARSE = ["/distilled_baslines/cgcnn_distilled", "/model", "/atom_init.json", "/crysgnn"]


def check_environment(python, need_build, strict=True):
    mods = REQUIRED + (BUILD_REQUIRED if need_build else [])
    out = subprocess.run([python, "-c", PROBE] + [m for m, _ in mods], capture_output=True, text=True)
    try:
        res = json.loads(out.stdout.strip().splitlines()[-1])
    except Exception:
        sys.exit(f"[ERROR] could not run {python}: {out.stderr[-300:]}")
    print(f"Environment ({python}):")
    missing = []
    for m, pip_name in mods:
        v = res.get(m)
        print(f"  {'ok     ' if v else 'MISSING'} {m:<16}{v or ''}")
        if not v:
            missing.append(pip_name)
    if "_cuda" in res:
        print(f"  CUDA available: {res['_cuda']}  {res.get('_gpu', '')}")
        if not res["_cuda"]:
            print("  [warning] no GPU visible to torch; CrysGNN will be extremely slow on CPU.")
    tools = [t for t in TOOLS if not shutil.which(t)]
    for t in tools:
        print(f"  MISSING tool: {t}  (needed to clone the CrysGNN code; on Ubuntu: sudo apt install git)")
    if (missing or tools) and strict:
        if missing:
            print("\nInstall the missing packages in this environment, e.g.\n  pip install " + " ".join(missing))
        sys.exit(1)
    return res


def run_logged(cmd, cwd, log_path, env=None, dry_run=False, label=""):
    shown = " ".join(shlex.quote(str(c)) if len(str(c)) < 200 else "<launcher shim code>" for c in cmd)
    print(f"  $ {shown}    (cwd={cwd})")
    if dry_run:
        return 0
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as log:
        rc = subprocess.run([str(c) for c in cmd], cwd=str(cwd), env=env, stdout=log,
                            stderr=subprocess.STDOUT).returncode
    if rc != 0:
        print(f"  [FAILED rc={rc}] {label}  see {log_path}")
        print("  " + "\n  ".join(Path(log_path).read_text(errors="replace").splitlines()[-8:]))
    return rc


def resolve_crysgnn_dir(arg, dry_run=False):
    cands = [arg] if arg else ["CrysGNN", "../CrysGNN", os.path.expanduser("~/CrysGNN")]
    for c in cands:
        p = Path(c)
        if ((p / "distilled_baslines" / "cgcnn_distilled" / "train.py").exists()
                and (p / "model").is_dir() and (p / "atom_init.json").exists()):
            return p.resolve()
    if arg:
        sys.exit(f"[ERROR] {arg} is not a CrysGNN folder (needs distilled_baslines/cgcnn_distilled/train.py, "
                 "model/ and atom_init.json).")
    if not shutil.which("git"):
        sys.exit(f"[ERROR] CrysGNN code not found and git is missing. Install git, or download {CRYSGNN_GIT} "
                 "yourself and set CRYSGNN_DIR.")
    print(f"Cloning the official CrysGNN code from {CRYSGNN_GIT} (code only, without the multi-GB data folders) ...")
    run_logged(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", CRYSGNN_GIT, "CrysGNN"],
               Path.cwd(), "setup_logs/git_clone.log", dry_run=dry_run, label="git clone")
    run_logged(["git", "-C", "CrysGNN", "sparse-checkout", "set", "--no-cone"] + SPARSE,
               Path.cwd(), "setup_logs/git_sparse.log", dry_run=dry_run, label="git sparse-checkout")
    p = Path("CrysGNN")
    if not dry_run and not (p / "distilled_baslines" / "cgcnn_distilled" / "train.py").exists():
        sys.exit("[ERROR] clone finished but CrysGNN/distilled_baslines/cgcnn_distilled/train.py is missing; see setup_logs/")
    return p.resolve()


def crysgnn_env():
    env = dict(os.environ)
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


TAG = "[CPGN-baseline patch]"

WRITE_PREDS = f'''
def write_predictions(loader, model, normalizer, use_cuda, path):  # {TAG}
    """Write id,target,prediction for every structure in `loader`."""
    import csv as _csv
    model.eval()
    with open(path, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["id", "target", "prediction"])
        for input, target, batch_cif_ids in loader:
            with torch.no_grad():
                if use_cuda:
                    input_var = (input[0].cuda(non_blocking=True),
                                 input[1].cuda(non_blocking=True),
                                 input[2].cuda(non_blocking=True),
                                 [c.cuda(non_blocking=True) for c in input[3]])
                else:
                    input_var = (input[0], input[1], input[2], input[3])
                output, _ = model(*input_var)
            pred = normalizer.denorm(output.data.cpu()).view(-1).tolist()
            for cid, t, p in zip(batch_cif_ids, target.view(-1).tolist(), pred):
                w.writerow([cid, t, p])

'''


def patch_file(path, edits):
    src = open(path).read()
    if TAG in src:
        print(f"  already patched: {path}")
        return
    new = src
    for old, rep in edits:
        n = new.count(old)
        if n != 1:
            sys.exit(f"[ERROR] {path}: anchor found {n} times (expected 1):\n{old!r}\n"
                     "Upstream code changed; no changes written to this file.")
        new = new.replace(old, rep)
    shutil.copy(path, path + ".orig")
    open(path, "w").write(new)
    print(f"  patched: {path}  ({len(edits)} edits, original saved as .orig)")


NO_SHUFFLE = ("        random.shuffle(self.id_prop_data)\n",
              f"        pass  # {TAG} keep id_prop.csv order (train -> val -> test)\n")


def patch_crysgnn(root):
    d = os.path.join(root, "distilled_baslines", "cgcnn_distilled")
    patch_file(os.path.join(d, "data.py"), [NO_SHUFFLE])
    patch_file(os.path.join(d, "train.py"), [
        ("import argparse\n", f"import argparse\nimport copy  # {TAG}\n"),
        ("    best_model=model\n",
         "    best_model=model\n"
         f"    best_val_mae = 1e10  # {TAG}\n"
         f"    best_state = None  # {TAG}\n"),
        ("        if test_mae_error < best_mae_error:\n"
         "            best_mae_error = min(test_mae_error, best_mae_error)\n"
         "            best_epoch = epoch\n"
         "            best_model = model\n",
         f"        if val_mae_error < best_val_mae:  # {TAG} select on VALIDATION MAE\n"
         "            best_val_mae = val_mae_error\n"
         "            best_mae_error = test_mae_error  # test MAE at the best-validation epoch\n"
         "            best_epoch = epoch\n"
         "            best_state = copy.deepcopy(model.state_dict())\n"),
        ("    out.writelines(\"Test error :\" + str(best_mae_error))\n",
         "    out.writelines(\"Test error :\" + str(best_mae_error))\n"
         f"    # {TAG} reload the best-validation weights and write the test set\n"
         "    model.load_state_dict(best_state)\n"
         "    pred_path = 'test_predictions_' + os.path.basename(os.path.normpath(args.data_path)) + '.csv'\n"
         "    write_predictions(test_loader, model, normalizer, args.cuda, pred_path)\n"
         "    print('Best-validation epoch', best_epoch, '; test predictions written to', pred_path)\n"),
        ("def validate(", WRITE_PREDS + "def validate("),
    ])


TEACHER_CODE = r'''
import sys, torch
sys.path.insert(0, ".")
from crysgnn.model import CrystalAE
m = CrystalAE(92, 41, atom_fea_len=64, n_conv=5)
ck = torch.load("../../model/crysgnn_state_checkpoint_v1.pth.tar", map_location="cpu")
m.load_state_dict(ck["state_dict"])
torch.save(m, "../../model/pretrain_model.pth")
print("teacher written:", sum(p.numel() for p in m.parameters()), "parameters")
'''


def ensure_teacher(cdir, python, env, logs, dry_run):
    out = cdir / "model" / "pretrain_model.pth"
    if out.exists():
        print("  [setup] teacher model already present")
        return
    print("  [setup] converting the shipped CrysGNN checkpoint into model/pretrain_model.pth ...")
    run_logged([python, "-c", TEACHER_CODE], cdir / "distilled_baslines" / "cgcnn_distilled",
               logs / "teacher.log", env=env, dry_run=dry_run, label="teacher conversion")
    if not dry_run:
        if not out.exists():
            sys.exit("[ERROR] teacher conversion did not produce model/pretrain_model.pth; see setup_logs/teacher.log")
        print("  [setup] teacher model written")


def patch_text(path, edits, tag, already=None):
    """Anchored, idempotent text patch with its own tag (so it also applies to files patched earlier)."""
    p = Path(path)
    src = p.read_text()
    if tag in src or (already and already in src):
        print(f"  already patched ({tag}): {p.name}")
        return
    new = src
    for old, rep in edits:
        n = new.count(old)
        if n != 1:
            sys.exit(f"[ERROR] {path}: anchor found {n} times (expected 1):\n{old!r}\nUpstream code changed; nothing written.")
        new = new.replace(old, rep)
    if not Path(str(p) + ".orig").exists():
        shutil.copy(p, str(p) + ".orig")
    p.write_text(new)
    print(f"  patched: {p}  ({tag})")


SIG_TAG = "[CPGN-baseline patch: teacher signature]"
DEV_TAG = "[CPGN-baseline patch: teacher device]"


def patch_teacher(cdir):
    """Two defects of the released code that make it unusable as shipped:
    1. CrystalAE.forward() declares a 5th argument `cuda_flag` that its body never uses, but train.py calls the
       teacher with 4 arguments (TypeError on the first batch) -> make it optional (no numerical change).
    2. With CUDA, train.py does torch.load(path) and never moves the teacher to the GPU, so it only works if the
       file was pickled from a GPU model -> move it explicitly (no-op when it is already there)."""
    wd = os.path.join(str(cdir), "distilled_baslines", "cgcnn_distilled")
    patch_text(os.path.join(wd, "crysgnn", "model.py"),
               [("    def forward(self, atom_fea, nbr_fea, nbr_fea_idx,crystal_atom_idx,cuda_flag):\n",
                 "    def forward(self, atom_fea, nbr_fea, nbr_fea_idx,crystal_atom_idx,cuda_flag=None):"
                 f"  # {SIG_TAG} cuda_flag is never used\n")], SIG_TAG, already="cuda_flag=None")
    patch_text(os.path.join(wd, "train.py"),
               [("        t_model = torch.load(checkpoint_file_path, map_location=torch.device('cpu'))\n",
                 "        t_model = torch.load(checkpoint_file_path, map_location=torch.device('cpu'))\n"
                 f"    if args.cuda:  # {DEV_TAG}\n        t_model = t_model.cuda()\n")], DEV_TAG, already="t_model = t_model.cuda()")


def setup_crysgnn(args):
    cdir = resolve_crysgnn_dir(args.crysgnn_dir, args.dry_run)
    logs = Path(args.work_dir) / "setup_logs"
    print(f"CrysGNN code: {cdir}")
    if not args.dry_run:
        patch_crysgnn(str(cdir))
        patch_teacher(cdir)
        ensure_teacher(cdir, args.train_python, crysgnn_env(), logs, False)
    else:
        print("  [dry-run] would patch the code and convert the teacher model")
    return cdir


# ============================================================================
# CrysGNN's own featurisation (verified identical to its CIFData by test)
# ============================================================================
class GaussianDistance:
    def __init__(self, dmin, dmax, step, var=None):
        assert dmin < dmax and dmax - dmin > step
        self.filter = np.arange(dmin, dmax + step, step)
        self.var = step if var is None else var

    def expand(self, distances):
        return np.exp(-(distances[..., np.newaxis] - self.filter) ** 2 / self.var ** 2)


def load_atom_init(path):
    with open(path) as f:
        return {int(k): np.array(v, dtype=float) for k, v in json.load(f).items()}


def featurise(structure, emb, gdf, radius=8, max_num_nbr=12):
    import scipy.sparse as sp
    atom_fea = np.vstack([emb[structure[i].specie.number] for i in range(len(structure))])
    all_nbrs = structure.get_all_neighbors(radius, include_index=True)
    all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
    nbr_fea_idx, nbr_fea = [], []
    for nbr in all_nbrs:
        if len(nbr) < max_num_nbr:
            nbr_fea_idx.append([x[2] for x in nbr] + [0] * (max_num_nbr - len(nbr)))
            nbr_fea.append([x[1] for x in nbr] + [radius + 1.0] * (max_num_nbr - len(nbr)))
        else:
            nbr_fea_idx.append([x[2] for x in nbr[:max_num_nbr]])
            nbr_fea.append([x[1] for x in nbr[:max_num_nbr]])
    nbr_fea_idx = np.array(nbr_fea_idx, dtype=np.int64)
    nbr_fea = gdf.expand(np.array(nbr_fea, dtype=float))
    return sp.csc_matrix(atom_fea), nbr_fea, nbr_fea_idx


def _pkl_chunk(items, atom_init_path):
    """items: (index, cif_path, target, jid, out_dir); writes CrysGNN's 5-field pickle per structure."""
    from pymatgen.core import Structure
    emb = load_atom_init(atom_init_path)
    gdf = GaussianDistance(0, 8, 0.2)
    for idx, cif, target, jid, out_dir in items:
        atom_fea, nbr_fea, nbr_idx = featurise(Structure.from_file(cif), emb, gdf)
        with open(os.path.join(out_dir, str(idx)), "wb") as f:
            pickle.dump([atom_fea, nbr_fea, nbr_idx, [float(target)], jid], f)
    return len(items)


# ============================================================================
# export: CIFs + label files (CGCNN format), pickles on demand
# ============================================================================
def export_cgcnn_format(records, manifest, work, props):
    from pymatgen.io.cif import CifWriter
    cdir = work / "cifs"
    cdir.mkdir(parents=True, exist_ok=True)
    needed = {r["jid"] for r in records if any(r["values"][nm] is not None for nm in props)}
    t0 = time.time()
    for n, r in enumerate(records):
        if r["jid"] in needed:
            CifWriter(r["structure"]).write_file(str(cdir / f"{r['jid']}.cif"))
        if (n + 1) % 10000 == 0:
            print(f"  wrote {n + 1:,} CIFs ({time.time() - t0:.0f}s)")
    kept = []
    for nm in props:
        c = manifest["counts"][nm]
        if min(c.values()) == 0:
            print(f"  [skip] {nm}: an empty split ({c}); CrysGNN cannot be run on it.")
            continue
        root = work / "data" / nm
        root.mkdir(parents=True, exist_ok=True)
        with open(root / "id_prop.csv", "w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            for sp_ in SPLITS:        # CrysGNN slices by position: train, then val, then test
                for r in records:
                    if r["split"] == sp_ and r["values"][nm] is not None:
                        w.writerow([f"../../cifs/{r['jid']}", repr(float(r["values"][nm]))])
        with open(root / "dataset_info.json", "w") as f:
            json.dump({"n_train": c["train"], "n_val": c["val"], "n_test": c["test"]}, f)
        kept.append(nm)
    print(f"Exported {len(needed):,} structures and {len(kept)} property datasets to {work}/")
    return kept


def make_pkls(work, name, atom_init, jobs, force=False, dry_run=False):
    """Build <work>/pkl/<name>/0..N-1 in id_prop.csv order. CrysGNN counts every file in the
    folder, so the completion marker lives outside it."""
    root, out, done = work / "data" / name, work / "pkl" / name, work / "pkl" / f"{name}.done"
    if done.exists() and out.is_dir() and not force:
        return out
    with open(root / "id_prop.csv", newline="") as f:
        rows = list(csv.reader(f))
    print(f"  building {len(rows):,} CrysGNN pickles for {name} with {jobs} process(es) ...")
    if dry_run:
        return out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    items = [(i, os.path.normpath(os.path.join(root, rel + ".cif")), tgt, os.path.basename(rel), str(out))
             for i, (rel, tgt) in enumerate(rows)]
    chunks = [items[k:k + 200] for k in range(0, len(items), 200)]
    t0 = time.time()
    os.environ["PYTHONWARNINGS"] = "ignore"          # worker processes inherit this
    if jobs > 1:
        from joblib import Parallel, delayed
        Parallel(n_jobs=jobs)(delayed(_pkl_chunk)(ch, str(atom_init)) for ch in chunks)
    else:
        for k, ch in enumerate(chunks):
            _pkl_chunk(ch, str(atom_init))
            if (k + 1) % 50 == 0:
                print(f"    {min((k + 1) * 200, len(items)):,}/{len(items):,} ({time.time() - t0:.0f}s)")
    n_files = len(os.listdir(out))
    if n_files != len(rows):
        sys.exit(f"[ERROR] wrote {n_files} pickles, expected {len(rows)}")
    done.write_text("ok")
    print(f"  pickles ready in {time.time() - t0:.0f}s")
    return out


# ============================================================================
# train: official CrysGNN (patched) per property and seed
# ============================================================================
LAUNCH = r'''
import os, random, runpy, sys
seed, script, rest = int(sys.argv[1]), sys.argv[2], sys.argv[3:]
sys.argv = [script] + rest
sys.path.insert(0, os.getcwd())
random.seed(seed)
try:
    import numpy as np
    np.random.seed(seed)
except Exception:
    pass
try:
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
except Exception:
    pass
print("[launcher] seeds set to", seed, flush=True)
runpy.run_path(script, run_name="__main__")
'''


def launch_cmd(python, seed, extra):
    return [python, "-c", LAUNCH, str(seed), "train.py"] + [str(x) for x in extra]


def run_cmd(cmd, cwd, log_path, dry_run, env=None, stream=False):
    shown = " ".join(shlex.quote(str(c)) if len(str(c)) < 200 else "<launcher shim code>" for c in cmd)
    print(f"  $ {shown}    (cwd={cwd})")
    if dry_run:
        return 0
    cmd = [str(c) for c in cmd]
    if not stream:
        with open(log_path, "w") as log:
            return subprocess.run(cmd, cwd=str(cwd), env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    with open(log_path, "w") as log, subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=subprocess.PIPE,
                                                      stderr=subprocess.STDOUT, text=True, bufsize=1) as p:
        for line in p.stdout:
            log.write(line)
            sys.stdout.write("    | " + line[-300:] if line.endswith("\n") else "    | " + line[-300:] + "\n")
        return p.wait()


def show_log_tail(path, n=8):
    try:
        lines = Path(path).read_text(errors="replace").splitlines()[-n:]
        print("    " + "\n    ".join(l[-200:] for l in lines))
    except Exception:
        pass


def pred_file(work, name, seed, selection):
    return work / "runs" / name / f"seed{seed}" / "predictions.csv"


def log_value(path, pattern):
    try:
        found = re.findall(pattern, Path(path).read_text(errors="replace"))
        return found[-1] if found else None
    except Exception:
        return None


def train_crysgnn(args, props, manifest, work):
    cdir = Path(args.crysgnn_dir).resolve()
    wd = cdir / "distilled_baslines" / "cgcnn_distilled"
    env = crysgnn_env()
    todo = [nm for nm in props if (work / "data" / nm / "id_prop.csv").exists()]
    print(f"{len(todo)} propert{'y' if len(todo) == 1 else 'ies'} x seeds {args.seeds}   "
          f"(resumable; {args.epochs} epochs, lr {args.lr}, Adam, batch {args.batch_size or 128})")
    failures = []
    for k, nm in enumerate(todo, 1):
        c = manifest["counts"][nm]
        pending = [s for s in args.seeds if args.force or not pred_file(work, nm, s, "best_val").exists()]
        print(f"\n[{k}/{len(todo)}] {nm}   (train {c['train']:,} / val {c['val']:,} / test {c['test']:,})")
        if not pending:
            print("  [done]")
            continue
        pkl = make_pkls(work, nm, cdir / "atom_init.json", args.jobs, args.force, args.dry_run)
        ok_all = True
        for seed in pending:
            out = work / "runs" / nm / f"seed{seed}"
            out.mkdir(parents=True, exist_ok=True)
            produced = wd / f"test_predictions_{nm}.csv"
            if produced.exists() and not args.dry_run:
                produced.unlink()
            extra = ["--epochs", args.epochs, "--data-path", f"{pkl}/", "--lr", args.lr, "--optim", "Adam",
                     "--train-size", c["train"], "--val-size", c["val"], "--test-size", c["test"]]
            if args.batch_size:
                extra += ["--batch-size", args.batch_size]
            print(f"  seed {seed}")
            t0 = time.time()
            rc = run_cmd(launch_cmd(args.train_python, seed, extra), wd, out / "train.log", args.dry_run, env, args.stream)
            if rc != 0 or (not args.dry_run and not produced.exists()):
                ok_all = False
                failures.append((nm, seed, f"training failed rc={rc}"))
                print(f"  [FAILED] see {out / 'train.log'}  (last lines:)")
                show_log_tail(out / "train.log")
                if not args.keep_going:
                    sys.exit(1)
                continue
            if not args.dry_run:
                shutil.move(str(produced), out / "predictions.csv")
                ep = log_value(out / "train.log", r"Best-validation epoch\s+(\d+)")
                te = log_value(out / "train.log", r"Test error\s*:\s*(?:tensor\()?([0-9.eE+-]+)")
                print(f"  trained in {(time.time() - t0) / 60:.1f} min; best-validation epoch {int(ep) + 1 if ep else ep}; "
                      f"CrysGNN's own test MAE at that epoch = {te}")
        if ok_all and not args.keep_pkl and not args.dry_run:
            shutil.rmtree(work / "pkl" / nm, ignore_errors=True)
            (work / "pkl" / f"{nm}.done").unlink(missing_ok=True)
    if failures:
        print("\nFailed runs:")
        for f_ in failures:
            print("  -", f_)
        sys.exit(1)


# ============================================================================
# collect: validate against the split, metrics, CSVs, plots
# ============================================================================
def norm_id(s):
    s = os.path.basename(str(s).strip())
    for ext in (".vasp", ".cif", ".json", ".xyz"):
        if s.endswith(ext):
            s = s[:-len(ext)]
    return s


def read_predictions(path):
    """CrysGNN writes 'id, target, prediction' with spaces after the commas."""
    out = {}
    with open(path, newline="") as f:
        r = csv.reader(f, skipinitialspace=True)
        header = [h.strip().lower() for h in next(r)]
        i_id, i_t, i_p = header.index("id"), header.index("target"), header.index("prediction")
        for row in r:
            if not row:
                continue
            jid = norm_id(row[i_id])
            if jid in out:
                raise ValueError(f"duplicate id {jid} in {path}")
            out[jid] = (float(row[i_p]), float(row[i_t]))
    return out


def load_alignment(path):
    ids, masks = set(), {}
    with open(path, newline="") as f:
        rd = csv.DictReader(f)
        mcols = [c for c in rd.fieldnames if c.startswith("mask_")]
        for c in mcols:
            masks[c[5:]] = set()
        for row in rd:
            ids.add(row["jid"])
            for c in mcols:
                if row[c] in ("1", "1.0"):
                    masks[c[5:]].add(row["jid"])
    return ids, masks


def metrics(y, p, name):
    err = p - y
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss = float(np.sum((y - y.mean()) ** 2))
    mad = float(np.mean(np.abs(y - y.mean())))
    return {"mae": mae, "rmse": rmse, "r2": (1 - float(np.sum(err ** 2)) / ss) if ss > 0 else float("nan"),
            "mad_test": mad, "mad_mae_test": mad / mae if mae > 0 else float("inf"),
            "mad_mae_paper": PAPER_MAD[name] / mae if (name in PAPER_MAD and mae > 0) else float("nan")}


def write_wide_csv(path, props, order, per_prop):
    """Same columns as cpgn_jarvis_test.csv: jid,true_ef,pred_ef,err_ef,true_X,pred_X,mask_X."""
    cols = ["jid"]
    for nm in props:
        cols += (["true_ef", "pred_ef", "err_ef"] if nm == "Ef"
                 else [f"true_{nm}", f"pred_{nm}", f"mask_{nm}"])
    lookup = {nm: dict(zip(per_prop[nm][0], zip(per_prop[nm][1], per_prop[nm][2])))
              for nm in props if nm in per_prop}
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(cols)
        for jid in order:
            if not any(jid in lookup.get(nm, {}) for nm in props):
                continue
            row = [jid]
            for nm in props:
                hit = lookup.get(nm, {}).get(jid)
                if nm == "Ef":
                    row += [hit[0], hit[1], hit[1] - hit[0]] if hit else ["", "", ""]
                else:
                    row += [hit[0], hit[1], 1] if hit else ["", "", 0]
            w.writerow(row)


def load_cpgn(path):
    with open(path, newline="") as f:
        return {row["jid"]: row for row in csv.DictReader(f)}


def cpgn_error(rows, name, ids):
    """MAE of cpgn_jarvis_test.csv on the given structures (only where the label exists)."""
    t, p = ("true_ef", "pred_ef") if name == "Ef" else (f"true_{name}", f"pred_{name}")
    errs = []
    for j in ids:
        r = rows.get(j)
        if r is None or (name != "Ef" and r.get(f"mask_{name}") not in ("1", "1.0")):
            continue
        try:
            errs.append(abs(float(r[p]) - float(r[t])))
        except (KeyError, ValueError):
            continue
    return (float(np.mean(errs)), len(errs)) if errs else (None, 0)


def collect(args, props, manifest, work):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    align = load_alignment(args.align_to) if args.align_to else None
    cpgn = load_cpgn(args.align_to) if args.align_to else None
    if align:
        print(f"Scoring on the {len(align[0]):,} test structures of {args.align_to}")
    summary, failures, compare = [], [], []
    seed_data = {s: {} for s in args.seeds}
    for nm in props:
        truth = manifest["test_truth"].get(nm, {})
        ids = [j for j in manifest["test_order"] if j in truth]
        if align:
            ids = [j for j in ids if j in align[0] and (nm == "Ef" or nm not in align[1] or j in align[1][nm])]
        if not ids:
            continue
        per_seed = []
        for s in args.seeds:
            path = pred_file(work, nm, s, args.selection)
            if not path.exists():
                failures.append(f"{nm} seed {s}: no {args.selection} predictions ({path.name}); run the train stage")
                continue
            try:
                preds = read_predictions(path)
            except Exception as ex:
                failures.append(f"{nm} seed {s}: unreadable ({ex})")
                continue
            missing = [j for j in ids if j not in preds]
            if missing:
                failures.append(f"{nm} seed {s}: {len(missing)} of {len(ids)} test ids missing -> not the same split")
                continue
            dt = max(abs(preds[j][1] - truth[j]) for j in ids)
            if dt > 1e-5 * max(1.0, max(abs(v) for v in truth.values())):
                failures.append(f"{nm} seed {s}: file targets differ from ours by {dt:.3g} -> rows misaligned")
                continue
            y = np.array([truth[j] for j in ids])
            p = np.array([preds[j][0] for j in ids])
            seed_data[s][nm] = (ids, y, p)
            per_seed.append(metrics(y, p, nm))
        if per_seed:
            c = manifest["counts"][nm]
            agg = {k: float(np.mean([m[k] for m in per_seed])) for k in per_seed[0]}
            row = {"model": "CrysGNN", "selection": args.selection, "property": nm, "n_train": c["train"],
                   "n_val": c["val"], "n_test_eval": len(ids), "n_seeds": len(per_seed),
                   "mae_mean": agg["mae"], "mae_std": float(np.std([m["mae"] for m in per_seed])),
                   "rmse": agg["rmse"], "r2": agg["r2"], "mad_test": agg["mad_test"],
                   "mad_mae_test": agg["mad_mae_test"], "mad_mae_paper": agg["mad_mae_paper"]}
            summary.append(row)
            if cpgn:
                cm, cn = cpgn_error(cpgn, nm, ids)
                if cm is not None:
                    compare.append({"property": nm, "n_test": cn, "crysgnn_mae": row["mae_mean"],
                                    "crysgnn_std": row["mae_std"], "cpgn_mae": cm,
                                    "crysgnn_over_cpgn": row["mae_mean"] / cm if cm > 0 else float("nan")})

    if summary:
        with open(work / "crysgnn_jarvis_summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(summary)
        hdr = "=" * 92
        print(f"\n{hdr}\n  CrysGNN JARVIS-DFT - test results   (checkpoint: {args.selection}; MAE mean +/- std over seeds)\n{hdr}")
        print(f"  {'Property':<9}{'n_train':>8}{'n_test':>8}{'seeds':>6}{'MAE':>16}{'RMSE':>10}{'R2':>8}{'MAD:MAE':>9}")
        for r in summary:
            mm = r["mad_mae_paper"] if not math.isnan(r["mad_mae_paper"]) else r["mad_mae_test"]
            print(f"  {r['property']:<9}{r['n_train']:>8,}{r['n_test_eval']:>8,}{r['n_seeds']:>6}"
                  f"{r['mae_mean']:>10.4f} +/-{r['mae_std']:.4f}{r['rmse']:>10.4f}{r['r2']:>8.3f}{mm:>9.2f}")
        print(hdr)
        print("  MAD:MAE uses the paper's MAD (Table 5) where defined, otherwise the test-set MAD.")
        if compare:
            with open(work / "crysgnn_vs_cpgn.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(compare[0]), lineterminator="\n")
                w.writeheader()
                w.writerows(compare)
            print(f"\n{hdr}\n  CrysGNN vs CPGN on the same test structures   (lower MAE is better)\n{hdr}")
            print(f"  {'Property':<9}{'n_test':>8}{'CrysGNN MAE':>13}{'CPGN MAE':>12}{'CrysGNN/CPGN':>13}   better")
            for r in compare:
                better = "CPGN" if r["cpgn_mae"] < r["crysgnn_mae"] else "CrysGNN"
                print(f"  {r['property']:<9}{r['n_test']:>8,}{r['crysgnn_mae']:>13.4f}{r['cpgn_mae']:>12.4f}"
                      f"{r['crysgnn_over_cpgn']:>13.2f}   {better}")
            print(hdr)
        wrote = []
        for s in args.seeds:
            if seed_data[s]:
                p = work / f"crysgnn_jarvis_test_seed{s}.csv"
                write_wide_csv(p, props, manifest["test_order"], seed_data[s])
                wrote.append(p)
        if wrote:
            shutil.copy(wrote[0], work / "crysgnn_jarvis_test.csv")
            print(f"  wrote {len(wrote)} per-seed CSV(s); crysgnn_jarvis_test.csv = first seed")
        first = next((s for s in args.seeds if seed_data[s]), None)
        if first is not None:
            sel = [nm for nm in ("Ef", "Eg_opt") if nm in seed_data[first]]
            if sel:
                fig, axes = plt.subplots(1, len(sel), figsize=(5 * len(sel), 4.5), squeeze=False)
                for ax, nm in zip(axes[0], sel):
                    _, y, p = seed_data[first][nm]
                    ax.scatter(y, p, s=3, alpha=0.4)
                    lim = [min(y.min(), p.min()), max(y.max(), p.max())]
                    ax.plot(lim, lim, "k--", lw=1)
                    ax.set_title(f"CrysGNN {nm} test  MAE={np.mean(np.abs(p - y)):.4f}")
                    ax.set_xlabel("true")
                    ax.set_ylabel("predicted")
                    ax.set_aspect("equal", adjustable="datalim")
                fig.tight_layout()
                fig.savefig(work / "CRYSGNN_JARVIS_parity_plots.png", dpi=200)
                plt.close(fig)
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar([r["property"] for r in summary], [r["mae_mean"] for r in summary],
                   yerr=[r["mae_std"] for r in summary], color="#E05C2A")
            ax.set_yscale("log")
            ax.set_ylabel("test MAE (property units, log scale)")
            ax.set_title("CrysGNN - JARVIS-DFT")
            fig.tight_layout()
            fig.savefig(work / "CRYSGNN_JARVIS_property_maes.png", dpi=200)
            plt.close(fig)
    if failures:
        print("\n[FAIL] runs not scored:")
        for f_ in failures:
            print("   -", f_)
        sys.exit(1)


# ============================================================================
# main
# ============================================================================
def resolve_props(spec):
    names = MAIN8 if spec == "main8" else [n for _, n, _ in ALL_PROPS] if spec == "all" \
        else [s.strip() for s in spec.split(",") if s.strip()]
    bad = [n for n in names if n not in KEY]
    if bad:
        sys.exit(f"[ERROR] unknown properties {bad}; choose from {[n for _, n, _ in ALL_PROPS]}")
    return names


def build_parser():
    p = argparse.ArgumentParser(description="CrysGNN on JARVIS-DFT with the CPGN protocol. Run with no "
                                "arguments to do everything with the CONFIG block at the top of the file.")
    p.add_argument("--work-dir", default=WORK_DIR)
    p.add_argument("--crysgnn-dir", default=CRYSGNN_DIR, help="CrysGNN folder (code-only clone made automatically if omitted)")
    p.add_argument("--train-python", default=TRAIN_PYTHON, help="python of CrysGNN's environment (default: this one)")
    p.add_argument("--dataset-json", default=DATASET_JSON)
    p.add_argument("--save-dataset", metavar="PATH", help="save jarvis dft_3d to PATH and exit")
    p.add_argument("--props", default=PROPS, help="main8 | all | comma list, e.g. Ef,Eg_opt,Kv")
    p.add_argument("--stage", choices=["check", "setup", "export", "train", "collect", "all"], default="all")
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--jobs", type=int, default=JOBS, help="featurisation processes (default: all CPUs)")
    p.add_argument("--keep-pkl", action="store_true", default=KEEP_PKL)
    p.add_argument("--align-to", default=ALIGN_TO if (ALIGN_TO and os.path.exists(ALIGN_TO)) else None,
                   help="cpgn_jarvis_test.csv: score only its test structures and compare with CPGN")
    p.add_argument("--n-train", type=int, default=N_TRAIN)
    p.add_argument("--n-val", type=int, default=N_VAL)
    p.add_argument("--n-test", type=int, default=N_TEST)
    p.add_argument("--split-seed", type=int, default=SEED)
    p.add_argument("--expect-total", type=int, default=EXPECTED_TOTAL)
    p.add_argument("--allow-other-total", action="store_true")
    p.add_argument("--smoke", type=int, default=0, help="keep only N structures per split (pipeline test)")
    p.add_argument("--force", action="store_true", help="redo runs even if results exist")
    p.add_argument("--stop-on-error", dest="keep_going", action="store_false", default=True)
    p.add_argument("--stream", action="store_true", help="show training output live (default: only in train.log)")
    p.add_argument("--dry-run", action="store_true", help="print the commands without running training")
    p.add_argument("--skip-check", action="store_true", help="do not verify the python environment first")
    return p


def main():
    args = build_parser().parse_args()
    args.train_python = args.train_python or sys.executable
    args.selection = "best_val"
    if not args.jobs:
        args.jobs = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    if args.save_dataset:
        data = load_dft3d(args.dataset_json, args.expect_total, args.allow_other_total)
        with open(args.save_dataset, "w") as f:
            json.dump(data, f)
        print(f"Saved {len(data):,} entries to {args.save_dataset}")
        return
    props = resolve_props(args.props)
    work = Path(args.work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    args.work_dir = str(work)
    stages = ["check", "setup", "export", "train", "collect"] if args.stage == "all" else [args.stage]
    if args.skip_check and args.stage == "all":
        stages.remove("check")
    print("=" * 70)
    print("  CrysGNN - JARVIS-DFT   (CPGN split and protocol)")
    print(f"  stages     : {' -> '.join(stages)}")
    print(f"  properties : {', '.join(props)}")
    print(f"  seeds      : {args.seeds}   epochs: {args.epochs}   lr: {args.lr}   checkpoint: best validation (patched)")
    print(f"  work dir   : {work}")
    print(f"  CPGN csv   : {args.align_to or 'not found (no alignment / comparison)'}")
    print("=" * 70)

    if "check" in stages:
        check_environment(args.train_python, need_build=False, strict=(args.stage == "check" or not args.dry_run))
    if "setup" in stages:
        args.crysgnn_dir = str(setup_crysgnn(args))
    elif "train" in stages and not args.crysgnn_dir:
        args.crysgnn_dir = str(resolve_crysgnn_dir(None, args.dry_run))

    manifest = None
    if "export" in stages:
        records, manifest = prepare(args, props, work)
        props = export_cgcnn_format(records, manifest, work, props)
        del records
    elif any(s in stages for s in ("train", "collect")):
        if not (work / "manifest.json").exists():
            sys.exit(f"[ERROR] {work / 'manifest.json'} not found: run the export stage first")
        with open(work / "manifest.json") as f:
            manifest = json.load(f)
        props = [p for p in props if p in manifest["props"]]
    if "train" in stages:
        train_crysgnn(args, props, manifest, work)
    if "collect" in stages and not args.dry_run:
        collect(args, props, manifest, work)
    print("\nDone.")


if __name__ == "__main__":
    main()
