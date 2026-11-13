#!/usr/bin/env python3
"""
PotNet - JARVIS-DFT baseline on the CPGN protocol          python potnet_jarvis_dft.py
=====================================================================================
Run it exactly like cpgn_jarvis_dft.py: no arguments needed. Edit the CONFIG block
below (or pass the same options on the command line) and run

        python potnet_jarvis_dft.py

Put it in the folder where cpgn_jarvis_dft.py was run. If cpgn_jarvis_test.csv is
there, PotNet is scored on exactly CPGN's test structures and a side-by-side table
is printed at the end. Everything is resumable: stop it any time and run it again.

What one run does  (stages: check -> setup -> export -> train -> collect)
-----------------------------------------------------------------------
 check    verifies the python environment (torch, torch_geometric, ignite, ...)
 setup    clones the OFFICIAL PotNet code (divelab/AIRS), builds the GSL library and
          the Cython summation module it needs, applies one numerically identical
          compatibility fix, and tests the summation against exact values
 export   loads JARVIS dft_3d, reproduces the CPGN split, writes PotNet's
          custom-dataset files for every property
 train    one official PotNet run per (property, seed), then scores the test set
          with the best-VALIDATION checkpoint (your CPGN script's protocol)
 collect  validates every prediction file against the split, writes
          potnet_jarvis_test.csv / potnet_jarvis_summary.csv / potnet_vs_cpgn.csv, plots

Same data and split as cpgn_jarvis_dft.py (checked against its source by test)
-----------------------------------------------------------------------------
  data    jarvis.db.figshare.data("dft_3d")          -> 75,993 entries (jarvis-tools 2026.4.2)
  split   random.Random(42).shuffle(range(len)); train = first min(37711, len-10000),
          then 5,000 val and 5,000 test (47,711 used; the other 28,282 are never used)
  labels  None / "na" / NaN / inf / unparsable = missing, as in extract_targets()

PotNet is trained per property on the labelled structures of each split (the
official custom-dataset mode). Its own configs/potnet.yaml is used unchanged
(500 epochs, lr 1e-3, batch 64, AdamW, OneCycle); only random_seed / epochs can be set.

Compatibility fixes (PotNet was written for Python 3.9, numpy 1.21, pydantic 1.x,
jarvis-tools 2022.9.16; a current environment needs these; each is printed when used)
------------------------------------------------------------------------------------
  * GSL + Cython extension must be built, and LD_LIBRARY_PATH must contain GSL's lib
    folder, otherwise "libgsl.so.27: cannot open shared object file"      (automated)
  * data.py uses `astype(np.int)`, removed in numpy 1.24 -> `astype(int)`  (identical)
  * jarvis-tools >= 2022.12 changed nearest_neighbor_edges(use_canonize) from False to
    True, which silently gives PotNet different graphs (e.g. SrTiO3: 94 vs 170 edges);
    the launcher restores False (verified identical to 2022.9.16 on 5 structures)
  * jarvis.core.graphs does `import dgl` at module level (PotNet needs two of its numpy functions
    only); DGL is not installable for a current torch, so a stand-in module is used
  * torch >= 2.6 refuses to load PyG files by default -> TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
  * pydantic >= 2 lacks BaseSettings -> PotNet's config is loaded through pydantic.v1
  * torch_sparse (only used by an option that is off) is stubbed when not installed

Dataset version warning: the split depends on len(dft_3d). jarvis-tools 2022.12.11 ...
2026.4.2 serve the 75,993-entry file used in your paper; 2022.9.16 serves 55,722 and
2026.6.12 a ~94k file. The script refuses any other length (--allow-other-total).

Dependencies (same venv as CPGN is fine): torch, torch_geometric, jarvis-tools,
pymatgen, pytorch-ignite, pandarallel, periodictable, pyyaml, scikit-learn, pandas,
matplotlib, scipy, joblib, tqdm; to build PotNet's extension: cython, setuptools,
gcc, make (and git to clone). Missing packages are listed by the check stage.
"""
import argparse
import csv
import json
import math
import os
import random
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import warnings

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen")   # CIF rounding / missing electronegativity

# ============================================================================
# CONFIG - edit here (all can also be given on the command line)
# ============================================================================
WORK_DIR = "potnet_jarvis"        # exports, runs and results are written here
POTNET_DIR = None                 # None: ./AIRS/OpenMat/PotNet (cloned automatically if missing)
TRAIN_PYTHON = None               # None: the python running this script
DATASET_JSON = None               # None: load with jarvis-tools like cpgn_jarvis_dft.py
PROPS = "main8"                   # main8 (paper Table 5) | all | e.g. "Ef,Eg_opt,Kv"
SEEDS = [42]                      # e.g. [42, 43, 44] for mean +/- std
EPOCHS = 100                     # None: PotNet's official 500
SELECTION = "best_val"            # best_val (CPGN's protocol) | last_epoch (PotNet's default)
ALIGN_TO = "cpgn_jarvis_test.csv"  # used when this file exists
REUSE_CACHE = True                # seeds of one property reuse PotNet's preprocessing

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
PRED_NAME = "potnet_target_prediction_results_test_set.csv"   # PotNet: <model>_<target>_...
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
        "meta": {"model": "PotNet", "total_after_parse": len(records), "parse_failures": len(failed),
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
              "cpgn_jarvis_dft.py silently uses Ef = 0.0 for them, PotNet excludes them.")
    print("\n  Labelled structures per property (train / val / test):")
    for nm in props:
        c = counts[nm]
        print(f"    {nm:<8} {c['train']:>7,} {c['val']:>7,} {c['test']:>7,}")
    return records, manifest


def export_potnet(records, manifest, work, props):
    from pymatgen.io.vasp import Poscar
    sdir = work / "structures"
    sdir.mkdir(parents=True, exist_ok=True)
    needed = {r["jid"] for r in records if any(r["values"][nm] is not None for nm in props)}
    t0 = time.time()
    for n, r in enumerate(records):
        if r["jid"] in needed:
            Poscar(r["structure"]).write_file(str(sdir / f"{r['jid']}.vasp"))
        if (n + 1) % 10000 == 0:
            print(f"  wrote {n + 1:,} POSCARs ({time.time() - t0:.0f}s)")
    kept = []
    for nm in props:
        c = manifest["counts"][nm]
        if min(c.values()) == 0:
            print(f"  [skip] {nm}: an empty split ({c}); PotNet cannot be run on it.")
            continue
        root = work / "data" / nm
        root.mkdir(parents=True, exist_ok=True)
        with open(root / "id_prop.csv", "w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            for sp in SPLITS:            # PotNet slices by position: train, then val, then test
                for r in records:
                    if r["split"] == sp and r["values"][nm] is not None:
                        w.writerow([f"../../structures/{r['jid']}.vasp", repr(float(r["values"][nm]))])
        with open(root / "dataset_info.json", "w") as f:
            json.dump({"n_train": c["train"], "n_val": c["val"], "n_test": c["test"]}, f)
        kept.append(nm)
    print(f"Exported {len(needed):,} structures and {len(kept)} property datasets to {work}/")
    return kept


# ============================================================================
# check + setup: environment, official PotNet code, GSL + Cython build, compat fix
# ============================================================================
REQUIRED = [("torch", "torch"), ("torch_geometric", "torch-geometric"), ("ignite", "pytorch-ignite"),
            ("pandarallel", "pandarallel"), ("periodictable", "periodictable"), ("scipy", "scipy"),
            ("sklearn", "scikit-learn"), ("pandas", "pandas"), ("tqdm", "tqdm"), ("yaml", "pyyaml"),
            ("joblib", "joblib"), ("pydantic", "pydantic"), ("jarvis", "jarvis-tools"),
            ("numpy", "numpy"), ("matplotlib", "matplotlib"), ("pymatgen", "pymatgen")]
BUILD_REQUIRED = [("Cython", "cython"), ("setuptools", "setuptools")]
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
POTNET_GIT = "https://github.com/divelab/AIRS.git"
ZETA_REF = 4 * 1.6449340668482264 * 0.915965594177219      # sum_{k!=0 in Z^2} 1/|k|^4 = 4 zeta(2) beta(2)


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
            print("  [warning] no GPU visible to torch; PotNet will be extremely slow on CPU.")
    tools = [t for t in ("git", "gcc", "make") if not shutil.which(t)]
    for t in tools:
        print(f"  MISSING tool: {t}  (needed to clone / build PotNet's extension; on Ubuntu: sudo apt install git build-essential)")
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


def resolve_potnet_dir(arg, dry_run=False):
    cands = [arg] if arg else ["AIRS/OpenMat/PotNet", "../AIRS/OpenMat/PotNet",
                               os.path.expanduser("~/AIRS/OpenMat/PotNet"), "PotNet"]
    for c in cands:
        p = Path(c)
        if (p / "main.py").exists() and (p / "configs" / "potnet.yaml").exists():
            return p.resolve()
    if arg:
        sys.exit(f"[ERROR] {arg} is not a PotNet folder (no main.py / configs/potnet.yaml).")
    if not shutil.which("git"):
        sys.exit("[ERROR] PotNet code not found and git is missing. Install git, or download "
                 f"{POTNET_GIT} yourself and set POTNET_DIR to AIRS/OpenMat/PotNet.")
    print(f"Cloning the official PotNet code from {POTNET_GIT} (only OpenMat/PotNet) ...")
    run_logged(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", POTNET_GIT, "AIRS"],
               Path.cwd(), "setup_logs/git_clone.log", dry_run=dry_run, label="git clone")
    run_logged(["git", "-C", "AIRS", "sparse-checkout", "set", "OpenMat/PotNet"],
               Path.cwd(), "setup_logs/git_sparse.log", dry_run=dry_run, label="git sparse-checkout")
    p = Path("AIRS/OpenMat/PotNet")
    if not dry_run and not (p / "main.py").exists():
        sys.exit("[ERROR] clone finished but AIRS/OpenMat/PotNet/main.py is missing; see setup_logs/")
    return p.resolve()


def potnet_env(pdir):
    env = dict(os.environ)
    lib = str(pdir / "functions" / "gsl" / "lib")
    env["LD_LIBRARY_PATH"] = lib + (os.pathsep + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def build_gsl(fdir, logs, dry_run):
    lib = fdir / "gsl" / "lib"
    if list(lib.glob("libgsl.so*")):
        print("  [setup] GSL already built")
        return
    print("  [setup] building GSL 2.7.1 from PotNet's tarball (about 4 minutes) ...")
    if dry_run:
        return
    with tarfile.open(fdir / "gsl-latest.tar.gz") as t:
        top = t.getnames()[0].split("/")[0]
        t.extractall(fdir)
    src = fdir / top
    jobs = str(max(1, len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)))
    for step in (["./configure", f"--prefix={fdir / 'gsl'}"], ["make", "-j", jobs], ["make", "install"]):
        if run_logged(step, src, logs / f"gsl_{step[0].strip('./')}.log", label="GSL " + step[0]) != 0:
            sys.exit("[ERROR] GSL build failed (needs gcc and make).")


def build_extension(fdir, python, env, logs, dry_run):
    if list(fdir.glob("series*.so")):
        print("  [setup] Cython summation module already built")
        return
    print("  [setup] compiling PotNet's Cython summation module ...")
    if run_logged([python, "setup.py", "build_ext", "--inplace"], fdir, logs / "cython_build.log",
                  env=env, dry_run=dry_run, label="build_ext") != 0:
        sys.exit("[ERROR] could not compile functions/series.pyx. Try: pip install -U cython setuptools")


def apply_compat_patch(pdir):
    """`astype(np.int)` -> `astype(int)`: np.int was an alias of int (removed in numpy 1.24)."""
    p = pdir / "data.py"
    if not p.exists():
        print(f"  [setup] {p} not found; skipping the compatibility fix")
        return
    txt = p.read_text()
    old = ".astype(np.int)"
    n = txt.count(old)
    if n == 0:
        print("  [setup] data.py compatibility fix: not needed")
        return
    if n != 2:
        print(f"  [setup] data.py has {n} occurrences of `{old}` (expected 2); leaving it, the launcher's numpy alias covers it")
        return
    shutil.copy(p, str(p) + ".orig")
    p.write_text(txt.replace(old, ".astype(int)"))
    print("  [setup] data.py: astype(np.int) -> astype(int)  (original saved as data.py.orig)")


def verify_algorithm(pdir, python, env, dry_run):
    if dry_run:
        return
    code = ("import numpy as np; from algorithm import zeta; "
            "x = zeta(np.array([[0.0, 0.0]]), np.eye(2), eps=1e-12, param=2.0, R=4, verbose=False); "
            "print(repr(float(np.ravel(x)[0])))")
    out = subprocess.run([python, "-c", code], cwd=str(pdir), env=env, capture_output=True, text=True)
    try:
        val = float(out.stdout.strip().splitlines()[-1])
    except Exception:
        sys.exit("[ERROR] PotNet's summation module does not load:\n" + out.stderr[-600:]
                 + "\nIf this mentions libgsl.so, the GSL build in functions/gsl is missing or incomplete.")
    if abs(val - ZETA_REF) > 1e-6:
        sys.exit(f"[ERROR] summation test gave {val}, expected {ZETA_REF}")
    print(f"  [setup] summation module verified (zeta = {val:.8f}, exact {ZETA_REF:.8f})")


def setup_potnet(args):
    pdir = resolve_potnet_dir(args.potnet_dir, args.dry_run)
    fdir = pdir / "functions"
    logs = Path(args.work_dir) / "setup_logs"
    env = potnet_env(pdir)
    print(f"PotNet code: {pdir}")
    build_gsl(fdir, logs, args.dry_run)
    build_extension(fdir, args.train_python, env, logs, args.dry_run)
    apply_compat_patch(pdir)
    verify_algorithm(pdir, args.train_python, env, args.dry_run)
    return pdir


# Launcher: runs the unmodified main.py after installing the compatibility fixes.
LAUNCH = r'''
import contextlib, os, runpy, sys, types
args = sys.argv[2:]
sys.argv = ["main.py"] + args
sys.path.insert(0, os.getcwd())
import numpy as np
for _n, _t in (("int", int), ("float", float), ("bool", bool), ("object", object), ("str", str), ("complex", complex)):
    if not hasattr(np, _n):
        setattr(np, _n, _t)
try:
    import dgl
except Exception:
    class _D(types.ModuleType):
        def __getattr__(self, n):
            if n.startswith("__"):
                raise AttributeError(n)
            c = type(n, (), {})
            setattr(self, n, c)
            return c
    sys.modules["dgl"] = _D("dgl")
    print("[shim] dgl not installed: stub used (jarvis.core.graphs imports it, PotNet only calls two numpy functions)", flush=True)
try:
    import jarvis.core.graphs as _g
    _orig = _g.nearest_neighbor_edges
    def _nne(atoms=None, cutoff=8, max_neighbors=12, id=None, use_canonize=False):
        return _orig(atoms=atoms, cutoff=cutoff, max_neighbors=max_neighbors, id=id, use_canonize=use_canonize)
    _g.nearest_neighbor_edges = _nne
    print("[shim] jarvis nearest_neighbor_edges: use_canonize defaults to False (as in jarvis-tools 2022.9.16)", flush=True)
except Exception as _e:
    print("[shim] jarvis graph shim skipped:", repr(_e)[:160], flush=True)
try:
    import torch_sparse
except Exception:
    _m = types.ModuleType("torch_sparse")
    _m.SparseTensor = type("SparseTensor", (), {})
    sys.modules["torch_sparse"] = _m
    print("[shim] torch_sparse not installed: stub used (PotNet only needs it for a disabled option)", flush=True)
@contextlib.contextmanager
def _pydantic_v1():
    import pydantic
    if int(pydantic.VERSION.split(".")[0]) < 2:
        yield
        return
    import pydantic.v1 as v1
    saved = {k: sys.modules.get(k) for k in ("pydantic", "pydantic.typing")}
    sys.modules["pydantic"] = v1
    sys.modules["pydantic.typing"] = v1.typing
    print("[shim] pydantic %s: loading PotNet's config through pydantic.v1" % pydantic.VERSION, flush=True)
    try:
        yield
    finally:
        for k, m in saved.items():
            if m is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = m
try:
    with _pydantic_v1():
        import models.config
except ModuleNotFoundError as _e:
    if getattr(_e, "name", "") not in ("models", "models.config"):
        print("[shim] could not pre-import models.config:", repr(_e)[:200], flush=True)
except Exception as _e:
    print("[shim] could not pre-import models.config:", repr(_e)[:200], flush=True)
runpy.run_path("main.py", run_name="__main__")
'''


def launch_cmd(python, extra):
    return [python, "-c", LAUNCH, "main.py"] + [str(x) for x in extra]


# ============================================================================
# train: official PotNet per property and seed, then best-validation test scoring
# ============================================================================
CKPT_RE = re.compile(r"_(\d+)_neg_mae=(-?\d+(?:\.\d+)?)\.pt$")


def set_yaml_key(txt, key, value):
    if re.search(rf"^{key}:", txt, flags=re.M):
        return re.sub(rf"^{key}:.*$", f"{key}: {value}", txt, flags=re.M)
    return txt.rstrip("\n") + f"\n{key}: {value}\n"


def make_config(base_text, epochs, seed, batch_size=None, warmup_steps=None):
    txt = base_text
    if epochs is not None:
        txt = re.sub(r"^epochs:.*$", f"epochs: {epochs}", txt, flags=re.M)
    if batch_size is not None:
        txt = set_yaml_key(txt, "batch_size", batch_size)
    if warmup_steps is not None:
        txt = set_yaml_key(txt, "warmup_steps", warmup_steps)
    if re.search(r"^random_seed:", txt, flags=re.M):
        txt = re.sub(r"^random_seed:.*$", f"random_seed: {seed}", txt, flags=re.M)
    else:
        txt = txt.rstrip("\n") + f"\nrandom_seed: {seed}\n"
    return txt


def pick_best_checkpoint(ckpt_dir):
    """PotNet keeps its 5 best checkpoints ranked by validation neg_mae, named
    checkpoint_<epoch>_neg_mae=<score>.pt (higher score = lower validation MAE)."""
    best = None
    for p in Path(ckpt_dir).glob("*.pt"):
        m = CKPT_RE.search(p.name)
        if m:
            key = (float(m.group(2)), int(m.group(1)))
            if best is None or key > best[0]:
                best = (key, p)
    return best[1] if best else None


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
    return work / "runs" / name / f"seed{seed}" / ("best_val_predictions.csv" if selection == "best_val"
                                                   else "last_epoch_predictions.csv")


def log_test_mae(log_path):
    try:
        found = re.findall(r"Test MAE:\s*([0-9.eE+-]+)", Path(log_path).read_text(errors="replace"))
        return float(found[-1]) if found else None
    except Exception:
        return None


def train_potnet(args, props, manifest, work):
    pdir = Path(args.potnet_dir).resolve()
    base_text = (pdir / "configs" / "potnet.yaml").read_text()
    cfg_py = pdir / "models" / "config.py"
    m_ = re.search(r"warmup_steps:\s*int\s*=\s*(\d+)", cfg_py.read_text()) if cfg_py.exists() else None
    default_warmup = int(m_.group(1)) if m_ else 2000
    m_ = re.search(r"^batch_size:\s*(\d+)", base_text, flags=re.M)
    default_batch = int(m_.group(1)) if m_ else 64
    env = potnet_env(pdir)
    failures = []
    runs = [(nm, s) for nm in props for s in args.seeds if (work / "data" / nm / "id_prop.csv").exists()]
    print(f"{len(runs)} PotNet run(s): {len(props)} propert{'y' if len(props) == 1 else 'ies'} x seeds {args.seeds}"
          f"  (resumable; PotNet trains {args.epochs or 500} epochs each)")
    for k, (nm, seed) in enumerate(runs, 1):
        root = work / "data" / nm
        out = work / "runs" / nm / f"seed{seed}"
        last, best, main_pred = out / "last_epoch_predictions.csv", out / "best_val_predictions.csv", out / PRED_NAME
        first_seed = args.seeds[0]
        c = manifest["counts"][nm]
        print(f"\n[{k}/{len(runs)}] {nm}  seed {seed}   (train {c['train']:,} / val {c['val']:,} / test {c['test']:,})")
        if pred_file(work, nm, seed, args.selection).exists() and not args.force:
            print("  [done]")
            continue
        out.mkdir(parents=True, exist_ok=True)
        cfg = out / "potnet_run.yaml"
        # PotNet's train and validation loaders use drop_last=True, and OneCycleLR needs warmup_steps <= total
        # steps. Full runs satisfy both; tiny splits (smoke tests, very rare properties) would crash PotNet.
        bs = default_batch
        if min(c["train"], c["val"]) < bs:
            bs = max(1, min(16, c["train"], c["val"]))
            print(f"  [note] a split has fewer than {default_batch} structures and PotNet drops incomplete batches; "
                  f"batch size set to {bs} for this property")
        total_steps = (args.epochs or 500) * max(1, c["train"] // bs)
        warm = None
        if default_warmup >= total_steps:
            warm = max(2, total_steps // 10)          # OneCycleLR needs at least 2 warmup steps
            print(f"  [note] PotNet's {default_warmup} warmup steps exceed the {total_steps} steps of this short run; "
                  f"warmup_steps set to {warm} so its OneCycleLR scheduler is valid")
            if total_steps < 4:
                print("  [note] this run is too short for PotNet's OneCycleLR scheduler; use more epochs")
        cfg.write_text(make_config(base_text, args.epochs, seed, bs if bs != default_batch else None, warm))
        t0 = time.time()
        if not last.exists() or args.force:
            if args.reuse_cache and seed != first_seed:
                src = work / "runs" / nm / f"seed{first_seed}"
                for d in ("cache", "processed"):
                    if (src / d).is_dir() and not (out / d).exists():
                        shutil.copytree(src / d, out / d)
            rc = run_cmd(launch_cmd(args.train_python, ["--config", cfg, "--output_dir", out, "--data_root", root.resolve()]),
                         pdir, out / "train.log", args.dry_run, env, args.stream)
            if rc != 0 or (not args.dry_run and not main_pred.exists()):
                failures.append((nm, seed, f"training failed rc={rc}"))
                print(f"  [FAILED] see {out / 'train.log'}  (last lines:)")
                show_log_tail(out / "train.log")
                if not args.keep_going:
                    sys.exit(1)
                continue
            if not args.dry_run:
                shutil.copy(main_pred, last)
                mae = log_test_mae(out / "train.log")
                print(f"  trained in {(time.time() - t0) / 60:.1f} min; last-epoch test MAE (PotNet's own) = "
                      f"{mae if mae is not None else 'n/a'}")
        if args.selection == "best_val" and (not best.exists() or args.force):
            ckpt = pick_best_checkpoint(out / "checkpoint") if not args.dry_run else Path("<best checkpoint>")
            if ckpt is None:
                failures.append((nm, seed, "no checkpoint named *_neg_mae=*.pt found"))
                print(f"  [FAILED] no checkpoint in {out / 'checkpoint'}")
                continue
            print(f"  scoring the test set with the best-validation checkpoint: {ckpt.name}")
            rc = run_cmd(launch_cmd(args.train_python, ["--config", cfg, "--output_dir", out, "--data_root", root.resolve(),
                                                       "--checkpoint", ckpt, "--testing"]),
                         pdir, out / "test_best.log", args.dry_run, env, args.stream)
            if rc != 0 or (not args.dry_run and not main_pred.exists()):
                failures.append((nm, seed, f"best-checkpoint test failed rc={rc}"))
                print(f"  [FAILED] see {out / 'test_best.log'}  (last lines:)")
                show_log_tail(out / "test_best.log")
                if not args.keep_going:
                    sys.exit(1)
                continue
            if not args.dry_run:
                shutil.copy(main_pred, best)
                print(f"  best-validation test MAE (PotNet's own) = {log_test_mae(out / 'test_best.log')}")
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
    """PotNet writes 'id, target, prediction' with spaces after the commas."""
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
            row = {"model": "PotNet", "selection": args.selection, "property": nm, "n_train": c["train"],
                   "n_val": c["val"], "n_test_eval": len(ids), "n_seeds": len(per_seed),
                   "mae_mean": agg["mae"], "mae_std": float(np.std([m["mae"] for m in per_seed])),
                   "rmse": agg["rmse"], "r2": agg["r2"], "mad_test": agg["mad_test"],
                   "mad_mae_test": agg["mad_mae_test"], "mad_mae_paper": agg["mad_mae_paper"]}
            summary.append(row)
            if cpgn:
                cm, cn = cpgn_error(cpgn, nm, ids)
                if cm is not None:
                    compare.append({"property": nm, "n_test": cn, "potnet_mae": row["mae_mean"],
                                    "potnet_std": row["mae_std"], "cpgn_mae": cm,
                                    "potnet_over_cpgn": row["mae_mean"] / cm if cm > 0 else float("nan")})

    if summary:
        with open(work / "potnet_jarvis_summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(summary)
        hdr = "=" * 92
        print(f"\n{hdr}\n  PotNet JARVIS-DFT - test results   (checkpoint: {args.selection}; MAE mean +/- std over seeds)\n{hdr}")
        print(f"  {'Property':<9}{'n_train':>8}{'n_test':>8}{'seeds':>6}{'MAE':>16}{'RMSE':>10}{'R2':>8}{'MAD:MAE':>9}")
        for r in summary:
            mm = r["mad_mae_paper"] if not math.isnan(r["mad_mae_paper"]) else r["mad_mae_test"]
            print(f"  {r['property']:<9}{r['n_train']:>8,}{r['n_test_eval']:>8,}{r['n_seeds']:>6}"
                  f"{r['mae_mean']:>10.4f} +/-{r['mae_std']:.4f}{r['rmse']:>10.4f}{r['r2']:>8.3f}{mm:>9.2f}")
        print(hdr)
        print("  MAD:MAE uses the paper's MAD (Table 5) where defined, otherwise the test-set MAD.")
        if compare:
            with open(work / "potnet_vs_cpgn.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(compare[0]), lineterminator="\n")
                w.writeheader()
                w.writerows(compare)
            print(f"\n{hdr}\n  PotNet vs CPGN on the same test structures   (lower MAE is better)\n{hdr}")
            print(f"  {'Property':<9}{'n_test':>8}{'PotNet MAE':>13}{'CPGN MAE':>12}{'PotNet/CPGN':>13}   better")
            for r in compare:
                better = "CPGN" if r["cpgn_mae"] < r["potnet_mae"] else "PotNet"
                print(f"  {r['property']:<9}{r['n_test']:>8,}{r['potnet_mae']:>13.4f}{r['cpgn_mae']:>12.4f}"
                      f"{r['potnet_over_cpgn']:>13.2f}   {better}")
            print(hdr)
        wrote = []
        for s in args.seeds:
            if seed_data[s]:
                p = work / f"potnet_jarvis_test_seed{s}.csv"
                write_wide_csv(p, props, manifest["test_order"], seed_data[s])
                wrote.append(p)
        if wrote:
            shutil.copy(wrote[0], work / "potnet_jarvis_test.csv")
            print(f"  wrote {len(wrote)} per-seed CSV(s); potnet_jarvis_test.csv = first seed")
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
                    ax.set_title(f"PotNet {nm} test  MAE={np.mean(np.abs(p - y)):.4f}")
                    ax.set_xlabel("true")
                    ax.set_ylabel("predicted")
                    ax.set_aspect("equal", adjustable="datalim")
                fig.tight_layout()
                fig.savefig(work / "POTNET_JARVIS_parity_plots.png", dpi=200)
                plt.close(fig)
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar([r["property"] for r in summary], [r["mae_mean"] for r in summary],
                   yerr=[r["mae_std"] for r in summary], color="#E05C2A")
            ax.set_yscale("log")
            ax.set_ylabel("test MAE (property units, log scale)")
            ax.set_title("PotNet - JARVIS-DFT")
            fig.tight_layout()
            fig.savefig(work / "POTNET_JARVIS_property_maes.png", dpi=200)
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
    p = argparse.ArgumentParser(description="PotNet on JARVIS-DFT with the CPGN protocol. Run with no "
                                "arguments to do everything with the CONFIG block at the top of the file.")
    p.add_argument("--work-dir", default=WORK_DIR)
    p.add_argument("--potnet-dir", default=POTNET_DIR, help="AIRS/OpenMat/PotNet (cloned automatically if omitted)")
    p.add_argument("--train-python", default=TRAIN_PYTHON, help="python of PotNet's environment (default: this one)")
    p.add_argument("--dataset-json", default=DATASET_JSON)
    p.add_argument("--save-dataset", metavar="PATH", help="save jarvis dft_3d to PATH and exit")
    p.add_argument("--props", default=PROPS, help="main8 | all | comma list, e.g. Ef,Eg_opt,Kv")
    p.add_argument("--stage", choices=["check", "setup", "export", "train", "collect", "all"], default="all")
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--epochs", type=int, default=EPOCHS, help="override PotNet's 500 epochs")
    p.add_argument("--selection", choices=["best_val", "last_epoch"], default=SELECTION)
    p.add_argument("--align-to", default=ALIGN_TO if (ALIGN_TO and os.path.exists(ALIGN_TO)) else None,
                   help="cpgn_jarvis_test.csv: score only its test structures and compare with CPGN")
    p.add_argument("--n-train", type=int, default=N_TRAIN)
    p.add_argument("--n-val", type=int, default=N_VAL)
    p.add_argument("--n-test", type=int, default=N_TEST)
    p.add_argument("--split-seed", type=int, default=SEED)
    p.add_argument("--expect-total", type=int, default=EXPECTED_TOTAL)
    p.add_argument("--allow-other-total", action="store_true")
    p.add_argument("--smoke", type=int, default=0, help="keep only N structures per split (pipeline test)")
    p.add_argument("--no-reuse-cache", dest="reuse_cache", action="store_false", default=REUSE_CACHE)
    p.add_argument("--force", action="store_true", help="redo runs even if results exist")
    p.add_argument("--stop-on-error", dest="keep_going", action="store_false", default=True)
    p.add_argument("--stream", action="store_true", help="show PotNet's output live (default: only in train.log)")
    p.add_argument("--dry-run", action="store_true", help="print the commands without running training")
    p.add_argument("--skip-check", action="store_true", help="do not verify the python environment first")
    return p


def main():
    args = build_parser().parse_args()
    args.train_python = args.train_python or sys.executable
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
    if args.skip_check and "check" in stages and args.stage == "all":
        stages.remove("check")
    print("=" * 70)
    print("  PotNet - JARVIS-DFT   (CPGN split and protocol)")
    print(f"  stages     : {' -> '.join(stages)}")
    print(f"  properties : {', '.join(props)}")
    print(f"  seeds      : {args.seeds}   epochs: {args.epochs or '500 (official)'}   checkpoint: {args.selection}")
    print(f"  work dir   : {work}")
    print(f"  CPGN csv   : {args.align_to or 'not found (no alignment / comparison)'}")
    print("=" * 70)

    if "check" in stages:
        try:
            pd_ = resolve_potnet_dir(args.potnet_dir) if (args.potnet_dir or Path("AIRS/OpenMat/PotNet/main.py").exists()) else None
        except SystemExit:
            pd_ = None
        built = bool(pd_ and list((pd_ / "functions").glob("series*.so")))
        check_environment(args.train_python, need_build=not built, strict=(args.stage == "check" or not args.dry_run))
    if "setup" in stages:
        args.potnet_dir = str(setup_potnet(args))
    elif "train" in stages and not args.potnet_dir:
        args.potnet_dir = str(resolve_potnet_dir(None, args.dry_run))

    manifest = None
    if "export" in stages:
        records, manifest = prepare(args, props, work)
        props = export_potnet(records, manifest, work, props)
        del records
    elif any(s in stages for s in ("train", "collect")):
        if not (work / "manifest.json").exists():
            sys.exit(f"[ERROR] {work / 'manifest.json'} not found: run the export stage first")
        with open(work / "manifest.json") as f:
            manifest = json.load(f)
        props = [p for p in props if p in manifest["props"]]
    if "train" in stages:
        train_potnet(args, props, manifest, work)
    if "collect" in stages and not args.dry_run:
        collect(args, props, manifest, work)
    print("\nDone.")


if __name__ == "__main__":
    main()
