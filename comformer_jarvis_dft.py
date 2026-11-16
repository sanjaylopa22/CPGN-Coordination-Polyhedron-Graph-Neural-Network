#!/usr/bin/env python3
"""
ComFormer - JARVIS-DFT baseline on the CPGN protocol          python comformer_jarvis_dft.py
=====================================================================================
Run it exactly like cpgn_jarvis_dft.py: no arguments needed. Edit the CONFIG block
below (or pass the same options on the command line) and run

        python comformer_jarvis_dft.py

Put it in the folder where cpgn_jarvis_dft.py was run. If cpgn_jarvis_test.csv is
there, ComFormer is scored on exactly CPGN's test structures and a side-by-side table
is printed at the end. Everything is resumable: stop it any time and run it again.

What "ComFormer" is here
------------------------
The OFFICIAL YKQ98/ComFormer code (ICLR 2024), by default the invariant variant
iComFormer (4.97M parameters); set MODEL = "eComformer" for the equivariant one.
Nothing is re-implemented. Settings are the repository's own JARVIS recipe
(scripts/train_jarvis.py): AdamW, lr 1e-3, batch 64, cutoff 4.0 A, 25 neighbours,
use_lattice=True, use_angle=False - with ONE change requested here: 100 epochs instead
of the official 500 (EPOCHS below). The schedule (OneCycleLR) is stretched over the
epochs you set, so a 100-epoch run is a complete, shorter schedule, not a truncated one.

What one run does  (stages: check -> setup -> export -> train -> collect)
-----------------------------------------------------------------------
 check    verifies the python environment (torch, torch_geometric, e3nn, ...)
 setup    clones the official code (code only) and applies the compatibility patches below
 export   loads JARVIS dft_3d, reproduces the CPGN split, writes one dataset file per property
 train    one official ComFormer run per (property, seed); resumable; scores the
          best-VALIDATION checkpoint
 collect  validates every prediction file against the split, writes
          comformer_jarvis_test.csv / comformer_jarvis_summary.csv / comformer_vs_cpgn.csv, plots

Same data and split as cpgn_jarvis_dft.py (checked against its source by test)
-----------------------------------------------------------------------------
  data    jarvis.db.figshare.data("dft_3d")          -> 75,993 entries (jarvis-tools 2026.4.2)
  split   random.Random(42).shuffle(range(len)); train = first min(37711, len-10000),
          then 5,000 val and 5,000 test (47,711 used; the other 28,282 are never used)
  labels  None / "na" / NaN / inf / unparsable = missing, as in extract_targets()

ComFormer is trained per property on the labelled structures of each split. Its own
data loader is fed your split through its official `dataset_array` hook (jdata is
replaced) and `keep_data_order=True`, so the exported train -> val -> test order is kept.

The public ComFormer code cannot be run as published; these explicit patches fix that
(each is anchored, idempotent, and keeps a .orig copy)
---------------------------------------------------------------------------------------
  * models/transformer.py imports from a package called `comforemr` (typo for `comformer`)
  * models/transformer.py calls super(MatformerConv, self), a name left over from the
    Matformer code it was forked from (NameError when the model is built)
  * data.py does `import imp`, removed in Python 3.12 (unused)
  * torch_scatter (compiled, not published for current PyTorch) is replaced by a
    TorchScript-compatible pure-PyTorch module with the same semantics for the three
    functions used (scatter, segment_csr, gather_csr); torch_sparse's SparseTensor was
    imported but never used
  * for regression the official train.py scores the LAST epoch and writes NO per-structure
    predictions (its CSV writer exists only for classification). It is patched to score the
    best-VALIDATION checkpoint (your CPGN protocol) and write id, target, prediction
Launcher-side (printed when used): pydantic >= 2 is bridged through pydantic.v1 for ComFormer's
config classes; numpy aliases removed in numpy 1.24 are restored; torch >= 2.6 needs
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1.

Unchanged official behaviour you should know: train and validation loaders use
drop_last=True (so a split smaller than the batch size cannot be trained; tiny splits get
a smaller batch automatically); labels are standardised with the training mean/std; no
early stopping.

Dataset version warning: the split depends on len(dft_3d). jarvis-tools 2022.12.11 ...
2026.4.2 serve the 75,993-entry file used in your paper; 2022.9.16 serves 55,722 and
2026.6.12 a ~94k file. The script refuses any other length (--allow-other-total).

Dependencies (same venv as CPGN is fine): torch, torch_geometric, jarvis-tools, pymatgen,
e3nn, pytorch-ignite, pandarallel, pydantic, pandas, scipy, scikit-learn, sympy, matplotlib,
tqdm. git is needed to clone the code. Missing packages are listed by the check stage.
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
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen")   # CIF rounding / missing electronegativity

# ============================================================================
# CONFIG - edit here (all can also be given on the command line)
# ============================================================================
WORK_DIR = "comformer_jarvis"     # exports, runs and results are written here
COMFORMER_DIR = None              # None: ./ComFormer (code-only clone made automatically if missing)
TRAIN_PYTHON = None               # None: the python running this script
DATASET_JSON = None               # None: load with jarvis-tools like cpgn_jarvis_dft.py
MODEL = "iComformer"              # iComformer (invariant, official JARVIS script) | eComformer (equivariant)
PROPS = "main8"                   # main8 (paper Table 5) | all | e.g. "Ef,Eg_opt,Kv"
SEEDS = [42]                      # e.g. [42, 43, 44] for mean +/- std
EPOCHS = 50                      # requested; the official ComFormer setting is 500
LR = None                         # None: official 0.001 (AdamW)
BATCH_SIZE = None                 # None: official 64
WORKERS = None                    # data-loading processes (None: CPU count, at most 8)
CUTOFF = 4.0                      # official JARVIS setting (Angstrom)
MAX_NEIGHBORS = 25                # official JARVIS setting
ALIGN_TO = "cpgn_jarvis_test.csv"  # used when this file exists

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
PRED_NAME = "comformer_target_prediction_results_test_set.csv"   # ComFormer: <model>_<target>_...
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
        "meta": {"model": "ComFormer", "total_after_parse": len(records), "parse_failures": len(failed),
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
              "cpgn_jarvis_dft.py silently uses Ef = 0.0 for them, ComFormer excludes them.")
    print("\n  Labelled structures per property (train / val / test):")
    for nm in props:
        c = counts[nm]
        print(f"    {nm:<8} {c['train']:>7,} {c['val']:>7,} {c['test']:>7,}")
    return records, manifest


# ============================================================================
# check + setup: environment, official ComFormer code, compatibility patches
# ============================================================================
REQUIRED = [("torch", "torch"), ("torch_geometric", "torch-geometric"), ("e3nn", "e3nn"), ("ignite", "pytorch-ignite"),
            ("pandarallel", "pandarallel"), ("pydantic", "pydantic"), ("pandas", "pandas"), ("scipy", "scipy"),
            ("sklearn", "scikit-learn"), ("sympy", "sympy"), ("tqdm", "tqdm"), ("jarvis", "jarvis-tools"),
            ("pymatgen", "pymatgen"), ("numpy", "numpy"), ("matplotlib", "matplotlib")]
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
COMFORMER_GIT = "https://github.com/YKQ98/ComFormer.git"
SPARSE = ["/comformer", "/README.md", "/setup.py"]


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
            print("  [warning] no GPU visible to torch; ComFormer will be extremely slow on CPU.")
    tools = [t for t in TOOLS if not shutil.which(t)]
    for t in tools:
        print(f"  MISSING tool: {t}  (needed to clone the ComFormer code; on Ubuntu: sudo apt install git)")
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


def resolve_comformer_dir(arg, dry_run=False):
    cands = [arg] if arg else ["ComFormer", "../ComFormer", os.path.expanduser("~/ComFormer")]
    for c in cands:
        p = Path(c)
        if (p / "comformer" / "train_props.py").exists() and (p / "comformer" / "data.py").exists():
            return p.resolve()
    if arg:
        sys.exit(f"[ERROR] {arg} is not a ComFormer folder (needs comformer/train_props.py).")
    if not shutil.which("git"):
        sys.exit(f"[ERROR] ComFormer code not found and git is missing. Install git, or download {COMFORMER_GIT} "
                 "yourself and set COMFORMER_DIR.")
    print(f"Cloning the official ComFormer code from {COMFORMER_GIT} (code only) ...")
    run_logged(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", COMFORMER_GIT, "ComFormer"],
               Path.cwd(), "setup_logs/git_clone.log", dry_run=dry_run, label="git clone")
    run_logged(["git", "-C", "ComFormer", "sparse-checkout", "set", "--no-cone"] + SPARSE,
               Path.cwd(), "setup_logs/git_sparse.log", dry_run=dry_run, label="git sparse-checkout")
    p = Path("ComFormer")
    if not dry_run and not (p / "comformer" / "train_props.py").exists():
        sys.exit("[ERROR] clone finished but ComFormer/comformer/train_props.py is missing; see setup_logs/")
    return p.resolve()


def comformer_env():
    env = dict(os.environ)
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


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


TAG = "[CPGN-baseline patch]"
COMPAT_SCATTER_SRC = r'''"""Pure-PyTorch versions of the three torch_scatter functions ComFormer uses (scatter, segment_csr, gather_csr).
torch_scatter is a compiled package that is not published for current PyTorch releases; if it is installed it is used.
Semantics follow torch_scatter: empty groups give 0 for every reduction; index / indptr are 1-D (how ComFormer calls them).
Written to be TorchScript-compatible, because ComFormer's softmax functions are decorated with @torch.jit.script."""
from typing import Optional

try:
    from torch_scatter import gather_csr, scatter, segment_csr  # noqa: F401  (the real package, if present)
except Exception:
    import torch
    from torch import Tensor

    def _expand(index: Tensor, src: Tensor, dim: int) -> Tensor:
        if index.dim() == src.dim():
            return index
        shape = [1] * src.dim()
        shape[dim] = -1
        return index.view(shape).expand_as(src)

    def scatter(src: Tensor, index: Tensor, dim: int = -1, out: Optional[Tensor] = None,
                dim_size: Optional[int] = None, reduce: str = "sum") -> Tensor:
        if out is not None:
            raise RuntimeError("scatter(out=...) is not used by ComFormer")
        if dim < 0:
            dim = dim + src.dim()
        if reduce == "add":
            reduce = "sum"
        size_n = 0
        if dim_size is not None:
            size_n = dim_size
        elif index.numel() > 0:
            size_n = int(index.max()) + 1
        size = list(src.size())
        size[dim] = size_n
        idx = _expand(index, src, dim)
        zeros = torch.zeros(size, dtype=src.dtype, device=src.device)
        if reduce == "sum":
            return zeros.scatter_add_(dim, idx, src)
        elif reduce == "mean":
            total = zeros.scatter_add_(dim, idx, src)
            count = torch.zeros(size, dtype=src.dtype, device=src.device).scatter_add_(dim, idx, torch.ones_like(src))
            return total / count.clamp(min=1)
        elif reduce == "max":
            return zeros.scatter_reduce_(dim, idx, src, "amax", include_self=False)
        elif reduce == "min":
            return zeros.scatter_reduce_(dim, idx, src, "amin", include_self=False)
        else:
            raise ValueError("unsupported reduce")

    def segment_csr(src: Tensor, indptr: Tensor, out: Optional[Tensor] = None, reduce: str = "sum") -> Tensor:
        if out is not None:
            raise RuntimeError("segment_csr(out=...) is not used by ComFormer")
        counts = indptr[1:] - indptr[:-1]
        seg = torch.repeat_interleave(torch.arange(counts.numel(), device=indptr.device), counts)
        return scatter(src, seg, 0, None, counts.numel(), reduce)

    def gather_csr(src: Tensor, indptr: Tensor, out: Optional[Tensor] = None) -> Tensor:
        if out is not None:
            raise RuntimeError("gather_csr(out=...) is not used by ComFormer")
        return torch.repeat_interleave(src, indptr[1:] - indptr[:-1], dim=0)
'''

_HEAD = ("    if (\n        config.write_predictions\n        and not classification\n"
         "        and config.model.output_features == 1\n    ):\n")
_A_OLD = _HEAD + "        net.eval()\n        targets = []\n        predictions = []\n        import time\n        t1 = time.time()\n"
_A_NEW = _HEAD + f'''        # {TAG} score the best-VALIDATION checkpoint (ignite: best_model_<epoch>_neg_mae=<score>.pt), not the last epoch
        import glob as _glob, re as _re
        _best = None
        for _p in _glob.glob(os.path.join(checkpoint_dir, "best_model_*_neg_mae=*.pt")):
            _m = _re.search(r"_(\\d+)_neg_mae=(-?\\d+(?:\\.\\d+)?)\\.pt$", _p)
            if _m and (_best is None or (float(_m.group(2)), int(_m.group(1))) > _best[0]):
                _best = ((float(_m.group(2)), int(_m.group(1))), _p)
        if _best is not None:
            _ck = torch.load(_best[1], map_location=device)
            net.load_state_dict(_ck["model"] if isinstance(_ck, dict) and isinstance(_ck.get("model"), dict) else _ck)
            print("Scoring the best-validation checkpoint:", os.path.basename(_best[1]))
        else:
            print("[warning] no best-validation checkpoint found; scoring the last-epoch model")
        net.eval()
        targets = []
        predictions = []
        import time
        t1 = time.time()
'''
_B_OLD = ('        predictions = np.array(predictions) * std_train\n'
          '        print("Test MAE:", mean_absolute_error(targets, predictions))\n        \n    return history\n')
_B_NEW = f'''        predictions = np.array(predictions) * std_train
        print("Test MAE:", mean_absolute_error(targets, predictions))
        # {TAG} per-structure predictions (original units), for the common scorer
        import csv as _csv
        _ids = list(test_loader.dataset.ids)
        _t = np.asarray(targets, dtype=float).reshape(-1) + float(mean_train)
        _p = np.asarray(predictions, dtype=float).reshape(-1) + float(mean_train)
        assert len(_ids) == len(_t) == len(_p), "test ids and predictions are misaligned"
        with open(os.path.join(config.output_dir, "prediction_results_test_set.csv"), "w", newline="") as _f:
            _w = _csv.writer(_f)
            _w.writerow(["id", "target", "prediction"])
            for _i, _a, _b in zip(_ids, _t, _p):
                _w.writerow([_i, _a, _b])
        
    return history
'''


def patch_comformer(cdir):
    pkg = Path(cdir) / "comformer"
    compat = pkg / "compat_scatter.py"
    if not compat.exists() or compat.read_text() != COMPAT_SCATTER_SRC:
        compat.write_text(COMPAT_SCATTER_SRC)
        print(f"  wrote: {compat}  (TorchScript-compatible torch_scatter replacement)")
    patch_text(pkg / "data.py", [("import imp\n", f"# import imp  {TAG} removed in Python 3.12 and unused\n")], TAG)
    patch_text(pkg / "models" / "comformer.py",
               [("from torch_scatter import scatter\n", f"from comformer.compat_scatter import scatter  # {TAG}\n")], TAG)
    patch_text(pkg / "models" / "utils.py",
               [("from torch_scatter import gather_csr, scatter, segment_csr\n",
                 f"from comformer.compat_scatter import gather_csr, scatter, segment_csr  # {TAG}\n")], TAG)
    patch_text(pkg / "models" / "transformer.py",
               [("from torch_sparse import SparseTensor\n", f"# SparseTensor import removed {TAG}: it was never used\n"),
                ("from comforemr.models.utils import softmax\n",
                 f"from comformer.models.utils import softmax  # {TAG} was 'comforemr' (typo)\n"),
                ("from torch_scatter import scatter\n", f"from comformer.compat_scatter import scatter  # {TAG}\n"),
                ("super(MatformerConv, self).__init__(node_dim=0, **kwargs)",
                 f"super(ComformerConv, self).__init__(node_dim=0, **kwargs)  # {TAG} was MatformerConv (undefined)")], TAG)
    patch_text(pkg / "train.py", [(_A_OLD, _A_NEW), (_B_OLD, _B_NEW)], TAG)


def setup_comformer(args):
    cdir = resolve_comformer_dir(args.comformer_dir, args.dry_run)
    print(f"ComFormer code: {cdir}")
    if not args.dry_run:
        patch_comformer(cdir)
    else:
        print("  [dry-run] would patch the code")
    return cdir


# ============================================================================
# export: one dataset file per property, in ComFormer's own input format
# ============================================================================
def export_datasets(records, manifest, work, props):
    from jarvis.core.atoms import pmg_to_atoms
    atoms = {}
    t0 = time.time()
    for n, r in enumerate(records):
        if any(r["values"][nm] is not None for nm in props):
            atoms[r["jid"]] = pmg_to_atoms(r["structure"]).to_dict()
        if (n + 1) % 10000 == 0:
            print(f"  converted {n + 1:,} structures ({time.time() - t0:.0f}s)")
    kept = []
    for nm in props:
        c = manifest["counts"][nm]
        if min(c.values()) == 0:
            print(f"  [skip] {nm}: an empty split ({c}); ComFormer cannot be run on it.")
            continue
        root = work / "data" / nm
        root.mkdir(parents=True, exist_ok=True)
        entries, ids = [], {sp: [] for sp in SPLITS}
        for sp in SPLITS:           # ComFormer slices by position (keep_data_order): train, then val, then test
            for r in records:
                if r["split"] == sp and r["values"][nm] is not None:
                    entries.append({"jid": r["jid"], "atoms": atoms[r["jid"]], KEY[nm]: float(r["values"][nm])})
                    ids[sp].append(r["jid"])
        with open(root / "dataset.json", "w") as f:
            json.dump(entries, f)
        with open(root / "ids.json", "w") as f:
            json.dump(ids, f)
        with open(root / "dataset_info.json", "w") as f:
            json.dump({"n_train": c["train"], "n_val": c["val"], "n_test": c["test"]}, f)
        kept.append(nm)
    print(f"Exported {len(atoms):,} structures and {len(kept)} property datasets to {work}/")
    return kept


# ============================================================================
# train: official ComFormer (patched) per property and seed
# ============================================================================
RUNNER = r'''
import contextlib, json, os, sys
cfg = json.load(open(sys.argv[1]))
sys.path.insert(0, cfg["repo"])
import numpy as np
for _n, _t in (("int", int), ("float", float), ("bool", bool), ("object", object), ("str", str), ("complex", complex)):
    if not hasattr(np, _n):
        setattr(np, _n, _t)


@contextlib.contextmanager
def pydantic_v1():
    import pydantic
    if int(pydantic.VERSION.split(".")[0]) < 2:
        yield
        return
    import pydantic.v1 as v1
    saved = {k: sys.modules.get(k) for k in ("pydantic", "pydantic.typing")}
    sys.modules["pydantic"] = v1
    sys.modules["pydantic.typing"] = v1.typing
    print("[shim] pydantic %s: loading ComFormer's config classes through pydantic.v1" % pydantic.VERSION, flush=True)
    try:
        yield
    finally:
        for k, m in saved.items():
            if m is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = m


with pydantic_v1():
    import comformer.config, comformer.models.comformer, comformer.train, comformer.train_props  # noqa: F401
import comformer.data as D
with open(cfg["data"]) as f:
    _data = json.load(f)
D.jdata = lambda name: _data          # ComFormer reads its dataset through this hook
_orig = D.get_id_train_val_test


def _forced(*a, **k):
    k["keep_data_order"] = True
    k["n_train"], k["n_val"], k["n_test"] = cfg["n_train"], cfg["n_val"], cfg["n_test"]
    tr, va, te = _orig(*a, **k)
    assert len(tr) == cfg["n_train"] and len(va) == cfg["n_val"] and len(te) == cfg["n_test"], "split sizes differ from the export"
    print("[split] keep_data_order enforced: %d / %d / %d" % (len(tr), len(va), len(te)), flush=True)
    return tr, va, te


D.get_id_train_val_test = _forced
from comformer.train_props import train_prop_model
train_prop_model(**cfg["kwargs"])
'''


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
    return work / "runs" / name / f"seed{seed}" / "prediction_results_test_set.csv"


def log_value(path, pattern):
    try:
        found = re.findall(pattern, re.sub(r"[^\x20-\x7e\n]", "", Path(path).read_text(errors="replace")))
        return found[-1] if found else None
    except Exception:
        return None


def check_split(out_dir, ids_json):
    """ComFormer records the ids it actually used; they must equal the exported split."""
    rec = out_dir / "ids_train_val_test.json"
    if not rec.exists():
        return "ComFormer did not record ids_train_val_test.json"
    with open(rec) as f:
        got = json.load(f)
    with open(ids_json) as f:
        exp = json.load(f)
    for sp, key in (("train", "id_train"), ("val", "id_val"), ("test", "id_test")):
        if list(got.get(key, [])) != exp[sp]:
            return f"ComFormer's {sp} ids differ from the export ({len(got.get(key, []))} vs {len(exp[sp])})"
    return None


def clean_checkpoints(out_dir):
    """Keep only the best-validation checkpoint (the one that was scored)."""
    best = None
    for p in out_dir.glob("best_model_*_neg_mae=*.pt"):
        m = re.search(r"_(\d+)_neg_mae=(-?\d+(?:\.\d+)?)\.pt$", p.name)
        if m and (best is None or (float(m.group(2)), int(m.group(1))) > best[0]):
            best = ((float(m.group(2)), int(m.group(1))), p)
    for p in list(out_dir.glob("checkpoint_*.pt")) + list(out_dir.glob("best_model_*.pt")):
        if best is None or p != best[1]:
            p.unlink()


def train_comformer(args, props, manifest, work):
    cdir = Path(args.comformer_dir).resolve()
    runner = work / "comformer_runner.py"
    runner.write_text(RUNNER)
    env = comformer_env()
    todo = [nm for nm in props if (work / "data" / nm / "dataset.json").exists()]
    print(f"{len(todo)} propert{'y' if len(todo) == 1 else 'ies'} x seeds {args.seeds}   "
          f"(resumable; {args.model}, {args.epochs} epochs, AdamW lr {args.lr or 0.001}, {args.workers} data-loading workers)")
    failures = []
    for k, nm in enumerate(todo, 1):
        c = manifest["counts"][nm]
        root = work / "data" / nm
        pending = [s for s in args.seeds if args.force or not pred_file(work, nm, s, "best_val").exists()]
        print(f"\n[{k}/{len(todo)}] {nm}   (train {c['train']:,} / val {c['val']:,} / test {c['test']:,})")
        if not pending:
            print("  [done]")
            continue
        bs = args.batch_size or 64
        if min(c["train"], c["val"]) < bs:       # drop_last=True on the train and validation loaders
            bs = max(1, min(16, c["train"], c["val"]))
            print(f"  [note] a split has fewer than {args.batch_size or 64} structures and ComFormer drops incomplete batches; "
                  f"batch size set to {bs} for this property")
        for seed in pending:
            out = work / "runs" / nm / f"seed{seed}"
            out.mkdir(parents=True, exist_ok=True)
            cfg = {"repo": str(cdir), "data": str(root / "dataset.json"), "n_train": c["train"], "n_val": c["val"],
                   "n_test": c["test"],
                   "kwargs": {"name": args.model, "dataset": "dft_3d", "prop": KEY[nm], "pyg_input": True,
                              "n_epochs": args.epochs, "max_neighbors": args.max_neighbors, "cutoff": args.cutoff,
                              "batch_size": bs, "use_lattice": True, "use_angle": False, "output_dir": str(out),
                              "save_dataloader": False, "use_save": False, "random_seed": seed,
                              "num_workers": args.workers, "learning_rate": args.lr or 0.001}}
            cfg_path = out / "run_config.json"
            cfg_path.write_text(json.dumps(cfg, indent=1))
            print(f"  seed {seed}")
            t0 = time.time()
            rc = run_cmd([args.train_python, "-W", "ignore", runner, cfg_path], out, out / "train.log",
                         args.dry_run, env, args.stream)
            pf = pred_file(work, nm, seed, "best_val")
            if rc != 0 or (not args.dry_run and not pf.exists()):
                failures.append((nm, seed, f"training failed rc={rc}"))
                print(f"  [FAILED] see {out / 'train.log'}  (last lines:)")
                show_log_tail(out / "train.log")
                if not args.keep_going:
                    sys.exit(1)
                continue
            if args.dry_run:
                continue
            problem = check_split(out, root / "ids.json")
            if problem:
                pf.rename(pf.with_suffix(".csv.rejected"))
                failures.append((nm, seed, problem))
                print(f"  [FAILED] {problem}")
                if not args.keep_going:
                    sys.exit(1)
                continue
            if not args.keep_checkpoints:
                clean_checkpoints(out)
            tm = log_value(out / "train.log", r"Test MAE:\s*([0-9.eE+-]+)")
            print(f"  trained in {(time.time() - t0) / 60:.1f} min; best-validation model, test MAE = {tm}   "
                  "(split verified against ComFormer's own record)")
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
    """ComFormer writes 'id, target, prediction' with spaces after the commas."""
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
            row = {"model": "ComFormer", "selection": args.selection, "property": nm, "n_train": c["train"],
                   "n_val": c["val"], "n_test_eval": len(ids), "n_seeds": len(per_seed),
                   "mae_mean": agg["mae"], "mae_std": float(np.std([m["mae"] for m in per_seed])),
                   "rmse": agg["rmse"], "r2": agg["r2"], "mad_test": agg["mad_test"],
                   "mad_mae_test": agg["mad_mae_test"], "mad_mae_paper": agg["mad_mae_paper"]}
            summary.append(row)
            if cpgn:
                cm, cn = cpgn_error(cpgn, nm, ids)
                if cm is not None:
                    compare.append({"property": nm, "n_test": cn, "comformer_mae": row["mae_mean"],
                                    "comformer_std": row["mae_std"], "cpgn_mae": cm,
                                    "comformer_over_cpgn": row["mae_mean"] / cm if cm > 0 else float("nan")})

    if summary:
        with open(work / "comformer_jarvis_summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(summary)
        hdr = "=" * 92
        print(f"\n{hdr}\n  ComFormer JARVIS-DFT - test results   (checkpoint: {args.selection}; MAE mean +/- std over seeds)\n{hdr}")
        print(f"  {'Property':<9}{'n_train':>8}{'n_test':>8}{'seeds':>6}{'MAE':>16}{'RMSE':>10}{'R2':>8}{'MAD:MAE':>9}")
        for r in summary:
            mm = r["mad_mae_paper"] if not math.isnan(r["mad_mae_paper"]) else r["mad_mae_test"]
            print(f"  {r['property']:<9}{r['n_train']:>8,}{r['n_test_eval']:>8,}{r['n_seeds']:>6}"
                  f"{r['mae_mean']:>10.4f} +/-{r['mae_std']:.4f}{r['rmse']:>10.4f}{r['r2']:>8.3f}{mm:>9.2f}")
        print(hdr)
        print("  MAD:MAE uses the paper's MAD (Table 5) where defined, otherwise the test-set MAD.")
        if compare:
            with open(work / "comformer_vs_cpgn.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(compare[0]), lineterminator="\n")
                w.writeheader()
                w.writerows(compare)
            print(f"\n{hdr}\n  ComFormer vs CPGN on the same test structures   (lower MAE is better)\n{hdr}")
            print(f"  {'Property':<9}{'n_test':>8}{'ComFormer MAE':>13}{'CPGN MAE':>12}{'ComFormer/CPGN':>13}   better")
            for r in compare:
                better = "CPGN" if r["cpgn_mae"] < r["comformer_mae"] else "ComFormer"
                print(f"  {r['property']:<9}{r['n_test']:>8,}{r['comformer_mae']:>13.4f}{r['cpgn_mae']:>12.4f}"
                      f"{r['comformer_over_cpgn']:>13.2f}   {better}")
            print(hdr)
        wrote = []
        for s in args.seeds:
            if seed_data[s]:
                p = work / f"comformer_jarvis_test_seed{s}.csv"
                write_wide_csv(p, props, manifest["test_order"], seed_data[s])
                wrote.append(p)
        if wrote:
            shutil.copy(wrote[0], work / "comformer_jarvis_test.csv")
            print(f"  wrote {len(wrote)} per-seed CSV(s); comformer_jarvis_test.csv = first seed")
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
                    ax.set_title(f"ComFormer {nm} test  MAE={np.mean(np.abs(p - y)):.4f}")
                    ax.set_xlabel("true")
                    ax.set_ylabel("predicted")
                    ax.set_aspect("equal", adjustable="datalim")
                fig.tight_layout()
                fig.savefig(work / "COMFORMER_JARVIS_parity_plots.png", dpi=200)
                plt.close(fig)
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar([r["property"] for r in summary], [r["mae_mean"] for r in summary],
                   yerr=[r["mae_std"] for r in summary], color="#E05C2A")
            ax.set_yscale("log")
            ax.set_ylabel("test MAE (property units, log scale)")
            ax.set_title("ComFormer - JARVIS-DFT")
            fig.tight_layout()
            fig.savefig(work / "COMFORMER_JARVIS_property_maes.png", dpi=200)
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
    p = argparse.ArgumentParser(description="ComFormer on JARVIS-DFT with the CPGN protocol. Run with no "
                                "arguments to do everything with the CONFIG block at the top of the file.")
    p.add_argument("--work-dir", default=WORK_DIR)
    p.add_argument("--comformer-dir", default=COMFORMER_DIR, help="ComFormer folder (code-only clone made automatically if omitted)")
    p.add_argument("--train-python", default=TRAIN_PYTHON, help="python of ComFormer's environment (default: this one)")
    p.add_argument("--dataset-json", default=DATASET_JSON)
    p.add_argument("--save-dataset", metavar="PATH", help="save jarvis dft_3d to PATH and exit")
    p.add_argument("--model", choices=["iComformer", "eComformer"], default=MODEL)
    p.add_argument("--props", default=PROPS, help="main8 | all | comma list, e.g. Ef,Eg_opt,Kv")
    p.add_argument("--stage", choices=["check", "setup", "export", "train", "collect", "all"], default="all")
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--epochs", type=int, default=EPOCHS, help="default 100 (the official setting is 500)")
    p.add_argument("--lr", type=float, default=LR, help="default: ComFormer's 0.001")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="default: ComFormer's 64")
    p.add_argument("--workers", type=int, default=WORKERS, help="data-loading processes (default: CPU count, max 8)")
    p.add_argument("--cutoff", type=float, default=CUTOFF)
    p.add_argument("--max-neighbors", type=int, default=MAX_NEIGHBORS)
    p.add_argument("--keep-checkpoints", action="store_true", help="keep every checkpoint (default: only the scored one)")
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
    if not args.workers:
        n_cpu = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
        args.workers = max(1, min(8, n_cpu))
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
    print(f"  ComFormer ({args.model}) - JARVIS-DFT   (CPGN split and protocol)")
    print(f"  stages     : {' -> '.join(stages)}")
    print(f"  properties : {', '.join(props)}")
    print(f"  seeds      : {args.seeds}   epochs: {args.epochs}   checkpoint: best validation (patched)")
    print(f"  work dir   : {work}")
    print(f"  CPGN csv   : {args.align_to or 'not found (no alignment / comparison)'}")
    print("=" * 70)

    if "check" in stages:
        check_environment(args.train_python, need_build=False, strict=(args.stage == "check" or not args.dry_run))
    if "setup" in stages:
        args.comformer_dir = str(setup_comformer(args))
    elif "train" in stages and not args.comformer_dir:
        args.comformer_dir = str(resolve_comformer_dir(None, args.dry_run))

    manifest = None
    if "export" in stages:
        records, manifest = prepare(args, props, work)
        props = export_datasets(records, manifest, work, props)
        del records
    elif any(s in stages for s in ("train", "collect")):
        if not (work / "manifest.json").exists():
            sys.exit(f"[ERROR] {work / 'manifest.json'} not found: run the export stage first")
        with open(work / "manifest.json") as f:
            manifest = json.load(f)
        props = [p for p in props if p in manifest["props"]]
    if "train" in stages:
        train_comformer(args, props, manifest, work)
    if "collect" in stages and not args.dry_run:
        collect(args, props, manifest, work)
    print("\nDone.")


if __name__ == "__main__":
    main()
