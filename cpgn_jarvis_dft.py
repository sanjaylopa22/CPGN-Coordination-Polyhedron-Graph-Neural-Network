"""
CPGN — Adapted for JARVIS-DFT Multi-Property Benchmark
========================================================
Retains all six fixes from the MP-upgraded version and extends the model
to cover the full suite of JARVIS-DFT properties described in:

  Choudhary et al., npj Comput. Mater. 2021  (ALIGNN paper, Table 1)

Properties predicted
--------------------
  PRIMARY (MAE loss, full weight):
    Ef      formation_energy_peratom   eV/atom
  AUXILIARY (MAE/BCE, weight λ=0.1):
    Eg_opt  optb88vdw_bandgap          eV
    Eg_mbj  mbj_bandgap                eV
    SLME    slme                       %          (solar-cell efficiency)
    Spill   spillage                   —          (topological spillage)
    eps_x   epsilon_avg (OPT)          —          (dielectric, no ionic)
    eps_xi  epsilon_ionic              —          (dielectric ionic contrib.)
    Kv      bulk_modulus_kv            GPa
    Gv      shear_modulus_gv           GPa
    Exfol   exfoliation_en             meV/atom   (2D only; NaN→0 for 3D)
    EFG     efg                        V/Å²
    eij     max_ir_mode                C/m²       (piezoelectric stress proxy)
    dij     min_ir_mode                pm/V       (piezoelectric strain proxy)
    ehull   ehull                      eV/atom    (energy above convex hull)
    See_n   n_Seebeck                  μV/K
    See_p   p_Seebeck                  μV/K
    PF_n    n_powerfactor              μW/(mK²)
    PF_p    p_powerfactor              μW/(mK²)
    me      meff_me                    mₑ         (electron effective mass)
    mh      meff_mh                    mₑ         (hole effective mass)

Data source
-----------
  jarvis-tools  (pip install jarvis-tools)
  >> from jarvis.db.figshare import data
  >> dft_3d = data('dft_3d')          # ~48 k 3D structures

Split: 37 711 / 5 000 / 5 000  (shuffled, SEED=42)
  (remaining entries used for train; adjust N_TRAIN/N_VAL/N_TEST as needed)

Published ALIGNN benchmarks (Table 1, npj Comput. Mater. 2021)
---------------------------------------------------------------
  Ef    : ALIGNN 0.022, CFID 0.104, CGCNN 0.063, MEGNet 0.030
  Eg_opt: ALIGNN 0.142, CGCNN 0.200, MEGNet 0.330
  Eg_mbj: ALIGNN 0.210, CFID  0.510
  SLME  : ALIGNN 1.42 %
  Kv    : ALIGNN 14.41 GPa
  Gv    : ALIGNN 10.65 GPa

Fixes retained from MP version
--------------------------------
  Fix 1  MAE loss for formation energy
  Fix 2  Shuffled split SEED=42
  Fix 3  Single-task primary + lightweight auxiliary heads (λ=0.1)
  Fix 4  Learned 92-dim elemental embedding
  Fix 5  Explicit angle encoding via line graph (3-body)
  Fix 6  500 epochs with cosine-annealing LR

Outputs
-------
  cpgn_jarvis_best.pt
  cpgn_jarvis_test.csv / cpgn_jarvis_val.csv
  CPGN_JARVIS_training_curve.png
  CPGN_JARVIS_parity_plots.png        (Ef + Eg_opt parity, val & test)
  CPGN_JARVIS_property_maes.png       (bar chart: all property MAEs)
  CPGN_JARVIS_error_hist.png
  CPGN_JARVIS_benchmark_table.png     (vs ALIGNN / CGCNN / MEGNet)

Dependencies
------------
  pip install torch torch-geometric pymatgen scikit-learn matplotlib jarvis-tools
"""

# ============================================================
# 0.  IMPORTS
# ============================================================
import os, sys, json, warnings, time, random, math, pickle
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pymatgen.core import Structure, Element, Lattice
from pymatgen.analysis.local_env import VoronoiNN

from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    accuracy_score, f1_score,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Data
from torch_geometric.data import Batch as PyGBatch
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch.utils.data import DataLoader as TorchLoader

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


# ============================================================
# 1.  CONFIG
# ============================================================
OUTPUT_CSV     = "cpgn_jarvis_test.csv"
OUTPUT_VAL_CSV = "cpgn_jarvis_val.csv"
CHECKPOINT     = "cpgn_jarvis_best.pt"
GRAPH_CACHE    = "cpgn_jarvis_graph_cache.pkl"
SKIP_IF_CKPT   = True

# ── Split ─────────────────────────────────────────────────────────────────────
N_TRAIN = 37711          # adjust to len(dft_3d) - N_VAL - N_TEST if dataset grows
N_VAL   =  5000
N_TEST  =  5000
SEED    = 42

# ── Training ──────────────────────────────────────────────────────────────────
EPOCHS        = 500
BATCH_SIZE    = 64
LEARNING_RATE = 3e-4
WEIGHT_DECAY  = 1e-5
PATIENCE      = 50
LAMBDA_AUX    = 0.1      # auxiliary loss weight (keeps Ef gradient dominant)

# ── Model ─────────────────────────────────────────────────────────────────────
N_ELEM        = 103      # learned embedding per element (Z=1..102)
ELEM_DIM      = 64
N_POLY_FEAT   = 7        # polyhedron geometric features
N_EDGE_FEAT   = 40       # RBF bond distances
N_ANGLE_FEAT  = 40       # RBF bond angles (line graph)
HIDDEN_DIM    = 256
N_LAYERS      = 4
PRED_DIM      = 128
CUTOFF        = 8.0      # Å
RBF_CENTRES   = 40

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device : {DEVICE}")

# ── JARVIS-DFT property keys  (jarvis-tools field names) ──────────────────────
# Each entry: (jarvis_key, display_name, unit, activation, is_positive)
#   activation: "none" | "softplus" | "sigmoid"
#   is_positive: True → clamp predictions to ≥0 via softplus
JARVIS_PROPS = [
    # key                      name        unit       activation   pos?
    ("optb88vdw_bandgap",    "Eg_opt",   "eV",      "softplus",  True ),
    ("mbj_bandgap",          "Eg_mbj",   "eV",      "softplus",  True ),
    ("slme",                 "SLME",     "%",       "softplus",  True ),
    ("spillage",             "Spill",    "—",       "none",      False),
    ("epsilon_avg",          "eps_x",    "—",       "softplus",  True ),
    ("epsilon_ionic",        "eps_xi",   "—",       "softplus",  True ),
    ("bulk_modulus_kv",      "Kv",       "GPa",     "softplus",  True ),
    ("shear_modulus_gv",     "Gv",       "GPa",     "softplus",  True ),
    ("exfoliation_en",       "Exfol",    "meV/at",  "softplus",  True ),
    ("efg",                  "EFG",      "V/Å²",    "none",      False),
    ("max_ir_mode",          "eij",      "C/m²",    "none",      False),
    ("min_ir_mode",          "dij",      "pm/V",    "none",      False),
    ("ehull",                "ehull",    "eV/at",   "softplus",  True ),
    ("n_Seebeck",            "See_n",    "μV/K",    "none",      False),
    ("p_Seebeck",            "See_p",    "μV/K",    "none",      False),
    ("n_powerfactor",        "PF_n",     "μW/mK²",  "softplus",  True ),
    ("p_powerfactor",        "PF_p",     "μW/mK²",  "softplus",  True ),
    ("meff_me",              "me",       "mₑ",      "softplus",  True ),
    ("meff_mh",              "mh",       "mₑ",      "softplus",  True ),
]
N_AUX = len(JARVIS_PROPS)   # 19 auxiliary regression targets

# ── ALIGNN published MAE values for benchmark table ───────────────────────────
ALIGNN_BENCH = {
    "Ef"    : 0.022,
    "Eg_opt": 0.142,
    "Eg_mbj": 0.210,
    "SLME"  : 1.42,
    "Kv"    : 14.41,
    "Gv"    : 10.65,
}
JARVIS_MAD = {          # from ALIGNN paper Table 1
    "Ef"    : 0.86,
    "Eg_opt": 1.14,
    "Eg_mbj": 1.55,
}


# ============================================================
# 2.  RBF EXPANSION
# ============================================================
class RBFExpansion(nn.Module):
    """Gaussian RBF — same as DimeNet++ / ALIGNN."""
    def __init__(self, low: float, high: float, n_centres: int):
        super().__init__()
        centres = torch.linspace(low, high, n_centres)
        self.register_buffer("centres", centres)
        self.gamma = 1.0 / (2.0 * ((high - low) / n_centres) ** 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.gamma * (x.unsqueeze(-1) - self.centres) ** 2)


# ============================================================
# 3.  JARVIS STRUCTURE PARSER
#     Converts jarvis-tools atom dict → pymatgen Structure
# ============================================================
def jarvis_to_structure(entry: dict) -> Structure:
    """
    Parse a JARVIS-DFT entry (from jarvis-tools data('dft_3d')) into
    a pymatgen Structure.  The 'atoms' dict contains:
      lattice_mat  (3×3 matrix)
      coords       (fractional or Cartesian)
      elements     (list of element symbols)
      cartesian    (bool)
    """
    atoms = entry.get("atoms", {})
    if not atoms:
        jid = entry.get("jid", "unknown")
        raise ValueError(f"No atoms dict: {jid}")

    lat_mat  = np.array(atoms["lattice_mat"], dtype=float)
    lattice  = Lattice(lat_mat)
    elements = atoms["elements"]
    coords   = np.array(atoms["coords"], dtype=float)
    cart     = atoms.get("cartesian", False)

    if cart:
        return Structure(lattice, elements, coords,
                         coords_are_cartesian=True)
    else:
        return Structure(lattice, elements, coords,
                         coords_are_cartesian=False)


# ============================================================
# 4.  ATOMIC NUMBER LOOKUP
# ============================================================
def get_atomic_number(element_str: str) -> int:
    sym = str(element_str).split("+")[0].split("-")[0].strip()
    try:
        z = Element(sym).Z
    except Exception:
        z = 1
    return max(1, min(z, N_ELEM - 1))


# ============================================================
# 5.  POLYHEDRON FEATURE VECTOR  (7-dim)
# ============================================================
ATOM_EN = {
    "H":2.20,"He":0.00,"Li":0.98,"Be":1.57,"B":2.04,"C":2.55,"N":3.04,
    "O":3.44,"F":3.98,"Ne":0.00,"Na":0.93,"Mg":1.31,"Al":1.61,"Si":1.90,
    "P":2.19,"S":2.58,"Cl":3.16,"K":0.82,"Ca":1.00,"Sc":1.36,"Ti":1.54,
    "V":1.63,"Cr":1.66,"Mn":1.55,"Fe":1.83,"Co":1.88,"Ni":1.91,"Cu":1.90,
    "Zn":1.65,"Ga":1.81,"Ge":2.01,"As":2.18,"Se":2.55,"Br":2.96,"Rb":0.82,
    "Sr":0.95,"Y":1.22,"Zr":1.33,"Nb":1.60,"Mo":2.16,"Ru":2.20,"Rh":2.28,
    "Pd":2.20,"Ag":1.93,"Cd":1.69,"In":1.78,"Sn":1.96,"Sb":2.05,"Te":2.10,
    "I":2.66,"Cs":0.79,"Ba":0.89,"La":1.10,"Ce":1.12,"Pr":1.13,"Nd":1.14,
    "Sm":1.17,"Eu":1.20,"Gd":1.20,"Hf":1.30,"Ta":1.50,"W":2.36,"Re":1.90,
    "Os":2.20,"Ir":2.20,"Pt":2.28,"Au":2.54,"Pb":2.33,"Bi":2.02,"U":1.38,
}

def get_poly_features(structure, site_idx, nn_info) -> np.ndarray:
    if not nn_info:
        return np.zeros(7, dtype=np.float32)
    center    = structure[site_idx]
    center_en = ATOM_EN.get(str(center.specie), 1.5)
    bond_lengths, en_diffs, positions = [], [], []
    for nb in nn_info:
        nb_site = nb["site"]
        bl = center.distance(nb_site)
        bond_lengths.append(bl)
        en_diffs.append(abs(center_en - ATOM_EN.get(str(nb_site.specie), 1.5)))
        positions.append(nb_site.coords)
    bl_arr  = np.array(bond_lengths, dtype=np.float32)
    cn      = len(bl_arr)
    mean_bl = bl_arr.mean()
    di      = float(bl_arr.std() / (mean_bl + 1e-8))
    angles  = []
    pos     = np.array(positions, dtype=np.float32)
    c_pos   = np.array(center.coords, dtype=np.float32)
    for i in range(cn):
        for j in range(i + 1, cn):
            v1 = pos[i] - c_pos; v2 = pos[j] - c_pos
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 < 1e-8 or n2 < 1e-8: continue
            cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
            angles.append(np.degrees(np.arccos(cos_a)))
    bav         = float(np.var(angles)) if len(angles) > 1 else 0.0
    vol_proxy   = float(mean_bl ** 3)
    en_mismatch = float(np.mean(en_diffs))
    ct_enc      = min(cn, 12) / 12.0
    return np.array([cn/12.0, di, bav/10000.0, vol_proxy/100.0,
                     mean_bl/8.0, en_mismatch, ct_enc], dtype=np.float32)


# ============================================================
# 6.  DUAL GRAPH BUILDER  (atom + line + poly)
# ============================================================
def build_dual_graphs(structure: Structure) -> dict:
    try:    vnn = VoronoiNN(cutoff=CUTOFF, allow_pathological=True)
    except: vnn = VoronoiNN(cutoff=CUTOFF)
    N = len(structure)

    atom_z = np.array([get_atomic_number(str(s.specie)) for s in structure],
                       dtype=np.int64)
    poly_x  = np.zeros((N, N_POLY_FEAT), dtype=np.float32)
    site_nn = []
    poly_nb = []
    for i in range(N):
        try:    nn = vnn.get_nn_info(structure, i)
        except: nn = []
        site_nn.append(nn)
        poly_x[i] = get_poly_features(structure, i, nn)
        poly_nb.append(set(nb["site_index"] for nb in nn))

    # atom graph
    a_src, a_dst, a_dist = [], [], []
    for i, nn in enumerate(site_nn):
        for nb in nn:
            j = nb["site_index"]
            d = structure[i].distance(nb["site"])
            if d > CUTOFF: continue
            a_src.append(i); a_dst.append(j); a_dist.append(d)
    if not a_src:
        for i in range(N):
            for j in range(N):
                if i == j: continue
                d = structure.get_distance(i, j)
                if d < CUTOFF:
                    a_src.append(i); a_dst.append(j); a_dist.append(d)

    E = len(a_src)
    atom_ei   = np.array([a_src, a_dst], dtype=np.int64)
    atom_dist = np.array(a_dist, dtype=np.float32)

    # line graph
    incoming = {i: [] for i in range(N)}
    for eid, (src, dst) in enumerate(zip(a_src, a_dst)):
        incoming[dst].append((src, eid))
    lg_src, lg_dst, lg_angles = [], [], []
    for i in range(N):
        nbrs = incoming[i]
        for p in range(len(nbrs)):
            for q in range(len(nbrs)):
                if p == q: continue
                src_p, eid_p = nbrs[p]
                src_q, eid_q = nbrs[q]
                v1 = np.array(structure[src_p].coords) - np.array(structure[i].coords)
                v2 = np.array(structure[src_q].coords) - np.array(structure[i].coords)
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if n1 < 1e-8 or n2 < 1e-8: continue
                cos_a = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
                lg_src.append(eid_p); lg_dst.append(eid_q)
                lg_angles.append(float(np.arccos(cos_a)))
    if lg_src:
        line_ei     = np.array([lg_src, lg_dst], dtype=np.int64)
        line_angles = np.array(lg_angles, dtype=np.float32)
    else:
        line_ei     = np.zeros((2, 0), dtype=np.int64)
        line_angles = np.zeros(0, dtype=np.float32)

    # poly graph
    p_src, p_dst, p_ea = [], [], []
    for i in range(N):
        for j in range(i + 1, N):
            shared = poly_nb[i] & poly_nb[j]
            ns = len(shared)
            if ns == 0: continue
            ct = 0.0 if ns == 1 else (0.5 if ns == 2 else 1.0)
            p_src += [i, j]; p_dst += [j, i]; p_ea += [[ct], [ct]]
    if not p_src:
        for i in range(N):
            best_j, best_d = -1, 1e9
            for j in range(N):
                if i == j: continue
                d = structure.get_distance(i, j)
                if d < best_d: best_d, best_j = d, j
            if best_j >= 0:
                p_src += [i, best_j]; p_dst += [best_j, i]
                p_ea  += [[0.5], [0.5]]
    poly_ei = (np.array([p_src, p_dst], dtype=np.int64)
               if p_src else np.zeros((2, 0), dtype=np.int64))
    poly_ea = (np.array(p_ea, dtype=np.float32)
               if p_ea else np.zeros((0, 1), dtype=np.float32))

    return {
        "atom_z"     : atom_z,
        "atom_ei"    : atom_ei,
        "atom_dist"  : atom_dist,
        "line_ei"    : line_ei,
        "line_angles": line_angles,
        "poly_x"     : poly_x,
        "poly_ei"    : poly_ei,
        "poly_ea"    : poly_ea,
        "n_atoms"    : N,
        "n_edges"    : E,
    }


# ============================================================
# 7.  JARVIS PROPERTY EXTRACTOR
#     Returns (ef, aux_vec) where aux_vec is float32 (N_AUX,)
#     Missing / "na" / NaN values → 0.0  (masked in loss)
# ============================================================
_SENTINEL = 0.0   # fill value for missing properties

def _safe_float(val, default=_SENTINEL) -> float:
    """Convert a JARVIS property value safely; return default if unavailable."""
    if val is None:
        return default
    try:
        v = float(val)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except (TypeError, ValueError):
        return default


def extract_targets(entry: dict) -> tuple:
    """
    Returns:
        ef       float          formation energy per atom (eV/atom)
        aux_vals np.ndarray     (N_AUX,) auxiliary targets
        aux_mask np.ndarray     (N_AUX,) bool — True where value is valid
    """
    ef = _safe_float(entry.get("formation_energy_peratom",
                    entry.get("formation_energy_per_atom", None)))

    aux_vals = np.zeros(N_AUX, dtype=np.float32)
    aux_mask = np.zeros(N_AUX, dtype=np.float32)   # 1.0 = valid, 0.0 = missing

    for idx, (key, name, unit, act, pos) in enumerate(JARVIS_PROPS):
        raw = entry.get(key, None)
        # JARVIS stores missing as "na" (string) or None
        if raw == "na" or raw is None:
            continue
        v = _safe_float(raw)
        if v == _SENTINEL and raw not in (0, 0.0):
            continue          # treat conversion failure as missing
        aux_vals[idx] = v
        aux_mask[idx] = 1.0

    return ef, aux_vals, aux_mask


# ============================================================
# 8.  PyG DATA OBJECT
# ============================================================
def graphs_to_pyg(graph: dict, ef: float,
                  aux_vals: np.ndarray, aux_mask: np.ndarray,
                  jid: str = "") -> Data:
    data = Data(
        x          = torch.tensor(graph["atom_z"],    dtype=torch.long),
        edge_index = torch.tensor(graph["atom_ei"],   dtype=torch.long),
        edge_dist  = torch.tensor(graph["atom_dist"], dtype=torch.float32),
        y_ef       = torch.tensor([ef],              dtype=torch.float32),
        y_aux      = torch.tensor(aux_vals,          dtype=torch.float32),   # (N_AUX,)
        y_mask     = torch.tensor(aux_mask,          dtype=torch.float32),   # (N_AUX,)
        num_nodes  = graph["n_atoms"],
    )
    data._poly_x_np      = graph["poly_x"]
    data._poly_ei_np     = graph["poly_ei"]
    data._poly_ea_np     = graph["poly_ea"]
    data._line_ei_np     = graph["line_ei"]
    data._line_angles_np = graph["line_angles"]
    data._n_edges        = graph["n_edges"]
    data.jid             = jid
    return data


# ============================================================
# 9.  SPLIT LOADER  (JARVIS version)
# ============================================================
def load_split(split_data: list, label: str = "") -> tuple:
    structs, ef_arr, aux_arr, mask_arr, jids = [], [], [], [], []
    skipped = 0
    for entry in split_data:
        jid = entry.get("jid", "unknown")
        try:
            s         = jarvis_to_structure(entry)
            ef, av, am = extract_targets(entry)
            if math.isnan(ef) or math.isinf(ef):
                raise ValueError("invalid Ef")
            structs.append(s)
            ef_arr.append(ef)
            aux_arr.append(av)
            mask_arr.append(am)
            jids.append(jid)
        except Exception as e:
            skipped += 1
            if skipped <= 10:
                print(f"  [{label}] skip {jid}: {e}")
    print(f"  [{label}] {len(structs):,} / {len(split_data):,}  skipped={skipped}")
    return (structs,
            np.array(ef_arr,   dtype=np.float32),
            np.array(aux_arr,  dtype=np.float32),   # (N, N_AUX)
            np.array(mask_arr, dtype=np.float32),   # (N, N_AUX)
            jids)


# ============================================================
# 10.  DATASET BUILDER
# ============================================================
def build_dataset(structs, ef, aux, mask, jids, label="") -> list:
    ds, skip = [], 0
    n = len(structs)
    print(f"\n  [{label}] Building dual+line graphs for {n:,} structures ...")
    t0 = time.time()
    for i in range(n):
        try:
            g    = build_dual_graphs(structs[i])
            data = graphs_to_pyg(g, float(ef[i]),
                                  aux[i], mask[i], jids[i])
            ds.append(data)
        except Exception:
            skip += 1
        if (i + 1) % 5000 == 0 or (i + 1) == n:
            print(f"    ... {i+1:,}/{n:,}  ({time.time()-t0:.0f}s)")
    print(f"  [{label}] Built {len(ds):,}  skipped {skip}")
    return ds


# ============================================================
# 11.  MESSAGE PASSING LAYERS
# ============================================================
class CPGNConv(MessagePassing):
    def __init__(self, node_dim, edge_dim, hidden):
        super().__init__(aggr="mean")
        self.msg_mlp = nn.Sequential(
            nn.Linear(node_dim*2 + edge_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, node_dim),
        )
        self.norm = nn.LayerNorm(node_dim)
    def forward(self, x, edge_index, edge_attr):
        if edge_index.shape[1] == 0: return x
        return self.norm(x + self.propagate(edge_index, x=x, edge_attr=edge_attr))
    def message(self, x_i, x_j, edge_attr):
        return self.msg_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))


class LineGraphConv(MessagePassing):
    """Angle-aware bond update — mirrors ALIGNN's eConv on line graph."""
    def __init__(self, edge_dim, angle_dim, hidden):
        super().__init__(aggr="mean")
        self.msg_mlp = nn.Sequential(
            nn.Linear(edge_dim*2 + angle_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, edge_dim),
        )
        self.norm = nn.LayerNorm(edge_dim)
    def forward(self, x, edge_index, edge_attr):
        if edge_index.shape[1] == 0: return x
        return self.norm(x + self.propagate(edge_index, x=x, edge_attr=edge_attr))
    def message(self, x_i, x_j, edge_attr):
        return self.msg_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))


# ============================================================
# 12.  CROSS-ATTENTION  (atom ↔ polyhedron)
# ============================================================
class CrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q_a = nn.Linear(dim, dim); self.k_p = nn.Linear(dim, dim)
        self.v_p = nn.Linear(dim, dim); self.q_p = nn.Linear(dim, dim)
        self.k_a = nn.Linear(dim, dim); self.v_a = nn.Linear(dim, dim)
        self.norm_a = nn.LayerNorm(dim); self.norm_p = nn.LayerNorm(dim)
        self.scale  = dim ** -0.5

    def _pool(self, h, batch, B, dev):
        ctx   = torch.zeros(B, h.shape[1], device=dev, dtype=h.dtype)
        count = torch.zeros(B, 1,          device=dev, dtype=h.dtype)
        ctx.scatter_add_(0, batch.long().unsqueeze(1).expand_as(h), h)
        count.scatter_add_(0, batch.long().unsqueeze(1),
                           torch.ones(len(batch), 1, device=dev, dtype=h.dtype))
        return ctx / (count + 1e-8)

    def forward(self, ah, ph, a_batch, p_batch):
        B   = int(a_batch.max().item()) + 1; dev = ah.device
        pc  = self._pool(ph, p_batch, B, dev)
        cpa = pc[a_batch.long()]
        w_a = torch.sigmoid((self.q_a(ah) * self.k_p(cpa)).sum(-1, keepdim=True) * self.scale)
        ah_new = self.norm_a(ah + w_a * self.v_p(cpa))
        ac  = self._pool(ah, a_batch, B, dev)
        cpp = ac[p_batch.long()]
        w_p = torch.sigmoid((self.q_p(ph) * self.k_a(cpp)).sum(-1, keepdim=True) * self.scale)
        ph_new = self.norm_p(ph + w_p * self.v_a(cpp))
        return ah_new, ph_new


# ============================================================
# 13.  CPGN MODEL  (JARVIS-DFT multi-property)
# ============================================================
class CPGN(nn.Module):
    """
    Multi-property CPGN for JARVIS-DFT.
    One primary head (Ef) + N_AUX auxiliary heads, one per JARVIS property.
    Each auxiliary head applies the appropriate activation (softplus / none).
    """
    def __init__(self):
        super().__init__()
        H = HIDDEN_DIM

        self.elem_embed  = nn.Embedding(N_ELEM, ELEM_DIM, padding_idx=0)
        self.dist_rbf    = RBFExpansion(0.0, CUTOFF,   RBF_CENTRES)
        self.angle_rbf   = RBFExpansion(0.0, math.pi, RBF_CENTRES)

        self.atom_in         = nn.Sequential(nn.Linear(ELEM_DIM, H), nn.SiLU(), nn.Linear(H, H))
        self.edge_in         = nn.Sequential(nn.Linear(N_EDGE_FEAT, H), nn.SiLU())
        self.poly_in         = nn.Sequential(nn.Linear(N_POLY_FEAT, H), nn.SiLU(), nn.Linear(H, H))
        self.poly_edge_proj  = nn.Linear(1, H)

        self.atom_convs  = nn.ModuleList([CPGNConv(H, H, H)               for _ in range(N_LAYERS)])
        self.line_convs  = nn.ModuleList([LineGraphConv(H, N_ANGLE_FEAT, H) for _ in range(N_LAYERS)])
        self.poly_convs  = nn.ModuleList([CPGNConv(H, H, H)               for _ in range(N_LAYERS)])
        self.cross_attns = nn.ModuleList([CrossAttention(H)                for _ in range(N_LAYERS)])
        self.dropout     = nn.Dropout(p=0.1)

        self.fusion = nn.Sequential(
            nn.Linear(H*2, H), nn.SiLU(),
            nn.Linear(H, PRED_DIM), nn.SiLU(),
        )

        # ── Primary head: Ef ────────────────────────────────────────────────
        self.head_ef = nn.Linear(PRED_DIM, 1)

        # ── Auxiliary heads: one per JARVIS property ────────────────────────
        # Store activation type per head so forward() can apply it
        self.aux_heads = nn.ModuleList([nn.Linear(PRED_DIM, 1) for _ in JARVIS_PROPS])
        self._aux_activations = [act for _, _, _, act, _ in JARVIS_PROPS]

    def forward(self, batch):
        dev     = batch.x.device
        atom_h  = self.atom_in(self.elem_embed(batch.x.to(dev)))
        dist_rbf = self.dist_rbf(batch.edge_dist.to(dev))
        bond_h   = self.edge_in(dist_rbf)
        poly_h   = self.poly_in(batch.poly_x.to(dev))
        p_ei     = batch.poly_ei.to(dev)
        p_ea     = batch.poly_ea.to(dev)
        p_ea_p   = (self.poly_edge_proj(p_ea)
                    if p_ea.shape[0] > 0
                    else torch.zeros(0, HIDDEN_DIM, device=dev, dtype=atom_h.dtype))
        line_ei    = batch.line_ei.to(dev)
        angle_rbf  = self.angle_rbf(batch.line_angles.to(dev))
        a_batch    = batch.batch

        for ac, lc, pc, ca in zip(self.atom_convs, self.line_convs,
                                   self.poly_convs, self.cross_attns):
            atom_h  = self.dropout(ac(atom_h, batch.edge_index, bond_h))
            bond_h  = self.dropout(lc(bond_h, line_ei, angle_rbf))
            poly_h  = self.dropout(pc(poly_h, p_ei, p_ea_p))
            atom_h, poly_h = ca(atom_h, poly_h, a_batch, a_batch)

        ha = global_mean_pool(atom_h, a_batch)
        hp = global_mean_pool(poly_h, a_batch)
        z  = self.fusion(torch.cat([ha, hp], dim=-1))

        # Primary
        ef_pred = self.head_ef(z).squeeze(-1)

        # Auxiliary — shape (B, N_AUX)
        aux_preds = []
        for head, act in zip(self.aux_heads, self._aux_activations):
            p = head(z).squeeze(-1)
            if act == "softplus":
                p = F.softplus(p)
            elif act == "sigmoid":
                p = torch.sigmoid(p)
            aux_preds.append(p)
        aux_pred = torch.stack(aux_preds, dim=-1)   # (B, N_AUX)

        return {"ef": ef_pred, "aux": aux_pred}


# ============================================================
# 14.  LOSS
#      Primary: MAE on Ef
#      Auxiliary: masked MAE on each available JARVIS property
#      Missing values (mask=0) contribute zero gradient
# ============================================================
class CPGNLoss(nn.Module):
    def forward(self, preds, batch):
        # Primary Ef MAE
        loss_ef = F.l1_loss(preds["ef"], batch.y_ef)

        # Masked auxiliary MAE
        # y_aux  : (B, N_AUX)   true values
        # y_mask : (B, N_AUX)   1.0 if valid, 0.0 if missing
        diff     = (preds["aux"] - batch.y_aux).abs()   # (B, N_AUX)
        masked   = diff * batch.y_mask                   # zero-out missing
        n_valid  = batch.y_mask.sum().clamp(min=1)
        loss_aux = masked.sum() / n_valid

        return loss_ef + LAMBDA_AUX * loss_aux


# ============================================================
# 15.  COLLATE / DATALOADER
# ============================================================
def custom_collate(data_list):
    batch = PyGBatch.from_data_list(data_list)
    poly_x_list, poly_ei_list, poly_ea_list = [], [], []
    line_ei_list, line_ang_list = [], []
    atom_cum = edge_cum = 0

    for d in data_list:
        n_a = d.num_nodes
        n_e = d._n_edges
        poly_x_list.append(torch.tensor(d._poly_x_np,  dtype=torch.float32))
        poly_ea_list.append(torch.tensor(d._poly_ea_np, dtype=torch.float32))
        ei_p = d._poly_ei_np
        if ei_p.shape[1] > 0:
            poly_ei_list.append(torch.tensor(ei_p + atom_cum, dtype=torch.long))
        lei = d._line_ei_np
        if lei.shape[1] > 0:
            line_ei_list.append(torch.tensor(lei + edge_cum, dtype=torch.long))
        line_ang_list.append(torch.tensor(d._line_angles_np, dtype=torch.float32))
        atom_cum += n_a
        edge_cum += n_e

    batch.poly_x      = torch.cat(poly_x_list, dim=0)
    batch.poly_ei     = (torch.cat(poly_ei_list, dim=1)
                         if poly_ei_list else torch.zeros(2, 0, dtype=torch.long))
    batch.poly_ea     = torch.cat(poly_ea_list, dim=0)
    batch.line_ei     = (torch.cat(line_ei_list, dim=1)
                         if line_ei_list else torch.zeros(2, 0, dtype=torch.long))
    batch.line_angles = torch.cat(line_ang_list, dim=0)
    batch.y_ef        = torch.cat([d.y_ef   for d in data_list])
    batch.y_aux       = torch.stack([d.y_aux  for d in data_list])   # (B, N_AUX)
    batch.y_mask      = torch.stack([d.y_mask for d in data_list])   # (B, N_AUX)
    batch.edge_dist   = torch.cat([d.edge_dist for d in data_list], dim=0)
    return batch


def make_loader(data_list, batch_size, shuffle):
    return TorchLoader(
        data_list, batch_size=batch_size, shuffle=shuffle,
        collate_fn=custom_collate, drop_last=False,
        num_workers=4 if DEVICE.startswith("cuda") else 0,
        pin_memory=DEVICE.startswith("cuda"),
        persistent_workers=DEVICE.startswith("cuda"),
    )


# ============================================================
# 16.  TRAINING LOOP
# ============================================================
def train_model(train_ds, val_ds):
    model     = CPGN().to(DEVICE)
    criterion = CPGNLoss()
    optimiser = AdamW(model.parameters(), lr=LEARNING_RATE,
                      weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimiser, T_max=EPOCHS, eta_min=1e-6)

    tr_loader = make_loader(train_ds, BATCH_SIZE, shuffle=True)
    vl_loader = make_loader(val_ds,   BATCH_SIZE, shuffle=False)

    best_val_mae = float("inf")
    best_state   = None
    wait         = 0
    tr_mae_hist, vl_mae_hist = [], []
    tr_loss_hist, vl_loss_hist = [], []

    n_p = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*70}")
    print("  CPGN JARVIS-DFT — Multi-property training")
    print(f"  Parameters  : {n_p:,}")
    print(f"  Train/Val   : {len(train_ds):,} / {len(val_ds):,}")
    print(f"  Aux targets : {N_AUX}  (masked MAE, λ={LAMBDA_AUX})")
    print(f"  Epochs      : {EPOCHS}  Batch : {BATCH_SIZE}  LR : {LEARNING_RATE}")
    print(f"  Checkpoint  : best val Ef MAE → {CHECKPOINT}")
    print(f"{'='*70}")
    print(f"  {'Ep':>5} | {'TrLoss':>8} | {'VlLoss':>8} | "
          f"{'TrMAE_Ef':>9} | {'VlMAE_Ef':>9} | {'LR':>9} | {'Time':>6}")
    print(f"  {'-'*65}")

    t0_total = time.time()
    for ep in range(1, EPOCHS + 1):
        model.train()
        tl = tm = n = 0; t0 = time.time()
        for b in tr_loader:
            b = b.to(DEVICE)
            optimiser.zero_grad(set_to_none=True)
            p  = model(b)
            ls = criterion(p, b)
            if not torch.isfinite(ls): continue
            ls.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            bs  = len(b.y_ef)
            tl += ls.item() * bs
            tm += F.l1_loss(p["ef"].detach(), b.y_ef).item() * bs
            n  += bs
        n = max(n, 1); tl /= n; tm /= n
        scheduler.step()

        model.eval()
        vl = vm = nv = 0
        with torch.no_grad():
            for b in vl_loader:
                b = b.to(DEVICE); p = model(b); bs = len(b.y_ef)
                vl += criterion(p, b).item() * bs
                vm += F.l1_loss(p["ef"], b.y_ef).item() * bs
                nv += bs
        nv = max(nv, 1); vl /= nv; vm /= nv

        tr_loss_hist.append(tl); vl_loss_hist.append(vl)
        tr_mae_hist.append(tm);  vl_mae_hist.append(vm)

        lr_now = optimiser.param_groups[0]["lr"]
        if ep % 10 == 0 or ep <= 5:
            print(f"  {ep:>5} | {tl:>8.4f} | {vl:>8.4f} | "
                  f"{tm:>9.4f} | {vm:>9.4f} | {lr_now:>9.2e} | "
                  f"{time.time()-t0:>5.1f}s")

        if vm < best_val_mae - 1e-5:
            best_val_mae = vm
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            torch.save({
                "model_state": best_state, "epoch": ep,
                "val_mae_ef" : best_val_mae,
                "config"     : {"HIDDEN_DIM": HIDDEN_DIM,
                                "N_LAYERS"  : N_LAYERS,
                                "PRED_DIM"  : PRED_DIM,
                                "ELEM_DIM"  : ELEM_DIM},
            }, CHECKPOINT)
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"  Early stopping ep {ep}  (best val Ef MAE={best_val_mae:.4f})")
                break

    print(f"\n  Total : {time.time()-t0_total:.1f}s  |  "
          f"Best val Ef MAE = {best_val_mae:.4f} eV/atom")
    if best_state:
        model.load_state_dict(best_state)
    return model, tr_loss_hist, vl_loss_hist, tr_mae_hist, vl_mae_hist


# ============================================================
# 17.  LOAD CHECKPOINT
# ============================================================
def load_checkpoint(path):
    ckpt = torch.load(path, map_location=DEVICE)
    cfg  = ckpt.get("config", {})
    global HIDDEN_DIM, N_LAYERS, PRED_DIM, ELEM_DIM
    HIDDEN_DIM = cfg.get("HIDDEN_DIM", HIDDEN_DIM)
    N_LAYERS   = cfg.get("N_LAYERS",   N_LAYERS)
    PRED_DIM   = cfg.get("PRED_DIM",   PRED_DIM)
    ELEM_DIM   = cfg.get("ELEM_DIM",   ELEM_DIM)
    model = CPGN().to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  Checkpoint : {path}")
    print(f"  Epoch      : {ckpt.get('epoch','?')}  "
          f"Val Ef MAE = {ckpt.get('val_mae_ef','?'):.4f}")
    return model


# ============================================================
# 18.  INFERENCE
# ============================================================
@torch.no_grad()
def predict_set(model, dataset, label=""):
    model.eval()
    loader = make_loader(dataset, batch_size=64, shuffle=False)
    ef_true, ef_pred = [], []
    aux_true_list, aux_pred_list, mask_list = [], [], []

    for b in loader:
        b = b.to(DEVICE)
        out = model(b)
        ef_true.extend(b.y_ef.cpu().tolist())
        ef_pred.extend(out["ef"].detach().cpu().tolist())
        aux_true_list.append(b.y_aux.cpu().numpy())
        aux_pred_list.append(out["aux"].detach().cpu().numpy())
        mask_list.append(b.y_mask.cpu().numpy())

    print(f"  [{label}] {len(ef_true):,} predictions")
    return {
        "ef_true"  : np.array(ef_true),
        "ef_pred"  : np.array(ef_pred),
        "aux_true" : np.vstack(aux_true_list),    # (N, N_AUX)
        "aux_pred" : np.vstack(aux_pred_list),    # (N, N_AUX)
        "aux_mask" : np.vstack(mask_list),        # (N, N_AUX)
    }


# ============================================================
# 19.  METRICS
# ============================================================
def compute_metrics(res, label=""):
    ef_mae  = mean_absolute_error(res["ef_true"], res["ef_pred"])
    ef_rmse = float(np.sqrt(mean_squared_error(res["ef_true"], res["ef_pred"])))
    ef_mad  = float(np.mean(np.abs(res["ef_true"] - res["ef_true"].mean())))
    ef_mad_mae = ef_mad / ef_mae if ef_mae > 0 else float("inf")

    hdr = "=" * 70
    print(f"\n{hdr}")
    print(f"  CPGN JARVIS-DFT — {label} Results")
    print(hdr)
    print(f"  Samples : {len(res['ef_true']):,}")
    print(f"\n  ── Formation Energy Ef (eV/atom) ──")
    print(f"  MAE      : {ef_mae:.4f}    (ALIGNN: 0.022)")
    print(f"  RMSE     : {ef_rmse:.4f}")
    print(f"  MAD      : {ef_mad:.4f}")
    print(f"  MAD:MAE  : {ef_mad_mae:.2f}  (ALIGNN: 39.1)")

    # Per-property auxiliary metrics (only where mask=1)
    aux_maes = {}
    print(f"\n  ── Auxiliary Properties ──")
    print(f"  {'Property':<10} {'Name':<10} {'Unit':<8} "
          f"{'N_valid':>8} {'MAE':>10} {'ALIGNN MAE':>12}")
    print(f"  {'-'*62}")
    for idx, (key, name, unit, act, pos) in enumerate(JARVIS_PROPS):
        valid_mask = res["aux_mask"][:, idx].astype(bool)
        n_valid    = valid_mask.sum()
        if n_valid < 5:
            aux_maes[name] = float("nan")
            print(f"  {key[:10]:<10} {name:<10} {unit:<8} {n_valid:>8} {'—':>10}")
            continue
        mae = mean_absolute_error(res["aux_true"][valid_mask, idx],
                                  res["aux_pred"][valid_mask, idx])
        aux_maes[name] = mae
        alignn_ref = f"{ALIGNN_BENCH[name]:.3f}" if name in ALIGNN_BENCH else "—"
        print(f"  {key[:10]:<10} {name:<10} {unit:<8} "
              f"{n_valid:>8} {mae:>10.4f} {alignn_ref:>12}")
    print(hdr)

    return {
        "ef_mae": ef_mae, "ef_rmse": ef_rmse,
        "ef_mad": ef_mad, "ef_mad_mae": ef_mad_mae,
        "aux_maes": aux_maes,
    }


# ============================================================
# 20.  SAVE CSV
# ============================================================
def save_csv(res, dataset, path):
    rows = []
    prop_names = [name for _, name, _, _, _ in JARVIS_PROPS]
    for i in range(len(res["ef_true"])):
        jid = dataset[i].jid if hasattr(dataset[i], "jid") else str(i)
        row = {
            "jid"       : jid,
            "true_ef"   : res["ef_true"][i],
            "pred_ef"   : res["ef_pred"][i],
            "err_ef"    : res["ef_pred"][i] - res["ef_true"][i],
        }
        for j, name in enumerate(prop_names):
            row[f"true_{name}"] = res["aux_true"][i, j] if res["aux_mask"][i, j] else float("nan")
            row[f"pred_{name}"] = res["aux_pred"][i, j]
            row[f"mask_{name}"] = int(res["aux_mask"][i, j])
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  Saved: {path}")


# ============================================================
# 21.  PLOTS
# ============================================================
BLUE   = "#2A7EC0"; CORAL  = "#E05C2A"; GREEN  = "#1D9E75"
PURPLE = "#534AB7"; DARK   = "#2C2C2A"; GRAY   = "#888780"
AMBER  = "#BA7517"

def plot_training_curve(tr_loss, vl_loss, tr_mae, vl_mae):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ep = range(1, len(tr_loss) + 1)
    axes[0].plot(ep, tr_loss, lw=1.5, color=BLUE,  label="Train loss")
    axes[0].plot(ep, vl_loss, lw=1.5, color=CORAL, ls="--", label="Val loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("CPGN JARVIS — Total Loss  (MAE Ef + masked aux)"); axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(ep, tr_mae, lw=1.5, color=BLUE,  label="Train MAE Ef")
    axes[1].plot(ep, vl_mae, lw=1.5, color=CORAL, ls="--", label="Val MAE Ef")
    for v, lbl, col in [(0.063,"CGCNN","gray"),(0.030,"MEGNet","purple"),(0.022,"ALIGNN","green")]:
        axes[1].axhline(v, color=col, lw=0.8, ls=":", alpha=0.7, label=lbl)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Ef MAE (eV/atom)")
    axes[1].set_title("CPGN JARVIS — Formation Energy MAE")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

    plt.suptitle(f"CPGN Training — JARVIS-DFT  ({N_TRAIN:,}/{N_VAL:,}/{N_TEST:,} shuffled)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("CPGN_JARVIS_training_curve.png", dpi=300); plt.close()
    print("  Saved: CPGN_JARVIS_training_curve.png")


def plot_parity(val_res, test_res, val_m, test_m):
    """Parity plots for Ef (primary) and Eg_opt (most-compared auxiliary)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    eg_idx = next((i for i, (k,*_) in enumerate(JARVIS_PROPS)
                   if k == "optb88vdw_bandgap"), None)

    for col, (res, m, lbl) in enumerate(
            [(val_res, val_m, f"Val ({N_VAL:,})"),
             (test_res, test_m, f"Test ({N_TEST:,})")]):

        # Ef parity
        ax = axes[0][col]
        yt, yp = res["ef_true"], res["ef_pred"]
        ax.scatter(yt, yp, alpha=0.35, s=5, color=BLUE if col == 0 else CORAL,
                   edgecolors="none", zorder=3)
        lo = min(yt.min(), yp.min()); hi = max(yt.max(), yp.max())
        mg = 0.05*(hi-lo)
        ax.plot([lo-mg, hi+mg],[lo-mg, hi+mg],"k--",lw=1.5)
        ax.set_xlim(lo-mg, hi+mg); ax.set_ylim(lo-mg, hi+mg)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("True Ef (eV/atom)"); ax.set_ylabel("Pred Ef (eV/atom)")
        ax.set_title(f"Ef — {lbl}\nMAE={m['ef_mae']:.4f}  MAD:MAE={m['ef_mad_mae']:.1f}")
        ax.grid(alpha=0.3)

        # Eg_opt parity (where available)
        ax = axes[1][col]
        if eg_idx is not None:
            msk  = res["aux_mask"][:, eg_idx].astype(bool)
            yt_g = res["aux_true"][msk, eg_idx]
            yp_g = res["aux_pred"][msk, eg_idx]
            mae_g = m["aux_maes"].get("Eg_opt", float("nan"))
            ax.scatter(yt_g, yp_g, alpha=0.35, s=5,
                       color=GREEN if col == 0 else AMBER,
                       edgecolors="none", zorder=3)
            lo_g = min(yt_g.min(), yp_g.min()); hi_g = max(yt_g.max(), yp_g.max())
            mg_g = 0.05*(hi_g-lo_g)
            ax.plot([lo_g-mg_g, hi_g+mg_g],[lo_g-mg_g, hi_g+mg_g],"k--",lw=1.5)
            ax.set_xlim(lo_g-mg_g, hi_g+mg_g); ax.set_ylim(lo_g-mg_g, hi_g+mg_g)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"Eg_opt — {lbl}\nMAE={mae_g:.4f}  (ALIGNN=0.142 eV)")
        ax.set_xlabel("True Eg_opt (eV)"); ax.set_ylabel("Pred Eg_opt (eV)")
        ax.grid(alpha=0.3)

    plt.suptitle("CPGN JARVIS-DFT — Parity Plots", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("CPGN_JARVIS_parity_plots.png", dpi=300); plt.close()
    print("  Saved: CPGN_JARVIS_parity_plots.png")


def plot_property_maes(test_m):
    """Horizontal bar chart of MAE for every JARVIS property."""
    names  = ["Ef"] + [name for _, name, _, _, _ in JARVIS_PROPS]
    ef_mae = test_m["ef_mae"]
    maes   = [ef_mae] + [test_m["aux_maes"].get(n, float("nan")) for n in names[1:]]
    units  = ["eV/at"] + [unit for _, _, unit, _, _ in JARVIS_PROPS]

    valid_idx = [i for i, v in enumerate(maes) if not math.isnan(v)]
    names_v   = [f"{names[i]} ({units[i]})" for i in valid_idx]
    maes_v    = [maes[i] for i in valid_idx]
    colors    = [CORAL if names[i] == "Ef" else BLUE for i in valid_idx]

    fig, ax = plt.subplots(figsize=(10, max(6, len(valid_idx) * 0.45)))
    bars = ax.barh(names_v, maes_v, color=colors, alpha=0.85, edgecolor="k", lw=0.4)
    for bar, v in zip(bars, maes_v):
        ax.text(v * 1.01, bar.get_y() + bar.get_height()/2,
                f"{v:.4g}", va="center", fontsize=8)
    ax.set_xlabel("MAE (property units)", fontsize=11)
    ax.set_title("CPGN JARVIS-DFT — Test MAE per Property", fontsize=12)
    ax.invert_yaxis(); ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig("CPGN_JARVIS_property_maes.png", dpi=300); plt.close()
    print("  Saved: CPGN_JARVIS_property_maes.png")


def plot_error_hist(val_res, test_res):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, res, label, color in [
        (axes[0], val_res,  f"Val ({N_VAL:,})",   BLUE),
        (axes[1], test_res, f"Test ({N_TEST:,})",  CORAL),
    ]:
        err = res["ef_pred"] - res["ef_true"]
        ax.hist(err, bins=60, color=color, alpha=0.75, edgecolor="k", lw=0.3)
        ax.axvline(0, color="r", lw=1.5, ls="--")
        ax.axvline(err.mean(), color=DARK, lw=1.2, ls=":",
                   label=f"Mean={err.mean():.4f}")
        ax.set_xlabel("Ef Error (eV/atom)"); ax.set_ylabel("Count")
        ax.set_title(f"Ef Error Distribution — {label}"); ax.legend(); ax.grid(alpha=0.3)
    plt.suptitle("CPGN JARVIS-DFT — Formation Energy Error Distributions",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("CPGN_JARVIS_error_hist.png", dpi=300); plt.close()
    print("  Saved: CPGN_JARVIS_error_hist.png")


def plot_benchmark_table(test_m):
    """Bar chart comparing CPGN vs ALIGNN / CGCNN / MEGNet on Ef and Eg_opt."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    benchmarks_ef  = {"CFID":0.104, "CGCNN":0.063, "MEGNet":0.030, "ALIGNN":0.022}
    benchmarks_eg  = {"CGCNN":0.200, "MEGNet":0.330, "ALIGNN":0.142}

    def _bar(ax, bench_dict, cpgn_val, xlabel, title):
        models = list(bench_dict.keys()) + ["CPGN (ours)"]
        vals   = list(bench_dict.values()) + [cpgn_val]
        cols   = [GRAY]*len(bench_dict) + [CORAL]
        bars   = ax.barh(models, vals, color=cols, alpha=0.88, edgecolor="k", lw=0.5)
        for bar, v in zip(bars, vals):
            ax.text(v + max(vals)*0.01, bar.get_y() + bar.get_height()/2,
                    f"{v:.4f}", va="center", fontsize=9)
        ax.set_xlabel(xlabel, fontsize=11); ax.set_title(title, fontsize=10)
        ax.invert_yaxis(); ax.grid(axis="x", alpha=0.3)

    _bar(axes[0], benchmarks_ef,
         test_m["ef_mae"],
         "MAE (eV/atom)",
         "Formation Energy Ef MAE — JARVIS-DFT")

    eg_mae = test_m["aux_maes"].get("Eg_opt", float("nan"))
    _bar(axes[1], benchmarks_eg,
         eg_mae if not math.isnan(eg_mae) else 0.0,
         "MAE (eV)",
         "OPT Band Gap Eg_opt MAE — JARVIS-DFT")

    plt.suptitle("CPGN vs State-of-the-Art — JARVIS-DFT Benchmark",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("CPGN_JARVIS_benchmark_table.png", dpi=300); plt.close()
    print("  Saved: CPGN_JARVIS_benchmark_table.png")


# ============================================================
# 22.  FINAL SUMMARY TABLE
# ============================================================
def print_final_summary(val_m, test_m):
    print("\n" + "=" * 70)
    print("  CPGN JARVIS-DFT — Final Benchmark Summary")
    print("  (mirrors Table 1, ALIGNN paper, npj Comput. Mater. 2021)")
    print("=" * 70)
    print(f"  {'Property':<12} {'Unit':<8} {'CPGN MAE':>10} {'ALIGNN':>10} {'CGCNN':>10}")
    print(f"  {'-'*54}")

    # Ef row
    print(f"  {'Ef':<12} {'eV/at':<8} {test_m['ef_mae']:>10.4f} "
          f"{'0.022':>10} {'0.063':>10}")

    prop_ref = {
        "Eg_opt" : (0.142, 0.200),
        "Eg_mbj" : (0.210, "—"),
        "SLME"   : (1.42,  "—"),
        "Kv"     : (14.41, "—"),
        "Gv"     : (10.65, "—"),
    }
    units_map = {k: u for _, k, u, _, _ in JARVIS_PROPS}
    for name, (alignn, cgcnn) in prop_ref.items():
        mae  = test_m["aux_maes"].get(name, float("nan"))
        unit = units_map.get(name, "—")
        mae_s = f"{mae:.4f}" if not math.isnan(mae) else "—"
        aln_s = f"{alignn}" if isinstance(alignn, float) else alignn
        cgn_s = f"{cgcnn}"  if isinstance(cgcnn,  float) else cgcnn
        print(f"  {name:<12} {unit:<8} {mae_s:>10} {aln_s:>10} {cgn_s:>10}")

    print("=" * 70)
    print(f"\n  Split   : Train={N_TRAIN:,} | Val={N_VAL:,} | Test={N_TEST:,}  (SEED={SEED})")
    print(f"  Val  Ef MAE  : {val_m['ef_mae']:.4f} eV/atom")
    print(f"  Test Ef MAE  : {test_m['ef_mae']:.4f} eV/atom")
    print(f"  Test MAD:MAE : {test_m['ef_mad_mae']:.2f}  (ALIGNN: 39.1)")
    print("=" * 70)


# ============================================================
# 23.  MAIN
# ============================================================
def main():
    print("=" * 70)
    print("  CPGN — JARVIS-DFT Multi-Property  (19 auxiliary targets)")
    print("  Fixes: MAE loss | shuffled split | line graph | learned embed")
    print("         500 epochs cosine LR | masked auxiliary loss")
    print("=" * 70)

    # ── Load JARVIS-DFT data via jarvis-tools ─────────────────────────────────
    print("\nLoading JARVIS-DFT 3D dataset via jarvis-tools ...")
    print("  (Downloads ~500 MB from Figshare on first run — cached afterwards)")
    try:
        from jarvis.db.figshare import data as jdata
        dft_3d = jdata("dft_3d")
    except ImportError:
        print("[ERROR] jarvis-tools not installed.  Run:")
        print("        pip install jarvis-tools")
        sys.exit(1)

    print(f"  Total entries : {len(dft_3d):,}")

    # ── Shuffled split ────────────────────────────────────────────────────────
    indices = list(range(len(dft_3d)))
    random.Random(SEED).shuffle(indices)
    n_total = len(dft_3d)
    n_tr    = min(N_TRAIN, n_total - N_VAL - N_TEST)
    train_idx = indices[:n_tr]
    val_idx   = indices[n_tr : n_tr + N_VAL]
    test_idx  = indices[n_tr + N_VAL : n_tr + N_VAL + N_TEST]

    train_data = [dft_3d[i] for i in train_idx]
    val_data   = [dft_3d[i] for i in val_idx]
    test_data  = [dft_3d[i] for i in test_idx]
    del dft_3d

    print(f"  Train : {len(train_data):,}  Val : {len(val_data):,}  "
          f"Test : {len(test_data):,}  (shuffled SEED={SEED})")

    # ── Parse ─────────────────────────────────────────────────────────────────
    print("\nParsing structures ...")
    tr_s, tr_ef, tr_aux, tr_mask, tr_ids = load_split(train_data, "TRAIN")
    vl_s, vl_ef, vl_aux, vl_mask, vl_ids = load_split(val_data,   "VAL")
    te_s, te_ef, te_aux, te_mask, te_ids = load_split(test_data,  "TEST")
    del train_data, val_data, test_data

    # Property availability diagnostics
    print("\n  Auxiliary property availability (train set):")
    for idx, (key, name, unit, act, pos) in enumerate(JARVIS_PROPS):
        n_val = int(tr_mask[:, idx].sum())
        pct   = 100 * n_val / max(len(tr_mask), 1)
        print(f"    {name:<10} {n_val:>6,}/{len(tr_mask):,}  ({pct:5.1f}%)  [{key}]")

    # ── Build / load graph cache ──────────────────────────────────────────────
    if os.path.isfile(GRAPH_CACHE):
        print(f"\n  Graph cache found — loading {GRAPH_CACHE} ...")
        with open(GRAPH_CACHE, "rb") as f:
            cache = pickle.load(f)
        train_ds = cache["train"]
        val_ds   = cache["val"]
        test_ds  = cache["test"]
        print(f"  Loaded  train={len(train_ds):,}  "
              f"val={len(val_ds):,}  test={len(test_ds):,}")
    else:
        train_ds = build_dataset(tr_s, tr_ef, tr_aux, tr_mask, tr_ids, "TRAIN")
        val_ds   = build_dataset(vl_s, vl_ef, vl_aux, vl_mask, vl_ids, "VAL")
        test_ds  = build_dataset(te_s, te_ef, te_aux, te_mask, te_ids, "TEST")
        print(f"\n  Saving graph cache → {GRAPH_CACHE} ...")
        with open(GRAPH_CACHE, "wb") as f:
            pickle.dump({"train": train_ds, "val": val_ds, "test": test_ds},
                        f, protocol=pickle.HIGHEST_PROTOCOL)
        print("  Graph cache saved.")

    del tr_s, vl_s, te_s

    # ── Train or load ─────────────────────────────────────────────────────────
    if SKIP_IF_CKPT and os.path.isfile(CHECKPOINT):
        print("\n  Checkpoint found — skipping training.")
        model = load_checkpoint(CHECKPOINT)
        tr_loss_h = vl_loss_h = tr_mae_h = vl_mae_h = []
    else:
        if os.path.isfile(CHECKPOINT):
            os.remove(CHECKPOINT)
        (model,
         tr_loss_h, vl_loss_h,
         tr_mae_h, vl_mae_h) = train_model(train_ds, val_ds)

    if tr_mae_h:
        print("\n  Plotting training curve ...")
        plot_training_curve(tr_loss_h, vl_loss_h, tr_mae_h, vl_mae_h)

    # ── Inference ─────────────────────────────────────────────────────────────
    print("\nPredicting on validation set ...")
    val_res = predict_set(model, val_ds,  "VAL")
    val_m   = compute_metrics(val_res, "Validation")
    save_csv(val_res, val_ds, OUTPUT_VAL_CSV)

    print("\nPredicting on test set ...")
    test_res = predict_set(model, test_ds, "TEST")
    test_m   = compute_metrics(test_res, "Test")
    save_csv(test_res, test_ds, OUTPUT_CSV)

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n  Generating plots ...")
    plot_parity(val_res, test_res, val_m, test_m)
    plot_error_hist(val_res, test_res)
    plot_property_maes(test_m)
    plot_benchmark_table(test_m)

    # ── Final summary ─────────────────────────────────────────────────────────
    print_final_summary(val_m, test_m)

    print("\n  Output files:")
    for f in [CHECKPOINT, OUTPUT_VAL_CSV, OUTPUT_CSV,
              "CPGN_JARVIS_training_curve.png",
              "CPGN_JARVIS_parity_plots.png",
              "CPGN_JARVIS_property_maes.png",
              "CPGN_JARVIS_error_hist.png",
              "CPGN_JARVIS_benchmark_table.png"]:
        tag = "✓" if os.path.exists(f) else "·"
        print(f"    [{tag}] {f}")
    print("\n  Done.")


if __name__ == "__main__":
    main()
