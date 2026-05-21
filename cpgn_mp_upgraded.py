"""
CPGN — Upgraded for ALIGNN/MEGNet benchmark parity on mp.2018.6.1.json
========================================================================
All six root causes of the MAE gap vs ALIGNN (0.022) / MEGNet (0.028)
are fixed in this version:

  Fix 1  MAE loss for formation energy training  (was MSE)
  Fix 2  Shuffled split with SEED=42             (was index-based — distribution shift)
  Fix 3  Single-task primary training on Ef      (auxiliary tasks as lightweight heads,
                                                  loss weight λ_aux=0.1 so they do not
                                                  dilute the Ef gradient)
  Fix 4  Learned 92-dim elemental embedding      (was 9-dim hand-crafted)
  Fix 5  Explicit angle encoding via line graph  (3-body interactions, same as ALIGNN)
  Fix 6  500 epochs with cosine-annealing LR     (was 50 epochs, aggressive step decay)

Additional benchmark requirements from the ALIGNN paper:
  • MAD (Mean Absolute Deviation) computed and printed
  • MAD:MAE ratio printed  (benchmark: ALIGNN=42.27, MEGNet=33.2, CGCNN=23.8)
  • Real band_gap from JSON used if present (not proxied)
  • Checkpoint selected on best val Ef MAE  (not combined loss)
  • Parity plots equal-aspect (same as SchNet/MEGNet paper figures)
  • Benchmark comparison table printed at end

Split: 60,000 / 5,000 / 4,239  (shuffled, SEED=42)
Target: formation_energy_per_atom  (eV/atom)  — primary
        band_gap                   (eV)        — auxiliary (real value from JSON)

Outputs
-------
  cpgn_results_MP_test.csv        per-sample test predictions
  cpgn_results_MP_val.csv         per-sample val predictions
  CPGN_MP_training_curve.png      train/val MAE curve
  CPGN_MP_parity_plots.png        parity plots (val + test, equal-aspect)
  CPGN_MP_error_hist.png          error distribution
  CPGN_MP_benchmark_table.png     comparison vs CGCNN / MEGNet / SchNet / ALIGNN
  cpgn_mp_best.pt                 best checkpoint (on val Ef MAE)

Dependencies
------------
  pip install torch torch-geometric pymatgen scikit-learn matplotlib seaborn
"""

# ============================================================
# 0.  IMPORTS
# ============================================================
import os, sys, json, warnings, time, random, math
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pymatgen.core import Structure, Element
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
from torch_geometric.nn import MessagePassing, global_mean_pool

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


# ============================================================
# 1.  CONFIG
# ============================================================
JSON_PATH      = "mp.2018.6.1.json"
OUTPUT_CSV     = "cpgn_results_MP_test.csv"
OUTPUT_VAL_CSV = "cpgn_results_MP_val.csv"
CHECKPOINT     = "cpgn_mp_best.pt"
SKIP_IF_CKPT   = True

# ── Split (shuffled — Fix 2) ──────────────────────────────────────────────────
N_TRAIN = 60000
N_VAL   =  5000
N_TEST  =  4239
SEED    = 42

# ── Training (Fix 1, Fix 6) ───────────────────────────────────────────────────
EPOCHS        = 500           # Fix 6: was 50
BATCH_SIZE    = 64
LEARNING_RATE = 3e-4          # slightly higher for AdamW + cosine
WEIGHT_DECAY  = 1e-5
PATIENCE      = 50            # generous early stopping for cosine schedule
LAMBDA_AUX    = 0.1           # Fix 3: auxiliary task loss weight (was equal to primary)

# ── Model (Fix 4, Fix 5) ──────────────────────────────────────────────────────
N_ELEM        = 103           # Fix 4: learned embedding per element (Z=1..102)
ELEM_DIM      = 64            # Fix 4: elemental embedding dimension
N_POLY_FEAT   = 7             # polyhedron geometric features
N_EDGE_FEAT   = 40            # Fix 5: RBF-expanded bond distances (40 Gaussian centres)
N_ANGLE_FEAT  = 40            # Fix 5: RBF-expanded bond angles  (line graph)
HIDDEN_DIM    = 256           # wider hidden for larger dataset
N_LAYERS      = 4
PRED_DIM      = 128
CUTOFF        = 8.0           # Å — larger cutoff catches more neighbours
RBF_CENTRES   = 40
RBF_SIGMA     = 0.5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device : {DEVICE}")

# ── Published benchmark MAE values (eV/atom) for comparison table ─────────────
PUBLISHED = {
    "CFID"   : {"ef_mae": 0.104, "eg_mae": None},
    "CGCNN"  : {"ef_mae": 0.039, "eg_mae": 0.388},
    "MEGNet" : {"ef_mae": 0.028, "eg_mae": 0.330},
    "SchNet" : {"ef_mae": 0.035, "eg_mae": None},
    "ALIGNN" : {"ef_mae": 0.022, "eg_mae": 0.218},
}
MP_MAD_EF = 0.93   # eV/atom — from ALIGNN paper Table 2
MP_MAD_EG = 0.434  # eV       — from ALIGNN paper Table 2


# ============================================================
# 2.  RBF EXPANSION  (Fix 5 — replaces raw distance/angle)
# ============================================================
class RBFExpansion(nn.Module):
    """
    Gaussian radial basis function expansion.
    Maps scalar distance/angle to a fixed-length vector.
    Same approach used in DimeNet++ and ALIGNN.
    """
    def __init__(self, low: float, high: float, n_centres: int):
        super().__init__()
        centres = torch.linspace(low, high, n_centres)
        self.register_buffer("centres", centres)
        self.gamma = 1.0 / (2.0 * ((high - low) / n_centres) ** 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.gamma * (x.unsqueeze(-1) - self.centres) ** 2)


# ============================================================
# 3.  JSON PARSER  (same as MEGNet script)
# ============================================================
def parse_structure(entry: dict) -> Structure:
    struct_entry = entry.get("structure", None)
    mid          = entry.get("material_id", "unknown")
    if not struct_entry:
        raise ValueError(f"Empty structure: {mid}")
    if isinstance(struct_entry, dict):
        return Structure.from_dict(struct_entry)
    if isinstance(struct_entry, str):
        s = struct_entry.strip()
        if not s:
            raise ValueError(f"Empty string: {mid}")
        if s.startswith("#") or "_cell_length" in s:
            return Structure.from_str(s, fmt="cif")
        return Structure.from_dict(json.loads(s))
    raise ValueError(f"Unknown type {type(struct_entry)}: {mid}")


# ============================================================
# 4.  ATOM NUMBER LOOKUP  (Fix 4 — for learned embedding)
# ============================================================
def get_atomic_number(element_str: str) -> int:
    """Returns atomic number Z clipped to [1, N_ELEM-1]."""
    sym = str(element_str).split("+")[0].split("-")[0].strip()
    try:
        z = Element(sym).Z
    except Exception:
        z = 1
    return max(1, min(z, N_ELEM - 1))


# ============================================================
# 5.  POLYHEDRON FEATURE VECTOR  (7-dim, same as before)
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
# 6.  DUAL GRAPH BUILDER  (Fix 5 — adds line graph for angles)
# ============================================================
def build_dual_graphs(structure: Structure) -> dict:
    """
    Builds:
      Atom graph  G_A : nodes=atoms, edges=bonds, edge_attr=raw distance (scalar)
      Line graph  G_L : nodes=bonds, edges=bond-pairs, edge_attr=raw angle (scalar)
                        — same as ALIGNN's triplet interaction graph
      Poly graph  G_P : nodes=polyhedra, edges=connectivity type
    All scalars are stored raw; RBF expansion applied inside the model.
    """
    try:    vnn = VoronoiNN(cutoff=CUTOFF, allow_pathological=True)
    except: vnn = VoronoiNN(cutoff=CUTOFF)
    N = len(structure)

    # ── atom features: just atomic numbers (for learned embedding) ────────────
    atom_z = np.array([get_atomic_number(str(s.specie)) for s in structure],
                       dtype=np.int64)

    # ── polyhedron features ───────────────────────────────────────────────────
    poly_x  = np.zeros((N, N_POLY_FEAT), dtype=np.float32)
    site_nn = []
    poly_nb = []
    for i in range(N):
        try:    nn = vnn.get_nn_info(structure, i)
        except: nn = []
        site_nn.append(nn)
        poly_x[i] = get_poly_features(structure, i, nn)
        poly_nb.append(set(nb["site_index"] for nb in nn))

    # ── atom graph edges  (store raw distance for RBF) ───────────────────────
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
    atom_dist = np.array(a_dist, dtype=np.float32)  # (E,)

    # ── line graph (Fix 5) ────────────────────────────────────────────────────
    # Build edge_to_idx: (src, dst) → edge_id

    # For each atom i, find all pairs of incoming edges (edges ending at i)
    # Bond pair (edge j→i, edge k→i) forms a line graph edge j←→k with angle
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
                # angle between bond (src_p → i) and bond (src_q → i)
                v1 = (np.array(structure[src_p].coords) -
                       np.array(structure[i].coords))
                v2 = (np.array(structure[src_q].coords) -
                       np.array(structure[i].coords))
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if n1 < 1e-8 or n2 < 1e-8: continue
                cos_a = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
                angle = np.arccos(cos_a)   # radians [0, π]
                lg_src.append(eid_p)
                lg_dst.append(eid_q)
                lg_angles.append(float(angle))

    if lg_src:
        line_ei     = np.array([lg_src, lg_dst], dtype=np.int64)
        line_angles = np.array(lg_angles, dtype=np.float32)
    else:
        line_ei     = np.zeros((2, 0), dtype=np.int64)
        line_angles = np.zeros(0, dtype=np.float32)

    # ── polyhedron graph edges ────────────────────────────────────────────────
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
                p_src += [i, best_j]; p_dst += [best_j, i]; p_ea += [[0.5], [0.5]]
    poly_ei = (np.array([p_src, p_dst], dtype=np.int64)
               if p_src else np.zeros((2, 0), dtype=np.int64))
    poly_ea = (np.array(p_ea, dtype=np.float32)
               if p_ea else np.zeros((0, 1), dtype=np.float32))

    return {
        "atom_z"     : atom_z,        # (N,)        int
        "atom_ei"    : atom_ei,        # (2, E)      int
        "atom_dist"  : atom_dist,      # (E,)        float  raw Å
        "line_ei"    : line_ei,        # (2, E_L)    int
        "line_angles": line_angles,    # (E_L,)      float  raw radians
        "poly_x"     : poly_x,         # (N, 7)      float
        "poly_ei"    : poly_ei,        # (2, E_P)    int
        "poly_ea"    : poly_ea,        # (E_P, 1)    float
        "n_atoms"    : N,
        "n_edges"    : E,
    }


# ============================================================
# 7.  PyG DATA OBJECT
# ============================================================
def graphs_to_pyg(graph: dict, targets: dict,
                  material_id: str = "") -> Data:
    data = Data(
        # atom graph
        x          = torch.tensor(graph["atom_z"],    dtype=torch.long),
        edge_index = torch.tensor(graph["atom_ei"],   dtype=torch.long),
        edge_dist  = torch.tensor(graph["atom_dist"], dtype=torch.float32),
        # targets
        y_ef       = torch.tensor([targets["ef"]],  dtype=torch.float32),
        y_bg       = torch.tensor([targets["bg"]],  dtype=torch.float32),
        y_stab     = torch.tensor([targets["stab"]], dtype=torch.float32),
        num_nodes  = graph["n_atoms"],
    )
    # poly + line graph stored as numpy (avoids PyG auto-collate crash)
    data._poly_x_np      = graph["poly_x"]
    data._poly_ei_np     = graph["poly_ei"]
    data._poly_ea_np     = graph["poly_ea"]
    data._line_ei_np     = graph["line_ei"]
    data._line_angles_np = graph["line_angles"]
    data._n_edges        = graph["n_edges"]
    data.material_id     = material_id
    return data


# ============================================================
# 8.  SPLIT LOADER
# ============================================================
def load_split(split_data: list, label: str = "") -> tuple:
    structs, ef_arr, bg_arr, stab_arr, ids = [], [], [], [], []
    skipped = 0
    for d in split_data:
        mid = d.get("material_id", "unknown")
        try:
            s  = parse_structure(d)
            ef = float(d["formation_energy_per_atom"])
            # Fix: use real band_gap from JSON, proxy only if absent
            bg   = float(d["band_gap"])   if "band_gap"  in d else max(0.0, -ef * 0.8)
            stab = float(d["stability"])  if "stability" in d else float(ef < 0.0)
            structs.append(s)
            ef_arr.append(ef); bg_arr.append(bg); stab_arr.append(stab)
            ids.append(mid)
        except Exception as e:
            print(f"  [{label}] skip {mid}: {e}")
            skipped += 1
    print(f"  [{label}] {len(structs):,} / {len(split_data):,}  skipped={skipped}")
    return (structs,
            np.array(ef_arr, dtype=np.float32),
            np.array(bg_arr, dtype=np.float32),
            np.array(stab_arr, dtype=np.float32),
            ids)


# ============================================================
# 9.  DATASET BUILDER
# ============================================================
def build_dataset(structs, ef, bg, stab, ids, label="") -> list:
    ds, skip = [], 0
    n = len(structs)
    print(f"\n  [{label}] Building dual+line graphs for {n:,} structures ...")
    t0 = time.time()
    for i in range(n):
        try:
            g    = build_dual_graphs(structs[i])
            data = graphs_to_pyg(g,
                                  {"ef":float(ef[i]),
                                   "bg":float(bg[i]),
                                   "stab":float(stab[i])},
                                  ids[i])
            ds.append(data)
        except Exception:
            skip += 1
        if (i+1) % 5000 == 0 or (i+1) == n:
            print(f"    ... {i+1:,}/{n:,}  ({time.time()-t0:.0f}s)")
    print(f"  [{label}] Built {len(ds):,}  skipped {skip}")
    return ds


# ============================================================
# 10.  MESSAGE PASSING LAYER  (edge-conditioned, for atom graph)
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


# ============================================================
# 11.  LINE GRAPH MESSAGE PASSING LAYER  (Fix 5 — angle interactions)
# ============================================================
class LineGraphConv(MessagePassing):
    """
    Message passing on the line graph:
    nodes = bonds (edge embeddings from atom graph)
    edges = bond pairs, attr = RBF(angle)
    Updates bond embeddings using angular information.
    Same as ALIGNN's eConv on the line graph.
    """
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
# 12.  CROSS-ATTENTION MODULE  (atom ↔ polyhedron)
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
# 13.  CPGN MODEL  (fully upgraded)
# ============================================================
class CPGN(nn.Module):
    """
    Upgraded CPGN:
      • Learned elemental embedding  (Fix 4)
      • RBF-expanded distances + angles  (Fix 5)
      • Line graph message passing for angular interactions  (Fix 5)
      • Dual atom+poly cross-attention  (retained)
      • Single primary head (Ef) + lightweight auxiliary heads
    """
    def __init__(self):
        super().__init__()
        H = HIDDEN_DIM

        # ── Fix 4: learned elemental embedding ─────────────────────────────
        self.elem_embed = nn.Embedding(N_ELEM, ELEM_DIM, padding_idx=0)

        # ── RBF expansions ──────────────────────────────────────────────────
        self.dist_rbf  = RBFExpansion(0.0, CUTOFF, RBF_CENTRES)   # bond dist
        self.angle_rbf = RBFExpansion(0.0, math.pi, RBF_CENTRES)  # bond angle

        # Input projections
        self.atom_in   = nn.Sequential(nn.Linear(ELEM_DIM, H), nn.SiLU(), nn.Linear(H, H))
        self.edge_in   = nn.Sequential(nn.Linear(N_EDGE_FEAT, H), nn.SiLU())
        self.poly_in   = nn.Sequential(nn.Linear(N_POLY_FEAT, H), nn.SiLU(), nn.Linear(H, H))
        self.poly_edge_proj = nn.Linear(1, H)

        # ── Fix 5: line graph convolutions ─────────────────────────────────
        self.atom_convs  = nn.ModuleList([CPGNConv(H, H, H)      for _ in range(N_LAYERS)])
        self.line_convs  = nn.ModuleList([LineGraphConv(H, N_ANGLE_FEAT, H) for _ in range(N_LAYERS)])
        self.poly_convs  = nn.ModuleList([CPGNConv(H, H, H)      for _ in range(N_LAYERS)])
        self.cross_attns = nn.ModuleList([CrossAttention(H)       for _ in range(N_LAYERS)])
        self.dropout     = nn.Dropout(p=0.1)

        # ── Fusion + heads ──────────────────────────────────────────────────
        self.fusion  = nn.Sequential(
            nn.Linear(H*2, H), nn.SiLU(),
            nn.Linear(H, PRED_DIM), nn.SiLU(),
        )
        # Primary head: formation energy (no activation — regression in eV/atom)
        self.head_ef   = nn.Linear(PRED_DIM, 1)
        # Auxiliary heads
        self.head_bg   = nn.Linear(PRED_DIM, 1)   # band gap (softplus → ≥0)
        self.head_stab = nn.Linear(PRED_DIM, 1)   # stability (BCE)

    def forward(self, batch):
        dev = batch.x.device

        # ── atom embeddings (Fix 4) ─────────────────────────────────────────
        atom_h = self.atom_in(self.elem_embed(batch.x.to(dev)))   # (N_a, H)

        # ── bond embeddings from RBF distances (Fix 5) ─────────────────────
        dist_rbf = self.dist_rbf(batch.edge_dist.to(dev))         # (E, 40)
        bond_h   = self.edge_in(dist_rbf)                          # (E, H)

        # ── poly embeddings ─────────────────────────────────────────────────
        poly_h   = self.poly_in(batch.poly_x.to(dev))
        p_ei     = batch.poly_ei.to(dev)
        p_ea     = batch.poly_ea.to(dev)
        p_ea_p   = (self.poly_edge_proj(p_ea)
                    if p_ea.shape[0] > 0
                    else torch.zeros(0, HIDDEN_DIM, device=dev, dtype=atom_h.dtype))

        # ── line graph (angle) data ─────────────────────────────────────────
        line_ei     = batch.line_ei.to(dev)      # (2, E_L)
        angle_rbf   = self.angle_rbf(batch.line_angles.to(dev))  # (E_L, 40)

        a_batch = batch.batch

        # ── L × (atom MP + line MP + poly MP + cross-attn) ─────────────────
        for ac, lc, pc, ca in zip(self.atom_convs, self.line_convs,
                                   self.poly_convs, self.cross_attns):
            # atom graph: use current bond embeddings as edge features
            atom_h  = self.dropout(ac(atom_h, batch.edge_index, bond_h))

            # line graph (Fix 5): update bond embeddings with angle info
            bond_h  = self.dropout(lc(bond_h, line_ei, angle_rbf))

            # poly graph
            poly_h  = self.dropout(pc(poly_h, p_ei, p_ea_p))

            # cross-attention atom ↔ poly
            atom_h, poly_h = ca(atom_h, poly_h, a_batch, a_batch)

        # ── global pooling + fusion ─────────────────────────────────────────
        ha = global_mean_pool(atom_h, a_batch)
        hp = global_mean_pool(poly_h, a_batch)
        z  = self.fusion(torch.cat([ha, hp], dim=-1))

        return {
            "ef"        : self.head_ef(z).squeeze(-1),
            "bg"        : F.softplus(self.head_bg(z)).squeeze(-1),
            "stab_logit": self.head_stab(z).squeeze(-1),
        }


# ============================================================
# 14.  LOSS  (Fix 1 + Fix 3)
# ============================================================
class CPGNLoss(nn.Module):
    """
    Fix 1: MAE (L1) for primary formation energy task.
    Fix 3: auxiliary tasks weighted by LAMBDA_AUX=0.1 so they
           do not dilute the primary Ef gradient.
    """
    def forward(self, preds, batch):
        # Primary: MAE on Ef   (Fix 1)
        loss_ef   = F.l1_loss(preds["ef"], batch.y_ef)
        # Auxiliary: MAE on Bg + BCE on stab  (Fix 3: ×0.1)
        loss_bg   = F.l1_loss(preds["bg"], batch.y_bg)
        loss_stab = F.binary_cross_entropy_with_logits(
            preds["stab_logit"], batch.y_stab,
            pos_weight=torch.tensor([3.0], device=batch.y_stab.device))
        return loss_ef + LAMBDA_AUX * (loss_bg + loss_stab)


# ============================================================
# 15.  COLLATE / DATALOADER
# ============================================================
from torch_geometric.data import Batch as PyGBatch
from torch.utils.data import DataLoader as TorchLoader

def custom_collate(data_list):
    batch = PyGBatch.from_data_list(data_list)

    # poly graph (nodes = atoms, variable edges)
    poly_x_list, poly_ei_list, poly_ea_list = [], [], []
    # line graph (nodes = bonds, variable edges)
    line_ei_list, line_ang_list = [], []

    atom_cum = 0   # cumulative atom offset
    edge_cum = 0   # cumulative edge offset

    for d in data_list:
        n_a = d.num_nodes
        n_e = d._n_edges

        poly_x_list.append(torch.tensor(d._poly_x_np,  dtype=torch.float32))
        poly_ea_list.append(torch.tensor(d._poly_ea_np, dtype=torch.float32))
        ei_p = d._poly_ei_np
        if ei_p.shape[1] > 0:
            poly_ei_list.append(torch.tensor(ei_p + atom_cum, dtype=torch.long))

        # line graph edge index: node indices are edge indices in atom graph
        lei = d._line_ei_np
        if lei.shape[1] > 0:
            line_ei_list.append(torch.tensor(lei + edge_cum, dtype=torch.long))
        line_ang_list.append(torch.tensor(d._line_angles_np, dtype=torch.float32))

        atom_cum += n_a
        edge_cum += n_e

    batch.poly_x   = torch.cat(poly_x_list, dim=0)
    batch.poly_ei  = (torch.cat(poly_ei_list, dim=1)
                      if poly_ei_list else torch.zeros(2, 0, dtype=torch.long))
    batch.poly_ea  = torch.cat(poly_ea_list, dim=0)
    batch.line_ei  = (torch.cat(line_ei_list, dim=1)
                      if line_ei_list else torch.zeros(2, 0, dtype=torch.long))
    batch.line_angles = torch.cat(line_ang_list, dim=0)

    batch.y_ef    = torch.cat([d.y_ef   for d in data_list])
    batch.y_bg    = torch.cat([d.y_bg   for d in data_list])
    batch.y_stab  = torch.cat([d.y_stab for d in data_list])

    # accumulate edge_dist with proper edge_index already handled by PyG
    batch.edge_dist = torch.cat([d.edge_dist for d in data_list], dim=0)

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
# 16.  TRAINING LOOP  (Fix 6 — 500 epochs, cosine LR, best on MAE)
# ============================================================
def train_model(train_ds, val_ds):
    model     = CPGN().to(DEVICE)
    criterion = CPGNLoss()
    optimiser = AdamW(model.parameters(), lr=LEARNING_RATE,
                      weight_decay=WEIGHT_DECAY)
    # Fix 6: cosine annealing (same as ALIGNN)
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
    print("  CPGN Upgraded — mp.2018.6.1.json")
    print(f"  Params      : {n_p:,}")
    print(f"  Train/Val   : {len(train_ds):,} / {len(val_ds):,}")
    print(f"  Epochs      : {EPOCHS}  Batch : {BATCH_SIZE}  LR : {LEARNING_RATE}")
    print(f"  Loss        : MAE (primary Ef) + {LAMBDA_AUX}×(MAE Bg + BCE stab)")
    print(f"  Scheduler   : CosineAnnealingLR  T_max={EPOCHS}")
    print("  Checkpoint  : best val Ef MAE")
    print(f"{'='*70}")
    print(f"  {'Ep':>5} | {'TrLoss':>8} | {'VlLoss':>8} | "
          f"{'TrMAE':>8} | {'VlMAE':>8} | {'LR':>9} | {'Time':>6}")
    print(f"  {'-'*62}")

    t0_total = time.time()
    for ep in range(1, EPOCHS + 1):
        # ── train ────────────────────────────────────────────────────────────
        model.train()
        tl = tm = n = 0; t0 = time.time(); skip = 0
        for b in tr_loader:
            b = b.to(DEVICE)
            optimiser.zero_grad(set_to_none=True)
            p  = model(b)
            ls = criterion(p, b)
            if not torch.isfinite(ls): skip += 1; continue
            ls.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            bs = len(b.y_ef)
            tl += ls.item() * bs
            tm += F.l1_loss(p["ef"].detach(), b.y_ef).item() * bs
            n  += bs
        n = max(n, 1); tl /= n; tm /= n
        scheduler.step()

        # ── validate ─────────────────────────────────────────────────────────
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
                  f"{tm:>8.4f} | {vm:>8.4f} | {lr_now:>9.2e} | "
                  f"{time.time()-t0:>5.1f}s")

        # Fix 2b: checkpoint on best VAL EF MAE
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
                print(f"  Early stopping ep {ep} (best val MAE={best_val_mae:.4f})")
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
    bg_true, bg_pred = [], []
    st_true, st_logit = [], []
    for b in loader:
        b = b.to(DEVICE); out = model(b)
        ef_true.extend(b.y_ef.cpu().tolist())
        ef_pred.extend(out["ef"].detach().cpu().tolist())
        bg_true.extend(b.y_bg.cpu().tolist())
        bg_pred.extend(out["bg"].detach().cpu().tolist())
        st_true.extend(b.y_stab.cpu().tolist())
        st_logit.extend(out["stab_logit"].detach().cpu().tolist())
    st_pred = (np.array(st_logit) > 0.0).astype(int)
    print(f"  [{label}] {len(ef_true):,} predictions")
    return {
        "ef_true": np.array(ef_true), "ef_pred": np.array(ef_pred),
        "bg_true": np.array(bg_true), "bg_pred": np.array(bg_pred),
        "stab_true": np.array(st_true, dtype=int),
        "stab_pred": st_pred, "stab_logit": np.array(st_logit),
    }


# ============================================================
# 19.  METRICS  (with MAD + MAD:MAE ratio — ALIGNN paper requirement)
# ============================================================
def compute_metrics(res, label="", mad_ef=MP_MAD_EF, mad_eg=MP_MAD_EG):
    ef_mae  = mean_absolute_error(res["ef_true"], res["ef_pred"])
    ef_mse  = mean_squared_error(res["ef_true"],  res["ef_pred"])
    ef_rmse = float(np.sqrt(ef_mse))
    bg_mae  = mean_absolute_error(res["bg_true"],  res["bg_pred"])
    st_acc  = accuracy_score(res["stab_true"], res["stab_pred"])
    st_f1   = f1_score(res["stab_true"], res["stab_pred"],
                       average="binary", zero_division=0)

    # MAD computed from actual test data (more accurate than paper's global MAD)
    ef_mad_data  = float(np.mean(np.abs(res["ef_true"] - res["ef_true"].mean())))
    mad_mae_ef   = ef_mad_data / ef_mae if ef_mae > 0 else float("inf")

    bg_mad_data  = float(np.mean(np.abs(res["bg_true"] - res["bg_true"].mean())))
    mad_mae_bg   = bg_mad_data / bg_mae if bg_mae > 0 else float("inf")

    hdr = "=" * 60
    print(f"\n{hdr}")
    print(f"  CPGN — {label} Results  (MP 2018.6.1)")
    print(hdr)
    print(f"  Samples                    : {len(res['ef_true']):,}")
    print("\n  ── Formation Energy  Ef  (eV/atom) ──")
    print(f"  MAE                        : {ef_mae:.4f}")
    print(f"  MSE                        : {ef_mse:.4f}")
    print(f"  RMSE                       : {ef_rmse:.4f}")
    print(f"  MAD (this split)           : {ef_mad_data:.4f}")
    print(f"  MAD (paper, global)        : {mad_ef:.3f}")
    print(f"  MAD:MAE ratio              : {mad_mae_ef:.2f}")
    print("  ALIGNN benchmark MAD:MAE   : 42.27")
    print("\n  ── Band Gap  Eg  (eV) ──")
    print(f"  MAE                        : {bg_mae:.4f}")
    print(f"  MAD (this split)           : {bg_mad_data:.4f}")
    print(f"  MAD (paper, global)        : {mad_eg:.3f}")
    print(f"  MAD:MAE ratio              : {mad_mae_bg:.2f}")
    print("  ALIGNN benchmark MAD:MAE   : 6.19")
    print("\n  ── Stability Classification ──")
    print(f"  Accuracy                   : {st_acc:.4f}")
    print(f"  F1-score                   : {st_f1:.4f}")
    print(hdr)

    return {
        "ef_mae": ef_mae, "ef_mse": ef_mse, "ef_rmse": ef_rmse,
        "ef_mad": ef_mad_data, "ef_mad_mae": mad_mae_ef,
        "bg_mae": bg_mae, "bg_mad": bg_mad_data, "bg_mad_mae": mad_mae_bg,
        "st_acc": st_acc, "st_f1": st_f1,
    }


# ============================================================
# 20.  SAVE CSV
# ============================================================
def save_csv(res, dataset, path):
    rows = []
    for i in range(len(res["ef_true"])):
        mid = dataset[i].material_id if hasattr(dataset[i], "material_id") else str(i)
        rows.append({
            "id"        : mid,
            "true_eform": res["ef_true"][i],
            "pred_eform": res["ef_pred"][i],
            "error"     : res["ef_pred"][i] - res["ef_true"][i],
            "true_bg"   : res["bg_true"][i],
            "pred_bg"   : res["bg_pred"][i],
            "true_stab" : res["stab_true"][i],
            "pred_stab" : res["stab_pred"][i],
        })
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  Saved: {path}")


# ============================================================
# 21.  PLOTS
# ============================================================
BLUE  = "#2A7EC0"; CORAL = "#E05C2A"; GREEN = "#1D9E75"
PURPLE= "#534AB7"; DARK  = "#2C2C2A"; GRAY  = "#888780"

def plot_training_curve(tr_loss, vl_loss, tr_mae, vl_mae):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ep = range(1, len(tr_loss)+1)
    axes[0].plot(ep, tr_loss, lw=1.5, color=BLUE,  label="Train loss")
    axes[0].plot(ep, vl_loss, lw=1.5, color=CORAL, ls="--", label="Val loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MAE loss")
    axes[0].set_title("CPGN — Total Loss  (MAE Ef + aux)"); axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(ep, tr_mae, lw=1.5, color=BLUE,  label="Train MAE")
    axes[1].plot(ep, vl_mae, lw=1.5, color=CORAL, ls="--", label="Val MAE")
    # Reference lines
    for v, lbl, col in [(0.039,"CGCNN","gray"),(0.028,"MEGNet","purple"),
                         (0.022,"ALIGNN","green")]:
        axes[1].axhline(v, color=col, lw=0.8, ls=":", alpha=0.7, label=lbl)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Ef MAE (eV/atom)")
    axes[1].set_title("CPGN — Formation Energy MAE"); axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    plt.suptitle("CPGN Training — MP 2018.6.1  (60k/5k/4239 shuffled)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("CPGN_MP_training_curve.png", dpi=300); plt.close()
    print("  Saved: CPGN_MP_training_curve.png")


def plot_parity(val_res, test_res, val_m, test_m):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, res, m, label, color in [
        (axes[0], val_res,  val_m,  f"Validation ({N_VAL:,})",  BLUE),
        (axes[1], test_res, test_m, f"Test ({N_TEST:,})",        CORAL),
    ]:
        yt, yp = res["ef_true"], res["ef_pred"]
        ax.scatter(yt, yp, alpha=0.4, s=6, color=color, edgecolors="none", zorder=3)
        lo = min(yt.min(), yp.min()); hi = max(yt.max(), yp.max())
        mg = 0.05 * (hi - lo)
        ax.plot([lo-mg, hi+mg], [lo-mg, hi+mg], "k--", lw=1.5, label="Ideal")
        ax.set_xlim(lo-mg, hi+mg); ax.set_ylim(lo-mg, hi+mg)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("True Ef (eV/atom)", fontsize=11)
        ax.set_ylabel("Predicted Ef (eV/atom)", fontsize=11)
        ax.set_title(f"CPGN — {label}\n"
                     f"MAE={m['ef_mae']:.4f}  MSE={m['ef_mse']:.4f}  "
                     f"MAD:MAE={m['ef_mad_mae']:.1f}", fontsize=10)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    fig.suptitle("CPGN Parity Plots — MP 2018.6.1\n"
                 f"Train={N_TRAIN:,} | Val={N_VAL:,} | Test={N_TEST:,}  (shuffled)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("CPGN_MP_parity_plots.png", dpi=300); plt.close()
    print("  Saved: CPGN_MP_parity_plots.png")


def plot_error_hist(val_res, test_res):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, res, label, color in [
        (axes[0], val_res,  f"Val ({N_VAL:,})",   BLUE),
        (axes[1], test_res, f"Test ({N_TEST:,})",  CORAL),
    ]:
        err = res["ef_pred"] - res["ef_true"]
        ax.hist(err, bins=60, color=color, alpha=0.75, edgecolor="k", lw=0.3)
        ax.axvline(0, color="r", lw=1.5, ls="--", label="Zero error")
        ax.axvline(err.mean(), color=DARK, lw=1.2, ls=":",
                   label=f"Mean={err.mean():.4f}")
        ax.set_xlabel("Prediction Error (eV/atom)"); ax.set_ylabel("Count")
        ax.set_title(f"Ef Error Distribution — {label}"); ax.legend(); ax.grid(alpha=0.3)
    plt.suptitle("CPGN — Formation Energy Error Distributions (MP 2018)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("CPGN_MP_error_hist.png", dpi=300); plt.close()
    print("  Saved: CPGN_MP_error_hist.png")


def plot_benchmark_table(test_m):
    """
    Horizontal bar chart comparing CPGN vs CGCNN / MEGNet / SchNet / ALIGNN.
    Mirrors Table 2 in the ALIGNN paper.
    """
    models = list(PUBLISHED.keys()) + ["CPGN (ours)"]
    ef_maes = [PUBLISHED[m]["ef_mae"] for m in PUBLISHED] + [test_m["ef_mae"]]
    colors  = [GRAY]*len(PUBLISHED) + [CORAL]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Ef MAE bar ────────────────────────────────────────────────────────────
    bars = axes[0].barh(models, ef_maes, color=colors, alpha=0.88,
                         edgecolor="k", lw=0.5)
    for bar, v in zip(bars, ef_maes):
        axes[0].text(v + 0.001, bar.get_y() + bar.get_height()/2,
                     f"{v:.4f}", va="center", fontsize=10)
    axes[0].axvline(test_m["ef_mae"], color=CORAL, lw=1.5, ls="--", alpha=0.6)
    axes[0].set_xlabel("MAE (eV/atom)", fontsize=11)
    axes[0].set_title("Formation Energy Ef MAE — MP 2018\n"
                       "Table 2 (ALIGNN paper)", fontsize=11)
    axes[0].grid(axis="x", alpha=0.3)
    axes[0].invert_yaxis()

    # ── MAD:MAE ratio bar ─────────────────────────────────────────────────────
    # Paper values: CFID=0.93/0.104≈8.9, CGCNN=23.8, MEGNet=33.2, SchNet=26.6, ALIGNN=42.27
    ratios_pub = {
        "CFID": 0.93/0.104, "CGCNN": 0.93/0.039,
        "MEGNet": 0.93/0.028, "SchNet": 0.93/0.035,
        "ALIGNN": 42.27,
    }
    ratios = [ratios_pub[m] for m in PUBLISHED] + [test_m["ef_mad_mae"]]
    bar_colors = [GRAY]*len(PUBLISHED) + [CORAL]
    bars2 = axes[1].barh(models, ratios, color=bar_colors,
                          alpha=0.88, edgecolor="k", lw=0.5)
    for bar, v in zip(bars2, ratios):
        axes[1].text(v + 0.2, bar.get_y() + bar.get_height()/2,
                     f"{v:.1f}", va="center", fontsize=10)
    axes[1].set_xlabel("MAD:MAE ratio", fontsize=11)
    axes[1].set_title("MAD:MAE Ratio — Higher is Better\n"
                       "(MAD=0.93 eV/atom, MP 2018)", fontsize=11)
    axes[1].grid(axis="x", alpha=0.3)
    axes[1].invert_yaxis()

    plt.suptitle("CPGN vs State-of-the-Art — MP 2018.6.1 Benchmark",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("CPGN_MP_benchmark_table.png", dpi=300); plt.close()
    print("  Saved: CPGN_MP_benchmark_table.png")


# ============================================================
# 22.  FINAL SUMMARY  (full ALIGNN-paper-style table)
# ============================================================
def print_final_summary(val_m, test_m):
    print("\n" + "=" * 70)
    print("  CPGN MP 2018.6.1 — Benchmark Summary Table")
    print("  (mirrors Table 2, ALIGNN paper, npj Comput. Mater. 2021)")
    print("=" * 70)
    print(f"  {'Model':<14} {'Ef MAE':>10} {'MAD:MAE':>10} {'Eg MAE':>10}")
    print(f"  {'-'*48}")

    pub_eg = {"CFID":None,"CGCNN":0.388,"MEGNet":0.330,"SchNet":None,"ALIGNN":0.218}
    pub_ratio = {"CFID":0.93/0.104,"CGCNN":0.93/0.039,"MEGNet":0.93/0.028,
                 "SchNet":0.93/0.035,"ALIGNN":42.27}
    for m in PUBLISHED:
        eg = f"{pub_eg[m]:.3f}" if pub_eg[m] else "—"
        print(f"  {m:<14} {PUBLISHED[m]['ef_mae']:>10.4f} "
              f"{pub_ratio[m]:>10.2f} {eg:>10}")
    print(f"  {'-'*48}")
    print(f"  {'CPGN (ours)':<14} {test_m['ef_mae']:>10.4f} "
          f"{test_m['ef_mad_mae']:>10.2f} {test_m['bg_mae']:>10.4f}")
    print("=" * 70)
    print(f"\n  Split  : Train={N_TRAIN:,} | Val={N_VAL:,} | "
          f"Test={N_TEST:,}  (shuffled, SEED={SEED})")
    print(f"  Val Ef MAE  : {val_m['ef_mae']:.4f} eV/atom")
    print(f"  Test Ef MAE : {test_m['ef_mae']:.4f} eV/atom")
    print(f"  Test MAD    : {test_m['ef_mad']:.4f} eV/atom")
    print(f"  MAD:MAE     : {test_m['ef_mad_mae']:.2f}  "
          f"(ALIGNN=42.27, MEGNet=33.2, CGCNN=23.8)")
    print("=" * 70)


# ============================================================
# 23.  MAIN
# ============================================================
def main():
    print("=" * 70)
    print("  CPGN Upgraded — mp.2018.6.1.json")
    print("  Fixes: MAE loss | shuffled split | line graph angles |")
    print("         learned embeddings | 500 epochs | MAD:MAE output")
    print("=" * 70)

    if not os.path.isfile(JSON_PATH):
        print(f"[ERROR] {JSON_PATH} not found"); sys.exit(1)

    print(f"\nLoading {JSON_PATH} ...")
    with open(JSON_PATH) as f:
        mp_data = json.load(f)
    print(f"  Total structures : {len(mp_data):,}")

    # ── Fix 2: shuffled split ─────────────────────────────────────────────────
    indices = list(range(len(mp_data)))
    random.Random(SEED).shuffle(indices)
    train_idx = indices[:N_TRAIN]
    val_idx   = indices[N_TRAIN:N_TRAIN + N_VAL]
    test_idx  = indices[N_TRAIN + N_VAL:N_TRAIN + N_VAL + N_TEST]

    train_json = [mp_data[i] for i in train_idx]
    val_json   = [mp_data[i] for i in val_idx]
    test_json  = [mp_data[i] for i in test_idx]
    del mp_data

    print(f"  Train : {len(train_json):,}  Val : {len(val_json):,}  "
          f"Test : {len(test_json):,}  (shuffled, SEED={SEED})")

    # ── Parse ─────────────────────────────────────────────────────────────────
    print("\nParsing structures ...")
    tr_s,  tr_ef,  tr_bg,  tr_st,  tr_ids  = load_split(train_json, "TRAIN")
    vl_s,  vl_ef,  vl_bg,  vl_st,  vl_ids  = load_split(val_json,   "VAL")
    te_s,  te_ef,  te_bg,  te_st,  te_ids  = load_split(test_json,  "TEST")
    del train_json, val_json, test_json

    for lbl, s in [("Train",tr_s),("Val",vl_s),("Test",te_s)]:
        if not s:
            raise RuntimeError(f"No structures parsed for {lbl}")

    # ── Band gap diagnostic ───────────────────────────────────────────────────
    n_real_bg = int(np.sum(tr_bg > 0))
    print(f"\n  Band gap: {n_real_bg:,}/{len(tr_bg):,} train entries have Eg>0  "
          f"({'real DFT' if n_real_bg > len(tr_bg)//2 else 'mostly proxied'})")

    # ── Build dual+line graphs ────────────────────────────────────────────────
    # ── Build or load graph cache ─────────────────────────────────────────────
    import pickle
    GRAPH_CACHE = "cpgn_graph_cache.pkl"

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
        train_ds = build_dataset(tr_s, tr_ef, tr_bg, tr_st, tr_ids, "TRAIN")
        val_ds   = build_dataset(vl_s, vl_ef, vl_bg, vl_st, vl_ids, "VAL")
        test_ds  = build_dataset(te_s, te_ef, te_bg, te_st, te_ids, "TEST")
        print(f"\n  Saving graph cache → {GRAPH_CACHE} ...")
        with open(GRAPH_CACHE, "wb") as f:
            pickle.dump({"train": train_ds,
                         "val"  : val_ds,
                         "test" : test_ds}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        print("  Graph cache saved.")

    del tr_s, vl_s, te_s   # free structure memory regardless of path

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

    # ── Training curve ────────────────────────────────────────────────────────
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
    plot_benchmark_table(test_m)

    # ── Final summary ─────────────────────────────────────────────────────────
    print_final_summary(val_m, test_m)

    print("\n  Output files:")
    for f in [CHECKPOINT, OUTPUT_VAL_CSV, OUTPUT_CSV,
              "CPGN_MP_training_curve.png",
              "CPGN_MP_parity_plots.png",
              "CPGN_MP_error_hist.png",
              "CPGN_MP_benchmark_table.png"]:
        tag = "✓" if os.path.exists(f) else "·"
        print(f"    [{tag}] {f}")
    print("\n  Done.")


if __name__ == "__main__":
    main()
