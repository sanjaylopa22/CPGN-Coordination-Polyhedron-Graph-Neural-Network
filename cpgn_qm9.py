"""
CPGN — Adapted for QM9 Molecular Dataset
=========================================
Crystal Polyhedron Graph Network re-purposed for molecular property
prediction on the QM9 quantum chemistry benchmark.

QM9 Reference
-------------
  Ramakrishnan et al., Scientific Data 2014
  Ruddigkeit et al., J. Chem. Inf. Model. 2012
  134 k organic molecules (C, H, O, N, F) computed at B3LYP/6-31G(2df,p)

Properties predicted  (all 11 standard QM9 targets)
----------------------------------------------------
  PRIMARY (MAE loss, full weight):
    HOMO    α_HOMO   highest occupied molecular orbital energy    eV
  AUXILIARY (MAE, weight λ=0.1):
    LUMO    ε_LUMO   lowest unoccupied molecular orbital energy   eV
    Gap     ε_gap    HOMO–LUMO energy gap (= LUMO − HOMO)         eV
    ZPVE    ZPVE     zero-point vibrational energy                 eV
    mu      μ        dipole moment                                 Debye
    alpha   α        isotropic polarisability                      Bohr³
    R2      〈R²〉    electronic spatial extent                     Bohr²
    U0      U₀       internal energy at 0 K                        eV
    U298    U₂₉₈    internal energy at 298.15 K                   eV
    H298    H₂₉₈    enthalpy at 298.15 K                          eV
    G298    G₂₉₈    Gibbs free energy at 298.15 K                 eV

Molecular representation
------------------------
  Atom graph  G_A : atoms as nodes (learned Z-embedding, 64-dim)
                    bonds as edges (pairwise distances, 40-dim RBF, r_cut = 5 Å)
  Line graph  G_L : bond-pair triplets encoding bond angles (40-dim RBF on θ)
  "Poly" graph G_P: local coordination environments around each heavy atom
                    (same 7-dim geometric feature vector; corner/edge/face
                     connectivity weights 0.0/0.5/1.0 as in the crystal model)

  For molecules the polyhedron graph is a coordination-shell graph rather
  than a true Voronoi polyhedron graph, but the formulation is identical.

Architecture / training
-----------------------
  4 interleaved message-passing layers (atom MP → line-graph conv →
  poly MP → scalar-gated bidirectional cross-attention)
  Hidden dim  d_h = 256  |  Latent dim  d_z = 128  |  ~1.2 M parameters
  AdamW  lr = 3e-4  |  cosine annealing  |  grad clip 1.0  |  early stop 50 ep
  Split : 110 000 train / 10 000 val / 10 831 test  (shuffled, SEED=42)

Published SOTA MAEs (meV / Debye / Bohr³ etc. — converted to same units)
--------------------------------------------------------------------------
  HOMO  : SchNet 41 meV, DimeNet++ 24 meV, SphereNet 23 meV
  LUMO  : SchNet 34 meV, DimeNet++ 19 meV, SphereNet 18 meV
  Gap   : SchNet 63 meV, DimeNet++ 32 meV, SphereNet 31 meV
  ZPVE  : SchNet  1.7 meV, DimeNet++ 1.21 meV
  mu    : SchNet 0.033 D,  DimeNet++ 0.030 D
  alpha : SchNet 0.235 Bohr³, DimeNet++ 0.044 Bohr³

Data source
-----------
  torch_geometric.datasets.QM9   (auto-downloads from PyG servers)
  or:  pip install torch-geometric
       python -c "from torch_geometric.datasets import QM9; QM9(root='qm9_data')"

Outputs
-------
  cpgn_qm9_best.pt
  cpgn_qm9_test.csv / cpgn_qm9_val.csv
  CPGN_QM9_training_curve.png
  CPGN_QM9_parity_plots.png        (HOMO + Gap parity, val & test)
  CPGN_QM9_property_maes.png       (bar chart: all 11 property MAEs)
  CPGN_QM9_error_hist.png          (error distributions for 4 key properties)
  CPGN_QM9_benchmark_table.png     (vs SchNet / DimeNet++ / SphereNet)

Dependencies
------------
  pip install torch torch-geometric pymatgen scikit-learn matplotlib rdkit-pypi
"""

# ============================================================
# 0.  IMPORTS
# ============================================================
import os, sys, math, time, random, warnings, pickle
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error

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
OUTPUT_CSV     = "cpgn_qm9_test.csv"
OUTPUT_VAL_CSV = "cpgn_qm9_val.csv"
CHECKPOINT     = "cpgn_qm9_best.pt"
GRAPH_CACHE    = "cpgn_qm9_graph_cache.pkl"
QM9_ROOT       = "qm9_data"          # PyG will download here
SKIP_IF_CKPT   = True

# ── Split ─────────────────────────────────────────────────────────────────────
# QM9 has 130 831 valid molecules; standard splits follow SchNet/DimeNet papers
N_TRAIN = 110000
N_VAL   =  10000
# N_TEST  = remainder (~10 831)
SEED    = 42

# ── Training ──────────────────────────────────────────────────────────────────
EPOCHS        = 1000     # ALIGNN paper: 1000 epochs on QM9
BATCH_SIZE    = 64
LEARNING_RATE = 3e-4
WEIGHT_DECAY  = 1e-5
PATIENCE      = 50
LAMBDA_AUX    = 0.1      # auxiliary loss weight (HOMO gradient dominates)

# ── Model ─────────────────────────────────────────────────────────────────────
N_ELEM        = 10       # QM9 atoms: H(1) C(6) N(7) O(8) F(9)  → pad to 10
ELEM_DIM      = 64       # learned embedding dimension per element
N_POLY_FEAT   = 7        # local environment feature vector (same as crystal)
N_EDGE_FEAT   = 40       # RBF on bond distances
N_ANGLE_FEAT  = 40       # RBF on bond angles (line graph)
HIDDEN_DIM    = 256
N_LAYERS      = 4
PRED_DIM      = 128
CUTOFF        = 5.0      # Å  (shorter than crystal: molecules are compact)
RBF_CENTRES   = 40

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device : {DEVICE}")

# ── QM9 target indices (torch_geometric.datasets.QM9 ordering) ───────────────
# PyG stores all 19 properties; we select the 11 standard ones.
# Index → (short_name, display_label, unit, pyg_idx, activation)
#   activation: "none" | "softplus"   (all QM9 energies can be negative)
QM9_PROPS = [
    # short    label        unit      pyg_idx  activation  primary?
    ("HOMO",  "ε_HOMO",   "eV",      2,       "none",     True ),   # PRIMARY
    ("LUMO",  "ε_LUMO",   "eV",      3,       "none",     False),
    ("Gap",   "ε_gap",    "eV",      4,       "softplus",  False),
    ("ZPVE",  "ZPVE",     "eV",      6,       "softplus",  False),
    ("mu",    "μ",        "D",       0,       "softplus",  False),
    ("alpha", "α",        "Bohr³",   1,       "softplus",  False),
    ("R2",    "〈R²〉",   "Bohr²",   5,       "softplus",  False),
    ("U0",    "U₀",       "eV",      7,       "none",     False),
    ("U298",  "U₂₉₈",    "eV",      8,       "none",     False),
    ("H298",  "H₂₉₈",    "eV",      9,       "none",     False),
    ("G298",  "G₂₉₈",    "eV",      10,      "none",     False),
]

PRIMARY_IDX = 0                       # HOMO is the primary target
N_PROPS     = len(QM9_PROPS)          # 11
N_AUX       = N_PROPS - 1            # 10 auxiliary targets

# Units for display (PyG stores values in their natural QM9 units)
# energies in Hartree → we convert to eV (1 Hartree = 27.2114 eV)
HARTREE_TO_EV = 27.2114

# PyG qm9_v3.zip (pre-processed) stores ALL values already in eV/D/Bohr³.
# HARTREE_INDICES_PYG kept for reference but NO conversion applied.
# Only atomisation correction is needed for U0/U298/H298/G298.
HARTREE_INDICES_PYG = {2, 3, 4, 6, 7, 8, 9, 10}   # documented only — NOT used for conversion

# ── Fix 1: Atomic reference energies (Hartree) for atomisation energy ─────────
# U0/U/H/G are TOTAL molecular energies (~-1000 eV for small molecules).
# All benchmarks (SchNet, DimeNet++, MPNN) use ATOMISATION energy instead:
#   atomisation_E = total_E − Σ_i E_ref(Z_i)
# This removes the huge per-atom offset and makes the target O(1 eV).
# Reference energies from Ramakrishnan et al. (QM9 paper), Table 3.
ATOM_REF_ENERGIES_HAR = {
    1:  -0.500273,   # H
    6:  -37.846772,  # C
    7:  -54.583861,  # N
    8:  -75.064579,  # O
    9:  -99.718730,  # F
}
# PyG target indices that need atomisation correction
ATOMISATION_INDICES_PYG = {7, 8, 9, 10}   # U0, U298, H298, G298

# ── Fix 2: Unit scale factors — train in meV / D / Bohr³ not eV ───────────────
# All published benchmarks report in these units.
# Training in meV keeps all targets O(10–1000) instead of O(0.01–1000).
# HOMO/LUMO/Gap/ZPVE: eV → meV (×1000)
# U0/U298/H298/G298:  eV → meV (×1000)  (after atomisation correction)
# mu:    Debye — no conversion needed (already O(1–5))
# alpha: Bohr³ — no conversion needed (already O(10–100))
# R2:    Bohr² — no conversion needed (already O(100–1000))
# Unit scale applied to targets before training; MAE printed in these units.
# All targets stored in eV (natural units). No scaling.
# ALIGNN paper reports QM9 MAEs in eV — we match that table directly.
PROP_SCALE = {k: 1.0 for k in
    ["HOMO","LUMO","Gap","ZPVE","mu","alpha","R2","U0","U298","H298","G298"]}
PROP_UNIT_LABEL = {
    "HOMO":"eV","LUMO":"eV","Gap":"eV","ZPVE":"eV",
    "mu":"D","alpha":"Bohr³","R2":"Bohr²",
    "U0":"eV","U298":"eV","H298":"eV","G298":"eV",
}

# Benchmark MAEs in eV (matching ALIGNN paper table exactly)
# Source: ALIGNN paper, Table "Regression model performances on QM9"
SCHNET_BENCH = {
    "HOMO":0.0410,"LUMO":0.0340,"Gap":0.0630,"ZPVE":0.00170,
    "mu":0.0330,"alpha":0.2350,"R2":0.073,
    "U0":0.0140,"U298":0.0190,"H298":0.0140,"G298":0.0140,
}
MEGNET_BENCH = {
    "HOMO":0.0430,"LUMO":0.0440,"Gap":0.0660,"ZPVE":0.00143,
    "mu":0.0500,"alpha":0.0810,"R2":0.302,
    "U0":0.0120,"U298":0.0130,"H298":0.0120,"G298":0.0120,
}
DIMENET_BENCH = {
    "HOMO":0.0246,"LUMO":0.0195,"Gap":0.0326,"ZPVE":0.00121,
    "mu":0.0297,"alpha":0.0435,"R2":0.331,
    "U0":0.00632,"U298":0.00628,"H298":0.00653,"G298":0.00756,
}
ALIGNN_BENCH = {
    "HOMO":0.0214,"LUMO":0.0195,"Gap":0.0381,"ZPVE":0.00310,
    "mu":0.0146,"alpha":0.0561,"R2":0.5432,
    "U0":0.0153,"U298":0.0144,"H298":0.0147,"G298":0.0144,
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
# 3.  ELEMENT LOOKUP
# ============================================================
# QM9 contains only H, C, N, O, F — map atomic number → compact index
QM9_ATOMIC_NUM = {1: 1, 6: 2, 7: 3, 8: 4, 9: 5}   # Z → embedding idx

def atomic_num_to_idx(z: int) -> int:
    """Return compact embedding index for a QM9 atomic number."""
    return QM9_ATOMIC_NUM.get(int(z), 1)   # fallback to H index


# Pauling electronegativities for polyhedron features
ATOM_EN = {1: 2.20, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98}


# ============================================================
# 4.  POLYHEDRON / LOCAL-ENVIRONMENT FEATURE VECTOR  (7-dim)
#     Same definition as in the crystal version; valid for molecules
#     because every atom has a well-defined local coordination shell.
# ============================================================
def get_local_env_features(positions: np.ndarray,
                            atomic_nums: np.ndarray,
                            center_idx: int,
                            neighbor_indices: list) -> np.ndarray:
    """
    Compute 7-dim local environment descriptor for atom center_idx
    given its neighbor list (from the radius graph at r_cut=CUTOFF).

    Features:
      0  CN / 12              normalised coordination number
      1  distortion index     std(bond_lengths) / mean(bond_lengths)
      2  bond-angle variance  var(angles) / 10 000
      3  volume proxy         mean_bond_length^3 / 100
      4  mean bond length     mean_bl / CUTOFF
      5  EN mismatch          mean |EN_center - EN_neighbor|
      6  coordination type    min(CN, 12) / 12
    """
    if len(neighbor_indices) == 0:
        return np.zeros(7, dtype=np.float32)

    c_pos   = positions[center_idx]
    c_en    = ATOM_EN.get(int(atomic_nums[center_idx]), 2.0)
    bl_list, en_diff_list, nb_pos_list = [], [], []

    for nb in neighbor_indices:
        nb_pos = positions[nb]
        bl = float(np.linalg.norm(nb_pos - c_pos))
        if bl < 1e-8:
            continue
        bl_list.append(bl)
        en_diff_list.append(abs(c_en - ATOM_EN.get(int(atomic_nums[nb]), 2.0)))
        nb_pos_list.append(nb_pos)

    if len(bl_list) == 0:
        return np.zeros(7, dtype=np.float32)

    bl_arr  = np.array(bl_list, dtype=np.float32)
    cn      = len(bl_arr)
    mean_bl = float(bl_arr.mean())
    di      = float(bl_arr.std() / (mean_bl + 1e-8))

    # Bond-angle variance
    angles = []
    for i in range(cn):
        for j in range(i + 1, cn):
            v1 = nb_pos_list[i] - c_pos
            v2 = nb_pos_list[j] - c_pos
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 < 1e-8 or n2 < 1e-8:
                continue
            cos_a = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
            angles.append(np.degrees(np.arccos(cos_a)))
    bav = float(np.var(angles)) if len(angles) > 1 else 0.0

    vol_proxy   = float(mean_bl ** 3)
    en_mismatch = float(np.mean(en_diff_list)) if en_diff_list else 0.0
    ct_enc      = min(cn, 12) / 12.0

    return np.array([
        cn / 12.0,
        di,
        bav / 10000.0,
        vol_proxy / 100.0,
        mean_bl / CUTOFF,
        en_mismatch,
        ct_enc,
    ], dtype=np.float32)


# ============================================================
# 5.  MOLECULAR DUAL GRAPH BUILDER
#     Builds atom graph G_A, line graph G_L, and poly graph G_P
#     from a PyG QM9 Data object (which already contains positions).
# ============================================================
def build_molecular_graphs(mol_data) -> dict:
    """
    Input
    -----
    mol_data : torch_geometric.data.Data
        QM9 molecule with fields:
          pos      (N, 3)  float  3D coordinates in Ångström
          z        (N,)    long   atomic numbers
          y        (1, 19) float  QM9 properties

    Returns
    -------
    dict with keys:
      atom_z, atom_ei, atom_dist,
      line_ei, line_angles,
      poly_x, poly_ei, poly_ea,
      n_atoms, n_edges,
      targets   np.ndarray (N_PROPS,)
    """
    pos = mol_data.pos.numpy().astype(np.float64)    # (N, 3)
    zs  = mol_data.z.numpy().astype(np.int64)        # (N,)
    N   = len(zs)

    # ── compact embedding indices ────────────────────────────────────────────
    atom_z = np.array([atomic_num_to_idx(z) for z in zs], dtype=np.int64)

    # ── atom graph: radius graph at r_cut ────────────────────────────────────
    a_src, a_dst, a_dist = [], [], []
    neighbor_map = {i: [] for i in range(N)}   # i → [j, ...]

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            d = float(np.linalg.norm(pos[i] - pos[j]))
            if d < CUTOFF:
                a_src.append(i)
                a_dst.append(j)
                a_dist.append(d)
                neighbor_map[i].append(j)

    # Fallback: connect to nearest neighbour if isolated
    if not a_src:
        for i in range(N):
            dists = [(float(np.linalg.norm(pos[i] - pos[j])), j)
                     for j in range(N) if j != i]
            if dists:
                d, j = min(dists)
                a_src += [i, j]; a_dst += [j, i]; a_dist += [d, d]
                neighbor_map[i].append(j); neighbor_map[j].append(i)

    E        = len(a_src)
    atom_ei  = np.array([a_src, a_dst], dtype=np.int64)
    atom_dist = np.array(a_dist,        dtype=np.float32)

    # ── line graph: bond-pair triplets (angle encoding) ──────────────────────
    incoming = {i: [] for i in range(N)}
    for eid, (src, dst) in enumerate(zip(a_src, a_dst)):
        incoming[dst].append((src, eid))

    lg_src, lg_dst, lg_angles = [], [], []
    for i in range(N):
        nbrs = incoming[i]
        for p in range(len(nbrs)):
            for q in range(len(nbrs)):
                if p == q:
                    continue
                src_p, eid_p = nbrs[p]
                src_q, eid_q = nbrs[q]
                v1 = pos[src_p] - pos[i]
                v2 = pos[src_q] - pos[i]
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if n1 < 1e-8 or n2 < 1e-8:
                    continue
                cos_a = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
                lg_src.append(eid_p)
                lg_dst.append(eid_q)
                lg_angles.append(float(np.arccos(cos_a)))

    if lg_src:
        line_ei     = np.array([lg_src, lg_dst], dtype=np.int64)
        line_angles = np.array(lg_angles,         dtype=np.float32)
    else:
        line_ei     = np.zeros((2, 0), dtype=np.int64)
        line_angles = np.zeros(0,      dtype=np.float32)

    # ── poly graph: local coordination environment graph ─────────────────────
    poly_x = np.zeros((N, N_POLY_FEAT), dtype=np.float32)
    for i in range(N):
        poly_x[i] = get_local_env_features(pos, zs, i, neighbor_map[i])

    # Connect polyhedral nodes when they share ≥1 common neighbour
    p_src, p_dst, p_ea = [], [], []
    for i in range(N):
        ni = set(neighbor_map[i])
        for j in range(i + 1, N):
            shared = ni & set(neighbor_map[j])
            ns = len(shared)
            if ns == 0:
                continue
            ct = 0.0 if ns == 1 else (0.5 if ns == 2 else 1.0)
            p_src += [i, j]; p_dst += [j, i]; p_ea += [[ct], [ct]]

    if not p_src:
        # Fallback: connect each atom to its nearest neighbour
        for i in range(N):
            if neighbor_map[i]:
                j = min(neighbor_map[i],
                        key=lambda nb: np.linalg.norm(pos[i] - pos[nb]))
                p_src += [i, j]; p_dst += [j, i]; p_ea += [[0.5], [0.5]]

    poly_ei = (np.array([p_src, p_dst], dtype=np.int64)
               if p_src else np.zeros((2, 0), dtype=np.int64))
    poly_ea = (np.array(p_ea, dtype=np.float32)
               if p_ea else np.zeros((0, 1),  dtype=np.float32))

    # ── extract QM9 targets ──────────────────────────────────────────────────
    # Fix 1: atomisation energy for U0/U298/H298/G298
    # Fix 2: unit scaling so all targets are O(1–1000) not O(1000)
    raw_y = mol_data.y[0].numpy()     # (19,) float  — all 19 QM9 properties

    # Compute total atomic reference energy for this molecule (Hartree)
    atom_ref_total_har = sum(
        ATOM_REF_ENERGIES_HAR.get(int(z), 0.0) for z in zs
    )

    targets = np.zeros(N_PROPS, dtype=np.float32)
    for k, (short, label, unit, pyg_idx, act, primary) in enumerate(QM9_PROPS):
        val = float(raw_y[pyg_idx])

        # NOTE: PyG qm9_v3.zip pre-processed stores values ALREADY in eV
        # (confirmed by diagnostic: HOMO raw=-10.55 eV, U0 raw=-1101 eV)
        # DO NOT multiply by HARTREE_TO_EV — values are already in eV/D/Bohr³

        # Step 1: atomisation correction for U0/U298/H298/G298 only
        # raw value = total molecular energy in eV (~-1100 eV)
        # subtract atomic reference energies (converted Har→eV) to get
        # atomisation energy (~-5 to -50 eV) matching published benchmarks
        if pyg_idx in ATOMISATION_INDICES_PYG:
            val -= atom_ref_total_har * HARTREE_TO_EV

        # Step 2: unit scaling (1.0 for all — train in eV/D/Bohr³)
        val *= PROP_SCALE[short]

        targets[k] = val

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
        "targets"    : targets,   # (N_PROPS,)
    }


# ============================================================
# 6.  PyG DATA OBJECT
# ============================================================
def graph_to_pyg(graph: dict, mol_id: int = 0) -> Data:
    """Pack a molecular graph dict into a PyG Data object."""
    data = Data(
        x          = torch.tensor(graph["atom_z"],    dtype=torch.long),
        edge_index = torch.tensor(graph["atom_ei"],   dtype=torch.long),
        edge_dist  = torch.tensor(graph["atom_dist"], dtype=torch.float32),
        y_primary  = torch.tensor([graph["targets"][PRIMARY_IDX]],
                                  dtype=torch.float32),          # HOMO scalar
        y_aux      = torch.tensor(
                         np.delete(graph["targets"], PRIMARY_IDX),
                         dtype=torch.float32),                   # (N_AUX,)
        num_nodes  = graph["n_atoms"],
    )
    data._poly_x_np      = graph["poly_x"]
    data._poly_ei_np     = graph["poly_ei"]
    data._poly_ea_np     = graph["poly_ea"]
    data._line_ei_np     = graph["line_ei"]
    data._line_angles_np = graph["line_angles"]
    data._n_edges        = graph["n_edges"]
    data.mol_id          = mol_id
    return data


# ============================================================
# 7.  DATASET BUILDER
# ============================================================
def build_dataset(qm9_subset, label: str = "") -> list:
    """
    Build PyG graph list from a list/slice of QM9 Data objects.
    Skips molecules that fail graph construction.
    """
    ds, skip = [], 0
    n = len(qm9_subset)
    print(f"\n  [{label}] Building molecular graphs for {n:,} molecules ...")
    t0 = time.time()
    for i, mol in enumerate(qm9_subset):
        try:
            g    = build_molecular_graphs(mol)
            data = graph_to_pyg(g, mol_id=i)
            ds.append(data)
        except Exception as exc:
            skip += 1
            if skip <= 5:
                print(f"    skip mol {i}: {exc}")
        if (i + 1) % 10000 == 0 or (i + 1) == n:
            print(f"    ... {i+1:,}/{n:,}  ({time.time()-t0:.0f}s)")
    print(f"  [{label}] Built {len(ds):,}  skipped {skip}")
    return ds


# ============================================================
# 8.  MESSAGE PASSING LAYERS
# ============================================================
class CPGNConv(MessagePassing):
    """Atom or polyhedron graph convolution (MeanAgg + MLP + LayerNorm)."""
    def __init__(self, node_dim: int, edge_dim: int, hidden: int):
        super().__init__(aggr="mean")
        self.msg_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, node_dim),
        )
        self.norm = nn.LayerNorm(node_dim)

    def forward(self, x, edge_index, edge_attr):
        if edge_index.shape[1] == 0:
            return x
        return self.norm(x + self.propagate(edge_index, x=x, edge_attr=edge_attr))

    def message(self, x_i, x_j, edge_attr):
        return self.msg_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))


class LineGraphConv(MessagePassing):
    """Angle-aware bond update via message passing on G_L."""
    def __init__(self, edge_dim: int, angle_dim: int, hidden: int):
        super().__init__(aggr="mean")
        self.msg_mlp = nn.Sequential(
            nn.Linear(edge_dim * 2 + angle_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, edge_dim),
        )
        self.norm = nn.LayerNorm(edge_dim)

    def forward(self, x, edge_index, edge_attr):
        if edge_index.shape[1] == 0:
            return x
        return self.norm(x + self.propagate(edge_index, x=x, edge_attr=edge_attr))

    def message(self, x_i, x_j, edge_attr):
        return self.msg_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))


# ============================================================
# 9.  CROSS-ATTENTION  (atom ↔ local environment)
# ============================================================
class CrossAttention(nn.Module):
    """
    Scalar-gated bidirectional cross-attention coupling the atom stream
    and the local-environment (poly) stream.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.q_a   = nn.Linear(dim, dim); self.k_p = nn.Linear(dim, dim)
        self.v_p   = nn.Linear(dim, dim); self.q_p = nn.Linear(dim, dim)
        self.k_a   = nn.Linear(dim, dim); self.v_a = nn.Linear(dim, dim)
        self.norm_a = nn.LayerNorm(dim)
        self.norm_p = nn.LayerNorm(dim)
        self.scale  = dim ** -0.5

    @staticmethod
    def _pool(h: torch.Tensor, batch: torch.Tensor,
              B: int, dev: torch.device) -> torch.Tensor:
        ctx   = torch.zeros(B, h.shape[1], device=dev, dtype=h.dtype)
        count = torch.zeros(B, 1,          device=dev, dtype=h.dtype)
        idx   = batch.long().unsqueeze(1).expand_as(h)
        ctx.scatter_add_(0, idx, h)
        count.scatter_add_(0, batch.long().unsqueeze(1),
                           torch.ones(h.shape[0], 1, device=dev, dtype=h.dtype))
        return ctx / (count + 1e-8)

    def forward(self, ah, ph, a_batch, p_batch):
        B   = int(a_batch.max().item()) + 1
        dev = ah.device

        # Atom queries → poly context
        pc     = self._pool(ph, p_batch, B, dev)
        cpa    = pc[a_batch.long()]
        w_a    = torch.sigmoid(
            (self.q_a(ah) * self.k_p(cpa)).sum(-1, keepdim=True) * self.scale)
        ah_new = self.norm_a(ah + w_a * self.v_p(cpa))

        # Poly queries → atom context
        ac     = self._pool(ah, a_batch, B, dev)
        cpp    = ac[p_batch.long()]
        w_p    = torch.sigmoid(
            (self.q_p(ph) * self.k_a(cpp)).sum(-1, keepdim=True) * self.scale)
        ph_new = self.norm_p(ph + w_p * self.v_a(cpp))

        return ah_new, ph_new


# ============================================================
# 10.  CPGN MODEL  (QM9 multi-property)
# ============================================================
class CPGN(nn.Module):
    """
    Crystal Polyhedron Graph Network adapted for QM9 molecular property
    prediction.

    Outputs
    -------
    {
      "primary" : (B,)         HOMO prediction
      "aux"     : (B, N_AUX)  remaining 10 QM9 properties
    }
    """
    def __init__(self):
        super().__init__()
        H = HIDDEN_DIM

        # ── Encoders ─────────────────────────────────────────────────────────
        self.elem_embed     = nn.Embedding(N_ELEM, ELEM_DIM, padding_idx=0)
        self.dist_rbf       = RBFExpansion(0.0, CUTOFF,    RBF_CENTRES)
        self.angle_rbf      = RBFExpansion(0.0, math.pi,   RBF_CENTRES)

        self.atom_in        = nn.Sequential(
            nn.Linear(ELEM_DIM, H), nn.SiLU(), nn.Linear(H, H))
        self.edge_in        = nn.Sequential(
            nn.Linear(N_EDGE_FEAT, H), nn.SiLU())
        self.poly_in        = nn.Sequential(
            nn.Linear(N_POLY_FEAT, H), nn.SiLU(), nn.Linear(H, H))
        self.poly_edge_proj = nn.Linear(1, H)

        # ── Interleaved message-passing layers ────────────────────────────────
        self.atom_convs  = nn.ModuleList(
            [CPGNConv(H, H, H)                for _ in range(N_LAYERS)])
        self.line_convs  = nn.ModuleList(
            [LineGraphConv(H, N_ANGLE_FEAT, H) for _ in range(N_LAYERS)])
        self.poly_convs  = nn.ModuleList(
            [CPGNConv(H, H, H)                for _ in range(N_LAYERS)])
        self.cross_attns = nn.ModuleList(
            [CrossAttention(H)                 for _ in range(N_LAYERS)])
        self.dropout     = nn.Dropout(p=0.1)

        # ── Feature fusion ────────────────────────────────────────────────────
        self.fusion = nn.Sequential(
            nn.Linear(H * 2, H), nn.SiLU(),
            nn.Linear(H, PRED_DIM), nn.SiLU(),
        )

        # ── Output heads ──────────────────────────────────────────────────────
        # Primary: HOMO  (can be negative → no activation)
        self.head_primary = nn.Linear(PRED_DIM, 1)

        # Auxiliary: one head per remaining QM9 property
        aux_props = [(s, l, u, pyg, act, pri)
                     for (s, l, u, pyg, act, pri) in QM9_PROPS
                     if not pri]
        self.aux_heads      = nn.ModuleList([nn.Linear(PRED_DIM, 1)
                                             for _ in aux_props])
        self._aux_acts      = [act for (_, _, _, _, act, _) in aux_props]

    def forward(self, batch):
        dev = batch.x.device

        # Node / edge initial features
        atom_h   = self.atom_in(self.elem_embed(batch.x.to(dev)))
        dist_rbf = self.dist_rbf(batch.edge_dist.to(dev))
        bond_h   = self.edge_in(dist_rbf)
        poly_h   = self.poly_in(batch.poly_x.to(dev))

        p_ei  = batch.poly_ei.to(dev)
        p_ea  = batch.poly_ea.to(dev)
        p_ea_p = (self.poly_edge_proj(p_ea)
                  if p_ea.shape[0] > 0
                  else torch.zeros(0, HIDDEN_DIM, device=dev, dtype=atom_h.dtype))

        line_ei    = batch.line_ei.to(dev)
        angle_rbf  = self.angle_rbf(batch.line_angles.to(dev))
        a_batch    = batch.batch

        # L interleaved layers
        for ac, lc, pc, ca in zip(self.atom_convs, self.line_convs,
                                   self.poly_convs, self.cross_attns):
            atom_h          = self.dropout(ac(atom_h, batch.edge_index, bond_h))
            bond_h          = self.dropout(lc(bond_h, line_ei, angle_rbf))
            poly_h          = self.dropout(pc(poly_h, p_ei, p_ea_p))
            atom_h, poly_h  = ca(atom_h, poly_h, a_batch, a_batch)

        # Global pooling + fusion
        ha = global_mean_pool(atom_h, a_batch)
        hp = global_mean_pool(poly_h, a_batch)
        z  = self.fusion(torch.cat([ha, hp], dim=-1))

        # Primary prediction (HOMO)
        primary_pred = self.head_primary(z).squeeze(-1)

        # Auxiliary predictions
        aux_preds = []
        for head, act in zip(self.aux_heads, self._aux_acts):
            p = head(z).squeeze(-1)
            if act == "softplus":
                p = F.softplus(p)
            elif act == "sigmoid":
                p = torch.sigmoid(p)
            aux_preds.append(p)
        aux_pred = torch.stack(aux_preds, dim=-1)   # (B, N_AUX)

        return {"primary": primary_pred, "aux": aux_pred}


# ============================================================
# 11.  LOSS
#      Primary : MAE on HOMO
#      Auxiliary: MAE on remaining 10 properties (all always available in QM9)
# ============================================================
class CPGNLoss(nn.Module):
    def forward(self, preds, batch):
        loss_primary = F.l1_loss(preds["primary"], batch.y_primary.squeeze(-1))
        loss_aux     = F.l1_loss(preds["aux"],     batch.y_aux)
        return loss_primary + LAMBDA_AUX * loss_aux


# ============================================================
# 12.  COLLATE / DATALOADER
# ============================================================
def custom_collate(data_list):
    batch     = PyGBatch.from_data_list(data_list)
    poly_x_l  = []; poly_ei_l = []; poly_ea_l = []
    line_ei_l = []; line_ang_l = []
    atom_cum = edge_cum = 0

    for d in data_list:
        n_a = d.num_nodes
        n_e = d._n_edges
        poly_x_l.append(torch.tensor(d._poly_x_np,  dtype=torch.float32))
        poly_ea_l.append(torch.tensor(d._poly_ea_np, dtype=torch.float32))
        ei_p = d._poly_ei_np
        if ei_p.shape[1] > 0:
            poly_ei_l.append(torch.tensor(ei_p + atom_cum, dtype=torch.long))
        lei = d._line_ei_np
        if lei.shape[1] > 0:
            line_ei_l.append(torch.tensor(lei + edge_cum, dtype=torch.long))
        line_ang_l.append(torch.tensor(d._line_angles_np, dtype=torch.float32))
        atom_cum += n_a
        edge_cum += n_e

    batch.poly_x      = torch.cat(poly_x_l,  dim=0)
    batch.poly_ei     = (torch.cat(poly_ei_l, dim=1)
                         if poly_ei_l else torch.zeros(2, 0, dtype=torch.long))
    batch.poly_ea     = torch.cat(poly_ea_l, dim=0)
    batch.line_ei     = (torch.cat(line_ei_l, dim=1)
                         if line_ei_l else torch.zeros(2, 0, dtype=torch.long))
    batch.line_angles = torch.cat(line_ang_l, dim=0)
    batch.y_primary   = torch.cat([d.y_primary for d in data_list])
    batch.y_aux       = torch.stack([d.y_aux   for d in data_list])   # (B, N_AUX)
    batch.edge_dist   = torch.cat([d.edge_dist for d in data_list], dim=0)
    return batch


def make_loader(data_list, batch_size: int, shuffle: bool):
    return TorchLoader(
        data_list, batch_size=batch_size, shuffle=shuffle,
        collate_fn=custom_collate, drop_last=False,
        num_workers=4 if DEVICE.startswith("cuda") else 0,
        pin_memory=DEVICE.startswith("cuda"),
        persistent_workers=DEVICE.startswith("cuda"),
    )


# ============================================================
# 13.  TRAINING LOOP
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
    tr_mae_h, vl_mae_h       = [], []
    tr_loss_h, vl_loss_h     = [], []

    n_p = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*70}")
    print("  CPGN QM9 — Multi-property molecular training")
    print(f"  Parameters    : {n_p:,}")
    print(f"  Train / Val   : {len(train_ds):,} / {len(val_ds):,}")
    print(f"  Primary       : HOMO  |  Auxiliary: {N_AUX} targets (λ={LAMBDA_AUX})")
    print(f"  Epochs        : {EPOCHS}  |  Batch : {BATCH_SIZE}  |  LR : {LEARNING_RATE}")
    print(f"  Checkpoint    : best val HOMO MAE (eV) → {CHECKPOINT}")
    print(f"{'='*70}")
    print(f"  {'Ep':>5} | {'TrLoss':>8} | {'VlLoss':>8} | "
          f"{'TrMAE_HOMO':>11} | {'VlMAE_HOMO':>11} | {'LR':>9} | {'s':>5}")
    print(f"  {'-'*68}")

    t0_total = time.time()
    for ep in range(1, EPOCHS + 1):
        model.train()
        tl = tm = n = 0; t0 = time.time()
        for b in tr_loader:
            b = b.to(DEVICE)
            optimiser.zero_grad(set_to_none=True)
            p  = model(b)
            ls = criterion(p, b)
            if not torch.isfinite(ls):
                continue
            ls.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            bs  = len(b.y_primary)
            tl += ls.item() * bs
            tm += F.l1_loss(p["primary"].detach(),
                            b.y_primary.squeeze(-1)).item() * bs
            n  += bs
        n = max(n, 1); tl /= n; tm /= n
        scheduler.step()

        model.eval()
        vl = vm = nv = 0
        with torch.no_grad():
            for b in vl_loader:
                b = b.to(DEVICE); p = model(b); bs = len(b.y_primary)
                vl += criterion(p, b).item() * bs
                vm += F.l1_loss(p["primary"],
                                b.y_primary.squeeze(-1)).item() * bs
                nv += bs
        nv = max(nv, 1); vl /= nv; vm /= nv

        tr_loss_h.append(tl); vl_loss_h.append(vl)
        tr_mae_h.append(tm);  vl_mae_h.append(vm)

        lr_now = optimiser.param_groups[0]["lr"]
        if ep % 10 == 0 or ep <= 5:
            print(f"  {ep:>5} | {tl:>8.4f} | {vl:>8.4f} | "
                  f"{tm:>12.5f} | {vm:>12.5f} | {lr_now:>9.2e} | "
                  f"{time.time()-t0:>4.1f}s")

        if vm < best_val_mae - 1e-6:
            best_val_mae = vm
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            torch.save({
                "model_state" : best_state,
                "epoch"       : ep,
                "val_mae_homo": best_val_mae,
                "config": {
                    "HIDDEN_DIM": HIDDEN_DIM,
                    "N_LAYERS"  : N_LAYERS,
                    "PRED_DIM"  : PRED_DIM,
                    "ELEM_DIM"  : ELEM_DIM,
                },
            }, CHECKPOINT)
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"  Early stopping at ep {ep}  "
                      f"(best val HOMO MAE = {best_val_mae:.2f} meV)")
                break

    elapsed = time.time() - t0_total
    print(f"\n  Total : {elapsed:.1f}s  |  Best val HOMO MAE = {best_val_mae:.4f} eV")
    if best_state:
        model.load_state_dict(best_state)
    return model, tr_loss_h, vl_loss_h, tr_mae_h, vl_mae_h


# ============================================================
# 14.  LOAD CHECKPOINT
# ============================================================
def load_checkpoint(path: str) -> CPGN:
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
    ep  = ckpt.get("epoch", "?")
    mae = ckpt.get("val_mae_homo", float("nan"))
    print(f"  Loaded checkpoint: {path}  (ep {ep}, val HOMO MAE = {mae:.5f} eV)")
    return model


# ============================================================
# 15.  INFERENCE
# ============================================================
@torch.no_grad()
def predict_set(model: CPGN, dataset: list, label: str = "") -> dict:
    model.eval()
    loader = make_loader(dataset, batch_size=128, shuffle=False)

    primary_true, primary_pred = [], []
    aux_true_list, aux_pred_list = [], []

    for b in loader:
        b = b.to(DEVICE)
        out = model(b)
        primary_true.extend(b.y_primary.squeeze(-1).cpu().tolist())
        primary_pred.extend(out["primary"].detach().cpu().tolist())
        aux_true_list.append(b.y_aux.cpu().numpy())
        aux_pred_list.append(out["aux"].detach().cpu().numpy())

    print(f"  [{label}] {len(primary_true):,} predictions")
    return {
        "primary_true": np.array(primary_true),
        "primary_pred": np.array(primary_pred),
        "aux_true"    : np.vstack(aux_true_list),   # (N, N_AUX)
        "aux_pred"    : np.vstack(aux_pred_list),   # (N, N_AUX)
    }


# ============================================================
# 16.  METRICS
# ============================================================
def compute_metrics(res: dict, label: str = "") -> dict:
    # Primary: HOMO  (values are already in meV after Fix 2)
    # All targets are in their benchmark units (meV / D / Bohr³)
    p_mae  = mean_absolute_error(res["primary_true"], res["primary_pred"])
    p_rmse = float(np.sqrt(mean_squared_error(res["primary_true"], res["primary_pred"])))
    p_mad  = float(np.mean(np.abs(res["primary_true"] - res["primary_true"].mean())))

    hdr = "=" * 72
    print(f"\n{hdr}")
    print(f"  CPGN QM9 — {label} Results")
    print(hdr)
    print(f"  Samples : {len(res['primary_true']):,}")
    print("\n  All 11 QM9 properties — MAE (eV / D / Bohr³)")
    print("  Matches ALIGNN paper: "
          "'Regression model performances on QM9 dataset'")
    print(f"  {'Target':<7} {'Unit':<7} {'CPGN':>8} "
          f"{'ALIGNN':>8} {'DimNet++':>9} {'MEGNet':>8} {'SchNet':>8}")
    print(f"  {'-'*60}")

    # Print HOMO (primary) first
    a_h = ALIGNN_BENCH.get("HOMO", float("nan"))
    d_h = DIMENET_BENCH.get("HOMO", float("nan"))
    m_h = MEGNET_BENCH.get("HOMO",  float("nan"))
    s_h = SCHNET_BENCH.get("HOMO",  float("nan"))
    print(f"  {'HOMO':<7} {'eV':<7} {p_mae:>8.4f} "
          f"{a_h:>8.4f} {d_h:>9.4f} {m_h:>8.4f} {s_h:>8.4f}")

    # Auxiliary properties
    aux_props = [(s, l, u, pyg, act, pri)
                 for (s, l, u, pyg, act, pri) in QM9_PROPS if not pri]
    aux_maes  = {}

    for idx, (short, label_p, unit_p, pyg_idx, act, pri) in enumerate(aux_props):
        mae_k = mean_absolute_error(res["aux_true"][:, idx],
                                    res["aux_pred"][:, idx])
        aux_maes[short] = mae_k
        unit_lbl = PROP_UNIT_LABEL.get(short, unit_p)
        a_b = ALIGNN_BENCH.get(short,   float("nan"))
        d_b = DIMENET_BENCH.get(short,  float("nan"))
        m_b = MEGNET_BENCH.get(short,   float("nan"))
        s_b = SCHNET_BENCH.get(short,   float("nan"))
        print(f"  {short:<7} {unit_lbl:<7} {mae_k:>8.4f} "
              f"{a_b:>8.4f} {d_b:>9.4f} {m_b:>8.4f} {s_b:>8.4f}")

    print(hdr)
    return {
        "primary_mae" : p_mae,
        "primary_rmse": p_rmse,
        "primary_mad" : p_mad,
        "aux_maes"    : aux_maes,
    }


# ============================================================
# 17.  SAVE CSV
# ============================================================
def save_csv(res: dict, dataset: list, path: str):
    aux_props = [(s, l, u, pyg, act, pri)
                 for (s, l, u, pyg, act, pri) in QM9_PROPS if not pri]
    rows = []
    for i in range(len(res["primary_true"])):
        mol_id = dataset[i].mol_id if hasattr(dataset[i], "mol_id") else i
        row = {
            "mol_id"      : mol_id,
            "true_HOMO"   : res["primary_true"][i],
            "pred_HOMO"   : res["primary_pred"][i],
            "err_HOMO"    : res["primary_pred"][i] - res["primary_true"][i],
        }
        for j, (short, *_) in enumerate(aux_props):
            row[f"true_{short}"] = res["aux_true"][i, j]
            row[f"pred_{short}"] = res["aux_pred"][i, j]
            row[f"err_{short}"]  = res["aux_pred"][i, j] - res["aux_true"][i, j]
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  Saved: {path}")


# ============================================================
# 18.  PLOTS
# ============================================================
BLUE   = "#2A7EC0"; CORAL  = "#E05C2A"; GREEN  = "#1D9E75"
PURPLE = "#534AB7"; DARK   = "#2C2C2A"; GRAY   = "#888780"
AMBER  = "#BA7517"; TEAL   = "#0F6E56"


def plot_training_curve(tr_loss, vl_loss, tr_mae, vl_mae):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ep = range(1, len(tr_loss) + 1)

    axes[0].plot(ep, tr_loss, lw=1.5, color=BLUE,  label="Train loss")
    axes[0].plot(ep, vl_loss, lw=1.5, color=CORAL, ls="--", label="Val loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("CPGN QM9 — Total Loss  (MAE HOMO + λ·aux MAE)")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(ep, tr_mae, lw=1.5, color=BLUE,  label="Train MAE HOMO")
    axes[1].plot(ep, vl_mae, lw=1.5, color=CORAL, ls="--", label="Val MAE HOMO")
    for v, lbl, col in [
        (0.0412, "SchNet",    GRAY),
        (0.0244, "DimeNet++", PURPLE),
        (0.0230, "SphereNet", GREEN),
    ]:
        axes[1].axhline(v, color=col, lw=0.8, ls=":", alpha=0.8, label=lbl)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("HOMO MAE (eV)")
    axes[1].set_title("CPGN QM9 — HOMO MAE vs Epoch")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

    plt.suptitle(
        f"CPGN Training — QM9  ({N_TRAIN:,}/{N_VAL:,}/test shuffled, SEED={SEED})",
        fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("CPGN_QM9_training_curve.png", dpi=300); plt.close()
    print("  Saved: CPGN_QM9_training_curve.png")


def plot_parity(val_res, test_res, val_m, test_m):
    """Parity plots for HOMO (primary) and Gap (most-compared auxiliary)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Find Gap index in auxiliary results
    aux_shorts = [s for (s, l, u, p, a, pr) in QM9_PROPS if not pr]
    gap_idx    = aux_shorts.index("Gap") if "Gap" in aux_shorts else 0

    datasets = [
        (val_res,  val_m,  "Validation"),
        (test_res, test_m, "Test"),
    ]
    for col, (res, m, split) in enumerate(datasets):
        # HOMO parity
        ax = axes[0][col]
        lo = min(res["primary_true"].min(), res["primary_pred"].min()) - 0.1
        hi = max(res["primary_true"].max(), res["primary_pred"].max()) + 0.1
        ax.scatter(res["primary_true"], res["primary_pred"],
                   s=2, alpha=0.3, color=BLUE, rasterized=True)
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax.set_xlabel("True HOMO (eV)"); ax.set_ylabel("Pred HOMO (eV)")
        ax.set_title(f"HOMO — {split}  MAE={m['primary_mae']:.4f} eV")
        ax.grid(alpha=0.3)

        # Gap parity
        ax = axes[1][col]
        gt = res["aux_true"][:, gap_idx]; gp = res["aux_pred"][:, gap_idx]
        lo = min(gt.min(), gp.min()) - 0.1
        hi = max(gt.max(), gp.max()) + 0.1
        ax.scatter(gt, gp, s=2, alpha=0.3, color=CORAL, rasterized=True)
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        gap_mae = m["aux_maes"].get("Gap", float("nan"))
        ax.set_xlabel("True Gap (eV)"); ax.set_ylabel("Pred Gap (eV)")
        ax.set_title(f"HOMO–LUMO Gap — {split}  MAE={gap_mae:.4f} eV")
        ax.grid(alpha=0.3)

    plt.suptitle("CPGN QM9 — Parity Plots", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("CPGN_QM9_parity_plots.png", dpi=300); plt.close()
    print("  Saved: CPGN_QM9_parity_plots.png")


def plot_property_maes(test_m: dict):
    """Horizontal bar chart of MAE for all 11 properties."""
    names, maes, units_list = [], [], []
    # Primary
    names.append("HOMO"); maes.append(test_m["primary_mae"]); units_list.append("eV")
    # Auxiliary
    aux_props = [(s, l, u, p, a, pr) for (s, l, u, p, a, pr) in QM9_PROPS if not pr]
    for short, label_p, unit, *_ in aux_props:
        names.append(short)
        maes.append(test_m["aux_maes"].get(short, float("nan")))
        units_list.append(unit)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = [BLUE if n == "HOMO" else CORAL for n in names]
    bars = ax.barh(names, maes, color=colors, alpha=0.85, edgecolor="k", lw=0.5)
    for bar, v, u in zip(bars, maes, units_list):
        if not math.isnan(v):
            ax.text(v + max(maes) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{v:.4f} {u}", va="center", fontsize=9)
    ax.set_xlabel("MAE (property units)", fontsize=11)
    ax.set_title("CPGN QM9 — Test MAE for all 11 properties", fontsize=12)
    ax.invert_yaxis(); ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig("CPGN_QM9_property_maes.png", dpi=300); plt.close()
    print("  Saved: CPGN_QM9_property_maes.png")


def plot_error_hist(val_res: dict, test_res: dict):
    """Error distribution histograms for HOMO, LUMO, Gap, ZPVE."""
    aux_shorts = [s for (s, l, u, p, a, pr) in QM9_PROPS if not pr]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    panels = [
        ("HOMO",  "primary", None,   BLUE),
        ("LUMO",  "aux",     aux_shorts.index("LUMO"),  CORAL),
        ("Gap",   "aux",     aux_shorts.index("Gap"),   GREEN),
        ("ZPVE",  "aux",     aux_shorts.index("ZPVE"),  PURPLE),
    ]
    for ax, (name, src, idx, col) in zip(axes.flat, panels):
        for res, ls, lbl in [
            (val_res, "-",  "Validation"),
            (test_res,"--", "Test"),
        ]:
            if src == "primary":
                err = res["primary_pred"] - res["primary_true"]
            else:
                err = res["aux_pred"][:, idx] - res["aux_true"][:, idx]
            mae = float(np.mean(np.abs(err)))
            ax.hist(err, bins=80, alpha=0.55, color=col,
                    ls=ls, label=f"{lbl}  MAE={mae:.4f}", density=True)
        ax.axvline(0, color="k", lw=0.8, ls=":")
        unit_lbl = PROP_UNIT_LABEL.get(name, "units")
        ax.set_xlabel(f"Error ({unit_lbl})")
        ax.set_ylabel("Density"); ax.set_title(name)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.suptitle("CPGN QM9 — Prediction Error Distributions",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("CPGN_QM9_error_hist.png", dpi=300); plt.close()
    print("  Saved: CPGN_QM9_error_hist.png")


def plot_benchmark_table(test_m: dict):
    """Bar chart comparing CPGN with SchNet, DimeNet++, SphereNet."""
    props_to_plot = ["HOMO", "LUMO", "Gap", "ZPVE", "mu", "alpha"]
    sph_bench = {
        "HOMO": 0.0232, "LUMO": 0.0178, "Gap": 0.0306,
        "ZPVE": 0.00115,"mu":   0.0267, "alpha": 0.0408,
    }

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, prop in zip(axes.flat, props_to_plot):
        if prop == "HOMO":
            cpgn_val = test_m["primary_mae"]
        else:
            cpgn_val = test_m["aux_maes"].get(prop, float("nan"))

        bench = {
            "SchNet"    : SCHNET_BENCH.get(prop, float("nan")),
            "DimeNet++" : DIMENET_BENCH.get(prop, float("nan")),
            "SphereNet" : sph_bench.get(prop,    float("nan")),
        }
        models = list(bench.keys()) + ["CPGN (ours)"]
        vals   = list(bench.values()) + [cpgn_val]
        cols   = [GRAY, GRAY, GRAY, CORAL]
        bars   = ax.barh(models, vals, color=cols, alpha=0.85,
                         edgecolor="k", lw=0.5)
        for bar, v in zip(bars, vals):
            if not math.isnan(v):
                ax.text(v + max(v for v in vals if not math.isnan(v)) * 0.01,
                        bar.get_y() + bar.get_height() / 2,
                        f"{v:.4f}", va="center", fontsize=9)
        unit = next((u for (s, l, u, p, a, pr) in QM9_PROPS if s == prop), "")
        ax.set_xlabel(f"MAE ({unit})", fontsize=10)
        ax.set_title(prop, fontsize=11)
        ax.invert_yaxis(); ax.grid(axis="x", alpha=0.3)

    plt.suptitle("CPGN vs SchNet / DimeNet++ / SphereNet — QM9 Benchmark",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("CPGN_QM9_benchmark_table.png", dpi=300); plt.close()
    print("  Saved: CPGN_QM9_benchmark_table.png")


# ============================================================
# 19.  FINAL SUMMARY TABLE
# ============================================================
def print_final_summary(val_m: dict, test_m: dict):
    print("\n" + "=" * 72)
    print("  CPGN QM9 — Final Benchmark Summary")
    print("  (compare against Table 1 in DimeNet++ / SchNet papers)")
    print("=" * 72)
    print(f"  {'Property':<8} {'Unit':<7} {'CPGN MAE':>10} "
          f"{'DimeNet++':>11} {'SchNet':>8}")
    print(f"  {'-'*48}")

    # Primary: HOMO
    print(f"  {'HOMO':<8} {'eV':<7} {test_m['primary_mae']:>8.4f} "
          f"{'0.0244':>11} {'0.0412':>8}")

    aux_props = [(s, l, u, p, a, pr) for (s, l, u, p, a, pr) in QM9_PROPS if not pr]
    for short, label_p, unit, *_ in aux_props:
        mae   = test_m["aux_maes"].get(short, float("nan"))
        mae_s = f"{mae:.4f}" if not math.isnan(mae) else "—"
        d_ref = f"{DIMENET_BENCH.get(short, float('nan')):.4f}" \
                if short in DIMENET_BENCH else "—"
        s_ref = f"{SCHNET_BENCH.get(short, float('nan')):.4f}"  \
                if short in SCHNET_BENCH  else "—"
        print(f"  {short:<8} {unit:<7} {mae_s:>10} {d_ref:>11} {s_ref:>8}")

    print("=" * 72)
    print(f"\n  Split   : Train={N_TRAIN:,} | Val={N_VAL:,} | Test=remainder  (SEED={SEED})")
    print(f"  Val  HOMO MAE : {val_m['primary_mae']:.4f} eV")
    print(f"  Test HOMO MAE : {test_m['primary_mae']:.4f} eV")
    print("=" * 72)


# ============================================================
# 20.  MAIN
# ============================================================
def main():
    print("=" * 72)
    print("  CPGN — QM9 Multi-Property Molecular Prediction")
    print("  Primary: HOMO  |  Auxiliary: LUMO, Gap, ZPVE, μ, α, R², U₀, U₂₉₈, H₂₉₈, G₂₉₈")
    print("  Architecture: atom graph + line graph (angles) + poly graph (local env)")
    print("  Training: MAE loss | AdamW cosine LR | grad clip | early stop")
    print("=" * 72)

    # ── Load QM9 via PyG ──────────────────────────────────────────────────────
    print(f"\nLoading QM9 dataset (root='{QM9_ROOT}') ...")
    print("  (First run auto-downloads ~1 GB from PyG servers)")
    try:
        from torch_geometric.datasets import QM9
        dataset = QM9(root=QM9_ROOT)
    except ImportError:
        print("[ERROR] torch_geometric not installed.  Run:")
        print("        pip install torch-geometric")
        sys.exit(1)

    print(f"  Total molecules : {len(dataset):,}")

    # ══════════════════════════════════════════════════════════════════════
    # DIAGNOSTIC — print raw PyG values vs converted targets for mol[0]
    # Reveals true storage units before any graph building
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("  UNIT DIAGNOSTIC — molecule 0 raw vs converted targets")
    print("="*70)
    mol0     = dataset[0]
    raw_y0   = mol0.y[0].numpy()
    zs0      = mol0.z.numpy().tolist()
    ref0_har = sum(ATOM_REF_ENERGIES_HAR.get(int(z), 0.0) for z in zs0)
    print(f"  Atoms Z={zs0}  ref_total={ref0_har:.4f} Har")
    print(f"  {'Prop':<8} {'pyg_idx':>7} {'raw_value':>14} {'converted':>12}")
    print(f"  {'-'*46}")
    for short, label, unit, pyg_idx, act, primary in QM9_PROPS:
        raw = float(raw_y0[pyg_idx])
        # PyG stores in eV already — only atomisation for U0/U/H/G
        val = raw
        if pyg_idx in ATOMISATION_INDICES_PYG:
            val -= ref0_har * HARTREE_TO_EV
        val *= PROP_SCALE[short]
        tag = ">>>" if primary else "   "
        print(f"  {tag} {short:<7}{pyg_idx:>7}  {raw:>14.6f}  {val:>12.4f}")
    print("="*70)
    print("  HOMO should be ≈ -6 to -9 eV after conversion")
    print("  U0   should be ≈ -5 to -50 eV after atomisation correction")
    print("="*70 + "\n")
    # ══════════════════════════════════════════════════════════════════════

    # ── Shuffled split ────────────────────────────────────────────────────────
    indices = list(range(len(dataset)))
    random.Random(SEED).shuffle(indices)
    train_idx = indices[:N_TRAIN]
    val_idx   = indices[N_TRAIN : N_TRAIN + N_VAL]
    test_idx  = indices[N_TRAIN + N_VAL :]

    print(f"  Train : {len(train_idx):,}  |  Val : {len(val_idx):,}  "
          f"|  Test : {len(test_idx):,}  (shuffled SEED={SEED})")

    train_mols = [dataset[i] for i in train_idx]
    val_mols   = [dataset[i] for i in val_idx]
    test_mols  = [dataset[i] for i in test_idx]
    del dataset

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
        train_ds = build_dataset(train_mols, "TRAIN")
        val_ds   = build_dataset(val_mols,   "VAL")
        test_ds  = build_dataset(test_mols,  "TEST")
        print(f"\n  Saving graph cache → {GRAPH_CACHE} ...")
        with open(GRAPH_CACHE, "wb") as f:
            pickle.dump({"train": train_ds, "val": val_ds, "test": test_ds},
                        f, protocol=pickle.HIGHEST_PROTOCOL)
        print("  Graph cache saved.")

    del train_mols, val_mols, test_mols

    # ══════════════════════════════════════════════════════════════════════
    # NORMALISATION: compute mean/std of each target from training set
    # and normalise stored targets. This guarantees training is stable
    # regardless of what units PyG stores values in.
    # Inverse transform applied at inference time before metric reporting.
    # ══════════════════════════════════════════════════════════════════════
    NORM_CACHE = GRAPH_CACHE.replace(".pkl", "_norm.pkl")
    if os.path.isfile(NORM_CACHE):
        with open(NORM_CACHE, "rb") as _f:
            norm_params = pickle.load(_f)
        print(f"  Normalisation params loaded from {NORM_CACHE}")
    else:
        print("  Computing per-target normalisation stats from training set ...")
        all_primary = np.array([d.y_primary.item() for d in train_ds])
        all_aux     = np.stack([d.y_aux.numpy()   for d in train_ds])   # (N, N_AUX)
        norm_params = {
            "primary_mean": float(all_primary.mean()),
            "primary_std" : float(all_primary.std()) + 1e-8,
            "aux_mean"    : all_aux.mean(axis=0),    # (N_AUX,)
            "aux_std"     : all_aux.std(axis=0) + 1e-8,
        }
        with open(NORM_CACHE, "wb") as _f:
            pickle.dump(norm_params, _f)
        print(f"  Normalisation params saved to {NORM_CACHE}")

    pm  = norm_params["primary_mean"]
    ps  = norm_params["primary_std"]
    am  = torch.tensor(norm_params["aux_mean"],  dtype=torch.float32)
    as_ = torch.tensor(norm_params["aux_std"],   dtype=torch.float32)

    print(f"  Primary (HOMO): mean={pm:.4f}  std={ps:.4f}  "
          f"[normalised range ≈ ±3]")
    print(f"  Aux std range : {norm_params['aux_std'].min():.4f} — "
          f"{norm_params['aux_std'].max():.4f}")

    # Apply normalisation to all datasets in-place
    def normalise_dataset(ds, pm, ps, am, as_):
        for d in ds:
            d.y_primary = (d.y_primary - pm) / ps
            d.y_aux     = (d.y_aux     - am) / as_
    normalise_dataset(train_ds, pm, ps, am, as_)
    normalise_dataset(val_ds,   pm, ps, am, as_)
    normalise_dataset(test_ds,  pm, ps, am, as_)
    print("  Datasets normalised.")
    # ══════════════════════════════════════════════════════════════════════

    # ── Train or load ─────────────────────────────────────────════════════
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
        print("\n  Plotting training curves ...")
        plot_training_curve(tr_loss_h, vl_loss_h, tr_mae_h, vl_mae_h)

    # ── Inference + de-normalise ─────────────────────────────────────────────
    def denorm_results(res):
        """Convert normalised predictions/targets back to original units."""
        return {
            "primary_true": res["primary_true"] * ps + pm,
            "primary_pred": res["primary_pred"] * ps + pm,
            "aux_true"    : res["aux_true"] * norm_params["aux_std"] + norm_params["aux_mean"],
            "aux_pred"    : res["aux_pred"] * norm_params["aux_std"] + norm_params["aux_mean"],
        }

    print("\nPredicting on validation set ...")
    val_res  = denorm_results(predict_set(model, val_ds,  "VAL"))
    val_m    = compute_metrics(val_res, "Validation")
    save_csv(val_res, val_ds, OUTPUT_VAL_CSV)

    print("\nPredicting on test set ...")
    test_res = denorm_results(predict_set(model, test_ds, "TEST"))
    test_m   = compute_metrics(test_res, "Test")
    save_csv(test_res, test_ds, OUTPUT_CSV)

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n  Generating visualisation plots ...")
    plot_parity(val_res, test_res, val_m, test_m)
    plot_error_hist(val_res, test_res)
    plot_property_maes(test_m)
    plot_benchmark_table(test_m)

    # ── Summary ───────────────────────────────────────────────────────────────
    print_final_summary(val_m, test_m)

    output_files = [
        CHECKPOINT, OUTPUT_VAL_CSV, OUTPUT_CSV,
        "CPGN_QM9_training_curve.png",
        "CPGN_QM9_parity_plots.png",
        "CPGN_QM9_property_maes.png",
        "CPGN_QM9_error_hist.png",
        "CPGN_QM9_benchmark_table.png",
    ]
    print("\n  Output files:")
    for f in output_files:
        tag = "✓" if os.path.exists(f) else "·"
        print(f"    [{tag}] {f}")
    print("\n  Done.")


if __name__ == "__main__":
    main()
