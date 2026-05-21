# CPGN-Coordination-Polyhedron-Graph-Neural-Network

A graph neural network for multi-property prediction of inorganic crystals and molecules, built on a dual-graph architecture that combines an **atom graph**, a **line graph** for explicit bond-angle interactions, and a novel **coordination polyhedron graph** with bidirectional cross-attention.

---

## Overview

CPGN introduces the coordination polyhedron graph $G_P$ as a structural prior that no prior discriminative GNN has used for property prediction. Each node in $G_P$ represents the coordination cage around an atom, carrying geometric descriptors — distortion index, bond-angle variance, polyhedral volume, mean bond length, electronegativity mismatch, and coordination type. Edges encode the topological relationship between adjacent polyhedra: face-sharing (1.0), edge-sharing (0.5), or corner-sharing (0.0) connectivity. This information governs ionic migration pathways, cooperative octahedral tilting in perovskites, and magnetic superexchange — phenomena entirely absent from all prior atom-only GNN architectures.

The model simultaneously predicts formation energy, band gap, and thermodynamic stability from a single shared latent representation, benefiting from implicit multi-task regularisation across all three tasks.

---

## Architecture

```
Crystal structure (CIF / POSCAR / JSON)
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│  Three parallel graphs                                      │
│  G_A  Atom graph    (92-dim CGCNN features, RBF distances)  │
│  G_L  Line graph    (bond pairs → RBF bond angles)          │
│  G_P  Poly graph    (7-dim cage features, sharing type)     │
└─────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 1 — 4 ALIGNN layers                                  │
│    angle info  →  bond embeddings  (line graph conv)        │
│    bond embeddings  →  atom embeddings  (atom graph conv)   │
│                                                             │
│  Phase 2 — 4 GCN-only layers  (atom graph)                  │
│                                                             │
│  Phase 3 — 4 CPGN layers  (poly graph + cross-attention)    │
│    poly graph MP  →  poly embeddings                        │
│    bidirectional cross-attention  (atom ↔ poly)             │
└─────────────────────────────────────────────────────────────┘
          │
          ▼
  Global mean pool (atom) + Global mean pool (poly)
          │  concat [h_atom ∥ h_poly]
          ▼
     MLP fusion  →  latent z  (dim=128)
          │
  ┌───────┼───────┐
  ▼       ▼       ▼
 Ef      Eg    Stability
(MAE)  (MAE)   (BCE)
```

**Parameters:** ~4.85 million  
**Hidden dim:** 256  
**RBF centres:** 40 (distances, [0, 8 Å]) + 40 (angles, [0, π])  
**Cutoff:** 8.0 Å  
**Optimiser:** AdamW, lr=3×10⁻⁴, cosine annealing, grad clip 1.0  
**Loss:** MAE(Ef) + 0.1 × (MAE(Eg) + BCE(stability))

---

## Repository Structure

```
CPGN/
├── README.md
│
├── Materials Project (MP 2018.6.1)
│   ├── cpgn_mp_upgraded.py          CPGN — formation energy + band gap + stability
│   ├── cgcnn_mp_json.py             CGCNN baseline (from scratch)
│   ├── schnet_mp_json.py            SchNet baseline (from scratch)
│   ├── megnet_mp_json.py            MEGNet baseline (from scratch)
│   └── alignn_mp_json.py            ALIGNN baseline (from scratch)
│
├── JARVIS-DFT
│   ├── cpgn_jarvis_dft.py           CPGN — 19 auxiliary properties, masked MAE
│   ├── cgcnn_jarvis_dft.py          CGCNN baseline (from scratch)
│   └── alignn_jarvis_dft.py         ALIGNN baseline (from scratch)
│
└── QM9
    └── cpgn_qm9.py                  CPGN — 11 quantum chemistry properties
```

All baseline scripts share identical data loading, split strategy, loss function structure, and evaluation pipeline as the CPGN scripts to ensure fair comparison.

---

## Installation

```bash
# Clone repository
git clone https://github.com/sanjaylopa22/CPGN.git
cd CPGN

# Create virtual environment
python -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\activate           # Windows

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install torch-geometric
pip install pymatgen scikit-learn matplotlib pandas
pip install jarvis-tools          # for JARVIS-DFT experiments only
```

**Hardware:** All experiments were run on a single NVIDIA RTX 5090 GPU with CUDA 12.8. PyTorch 2.7 is required for RTX 5090 support.

**Python:** 3.12

---

## Running Experiments

### Materials Project

Download `mp.2018.6.1.json` from the Materials Project and place it in the MP directory, then:

```bash
cd "Materials Project"
python cpgn_mp_upgraded.py
```

On first run, all graphs are built and cached to `cpgn_graph_cache.pkl` (~3–4 GB). Subsequent runs load the cache instantly.

### JARVIS-DFT

```bash
cd JARVIS-DFT
python cpgn_jarvis_dft.py
```

The JARVIS-DFT dataset (~500 MB) is downloaded automatically via `jarvis-tools` on first run and cached locally.

### QM9

```bash
cd QM9
python cpgn_qm9.py
```

The QM9 dataset (~1 GB) is downloaded automatically via PyTorch Geometric on first run.

**Important:** Delete `*.pkl` and `*.pt` files when switching between script versions to force a clean rebuild of graph caches and checkpoints.

---

## Output Files

Each training script produces the following outputs in its working directory:

| File | Description |
|---|---|
| `*_best.pt` | Best model checkpoint (selected on val Ef MAE) |
| `*_test.csv` | Per-sample test predictions and errors |
| `*_val.csv` | Per-sample validation predictions and errors |
| `*_training_curve.png` | Train/val loss and MAE curves with benchmark reference lines |
| `*_parity_plots.png` | Equal-aspect parity plots (val + test) |
| `*_error_hist.png` | Error distribution histograms |
| `*_property_maes.png` | Bar chart comparing all property MAEs to published benchmarks |
| `*_benchmark_table.png` | Horizontal bar chart: MAE and MAD:MAE vs CGCNN/MEGNet/SchNet/ALIGNN |

---

## Key Design Decisions

**Coordination polyhedron graph.** The face/edge/corner-sharing connectivity between coordination cages encodes structural topology that governs ionic migration, octahedral tilting in perovskites, and magnetic superexchange. This information is absent from all previous atom-only GNN architectures and is CPGN's primary architectural contribution.

**ALIGNN backbone.** Phases 1 and 2 of the forward pass are identical to ALIGNN — four ALIGNN layers (line graph angle updates → atom updates) followed by four GCN-only layers. This means CPGN subsumes ALIGNN's angular interaction mechanism and adds the polyhedron stream on top rather than replacing the atom+line graph backbone.

**Bidirectional cross-attention.** After each poly-graph convolution layer, atom embeddings attend over the graph-level mean of polyhedron embeddings and vice versa. This couples the two streams at every depth so that structural topology continuously informs atomic representations.

**Multi-task learning.** Formation energy is the primary task (full loss weight). Band gap and stability classification are auxiliary (λ=0.1), contributing regularisation without diluting the primary gradient signal. The shared backbone learns representations useful for all three tasks simultaneously.

**Graph caching.** Building graphs for 60,000 MP structures takes ~2.5 hours. All scripts save graphs to a `.pkl` cache on first run and reload from cache on subsequent runs. The cache is separate for each script to avoid cross-contamination between different graph types (CPGN includes poly+line graphs; CGCNN does not).

---
