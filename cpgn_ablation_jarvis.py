#!/usr/bin/env python3
"""
CPGN component ablation on JARVIS-DFT  -  python cpgn_ablation_jarvis.py
=======================================================================
Put this file, cpgn_ablation_core.py and your JARVIS training script (cpgn_jarvis_dft.py) in one
folder, ideally the one that already holds cpgn_jarvis_graph_cache.pkl (the graphs are then reused
and nothing is rebuilt).

    python cpgn_ablation_jarvis.py --params-only          # parameter counts of every variant, no data needed
    python cpgn_ablation_jarvis.py                        # variants A,B,C,D,G x seeds 42 43 44 (resumable)
    python cpgn_ablation_jarvis.py --variants E1,E2,E3    # topology controls (appendix table)
    python cpgn_ablation_jarvis.py --variants B --seeds 43   # one run, e.g. to spread work over GPUs

Every run uses the base script's own train_model(): AdamW, cosine schedule, early stopping and
checkpointing on validation Ef - only the model class differs. Results: ablation_results/jarvis/*.json.
Time per run is about that of your normal JARVIS run (~2.6 h on your GPU log), so the default sweep
(5 variants x 3 seeds) is ~15 runs; variants B, C, D are cheaper per epoch than A.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cpgn_ablation_core import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main("jarvis", "cpgn_jarvis_dft.py", "cpgn_jarvis_graph_cache.pkl"))
