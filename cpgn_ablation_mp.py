#!/usr/bin/env python3
"""
CPGN component ablation on Materials Project (mp.2018.6.1)  -  python cpgn_ablation_mp.py
========================================================================================
Put this file, cpgn_ablation_core.py and your MP training script (cpgn_upgrade_1.py) in one folder,
ideally the one that already holds cpgn_alignn_graph_cache.pkl and mp.2018.6.1.json.

    python cpgn_ablation_mp.py --params-only            # parameter counts of every variant, no data needed
    python cpgn_ablation_mp.py                          # variants A,B,C,D,G x seeds 42 43 44 (resumable)
    python cpgn_ablation_mp.py --variants E1,E2,E3      # topology controls (appendix table)
    python cpgn_ablation_mp.py --variants B --seeds 43  # one run, e.g. to spread work over GPUs

IMPORTANT: cpgn_upgrade_1.py has 5,839,875 parameters (4 ALIGNN + 4 GCN atom layers), not the
4,852,419 reported in the paper, which is the JARVIS-style model with MP's heads. Run the ablation on the
script that produced your MP numbers (--base-script), and check --params-only first.
Every run uses the base script's own train_model(); only the model class differs.
Time per run is about that of your normal MP run (up to ~7.5 h), so the default sweep is ~15 long runs.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cpgn_ablation_core import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main("mp", "cpgn_upgrade.py", "cpgn_mp_graph_cache.pkl"))
