#!/usr/bin/env python3
"""
test_ablation_engine.py - quick self-checks (seconds, no GPU, no base script, no data)
    python test_ablation_engine.py
The numbers used here are FABRICATED test values for checking arithmetic; they are not results.
"""
import json
import os
import statistics as st
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cpgn_ablation_core as core  # noqa: E402
import cpgn_ablation_table as tab  # noqa: E402


def directed(pairs, attrs):
    src, dst, att = [], [], []
    for (a, b), v in zip(pairs, attrs):
        src += [a, b]; dst += [b, a]; att += [[v], [v]]
    return np.array([src, dst], dtype=np.int64), np.array(att, dtype=np.float32)


class ShuffleTests(unittest.TestCase):
    def check_invariants(self, ei, ea, nei, nea):
        sym = {(int(a), int(b)) for a, b in zip(nei[0], nei[1])}
        self.assertEqual(Counter(ei[0].tolist()), Counter(nei[0].tolist()))          # node degrees
        self.assertEqual(ei.shape, nei.shape)                                         # number of edges
        self.assertEqual(sorted(ea.ravel().tolist()), sorted(nea.ravel().tolist()))   # attribute multiset
        self.assertTrue(all((b, a) in sym for a, b in sym))                           # symmetric
        self.assertTrue(all(a != b for a, b in sym))                                  # no self loops
        self.assertEqual(len(sym), nei.shape[1])                                      # no duplicate edges

    def test_sparse_graphs_are_rewired_and_valid(self):
        graphs = {"ring": [(i, (i + 1) % 10) for i in range(10)], "path": [(i, i + 1) for i in range(7)]}
        for name, prs in graphs.items():
            prs = [tuple(sorted(p)) for p in prs]
            ei, ea = directed(prs, [[0.0, 0.5, 1.0][k % 3] for k in range(len(prs))])
            for seed in range(10):
                nei, nea, changed = core.shuffle_poly_edges(ei, ea, np.random.default_rng(seed))
                self.assertTrue(changed, name)
                self.check_invariants(ei, ea, nei, nea)

    def test_graphs_with_a_unique_labelling_are_unchanged(self):
        complete = [(i, j) for i in range(5) for j in range(i + 1, 5)]
        ei, ea = directed(complete, [0.5] * len(complete))
        nei, nea, changed = core.shuffle_poly_edges(ei, ea, np.random.default_rng(0))
        self.assertFalse(changed)
        self.check_invariants(ei, ea, nei, nea)

    def test_deterministic_for_a_fixed_seed(self):
        prs = [(i, (i + 1) % 12) for i in range(12)]
        prs = [tuple(sorted(p)) for p in prs]
        ei, ea = directed(prs, [0.0] * len(prs))
        r1 = core.shuffle_poly_edges(ei, ea, np.random.default_rng(7))
        r2 = core.shuffle_poly_edges(ei, ea, np.random.default_rng(7))
        self.assertTrue(np.array_equal(r1[0], r2[0]))


class VariantTests(unittest.TestCase):
    def test_definitions_are_consistent(self):
        for v, c in core.VARIANTS.items():
            if c["use_ca"]:
                self.assertTrue(c["use_poly"], v)       # cross-attention needs the polyhedron stream
        self.assertEqual({v for v, c in core.VARIANTS.items() if c["width"] == "match"}, {"G"})


class AggregationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        fabricated = {  # variant -> (Ef per seed, Eg per seed): FABRICATED
            "A": ([0.054, 0.056, 0.052], [0.250, 0.252, 0.248]),
            "B": ([0.063, 0.066, 0.060], [0.258, 0.256, 0.260]),     # clearly worse on Ef, small on Eg
            "C": ([0.055, 0.056, 0.054], [0.251, 0.253, 0.249]),     # inside the seed noise
            "D": ([0.050, 0.051, 0.049], [0.250, 0.252, 0.248])}     # clearly BETTER than A on Ef
        for ds in ("mp", "jarvis"):
            (root / ds).mkdir()
            for v, (efs, egs) in fabricated.items():
                for k, seed in enumerate((42, 43, 44)):
                    (root / ds / f"{v}_seed{seed}.json").write_text(json.dumps(
                        {"dataset": ds, "variant": v, "seed": seed, "params": 4854612, "test": {"Ef": efs[k], "Eg": egs[k]}}))
        self.data, self.manifests = tab.load_results(root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_mean_and_sample_std(self):
        vals = tab.values(self.data, "mp", "A", "Ef")
        mean, std = tab.mean_std(vals.values())
        self.assertAlmostEqual(mean, st.mean([0.054, 0.056, 0.052]))
        self.assertAlmostEqual(std, st.stdev([0.054, 0.056, 0.052]))                       # ddof = 1
        self.assertAlmostEqual(tab.mean_std(vals.values(), population=True)[1], st.pstdev([0.054, 0.056, 0.052]))
        self.assertEqual(tab.mean_std([0.1])[1], None)                                     # one seed: no std

    def test_paired_verdicts_have_the_right_direction(self):
        worse = tab.paired(self.data, "mp", "B", "Ef", False)
        self.assertIn("WORSE", worse["verdict"]); self.assertEqual(worse["seeds_worse"], 3)
        self.assertAlmostEqual(worse["rel_pct"], 100 * (0.063 / 0.054 - 1), places=6)
        noise = tab.paired(self.data, "mp", "C", "Ef", False)
        self.assertIn("no measurable effect", noise["verdict"])
        better = tab.paired(self.data, "mp", "D", "Ef", False)
        self.assertIn("BETTER", better["verdict"]); self.assertEqual(better["seeds_worse"], 0)

    def test_cell_formatting(self):
        vals = tab.values(self.data, "mp", "A", "Ef")
        self.assertEqual(tab.fmt_cell(vals, 3, 3, False), "0.054$\\pm$0.002")
        self.assertEqual(tab.fmt_cell({}, 3, 3, False), tab.TBD)
        self.assertIn("n=2", tab.fmt_cell({42: 0.05, 43: 0.06}, 3, 3, False))             # fewer runs than the others


if __name__ == "__main__":
    unittest.main(verbosity=2)
