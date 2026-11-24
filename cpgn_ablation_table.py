#!/usr/bin/env python3
"""
cpgn_ablation_table.py - turn the sweep results into the paper's ablation tables
================================================================================
    python cpgn_ablation_table.py                       # reads ablation_results/{jarvis,mp}/*.json
    python cpgn_ablation_table.py --results my_results --decimals 3 --ef-decimals 4

Writes (into --results):
  ablation_table_components.tex   Table: A, B, C, D, G         (main paper)
  ablation_table_topology.tex     Table: A, E1, E2, E3         (appendix; only if E runs exist)
  ablation_summary.csv            every cell: n, mean, std, per-seed values
  ablation_paired.csv             paired comparison of each variant with A (see below)
and prints the same to the console.

Conventions
-----------
* mean +- std over the available seeds; std is the SAMPLE standard deviation (ddof=1) unless
  --population-std is given. State the convention in the caption (the generated caption does).
* A cell with fewer runs than the others is marked $^{n=k}$; a missing cell is a red TBD.
* Paired comparison: the variant is compared with A under the SAME training seed.
    rel   = (mean_variant / mean_A - 1) * 100
    gap   = mean_variant - mean_A
    rule  = WORSE/BETTER (by the sign of the gap) if |gap| > max(std_variant, std_A), borderline if equal,
            otherwise "no measurable effect"
  (a gap that does not exceed the larger of the two seed standard deviations should be reported as
  "no measurable effect", not as an improvement or a degradation; lower MAE is better, so a negative
  gap means the variant is BETTER than A, which would contradict the claim that the component helps).
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

CELLS = [("mp", "Ef"), ("mp", "Eg"), ("jarvis", "Ef"), ("jarvis", "Eg")]
CELL_NAME = {("mp", "Ef"): "MP Ef", ("mp", "Eg"): "MP Eg", ("jarvis", "Ef"): "JARVIS Ef", ("jarvis", "Eg"): "JARVIS Eg_opt"}
COMPONENT_ROWS = ["A", "B", "C", "D", "G"]
TOPOLOGY_ROWS = ["A", "E1", "E2", "E3"]
LABELS = {
    "A": r"A\;\; full CPGN",
    "B": r"B\;\; $-G_P$, $-$CA",
    "C": r"C\;\; $-G_L$",
    "D": r"D\;\; $G_P$ without CA",
    "G": r"G\;\; B widened ($H{=}%s$)",
    "E1": r"E1\; $c_{pq}\equiv 0$ (type removed)",
    "E2": r"E2\; no $G_P$ edges (topology removed)",
    "E3": r"E3\; shuffled $G_P$ edges",
}
TBD = r"\textcolor{red}{TBD}"


def load_results(root):
    data = {}
    for ds in ("mp", "jarvis"):
        for f in sorted((Path(root) / ds).glob("*_seed*.json")):
            r = json.loads(f.read_text())
            data.setdefault(ds, {}).setdefault(r["variant"], {})[r["seed"]] = r
    manifests = {ds: json.loads((Path(root) / ds / "manifest.json").read_text())
                 for ds in ("mp", "jarvis") if (Path(root) / ds / "manifest.json").exists()}
    return data, manifests


def values(data, ds, variant, target):
    runs = data.get(ds, {}).get(variant, {})
    return {s: r["test"][target] for s, r in sorted(runs.items()) if r["test"].get(target) is not None}


def mean_std(vals, population=False):
    v = list(vals)
    n = len(v)
    if n == 0:
        return None, None
    mean = sum(v) / n
    if n == 1:
        return mean, None
    ddof = 0 if population else 1
    return mean, math.sqrt(sum((x - mean) ** 2 for x in v) / (n - ddof))


ZERO_STD_CELLS = []


def fmt_cell(vals, dec, n_ref, population):
    if not vals:
        return TBD
    mean, std = mean_std(vals.values(), population)
    if std is not None and 0 < std < 0.5 * 10 ** (-dec):
        ZERO_STD_CELLS.append((round(mean, dec + 2), std))
    s = f"{mean:.{dec}f}" + (f"$\\pm${std:.{dec}f}" if std is not None else r"$\pm$--")
    if len(vals) != n_ref:
        s += f"$^{{n={len(vals)}}}$"
    return s


def params_cell(data, manifests, variant):
    for ds in ("jarvis", "mp"):
        runs = data.get(ds, {}).get(variant, {})
        if runs:
            return f"{next(iter(runs.values()))['params'] / 1e6:.3f}"
        if ds in manifests and variant in manifests[ds]["params"]:
            return f"{manifests[ds]['params'][variant] / 1e6:.3f}"
    return TBD


def width_label(manifests):
    w = manifests.get("jarvis", manifests.get("mp", {})).get("g_width")
    return str(w) if w else "?"


def make_table(rows, caption, label, data, manifests, args):
    n_all = [len(values(data, ds, v, t)) for v in rows for ds, t in CELLS]
    n_ref = max(n_all) if n_all else 0
    n_min = min([n for n in n_all if n > 0], default=0)
    seeds_txt = (f"{n_ref} training seeds" if n_min == n_ref else f"{n_min}--{n_ref} training seeds (cells marked $n{{=}}k$ have fewer runs)")
    out = [r"\begin{table}[hbt!]", r"\centering",
           caption.replace("@SEEDS@", seeds_txt).replace("@STD@", "population" if args.population_std else "sample"),
           f"\\label{{{label}}}", r"\scriptsize", r"\begin{tabular}{|l|c|c|c|c|c|}", r"\hline",
           r" &  & \multicolumn{2}{c|}{Materials Project} & \multicolumn{2}{c|}{JARVIS-DFT} \\ \hline",
           r"Variant & Params (M) & $E_f$ (eV/atom) & $E_g$ (eV) & $E_f$ (eV/atom) & $E_g^{\text{opt}}$ (eV) \\ \hline"]
    for v in rows:
        lab = LABELS[v] % width_label(manifests) if v == "G" else LABELS[v]
        cells = []
        for ds, t in CELLS:
            dec = args.ef_decimals if (t == "Ef" and args.ef_decimals) else args.decimals
            cells.append(fmt_cell(values(data, ds, v, t), dec, n_ref, args.population_std))
        out.append(f"{lab:<32} & {params_cell(data, manifests, v)} & " + " & ".join(cells) + r" \\ \hline")
    out += [r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)


def paired(data, ds, variant, target, population):
    a, v = values(data, ds, "A", target), values(data, ds, variant, target)
    common = sorted(set(a) & set(v))
    if not common or not v or not a:
        return None
    ma, sa = mean_std(a.values(), population)
    mv, sv = mean_std(v.values(), population)
    diffs = [v[s] - a[s] for s in common]
    std_max = max([x for x in (sa, sv) if x is not None], default=None)
    gap = mv - ma
    if std_max is None:
        verdict = "n/a (need >= 2 seeds)"
    elif abs(gap) > std_max + 1e-12:
        verdict = "WORSE than A (gap exceeds std)" if gap > 0 else "BETTER than A (gap exceeds std)"
    elif abs(abs(gap) - std_max) <= 1e-12:
        verdict = "borderline (gap = std)"
    else:
        verdict = "no measurable effect (gap within std)"
    return {"rel_pct": 100 * (mv / ma - 1), "gap": gap, "std_max": std_max, "seeds_worse": sum(d > 0 for d in diffs),
            "seeds_paired": len(common), "verdict": verdict, "n_a": len(a), "n_v": len(v)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="ablation_results")
    p.add_argument("--decimals", type=int, default=3)
    p.add_argument("--ef-decimals", type=int, default=None, help="decimals for the E_f columns (default: --decimals)")
    p.add_argument("--population-std", action="store_true", help="use ddof=0 instead of the sample std (ddof=1)")
    args = p.parse_args()
    root = Path(args.results)
    data, manifests = load_results(root)
    if not data:
        sys.exit(f"[ERROR] no result files under {root}/{{mp,jarvis}}/")

    cap_c = (r"\caption{CPGN Component ablation (test MAE, mean $\pm$ std over @SEEDS@ with the data split fixed; "
             r"std is the @STD@ standard deviation; lower is better). Variants are defined in Appendix "
             r"Table~\ref{tab:abl_variants}. Parameter counts are for the JARVIS-DFT configuration.}")
    cap_t = (r"\caption{Does the polyhedron sharing structure matter? Test MAE (mean $\pm$ std over @SEEDS@ with the data "
             r"split fixed; std is the @STD@ standard deviation). E1 removes the sharing type only, E2 removes the edges, "
             r"E3 keeps the edges but rewires them.}")
    tables = {"ablation_table_components.tex": make_table(COMPONENT_ROWS, cap_c, "tab:abl_components", data, manifests, args)}
    if any(v in data.get(ds, {}) for ds in data for v in ("E1", "E2", "E3")):
        tables["ablation_table_topology.tex"] = make_table(TOPOLOGY_ROWS, cap_t, "tab:abl_topology", data, manifests, args)
    for name, tex in tables.items():
        (root / name).write_text(tex + "\n")
        print(f"\n% ===== {root / name}\n{tex}")

    with open(root / "ablation_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "variant", "target", "n", "mean", "std", "per_seed"])
        for ds in ("mp", "jarvis"):
            for v in data.get(ds, {}):
                for t in ("Ef", "Eg"):
                    vals = values(data, ds, v, t)
                    if vals:
                        mean, std = mean_std(vals.values(), args.population_std)
                        w.writerow([ds, v, t, len(vals), f"{mean:.6f}", "" if std is None else f"{std:.6f}",
                                    ";".join(f"{s}:{x:.6f}" for s, x in vals.items())])

    print("\n" + "=" * 100)
    print(f"  Paired comparison with A (same training seed); std = {'population' if args.population_std else 'sample'}")
    print("=" * 100)
    print(f"  {'variant':<8}{'cell':<15}{'rel. change':>12}{'gap':>10}{'max std':>9}{'seeds worse':>13}   verdict")
    rows = []
    for v in ["B", "C", "D", "G", "E1", "E2", "E3"]:
        for ds, t in CELLS:
            r = paired(data, ds, v, t, args.population_std)
            if r is None:
                continue
            rows.append([v, CELL_NAME[(ds, t)], r])
            sd = "n/a" if r["std_max"] is None else f"{r['std_max']:.4f}"
            print(f"  {v:<8}{CELL_NAME[(ds, t)]:<15}{r['rel_pct']:>+11.1f}%{r['gap']:>+10.4f}{sd:>9}"
                  f"{r['seeds_worse']:>9}/{r['seeds_paired']:<3}   {r['verdict']}")
    with open(root / "ablation_paired.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "cell", "rel_change_pct", "gap", "max_std", "seeds_worse", "seeds_paired", "verdict"])
        for v, c, r in rows:
            w.writerow([v, c, f"{r['rel_pct']:.3f}", f"{r['gap']:.6f}", "" if r["std_max"] is None else f"{r['std_max']:.6f}",
                        r["seeds_worse"], r["seeds_paired"], r["verdict"]])
    e3 = [r for ds in data for r in data[ds].get("E3", {}).values() if "e3_changed_fraction" in r]
    if e3:
        print("\n  E3 shuffle changed the graph for (fraction of structures):")
        for ds in ("mp", "jarvis"):
            rs = [r for r in e3 if r["dataset"] == ds]
            if rs:
                fr = rs[0]["e3_changed_fraction"]
                print(f"    {ds}: " + ", ".join(f"{k} {100 * v:.1f}%" for k, v in fr.items()))
    if ZERO_STD_CELLS:
        print(f"\n  [note] {len(ZERO_STD_CELLS)} table cell(s) have a std below half of the last printed digit and show as 0.000;"
              " rerun with a larger --decimals (and --ef-decimals) so the +- means something.")
    print(f"\n  wrote {root}/ablation_summary.csv and ablation_paired.csv")


if __name__ == "__main__":
    main()
