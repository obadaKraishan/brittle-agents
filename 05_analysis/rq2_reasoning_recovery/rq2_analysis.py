#!/usr/bin/env python3
"""
=============================================================================
 BRITTLE AGENTS | 05_analysis/rq2_reasoning_recovery/rq2_analysis.py
=============================================================================
 RQ2: Do reasoning models handle tool failures better than matched
      non-reasoning siblings?

 Three outcomes are compared within each model family, so the contrast is
 always reasoning vs instruct on the same weights lineage:
   detection    noticed the failure
   replanned    next action differed from the failed action
   recovered    reached the state this model reaches on its own clean run

 Design: paired by (task, fault, seed) within family. Inference per family
 is a binomial GEE clustered on task on the matched trials, Holm-corrected
 across families, with task-clustered bootstrap CIs for the difference;
 paired t-tests are kept as descriptive comparisons. A pooled GEE on all
 fault trials adds fault type as a covariate.

 Usage:
   python 05_analysis/rq2_reasoning_recovery/rq2_analysis.py

 Saves: outputs/rq2_results.csv, rq2_results.json,
        rq2_report.txt
=============================================================================
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import pandas as pd  # noqa: E402

from analysis_utils import (load_master, paired_frame, paired_gee,
                            paired_tests, cluster_boot_ci, gee_terms, holm,
                            gee_logit, apa_model, fmt_p, f2, pct, table)  # noqa: E402

OUT, REP = HERE / "outputs", []
OUTCOMES = [("detection", "detection"), ("replanned", "replanning"),
            ("recovered", "recovery")]


def rep(s=""):
    print(s, flush=True)
    REP.append(s)


def main():
    df, cfg = load_master()
    OUT.mkdir(parents=True, exist_ok=True)
    rep("=" * 76)
    rep(" RQ2 -- REASONING VS MATCHED NON-REASONING MODELS UNDER FAULTS")
    rep(f" Generated: {datetime.now(timezone.utc).isoformat()}")
    rep("=" * 76)

    fault = df[df["is_fault"]].copy()
    fams = sorted(fault["pair"].dropna().unique())
    rep(f"\n Families: {fams}")
    rep(" Trials are paired within family on (task, fault, seed); the "
        "scaffolding arm is excluded so the comparison is like for like.")
    fault = fault[fault["scaffolding"] == "off"]

    results, rows_csv = {}, []
    keys = ["task_id", "fault", "seed"]
    for col, label in OUTCOMES:
        rep(f"\n--- Outcome: {label} ---")
        keep = []
        for fam in fams:
            sub = fault[fault["pair"] == fam]
            if sub["reasoning"].nunique() < 2:
                continue
            pairs = paired_frame(sub, "reasoning", True, False, col, keys)
            g = paired_gee(sub, "reasoning", True, False, col, keys)
            t = paired_tests(pairs["a"], pairs["b"])
            ci = cluster_boot_ci(pairs, "d")
            keep.append(dict(family=fam, n=len(pairs),
                             mean_reasoning=float(pairs["a"].mean()),
                             mean_instruct=float(pairs["b"].mean()),
                             mean_diff=float(pairs["d"].mean()),
                             ci_lo_task=ci[0], ci_hi_task=ci[1],
                             or_gee=g["OR"], z_gee=g["z"], p_gee=g["p"],
                             t=t["t"], df=t["df"], p_t=t["p"], dz=t["dz"]))
        for r, a_, b_ in zip(keep, holm([r["p_gee"] for r in keep]),
                             holm([r["p_t"] for r in keep])):
            r["p_holm_gee"], r["p_holm_t"] = float(a_), float(b_)
        rows = [[r["family"], pct(r["mean_reasoning"], 1),
                 pct(r["mean_instruct"], 1),
                 f"{100 * r['mean_diff']:+.1f}",
                 f"[{100 * r['ci_lo_task']:+.1f}, {100 * r['ci_hi_task']:+.1f}]",
                 f2(r["or_gee"]), f2(r["z_gee"]),
                 fmt_p(r["p_holm_gee"]).replace("p ", ""), r["n"]]
                for r in keep]
        rep("\n" + table(rows, ["family", "reasoning", "instruct", "diff",
                                "95% CI (task-clustered)", "OR", "z",
                                "p_holm (GEE)", "pairs"]))
        rep("\n Task-clustered GEE on the matched trials (outcome ~ "
            "reasoning, clustered on task), Holm-corrected across families.")
        rep(" Descriptive paired t-tests:")
        for r in keep:
            rep(f"  {r['family']}: t({r['df']}) = {f2(r['t'])}, "
                f"{fmt_p(r['p_t'])}, Holm-corrected {fmt_p(r['p_holm_t'])}, "
                f"dz = {f2(r['dz'])}")
        for r in keep:
            rows_csv.append({"outcome": col, **r})

        # pooled across families
        pooled_pairs = pd.concat(
            [paired_frame(fault[fault["pair"] == fam], "reasoning", True,
                          False, col, keys) for fam in fams])
        pooled = paired_tests(pooled_pairs["a"], pooled_pairs["b"])
        ci = cluster_boot_ci(pooled_pairs, "d")
        pooled.update(ci_task=ci)
        rep(f"\n  POOLED ({label}): mean difference = "
            f"{100 * pooled['mean_diff']:+.1f} points, task-clustered 95% CI "
            f"[{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}] "
            f"(n = {pooled['n']} pairs); descriptive t({pooled['df']}) = "
            f"{f2(pooled['t'])}, {fmt_p(pooled['p'])}.")

        m = gee_logit(fault, col, "C(reasoning) + C(fault)")
        for line in apa_model(m, f"GEE for {label} (all fault trials)"):
            rep("  " + line)
        results[col] = {"families": {r["family"]: r for r in keep},
                        "pooled": pooled, "gee": gee_terms(m)}

    pd.DataFrame(rows_csv).to_csv(OUT / "rq2_results.csv", index=False)
    (OUT / "rq2_results.json").write_text(
        json.dumps(results, indent=2, default=float), encoding="utf-8")
    (OUT / "rq2_report.txt").write_text("\n".join(REP) + "\n",
                                              encoding="utf-8")
    rep("\nSaved: rq2_results.csv, rq2_results.json, rq2_report.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
