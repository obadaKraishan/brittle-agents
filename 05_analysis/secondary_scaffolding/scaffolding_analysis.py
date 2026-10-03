#!/usr/bin/env python3
"""
=============================================================================
 BRITTLE AGENTS | 05_analysis/secondary_scaffolding/scaffolding_analysis.py
=============================================================================
 Secondary analysis: does one line of metacognitive scaffolding -- "after every tool call,
      briefly check whether the result is what you expected" -- improve
      failure detection and recovery?

 The scaffolding arm was run on a subset of the panel (a matched
 reasoning/instruct pair), so the contrast is within model: the same model,
 same tasks, same faults, with and without the instruction. Trials are
 paired on (model, task, fault, seed).

 The interesting test is the interaction: scaffolding should help most
 where the failure is quiet, because that is where checking the result adds
 information the error channel does not provide.

 Usage:
   python 05_analysis/secondary_scaffolding/scaffolding_analysis.py

 Saves: outputs/scaffolding_results.csv, scaffolding_results.json,
        scaffolding_report.txt
=============================================================================
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis_utils import (load_master, paired_by, paired_tests, holm,
                            gee_logit, gee_terms, apa_paired, apa_model,
                            fmt_p, f2, pct, table)  # noqa: E402

OUT, REP = HERE / "outputs", []
OUTCOMES = [("detection", "detection"), ("recovered", "recovery"),
            ("perseveration", "perseveration"),
            ("false_alarm", "false alarms on clean trials")]


def rep(s=""):
    print(s, flush=True)
    REP.append(s)


def main():
    df, cfg = load_master()
    OUT.mkdir(parents=True, exist_ok=True)
    rep("=" * 76)
    rep(" SECONDARY ANALYSIS -- METACOGNITIVE SCAFFOLDING: 'CHECK THE RESULT' ")
    rep(f" Generated: {datetime.now(timezone.utc).isoformat()}")
    rep("=" * 76)

    models = sorted(df[df["scaffolding"] == "on"]["model"].unique())
    if not models:
        rep("\n[ERROR] no scaffolding trials found.")
        return 1
    sub = df[df["model"].isin(models)].copy()
    rep(f"\n Models carrying the scaffolding arm: {models}")
    rep(f" Trials: {len(sub):,} "
        f"({(sub['scaffolding'] == 'on').sum():,} with the instruction)")

    results, rows_csv = {}, []
    for col, label in OUTCOMES:
        pool = sub[sub["is_fault"]] if col != "false_alarm" \
            else sub[(sub["fault"] == "clean") & (sub["target_ok"] == True)]  # noqa: E712
        if pool[col].dropna().empty:
            continue
        rep(f"\n--- Outcome: {label} ---")
        hdr = ["model", "off", "on", "diff", "t", "df", "p", "p_holm", "dz",
               "pairs"]
        keep, ps = [], []
        for m in models:
            g = pool[pool["model"] == m]
            a, b, n = paired_by(g, ["task_id", "fault", "seed"],
                                "scaffolding", "on", "off", col)
            r = paired_tests(a, b)
            r.update(model=m, mean_on=float(np.nanmean(a)),
                     mean_off=float(np.nanmean(b)))
            keep.append(r)
            ps.append(r["p"])
        rows = []
        for r, a_ in zip(keep, holm(ps)):
            r["p_holm"] = float(a_)
            rows.append([r["model"], pct(r["mean_off"], 1),
                         pct(r["mean_on"], 1),
                         f"{100 * r['mean_diff']:+.1f} pts", f2(r["t"]),
                         r["df"], fmt_p(r["p"]).replace("p ", ""),
                         fmt_p(r["p_holm"]).replace("p ", ""), f2(r["dz"]),
                         r["n"]])
            rows_csv.append({"outcome": col, **{k: r[k] for k in
                             ("model", "mean_on", "mean_off", "mean_diff",
                              "t", "df", "p", "p_holm", "dz", "n")},
                             "ci_lo": r["ci"][0], "ci_hi": r["ci"][1]})
        rep("\n" + table(rows, hdr))
        a_all, b_all = [], []
        for m in models:
            g = pool[pool["model"] == m]
            a, b, _ = paired_by(g, ["task_id", "fault", "seed"],
                                "scaffolding", "on", "off", col)
            a_all += list(a)
            b_all += list(b)
        pooled = paired_tests(a_all, b_all)
        rep("\nAPA sentences (positive difference = scaffolding increases "
            "the outcome):")
        for r in keep:
            rep("  " + apa_paired(f"{r['model']} ({label})", r)
                + f" Holm-corrected {fmt_p(r['p_holm'])}.")
        rep("  " + apa_paired(f"POOLED ({label})", pooled))
        results[col] = {"models": {r["model"]: r for r in keep},
                        "pooled": pooled}

    # ------------------------------------- interaction with visibility --
    rep("\n--- Does scaffolding help most where failures are quiet? ---")
    fl = sub[sub["is_fault"]].copy()
    fl["visibility"] = np.where(fl["target_ok"] == True, "quiet", "loud")  # noqa: E712
    hdr = ["visibility", "off", "on", "difference (pts)", "n off", "n on"]
    rows = []
    for v, g in fl.groupby("visibility"):
        off = g[g["scaffolding"] == "off"]["detection"].dropna()
        on = g[g["scaffolding"] == "on"]["detection"].dropna()
        rows.append([v, pct(off.mean(), 1), pct(on.mean(), 1),
                     f"{100 * (on.mean() - off.mean()):+.1f}",
                     len(off), len(on)])
        rows_csv.append({"outcome": "detection_by_visibility",
                         "visibility": v, "mean_off": float(off.mean()),
                         "mean_on": float(on.mean()),
                         "diff": float(on.mean() - off.mean()),
                         "n_off": len(off), "n_on": len(on)})
    rep("\n" + table(rows, hdr))

    m = gee_logit(fl, "detection", "C(scaffolding) * C(visibility)")
    results["gee_interaction"] = gee_terms(m)
    for line in apa_model(m, "Detection ~ scaffolding x visibility"):
        rep("  " + line)
    rep("  (The interaction term tests whether the scaffolding benefit "
        "differs between quiet and loud failures.)")

    pd.DataFrame(rows_csv).to_csv(OUT / "scaffolding_results.csv", index=False)
    (OUT / "scaffolding_results.json").write_text(
        json.dumps(results, indent=2, default=float), encoding="utf-8")
    (OUT / "scaffolding_report.txt").write_text("\n".join(REP) + "\n",
                                              encoding="utf-8")
    rep("\nSaved: scaffolding_results.csv, scaffolding_results.json, scaffolding_report.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
