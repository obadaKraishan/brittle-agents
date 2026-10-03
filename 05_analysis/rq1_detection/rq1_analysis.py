#!/usr/bin/env python3
"""
=============================================================================
 BRITTLE AGENTS | 05_analysis/rq1_detection/rq1_analysis.py
=============================================================================
 RQ1: What fraction of injected tool failures does the agent detect, and
      does detection depend on how visible the failure is?

 The central contrast is fault VISIBILITY, not fault name:
   loud      the observation carries an explicit error (timeout, missing
             tool, rejected argument)
   quiet     the observation still looks like a success but the value is
             wrong (silent corruption)
   baseline  clean trials whose target call succeeded -- the false-alarm
             rate, i.e. how often an agent reports a problem when there is
             none

 Detection is the two-way coding (noticed vs not), the only scheme that
 reached acceptable agreement with hand-coding.

 Statistics: proportions with task-clustered bootstrap CIs, two-proportion
 z-tests with Cohen's h, and a binomial GEE clustered on task.

 Usage:
   python 05_analysis/rq1_detection/rq1_analysis.py

 Saves: outputs/rq1_results.csv, rq1_results.json, rq1_report.txt
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

from analysis_utils import (load_master, cluster_boot_ci, prop_test, holm,
                            gee_logit, gee_terms, apa_prop, apa_model,
                            fmt_p, f2, pct, table)  # noqa: E402

OUT, REP = HERE / "outputs", []


def rep(s=""):
    print(s, flush=True)
    REP.append(s)


def main():
    df, cfg = load_master()
    OUT.mkdir(parents=True, exist_ok=True)
    rep("=" * 76)
    rep(" RQ1 -- DETECTION OF TOOL FAILURES BY FAULT VISIBILITY")
    rep(f" Generated: {datetime.now(timezone.utc).isoformat()}")
    rep("=" * 76)
    rep("\n Detection = agent explicitly treated the tool result as "
        "problematic")
    rep(" (two-way judge coding; agreement with hand-coding in "
        "04_codebook/outputs/judge_agreement.txt).")

    # baseline uses only clean trials whose target call actually succeeded
    base = df[(df["fault"] == "clean") & (df["target_ok"] == True)]  # noqa: E712
    loud = df[(df["is_fault"]) & (df["target_ok"] == False)]  # noqa: E712
    quiet = df[(df["is_fault"]) & (df["target_ok"] == True)]  # noqa: E712

    results, rows_csv = {}, []

    # ------------------------------------------------- visibility -------
    rep("\n--- Detection by visibility ---")
    hdr = ["condition", "n", "detected", "rate", "95% CI (task-clustered)"]
    rows = []
    for name, g in [("baseline (no fault)", base), ("loud fault", loud),
                    ("quiet fault", quiet)]:
        v = g["detection"].dropna()
        ci = cluster_boot_ci(g, "detection")
        rows.append([name, len(v), int(v.sum()), pct(v.mean(), 1),
                     f"[{pct(ci[0], 1)}, {pct(ci[1], 1)}]"])
        rows_csv.append({"analysis": "visibility", "condition": name,
                         "n": len(v), "k": int(v.sum()),
                         "rate": float(v.mean()), "ci_lo": ci[0],
                         "ci_hi": ci[1]})
    rep("\n" + table(rows, hdr))

    lv = loud["detection"].dropna()
    qv = quiet["detection"].dropna()
    bv = base["detection"].dropna()
    t_lq = prop_test(lv.sum(), len(lv), qv.sum(), len(qv))
    t_qb = prop_test(qv.sum(), len(qv), bv.sum(), len(bv))
    t_lb = prop_test(lv.sum(), len(lv), bv.sum(), len(bv))
    adj = holm([t_lq["p"], t_qb["p"], t_lb["p"]])
    for r, a in zip((t_lq, t_qb, t_lb), adj):
        r["p_holm"] = float(a)
    results["visibility"] = {"loud_vs_quiet": t_lq, "quiet_vs_baseline": t_qb,
                             "loud_vs_baseline": t_lb}
    rep("\nAPA sentences:")
    rep("  " + apa_prop("Loud vs quiet faults", t_lq)
        + f" Holm-corrected {fmt_p(t_lq['p_holm'])}.")
    rep("  " + apa_prop("Quiet faults vs no-fault baseline", t_qb)
        + f" Holm-corrected {fmt_p(t_qb['p_holm'])}.")
    rep("  " + apa_prop("Loud faults vs no-fault baseline", t_lb)
        + f" Holm-corrected {fmt_p(t_lb['p_holm'])}.")

    # ------------------------------------------------- per fault type ---
    rep("\n--- Detection by fault type ---")
    hdr = ["fault", "n", "rate", "95% CI", "vs baseline (z, p_holm, h)"]
    rows, ps, keeps = [], [], []
    for f_, g in df[df["is_fault"]].groupby("fault"):
        v = g["detection"].dropna()
        ci = cluster_boot_ci(g, "detection")
        t = prop_test(v.sum(), len(v), bv.sum(), len(bv))
        ps.append(t["p"])
        keeps.append((f_, len(v), v.mean(), ci, t))
    adj = holm(ps)
    for (f_, n, rate, ci, t), a in zip(keeps, adj):
        t["p_holm"] = float(a)
        rows.append([f_, n, pct(rate, 1),
                     f"[{pct(ci[0], 1)}, {pct(ci[1], 1)}]",
                     f"z = {f2(t['z'])}, {fmt_p(a)}, h = {f2(t['h'])}"])
        rows_csv.append({"analysis": "fault_type", "condition": f_, "n": n,
                         "rate": float(rate), "ci_lo": ci[0], "ci_hi": ci[1],
                         "z_vs_baseline": t["z"], "p_holm": float(a),
                         "cohens_h": t["h"]})
    rep("\n" + table(rows, hdr))
    results["fault_type"] = {k[0]: k[4] for k in keeps}

    # ------------------------------------------------------- model -----
    rep("\n--- Binomial GEE (task-clustered): detection ~ visibility ---")
    d = df[df["is_fault"] | (df["fault"] == "clean")].copy()
    d["visibility"] = np.where(~d["is_fault"], "baseline",
                               np.where(d["target_ok"] == True,  # noqa: E712
                                        "quiet", "loud"))
    d = d[(d["visibility"] != "baseline") | (d["target_ok"] == True)]  # noqa: E712
    d["visibility"] = pd.Categorical(d["visibility"],
                                     ["baseline", "quiet", "loud"])
    m = gee_logit(d, "detection", "C(visibility)")
    for line in apa_model(m, "Detection by visibility"):
        rep("  " + line)

    # fault trials only: quiet vs loud, adding reasoning capability
    dfault = d[d["is_fault"]].copy()
    dfault["visibility"] = pd.Categorical(
        dfault["visibility"].astype(str), ["quiet", "loud"])
    m2 = gee_logit(dfault, "detection", "C(visibility) + C(reasoning)")
    results["gee_visibility"] = gee_terms(m)
    results["gee_visibility_reasoning"] = gee_terms(m2)
    for line in apa_model(m2, "Fault trials only, adding reasoning "
                              "capability (quiet = reference)"):
        rep("  " + line)

    pd.DataFrame(rows_csv).to_csv(OUT / "rq1_results.csv", index=False)
    (OUT / "rq1_results.json").write_text(
        json.dumps(results, indent=2, default=float), encoding="utf-8")
    (OUT / "rq1_report.txt").write_text("\n".join(REP) + "\n",
                                              encoding="utf-8")
    rep("\nSaved: rq1_results.csv, rq1_results.json, rq1_report.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
