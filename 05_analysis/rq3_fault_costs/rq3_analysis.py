#!/usr/bin/env python3
"""
=============================================================================
 BRITTLE AGENTS | 05_analysis/rq3_fault_costs/rq3_analysis.py
=============================================================================
 RQ3: What does a single tool failure cost in recovery, repeated calls, and
      effort?

 Outcomes per fault type:
   recovered          end state matches the model's own clean run of the
                      task (same seed). The clean baseline scores each
                      scaffolding-off clean run against the OTHER seed's
                      clean run, i.e. clean-vs-clean agreement.
   perseveration      three or more consecutive calls to the same tool after
                      the fault (arguments may differ)
   repeat_identical   three or more consecutive identical calls (same tool
                      and arguments) after the fault
   budget_exhausted   the trial ended by hitting the step cap
   recovery_cost      extra steps relative to the model's clean runs

 Repetition is defined relative to the fault step, so it is reported for
 fault trials only and never compared with clean trials.

 Also reported: when the fault fired relative to the task's reference
 trajectory, and whether noticing a failure predicts recovery.

 Usage:
   python 05_analysis/rq3_fault_costs/rq3_analysis.py

 Saves: outputs/rq3_results.csv, rq3_results.json,
        rq3_report.txt
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

from analysis_utils import (ROOT, load_master, cluster_boot_ci, prop_test,
                            holm, gee_logit, gee_terms, apa_prop,
                            apa_model, fmt_p, f2, pct, table, boot_ci)  # noqa: E402

OUT, REP = HERE / "outputs", []
FAULTS = ["timeout", "missing_tool", "schema_drift", "silent_corruption"]


def rep(s=""):
    print(s, flush=True)
    REP.append(s)


def ci_txt(ci):
    return f"[{pct(ci[0], 1)}, {pct(ci[1], 1)}]"


def reference_calls(cfg):
    """Number of ground-truth calls per task (the reference trajectory)."""
    suite = json.loads((ROOT / cfg["paths"]["task_suite"]).read_text())
    out = {}
    for t in suite:
        out[t["task_id"]] = sum(len(turn) if isinstance(turn, list) else 1
                                for turn in (t.get("ground_truth") or []))
    return out


def main():
    df, cfg = load_master()
    OUT.mkdir(parents=True, exist_ok=True)
    rep("=" * 76)
    rep(" RQ3 -- WHAT A SINGLE FAULT COSTS: RECOVERY, REPETITION, EFFORT")
    rep(f" Generated: {datetime.now(timezone.utc).isoformat()}")
    rep("=" * 76)

    clean = df[df["fault"] == "clean"]
    fl = df[df["is_fault"]]
    results, rows_csv = {}, []

    # ------------------------------------------- clean-vs-clean baseline --
    rep("\n--- Clean baseline: agreement between clean runs ---")
    rep(" Scaffolding-off clean runs are scored against the other seed's clean"
        "\n run; scaffolding-on clean runs against the scaffolding-off clean run"
        "\n with the same seed.")
    rows, base = [], {}
    for name, g in [("all clean trials", clean),
                    ("scaffolding off (seed vs seed)",
                     clean[clean["scaffolding"] == "off"]),
                    ("scaffolding on (vs off run)",
                     clean[clean["scaffolding"] == "on"])]:
        v = g["recovered"].dropna()
        ci = cluster_boot_ci(g, "recovered")
        rows.append([name, len(v), pct(v.mean(), 1), ci_txt(ci)])
        base[name] = dict(n=len(v), rate=float(v.mean()), ci=ci)
    for m, g in clean.groupby("model"):
        v = g["recovered"].dropna()
        ci = cluster_boot_ci(g, "recovered")
        rows.append([f"  {m}", len(v), pct(v.mean(), 1), ci_txt(ci)])
        base[m] = dict(n=len(v), rate=float(v.mean()), ci=ci)
        rows_csv.append({"outcome": "clean_agreement", "group": m,
                         "n": len(v), "rate": float(v.mean()),
                         "ci_lo": ci[0], "ci_hi": ci[1]})
    rep("\n" + table(rows, ["group", "n", "agreement",
                            "95% CI (task-clustered)"]))
    results["clean_baseline"] = base
    cv = clean["recovered"].dropna()

    # ------------------------------------------------ recovery ----------
    rep("\n--- Recovery by fault type against the clean baseline ---")
    d = df[df["recovered"].notna()].copy()
    d["condition"] = pd.Categorical(d["fault"], ["clean"] + FAULTS)
    m0 = gee_logit(d, "recovered", "C(condition)")
    m1 = gee_logit(d, "recovered", "C(condition) + C(model)")
    t0, t1 = gee_terms(m0), gee_terms(m1)
    p0 = holm([t0[f"C(condition)[T.{f}]"]["p"] for f in FAULTS])
    p1 = holm([t1[f"C(condition)[T.{f}]"]["p"] for f in FAULTS])
    rows, rec = [], {}
    for f_, a0, a1 in zip(FAULTS, p0, p1):
        g = df[df["fault"] == f_]
        v = g["recovered"].dropna()
        ci = cluster_boot_ci(g, "recovered")
        h = prop_test(v.sum(), len(v), cv.sum(), len(cv))["h"]
        e0, e1 = t0[f"C(condition)[T.{f_}]"], t1[f"C(condition)[T.{f_}]"]
        rec[f_] = dict(n=len(v), rate=float(v.mean()), ci=ci, h=h,
                       ci_excludes_baseline=bool(ci[1] < cv.mean()
                                                 or ci[0] > cv.mean()),
                       gee=dict(e0, p_holm=float(a0)),
                       gee_model=dict(e1, p_holm=float(a1)))
        rows.append([f_, len(v), pct(v.mean(), 1), ci_txt(ci), f2(h),
                     f"{f2(e0['OR'])} ({fmt_p(a0).replace('p ', '')})",
                     f"{f2(e1['OR'])} ({fmt_p(a1).replace('p ', '')})"])
        rows_csv.append({"outcome": "recovered", "fault": f_, "n": len(v),
                         "rate": float(v.mean()), "ci_lo": ci[0],
                         "ci_hi": ci[1], "h": h, "or_gee": e0["OR"],
                         "z_gee": e0["z"], "p_holm_gee": float(a0),
                         "or_gee_model": e1["OR"], "z_gee_model": e1["z"],
                         "p_holm_gee_model": float(a1)})
    rep(f"\n clean baseline: {pct(cv.mean(), 1)} (n = {len(cv)}), "
        f"task-clustered CI {ci_txt(base['all clean trials']['ci'])}")
    rep("\n" + table(rows, ["fault", "n", "recovered", "95% CI", "h",
                            "GEE OR (p_holm)", "+ model OR (p_holm)"]))
    rep(" GEE: recovered ~ condition (clean = reference), clustered on task;"
        "\n second column adds model as a covariate. Holm across the four "
        "faults.")
    results["recovery"] = rec
    for line in apa_model(m0, "Recovery ~ condition"):
        rep("  " + line)
    for line in apa_model(m1, "Recovery ~ condition + model"):
        rep("  " + line)

    # ------------------------------------- repetition and effort ---------
    rep("\n--- Repetition and effort ---")
    rows = []
    for f_ in ["clean"] + FAULTS:
        g = df[df["fault"] == f_]
        rc = g["recovery_cost"].dropna()
        ci = boot_ci(rc)
        is_f = f_ != "clean"
        same = g["perseveration"].mean() if is_f else np.nan
        ident = g["repeat_identical"].mean() if is_f else np.nan
        rows.append([f_, pct(g["recovered"].mean(), 1),
                     pct(same, 1) if is_f else "--",
                     pct(ident, 1) if is_f else "--",
                     pct(g["budget_exhausted"].mean(), 1),
                     f2(g["n_steps"].mean(), 1),
                     f"{np.mean(rc):+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]"])
        rows_csv.append({"outcome": "effort", "fault": f_,
                         "recovered": float(g["recovered"].mean()),
                         "same_tool_repeat": float(same),
                         "identical_repeat": float(ident),
                         "budget_exhausted": float(
                             g["budget_exhausted"].mean()),
                         "mean_steps": float(g["n_steps"].mean()),
                         "recovery_cost": float(np.mean(rc))})
    rep("\n" + table(rows, ["condition", "recovered", "same tool x3",
                            "identical x3", "cap hit", "steps",
                            "extra steps [95% CI]"]))
    rep(" Repetition counts calls after the fault step, so it is undefined "
        "for clean trials.")

    # ------------------------------------------------ injection timing ---
    rep("\n--- When the fault fired ---")
    ref = reference_calls(cfg)
    ft = fl[["fault", "task_id", "fault_step"]].copy()
    ft["ref_calls"] = ft["task_id"].map(ref)
    ft["early"] = ft["fault_step"] <= 2
    ft["before_mid"] = ft["fault_step"] < ft["ref_calls"] / 2
    rows, timing = [], {}
    for name, g in [("all faults", ft)] + list(ft.groupby("fault")):
        timing[name] = dict(n=len(g), median=float(g["fault_step"].median()),
                            step_1_2=float(g["early"].mean()),
                            before_ref_mid=float(g["before_mid"].mean()))
        rows.append([name, len(g), f"{g['fault_step'].median():.0f}",
                     pct(g["early"].mean(), 1),
                     pct(g["before_mid"].mean(), 1)])
    rep("\n" + table(rows, ["fault", "n", "median step", "at step 1 or 2",
                            "before reference midpoint"]))
    rep(f" Reference trajectories: median {np.median(list(ref.values())):.0f}"
        f" ground-truth calls (range {min(ref.values())}-"
        f"{max(ref.values())}). The target step is half the number of user "
        "turns.")
    results["timing"] = timing

    # ------------------------------------ detection x recovery ----------
    rep("\n--- Does noticing help? Detection x recovery on fault trials ---")
    fd = fl.dropna(subset=["detection", "recovered"])
    rec_det = fd[fd["detection"] == 1]["recovered"]
    rec_und = fd[fd["detection"] == 0]["recovered"]
    t = prop_test(rec_det.sum(), len(rec_det), rec_und.sum(), len(rec_und))
    rep(f"\n      recovery when the failure was noticed:     "
        f"{pct(rec_det.mean(), 1)} (n = {len(rec_det)})")
    rep(f"      recovery when the failure went unnoticed:  "
        f"{pct(rec_und.mean(), 1)} (n = {len(rec_und)})")
    rep("  " + apa_prop("Noticed vs unnoticed", t))
    models = {}
    for key, rhs in [("raw", "detection"),
                     ("fault", "detection + C(fault)"),
                     ("fault_model", "detection + C(fault) + C(model)")]:
        m = gee_logit(fd, "recovered", rhs)
        models[key] = gee_terms(m)["detection"]
        for line in apa_model(m, f"Recovery ~ {rhs}"):
            rep("  " + line)
    results["detection_x_recovery"] = {"test": t, "gee": models}

    pd.DataFrame(rows_csv).to_csv(OUT / "rq3_results.csv", index=False)
    (OUT / "rq3_results.json").write_text(
        json.dumps(results, indent=2, default=float), encoding="utf-8")
    (OUT / "rq3_report.txt").write_text("\n".join(REP) + "\n",
                                              encoding="utf-8")
    rep("\nSaved: rq3_results.csv, rq3_results.json, rq3_report.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
