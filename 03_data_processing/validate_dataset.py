#!/usr/bin/env python3
"""
=============================================================================
 BRITTLE AGENTS | 03_data_processing/validate_dataset.py
=============================================================================
 Data-quality checks on master_trials.csv.

 Checks:
   1. GRID COMPLETENESS   -- every model x fault x scaffolding x task x seed
                             cell present
   2. TRIGGER RATES       -- how often each fault actually fired (reported,
                             not enforced: victim-based faults fire only when
                             the agent calls the victim tool)
   3. MEASURE SANITY      -- rates in range, recovery defined, clean trials
                             never flagged as faulted
   4. DATA QUALITY        -- step-budget exhaustion, tool-call text leakage,
                             judge coverage

 Judge-human agreement is computed by 04_codebook/judge_agreement.py.

 Exit code 0 = analysis-ready.

 Usage:
   python 03_data_processing/validate_dataset.py

 Saves:
   03_data_processing/outputs/validation_report.txt
=============================================================================
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LOG = []


def log(s: str = "") -> None:
    print(s, flush=True)
    LOG.append(s)


def check(label: str, ok: bool, detail: str = "") -> bool:
    log(f"  [{'✓' if ok else '✗'}] {label}" + (f"  -- {detail}" if detail else ""))
    return ok


def main() -> int:
    import yaml
    import pandas as pd

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    out_dir = ROOT / cfg["paths"]["processing_dir"]
    master = out_dir / "master_trials.csv"
    if not master.exists():
        log("[ERROR] master_trials.csv not found. Run score_trajectories.py "
            "first.")
        return 1
    df = pd.read_csv(master)
    suite = json.loads((ROOT / cfg["paths"]["task_suite"]).read_text())

    log("=" * 74)
    log(" BRITTLE AGENTS -- VALIDATE DATASET")
    log(f" Timestamp : {datetime.now(timezone.utc).isoformat()}")
    log("=" * 74)
    all_ok = True

    # ----------------------------------------------------- 1 completeness --
    log(f"\n[1/4] Grid completeness ({len(df):,} trials)")
    n_tasks = df["task_id"].nunique()
    n_seeds = cfg["agent"]["n_seeds"]
    n_cells = 1 + (len(cfg["faults"]) - 1) * len(cfg["fault_positions"])
    scaff_models = set(cfg["agent"].get("scaffolding_models", []))
    for m in sorted(df["model"].unique()):
        sub = df[df["model"] == m]
        arms = 2 if m in scaff_models else 1
        expected = n_tasks * n_cells * n_seeds * arms
        all_ok &= check(f"{m:<28} {len(sub)}/{expected} trials",
                        len(sub) == expected)
    log(f"      tasks in data: {n_tasks} (suite has {len(suite)}); "
        f"seeds: {sorted(df['seed'].unique())}; "
        f"scaffolding arms: {sorted(df['scaffolding'].unique())}")

    # -------------------------------------------------------- 2 triggers --
    log("\n[2/4] Fault trigger rates (reported, not enforced)")
    for f_, g in df[df["fault"] != "clean"].groupby("fault"):
        rate = g["fault_fired"].mean()
        log(f"      {f_:<20}{g['fault_fired'].sum():>5}/{len(g):<5} "
            f"{rate:>5.0%}"
            + ("   (victim-based: fires only if the agent calls the "
               "victim tool)" if g["victim_tool"].notna().any() else ""))
    clean_fired = df[(df["fault"] == "clean")]["fault_fired"].sum()
    all_ok &= check("no clean trial recorded a fault", clean_fired == 0,
                    f"{clean_fired} found" if clean_fired else "")

    # --------------------------------------------------------- 3 sanity ---
    log("\n[3/4] Measure sanity")
    for col in ["detection", "replanned", "perseveration", "false_alarm"]:
        if col in df:
            v = df[col].dropna()
            ok = v.isin([True, False, 0, 1]).all()
            all_ok &= check(f"{col} is boolean", bool(ok))
    for col in ["state_agreement_ref", "state_agreement_own"]:
        v = df[col].dropna()
        ok = ((v >= 0) & (v <= 1)).all() if len(v) else True
        all_ok &= check(f"{col} within [0, 1]", bool(ok),
                        f"n = {len(v)}")
    rec = df["recovered"].dropna()
    all_ok &= check(f"recovery defined for {len(rec):,} trials",
                    len(rec) > 0.5 * len(df),
                    f"{len(rec) / len(df):.0%} of trials have a clean-run "
                    f"baseline")
    nofire = df[(df["fault"] != "clean") & (~df["fault_fired"])]
    log(f"      {len(nofire):,} trials excluded from analysis "
        f"(fault never fired) -- analysable n = {len(df) - len(nofire):,}")

    # ------------------------------------------------------- 4 quality ----
    log("\n[4/4] Data quality")
    cap = df["budget_exhausted"].mean()
    log(f"      step-budget exhausted: {cap:.0%} of trials "
        f"(cap = {cfg['agent']['max_steps']} steps)")
    if "leaked_tool_calls" in df:
        leak = df.groupby("model")["leaked_tool_calls"].mean()
        bad = leak[leak > 0.1]
        if len(bad):
            log("      [!] tool-call text leakage (calls written as text, "
                "never executed):")
            for m, v in bad.items():
                log(f"          {m:<28}{v:>6.2f} per trial")
    jc = df["judge_code"].notna().mean() if "judge_code" in df else 0
    log(f"      judge coverage: {jc:.0%} of trials coded")

    log("\n" + "=" * 74)
    if all_ok:
        log(" RESULT: DATASET VALID ✓")
    else:
        log(" RESULT: ISSUES FOUND ✗ -- review the flagged items above.")
    log("=" * 74)
    (out_dir / "validation_report.txt").write_text("\n".join(LOG) + "\n",
                                                   encoding="utf-8")
    print(f"\nReport saved -> "
          f"{(out_dir / 'validation_report.txt').relative_to(ROOT)}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
