#!/usr/bin/env python3
"""
=============================================================================
 BRITTLE AGENTS | 04_codebook/generate_codebook.py
=============================================================================
 Documents every variable in master_trials.csv: name, type, range or value
 set, missingness, and how it was derived.

 Usage:
   python 04_codebook/generate_codebook.py

 Saves: 04_codebook/outputs/codebook.csv, codebook.md
=============================================================================
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DESC = {
    "trial_id": "Unique key: model|fault|position|scaffolding|task|seed.",
    "model": "Model name from config.yaml (6 models, 3 matched families).",
    "reasoning": "True if the model runs an extended thinking phase.",
    "pair": "Model family, so reasoning/instruct contrasts stay within "
            "lineage.",
    "fault": "Injected fault type, or 'clean' for the no-fault baseline.",
    "position": "Injection setting: 'late' = first eligible call at or "
                "after step floor(T/2), T = user turns; 'na' for clean.",
    "scaffolding": "'on' if the system prompt asked the agent to check each "
                   "tool result; run on one model pair (secondary "
                   "analysis).",
    "task_id": "Frozen task suite identifier (T001-T024).",
    "domain": "Environment classes involved in the task.",
    "seed": "Repetition index (0 or 1). It seeds the injection layer "
            "only; model sampling is not seeded.",
    "fault_fired": "True if the fault actually triggered. Victim-based "
                   "faults fire only when the agent calls the victim tool; "
                   "trials where it never fired are excluded from analysis.",
    "fault_step": "Step at which the fault fired.",
    "fault_tool": "Tool call the fault landed on.",
    "victim_tool": "Pre-selected target for victim-based faults, drawn from "
                   "the task's ground-truth call path.",
    "n_steps": "Tool calls executed in the trial.",
    "n_model_calls": "Assistant turns in the trial.",
    "finish_reason": "'completed' or 'step_budget'.",
    "budget_exhausted": "True if the trial ended by hitting the step cap.",
    "detection_rule": "Keyword heuristic for problem language (superseded "
                      "by the judge coding; retained for transparency).",
    "detection_judge": "Judge coding collapsed to noticed vs not.",
    "judge_code": "Three-way judge coding: none / acknowledged / "
                  "rationalized. Descriptive only (three-way kappa = "
                  ".47 against hand-coding).",
    "judge_evidence": "Short quotation the judge cited for its code.",
    "detection": "PRIMARY MEASURE. Agent treated the tool result as "
                 "problematic (noticed vs not; kappa = .69 against "
                 "hand-coding on the 92 trials both coded). Falls back to "
                 "detection_rule where the judge did not code a trial.",
    "rationalized": "Judge coded the response as noticing an anomaly then "
                    "explaining it away. Descriptive only.",
    "replanned": "First post-fault action differed from the failed action "
                 "(different tool, or same tool with different arguments).",
    "perseveration": "Three or more consecutive calls to the same tool "
                     "after the fault (arguments may differ). Undefined "
                     "(False) for clean trials.",
    "repeat_identical": "Three or more consecutive identical calls (same "
                        "tool and arguments) after the fault. Undefined "
                        "(False) for clean trials.",
    "n_calls_after_fault": "Tool calls made after the fault fired.",
    "state_agreement_ref": "Fraction of the benchmark ground-truth end "
                           "state the agent's final state matches.",
    "state_agreement_own": "Fraction of the reference clean run's end "
                           "state that the final state matches. Reference: "
                           "the same model's scaffolding-off clean run of "
                           "the task with the same seed; for scaffolding-off "
                           "clean trials, the run with the other seed.",
    "recovered": "True if state_agreement_own = 1. On clean trials this "
                 "is the agreement between two fault-free runs.",
    "recovery_cost": "Steps used minus the model's mean steps on the same "
                     "task without a fault.",
    "false_alarm": "Clean trial whose target call SUCCEEDED but which the "
                   "agent still treated as problematic.",
    "fault_visibility": "'loud' (observation carries an explicit error), "
                        "'quiet' (observation looks successful but the "
                        "value is wrong), or 'baseline' (clean).",
    "target_tool": "Tool call the coding focused on (the fault step, or the "
                   "trajectory midpoint for clean trials).",
    "target_ok": "Whether that call returned successfully.",
    "prompt_tokens": "Input tokens consumed by the trial.",
    "completion_tokens": "Output tokens produced by the trial.",
    "latency_s": "Wall-clock seconds for the trial.",
    "leaked_tool_calls": "Assistant turns that wrote tool-call syntax as "
                         "plain text instead of emitting a structured call; "
                         "these never execute.",
    "is_fault": "Derived in analysis: fault != 'clean'.",
}


def main() -> int:
    import yaml
    import pandas as pd

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    master = ROOT / cfg["paths"]["processing_dir"] / "master_trials.csv"
    out_dir = ROOT / cfg["paths"]["codebook_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    if not master.exists():
        print("[ERROR] master_trials.csv not found -- run 03_data_processing "
              "first.")
        return 1
    df = pd.read_csv(master)

    print("=" * 74)
    print(" BRITTLE AGENTS -- GENERATE CODEBOOK")
    print(f" {datetime.now(timezone.utc).isoformat()}")
    print(f" Source: {master.relative_to(ROOT)} "
          f"({df.shape[0]:,} rows x {df.shape[1]} cols)")
    print("=" * 74)

    rows = []
    for col in df.columns:
        s = df[col]
        if s.dtype.kind in "if":
            rng = (f"[{s.min():.3f}, {s.max():.3f}]" if s.notna().any()
                   else "all missing")
        elif s.dtype == bool:
            rng = f"True: {int(s.sum())}, False: {int((~s).sum())}"
        else:
            vals = sorted(map(str, s.dropna().unique()))
            rng = ", ".join(vals[:6]) + (", ..." if len(vals) > 6 else "")
        rows.append({"variable": col, "dtype": str(s.dtype),
                     "n_missing": int(s.isna().sum()),
                     "range_or_values": rng,
                     "description": DESC.get(col, "(add description)")})
        print(f"  {col:<24}{str(s.dtype):<9}miss={s.isna().sum():<5}"
              f"{rng[:42]}")

    pd.DataFrame(rows).to_csv(out_dir / "codebook.csv", index=False)
    md = ["# Brittle Agents -- Variable Codebook",
          f"\nGenerated {datetime.now(timezone.utc).date()} from "
          f"`master_trials.csv` ({df.shape[0]:,} trials).\n",
          "Primary outcome is `detection` (noticed vs not, Cohen's kappa = "
          ".69 against hand-coding). The three-way `judge_code` did not "
          "reach acceptable agreement (kappa = .47) and is reported "
          "descriptively only.\n",
          "| Variable | Type | Missing | Range / values | Description |",
          "|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| `{r['variable']}` | {r['dtype']} | {r['n_missing']} | "
                  f"{r['range_or_values']} | {r['description']} |")
    (out_dir / "codebook.md").write_text("\n".join(md) + "\n",
                                         encoding="utf-8")
    print(f"\n Saved: codebook.csv, codebook.md -> "
          f"{out_dir.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
