#!/usr/bin/env python3
"""
=============================================================================
 BRITTLE AGENTS | reproduce.py
=============================================================================
 Reproduces every number, table, and figure in the paper from the released
 trajectories and cached judge codings. No API calls are made.

   1. score trajectories (cached judge codings only)   03_data_processing
   2. validate the scored dataset                      03_data_processing
   3. codebook and judge-human agreement               04_codebook
   4. RQ1, RQ2, RQ3, secondary analysis                05_analysis
   5. tables and figures                               06_figures_tables
   6. independent check of the paper's numbers         tests

 Options:
   --self-test   first fetch the BFCL environment code at the pinned commit
                 (network: GitHub) and run the fault-injection self-test
   --tex PATH    also compare the paper source's tables with the recomputed
                 values and fail on any TBD left in it

 Usage:
   python reproduce.py
   python reproduce.py --self-test --tex path/to/main.tex
=============================================================================
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STEPS = [
    ["03_data_processing/score_trajectories.py", "--cache-only"],
    ["03_data_processing/validate_dataset.py"],
    ["04_codebook/generate_codebook.py"],
    ["04_codebook/judge_agreement.py"],
    ["05_analysis/rq1_detection/rq1_analysis.py"],
    ["05_analysis/rq2_reasoning_recovery/rq2_analysis.py"],
    ["05_analysis/rq3_fault_costs/rq3_analysis.py"],
    ["05_analysis/secondary_scaffolding/scaffolding_analysis.py"],
    ["06_figures_tables/make_tables.py"],
    ["06_figures_tables/make_figures.py"],
]


def run(cmd):
    print(f"\n>>> python {' '.join(cmd)}", flush=True)
    done = subprocess.run([sys.executable, *cmd], cwd=ROOT,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True)
    tail = done.stdout.strip().splitlines()[-4:]
    print("\n".join("    " + line for line in tail))
    if done.returncode != 0:
        print(done.stdout)
        sys.exit(f"[FAILED] {' '.join(cmd)} (exit {done.returncode})")


def main():
    ap = argparse.ArgumentParser(description="Reproduce the paper")
    ap.add_argument("--self-test", action="store_true",
                    help="fetch the BFCL environments and run the self-test")
    ap.add_argument("--tex", type=Path, help="paper source to compare")
    args = ap.parse_args()

    if args.self_test:
        sys.path.insert(0, str(ROOT / "01_environment"))
        import yaml
        from build_task_suite import ensure_gorilla_repo
        tasks = yaml.safe_load((ROOT / "config.yaml").read_text())["tasks"]
        if not ensure_gorilla_repo(tasks["gorilla_repo"],
                                   tasks["gorilla_commit"]):
            sys.exit("[FAILED] could not fetch the BFCL environment code")
        run(["01_environment/fault_injection.py", "--self-test"])

    for step in STEPS:
        run(step)
    test = ["tests/verify_paper_numbers.py"]
    if args.tex:
        test += ["--tex", str(args.tex.resolve())]
    run(test)
    print("\nAll steps completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
