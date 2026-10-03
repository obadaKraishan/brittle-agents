#!/usr/bin/env python3
"""
=============================================================================
 BRITTLE AGENTS | 01_environment/build_task_suite.py
=============================================================================
 Builds and freezes the task suite, after checking that the BFCL
 multi-turn environments can be driven locally, outside the official
 evaluation harness.

 What it does:
   1. Clones (sparse) the gorilla repo at the pinned commit to get the
      executable environment classes, and downloads the multi-turn task
      files from Hugging Face at the pinned dataset revision.
   2. Checks every environment class: instantiate, load a scenario,
      confirm methods are present.
   3. Filters tasks to those whose classes are all executable, with a
      number of user turns in the allowed range and a ground-truth answer.
   4. Samples n_tasks balanced across environment domains (seeded) and
      freezes them with stable IDs.

 The suite is written once; the script refuses to overwrite it without
 --overwrite.

 Usage:
   python 01_environment/build_task_suite.py
   python 01_environment/build_task_suite.py --overwrite
   python 01_environment/build_task_suite.py --gate-only   # just the checks

 Saves (in 01_environment/outputs/):
   task_suite.json / task_suite.csv    frozen task set
   suite_summary.txt                   selection log + gate results
=============================================================================
"""

import argparse
import json
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GORILLA_DIR = ROOT / "third_party" / "gorilla"
BFCL_PKG = GORILLA_DIR / "berkeley-function-call-leaderboard"
SPARSE_PATHS = [
    "berkeley-function-call-leaderboard/bfcl_eval/eval_checker/"
    "multi_turn_eval/func_source_code",
    "berkeley-function-call-leaderboard/bfcl_eval/constants",
]
TASK_FILES = {
    "base": "BFCL_v3_multi_turn_base.json",
    "composite": "BFCL_v3_multi_turn_composite.json",
}
# Executable ground-truth call sequences (with arguments), used to compute
# each task's reference end-state for recovery scoring.
ANSWER_FILES = {
    "base": "possible_answer/BFCL_v3_multi_turn_base.json",
    "composite": "possible_answer/BFCL_v3_multi_turn_composite.json",
}

LOG = []


def log(line: str = "") -> None:
    print(line, flush=True)
    LOG.append(line)


# ---------------------------------------------------------------- repo -----
def ensure_gorilla_repo(url: str, commit: str) -> bool:
    """Sparse-clone the environment source at a pinned commit (idempotent)."""
    if BFCL_PKG.exists():
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=GORILLA_DIR,
                              capture_output=True, text=True).stdout.strip()
        log(f"      gorilla repo already present at "
            f"{GORILLA_DIR.relative_to(ROOT)} (commit {head[:12]})")
        if head != commit:
            log(f"      [WARN] expected commit {commit[:12]}")
        return True
    GORILLA_DIR.parent.mkdir(parents=True, exist_ok=True)
    log(f"      cloning {url} at {commit[:12]} (sparse) ...")
    try:
        subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout",
                        url, str(GORILLA_DIR)],
                       check=True, capture_output=True, text=True)
        subprocess.run(["git", "sparse-checkout", "set", *SPARSE_PATHS],
                       cwd=GORILLA_DIR, check=True, capture_output=True,
                       text=True)
        subprocess.run(["git", "checkout", "--quiet", commit],
                       cwd=GORILLA_DIR, check=True, capture_output=True,
                       text=True)
        log("      clone OK")
        return True
    except subprocess.CalledProcessError as e:
        log(f"      [ERROR] clone failed: {e.stderr[:200]}")
        return False
    except FileNotFoundError:
        log("      [ERROR] git not found on PATH")
        return False


# ------------------------------------------------------------ gate ---------
def run_gate(initial_configs_by_class) -> tuple[dict, bool]:
    """Instantiate every environment class and prove it runs locally.

    Returns (per-class result dict, all_ok).
    """
    sys.path.insert(0, str(BFCL_PKG))
    from bfcl_eval.constants.executable_backend_config import (
        CLASS_FILE_PATH_MAPPING, STATELESS_CLASSES)
    import importlib

    results, all_ok = {}, True
    for cls_name, module_path in CLASS_FILE_PATH_MAPPING.items():
        entry = {"module": module_path,
                 "stateless": cls_name in STATELESS_CLASSES}
        try:
            mod = importlib.import_module(module_path)
            cls = getattr(mod, cls_name)
            inst = cls()
            cfg = initial_configs_by_class.get(cls_name)
            if cfg is not None and hasattr(inst, "_load_scenario"):
                inst._load_scenario(cfg)
                entry["scenario_loaded"] = True
            else:
                entry["scenario_loaded"] = False
            methods = [m for m in dir(inst)
                       if not m.startswith("_") and callable(getattr(inst, m))]
            entry["n_methods"] = len(methods)
            entry["ok"] = len(methods) > 0
        except Exception as e:  # import or instantiation failure
            entry.update(ok=False, error=repr(e)[:160])
        results[cls_name] = entry
        mark = "✓" if entry.get("ok") else "✗"
        detail = (f"{entry.get('n_methods', 0)} methods"
                  + (", scenario loaded" if entry.get("scenario_loaded")
                     else "")
                  + (" [stateless]" if entry["stateless"] else ""))
        if not entry.get("ok"):
            detail = entry.get("error", "failed")
            all_ok = False
        log(f"      [{mark}] {cls_name:<20} {detail}")
    return results, all_ok


# ------------------------------------------------------------ main ---------
def main() -> int:
    ap = argparse.ArgumentParser(description="Build and freeze the task suite")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow overwriting an existing frozen suite")
    ap.add_argument("--gate-only", action="store_true",
                    help="run the local-execution gate and stop")
    args = ap.parse_args()

    import yaml
    import pandas as pd
    from huggingface_hub import hf_hub_download

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    tcfg = cfg["tasks"]
    out_dir = ROOT / cfg["paths"]["env_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    suite_path = ROOT / cfg["paths"]["task_suite"]

    log("=" * 74)
    log(" BRITTLE AGENTS -- BUILD TASK SUITE")
    log(f" Timestamp : {datetime.now(timezone.utc).isoformat()}")
    log(f" Seed      : {cfg['project']['seed']} | target tasks: "
        f"{tcfg['n_tasks']}")
    log("=" * 74)

    if suite_path.exists() and not args.overwrite and not args.gate_only:
        log(f"\n[ABORT] A frozen suite already exists at "
            f"{suite_path.relative_to(ROOT)}.")
        log("        Rerun with --overwrite to replace it.")
        return 1

    # ------------------------------------------------------------- 1 source -
    log("\n[1/5] Sources")
    if not ensure_gorilla_repo(tcfg["gorilla_repo"], tcfg["gorilla_commit"]):
        return 1
    rows = []
    for split, fname in TASK_FILES.items():
        try:
            p = hf_hub_download(tcfg["hf_repo"], fname, repo_type="dataset",
                                revision=tcfg["hf_revision"])
            n = 0
            for line in Path(p).read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    r["_split"] = split
                    rows.append(r)
                    n += 1
            log(f"      {fname:<40} {n} tasks")
        except Exception as e:
            log(f"      [WARN] could not load {fname}: {repr(e)[:120]}")
    if not rows:
        log("[ERROR] no tasks downloaded.")
        return 1
    log(f"      total candidate tasks: {len(rows)}")

    answers = {}
    for split, fname in ANSWER_FILES.items():
        try:
            p = hf_hub_download(tcfg["hf_repo"], fname, repo_type="dataset",
                                revision=tcfg["hf_revision"])
            n = 0
            for line in Path(p).read_text().splitlines():
                if line.strip():
                    a = json.loads(line)
                    answers[a["id"]] = a.get("ground_truth")
                    n += 1
            log(f"      {fname:<48} {n} ground-truth answers")
        except Exception as e:
            log(f"      [WARN] could not load {fname}: {repr(e)[:110]}")
    log(f"      ground-truth answers available for {len(answers)} tasks")

    # --------------------------------------------------------------- 2 gate -
    log("\n[2/5] Do the environments run locally?")
    sample_cfg = {}
    for r in rows:                      # collect one config per class
        for cls, c in (r.get("initial_config") or {}).items():
            sample_cfg.setdefault(cls, c)
    gate, _ = run_gate(sample_cfg)
    executable = {c for c, e in gate.items() if e.get("ok")}
    stateful = {c for c in executable if not gate[c]["stateless"]}
    log(f"      executable classes: {len(executable)}/{len(gate)} "
        f"({len(stateful)} stateful)")

    # The gate is judged on COVERAGE, not on every class importing: some
    # classes are optional extras (web search, vector memory) that need
    # paid APIs or heavy models and are not used by the multi-turn base or
    # composite tasks. What matters is how many tasks remain runnable.
    runnable = [r for r in rows
                if set(r.get("involved_classes") or []) <= executable]
    coverage = len(runnable) / max(len(rows), 1)
    log(f"      task coverage: {len(runnable)}/{len(rows)} "
        f"({coverage:.0%}) runnable with the executable classes")
    failed = {c: gate[c].get("error", "") for c in gate if not gate[c].get("ok")}
    if failed:
        log("      excluded classes (optional extras, not required by the "
            "base/composite splits):")
        for c, e in failed.items():
            log(f"        - {c}: {e[:70]}")
    gate_ok = coverage >= 0.5 and len(stateful) >= 4
    if not gate_ok:
        log("\n      [!] Coverage too low. If most tasks are unrunnable, "
            "fall back to the purpose-built tool suite described in the "
            "roadmap.")

    if args.gate_only:
        log("\n" + "=" * 74)
        log(" GATE ONLY -- suite not written." +
            ("  RESULT: PASSED ✓" if gate_ok else "  RESULT: ISSUES ✗"))
        log("=" * 74)
        (out_dir / "suite_summary.txt").write_text("\n".join(LOG) + "\n",
                                                   encoding="utf-8")
        return 0 if gate_ok else 1

    # ------------------------------------------------------------- 3 filter -
    log("\n[3/5] Filtering to eligible tasks")
    eligible, reasons = [], {"non_executable": 0, "no_stateful": 0,
                             "too_short": 0, "too_long": 0,
                             "no_ground_truth": 0}
    for r in rows:
        classes = set(r.get("involved_classes") or [])
        n_turns = len(r.get("question") or [])
        if not classes <= executable:
            reasons["non_executable"] += 1
            continue
        if not (classes & stateful):
            reasons["no_stateful"] += 1
            continue
        if n_turns < tcfg["min_turns"]:
            reasons["too_short"] += 1
            continue
        if n_turns > tcfg["max_turns"]:
            reasons["too_long"] += 1
            continue
        if not answers.get(r["id"]):
            reasons["no_ground_truth"] = reasons.get("no_ground_truth", 0) + 1
            continue
        eligible.append(r)
    log(f"      eligible: {len(eligible)}  "
        f"(excluded: {reasons['non_executable']} non-executable class, "
        f"{reasons['no_stateful']} stateless-only, "
        f"{reasons['too_short']} too short, {reasons['too_long']} too long, "
        f"{reasons.get('no_ground_truth', 0)} no ground truth)")
    if len(eligible) < tcfg["n_tasks"]:
        log(f"      [WARN] only {len(eligible)} eligible tasks for a target "
            f"of {tcfg['n_tasks']}; the suite will use all of them.")

    # ------------------------------------------------------------- 4 sample -
    log(f"\n[4/5] Sampling {tcfg['n_tasks']} tasks balanced by "
        f"{tcfg['balance_by']}")
    rng = random.Random(cfg["project"]["seed"])
    by_domain = {}
    for r in eligible:
        key = "+".join(sorted(r["involved_classes"]))
        by_domain.setdefault(key, []).append(r)
    for v in by_domain.values():
        rng.shuffle(v)
    domains = sorted(by_domain)
    rng.shuffle(domains)

    chosen, i = [], 0
    while len(chosen) < min(tcfg["n_tasks"], len(eligible)):
        d = domains[i % len(domains)]
        if by_domain[d]:
            chosen.append(by_domain[d].pop())
        i += 1
        if i > 10000:
            break

    suite = []
    for n, r in enumerate(chosen, start=1):
        suite.append({
            "task_id": f"T{n:03d}",
            "bfcl_id": r["id"],
            "split": r["_split"],
            "domain": "+".join(sorted(r["involved_classes"])),
            "involved_classes": sorted(r["involved_classes"]),
            "initial_config": r["initial_config"],
            "turns": r["question"],
            "n_turns": len(r["question"]),
            "reference_path": r.get("path"),
            "ground_truth": answers.get(r["id"]),
        })

    counts = {}
    for s in suite:
        counts[s["domain"]] = counts.get(s["domain"], 0) + 1
    for d, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        log(f"      {d:<44} {c}")
    turn_counts = sorted(s["n_turns"] for s in suite)
    log(f"      turns per task: min {turn_counts[0]}, "
        f"median {turn_counts[len(turn_counts) // 2]}, "
        f"max {turn_counts[-1]}")

    # --------------------------------------------------------------- 5 save -
    suite_path.write_text(json.dumps(suite, indent=2), encoding="utf-8")
    flat = [{k: v for k, v in s.items()
             if k not in ("initial_config", "turns", "reference_path")}
            for s in suite]
    pd.DataFrame(flat).to_csv(out_dir / "task_suite.csv", index=False)
    (out_dir / "gate_report.json").write_text(json.dumps(gate, indent=2),
                                              encoding="utf-8")
    log(f"\n[5/5] Suite FROZEN: {len(suite)} tasks")
    log(f"      JSON -> {suite_path.relative_to(ROOT)}")
    log(f"      CSV  -> {(out_dir / 'task_suite.csv').relative_to(ROOT)}")

    log("\n" + "=" * 74)
    log(" RESULT: SUITE BUILT ✓")
    log("=" * 74)
    (out_dir / "suite_summary.txt").write_text("\n".join(LOG) + "\n",
                                               encoding="utf-8")
    print(f"\nSummary saved -> "
          f"{(out_dir / 'suite_summary.txt').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
