#!/usr/bin/env python3
"""
=============================================================================
 BRITTLE AGENTS | 00_setup/setup_check.py
=============================================================================
 Environment check. Only step 6 (--ping) makes a network call to the model
 provider; an API key is needed only to collect new trajectories or code new
 trials with the judge, not to reproduce the paper.

 Checks, in order:
   1. Python version and required packages
   2. config.yaml loads and contains every required section, with a
      consistency pass over the design grid (models, faults, conditions)
   3. .env present and the API key set (reported, not required)
   4. All output directories exist (creates missing ones)
   5. Task source reachable: the BFCL dataset repo on HuggingFace
   6. (optional, --ping) one live API call to confirm the key authenticates

 Usage:
   python 00_setup/setup_check.py            # offline checks + HF reachability
   python 00_setup/setup_check.py --ping     # + a live API ping

 Saves:
   00_setup/setup_report.txt   (local; not part of the release)
=============================================================================
"""

import argparse
import importlib
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REPORT = []


def log(line: str = "") -> None:
    print(line, flush=True)
    REPORT.append(line)


def check(label: str, ok: bool, detail: str = "") -> bool:
    log(f"  [{'✓' if ok else '✗'}] {label}" + (f"  -- {detail}" if detail else ""))
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="Brittle Agents pre-flight check")
    ap.add_argument("--ping", action="store_true",
                    help="also make one live API call")
    args = ap.parse_args()

    log("=" * 72)
    log(" BRITTLE AGENTS -- SETUP CHECK")
    log(f" Repo root : {ROOT}")
    log(f" Timestamp : {datetime.now(timezone.utc).isoformat()}")
    log("=" * 72)
    all_ok = True

    # ------------------------------------------------------------------ 1 ---
    log("\n[1/6] Python & packages")
    all_ok &= check(f"Python 3.10 to 3.12 (found {sys.version.split()[0]})",
                    (3, 10) <= sys.version_info[:2] <= (3, 12))
    for pkg in ["yaml", "dotenv", "httpx", "pandas", "numpy", "scipy",
                "statsmodels", "matplotlib", "huggingface_hub"]:
        try:
            importlib.import_module(pkg)
            check(f"import {pkg}", True)
        except ImportError as e:
            all_ok &= check(f"import {pkg}", False, str(e))

    # ------------------------------------------------------------------ 2 ---
    log("\n[2/6] config.yaml")
    cfg = None
    cfg_path = ROOT / "config.yaml"
    if not cfg_path.exists():
        all_ok &= check("config.yaml exists", False, f"missing at {cfg_path}")
    else:
        check("config.yaml exists", True)
        import yaml
        try:
            cfg = yaml.safe_load(cfg_path.read_text())
            check("config.yaml parses", True)
        except yaml.YAMLError as e:
            all_ok &= check("config.yaml parses", False, str(e))

    if cfg:
        for sec in ["project", "paths", "tasks", "faults", "fault_positions",
                    "agent", "coding", "models", "providers"]:
            all_ok &= check(f"section '{sec}' present", sec in cfg)

        all_ok &= check("'clean' baseline present in faults",
                        "clean" in cfg.get("faults", []))
        n_models = len(cfg.get("models", []))
        n_reason = sum(1 for m in cfg["models"] if m.get("reasoning"))
        pairs = {m.get("pair") for m in cfg.get("models", [])}
        log(f"      -> {n_models} models ({n_reason} reasoning, "
            f"{n_models - n_reason} instruct) in {len(pairs)} families")

        # every family should have a reasoning + instruct member for RQ2
        for fam in sorted(pairs):
            fam_models = [m for m in cfg["models"] if m.get("pair") == fam]
            has_r = any(m.get("reasoning") for m in fam_models)
            has_i = any(not m.get("reasoning") for m in fam_models)
            check(f"family '{fam}': reasoning={has_r}, instruct={has_i}",
                  has_r, "" if has_i else "no instruct sibling (RQ2 uses "
                                          "cross-family comparator)")

        # design grid size, printed so it can be sanity-checked early
        n_fault_cells = 1 + (len(cfg["faults"]) - 1) * len(cfg["fault_positions"])
        scaff_models = set(cfg["agent"].get("scaffolding_models", []))
        base = cfg["tasks"]["n_tasks"] * n_fault_cells * cfg["agent"]["n_seeds"]
        n_trials = sum(base * (2 if m["name"] in scaff_models else 1)
                       for m in cfg["models"])
        log(f"      -> design grid: {cfg['tasks']['n_tasks']} tasks x "
            f"{n_fault_cells} fault cells x {cfg['agent']['n_seeds']} seeds "
            f"= {base} trials/model")
        log(f"         {n_models} models, scaffolding arm on "
            f"{len(scaff_models)} of them -> {n_trials:,} trials total")
        for nm in scaff_models:
            if nm not in {m["name"] for m in cfg["models"]}:
                all_ok &= check(f"scaffolding_models entry '{nm}' is a real "
                                f"model name", False)

    # ------------------------------------------------------------------ 3 ---
    log("\n[3/6] Environment / API keys")
    from dotenv import load_dotenv
    env_path = ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        check(".env found and loaded", True)
    else:
        check(".env found", False, "only needed for new API runs; copy "
              ".env.example -> .env and add your key")

    if cfg:
        for prov, pconf in cfg["providers"].items():
            val = os.getenv(pconf["env_key"], "")
            ok = bool(val)
            check(f"{pconf['env_key']} set ({prov})", ok,
                  "" if ok else "missing; only needed for new API runs")

    # ------------------------------------------------------------------ 4 ---
    log("\n[4/6] Directory structure")
    for d in ["01_environment/outputs",
              "02_data_collection/outputs/trajectories",
              "03_data_processing/outputs", "04_codebook/outputs",
              "05_analysis/rq1_detection/outputs",
              "05_analysis/rq2_reasoning_recovery/outputs",
              "05_analysis/rq3_fault_costs/outputs",
              "05_analysis/secondary_scaffolding/outputs",
              "06_figures_tables/outputs"]:
        path = ROOT / d
        created = not path.exists()
        path.mkdir(parents=True, exist_ok=True)
        check(f"{d}/", True, "created" if created else "")

    # ------------------------------------------------------------------ 5 ---
    log("\n[5/6] Task source reachability")
    if cfg:
        repo = cfg["tasks"]["hf_repo"]
        try:
            from huggingface_hub import list_repo_files
            files = list_repo_files(repo, repo_type="dataset")
            multi = [f for f in files if "multi_turn" in f.lower()]
            all_ok &= check(f"HF dataset '{repo}' reachable", True,
                            f"{len(files)} files")
            check("multi-turn files present", len(multi) > 0,
                  f"{len(multi)} found"
                  + ("" if multi else "; build_task_suite.py will list the "
                                      "available splits"))
        except Exception as e:
            all_ok &= check(f"HF dataset '{repo}' reachable", False,
                            repr(e)[:110])
        log(f"      note: executable environment classes come from the "
            f"gorilla repo\n            ({cfg['tasks']['gorilla_repo']}); "
            f"build_task_suite.py verifies them.")

    # ------------------------------------------------------------------ 6 ---
    log("\n[6/6] Live API ping" + ("" if args.ping else "  (skipped -- use --ping)"))
    if args.ping and cfg:
        import httpx
        pconf = cfg["providers"]["openrouter"]
        key = os.getenv(pconf["env_key"], "")
        if not key:
            all_ok &= check("ping openrouter", False, "no key")
        else:
            try:
                r = httpx.get(f"{pconf['base_url']}/models",
                              headers={"Authorization": f"Bearer {key}"},
                              timeout=30)
                all_ok &= check("ping openrouter", r.status_code == 200,
                                f"HTTP {r.status_code}")
                if r.status_code == 200:
                    ids = {m["id"] for m in r.json().get("data", [])}
                    for m in cfg["models"]:
                        check(f"model id '{m['model_id']}' available",
                              m["model_id"] in ids)
                    jm = cfg["coding"]["judge_model"]
                    check(f"judge model '{jm}' available", jm in ids)
            except httpx.HTTPError as e:
                all_ok &= check("ping openrouter", False, repr(e)[:110])

    # ----------------------------------------------------------------- done --
    log("\n" + "=" * 72)
    if all_ok:
        log(" RESULT: ALL CHECKS PASSED ✓")
    else:
        log(" RESULT: SOME CHECKS FAILED ✗ -- fix the items above and re-run.")
    log("=" * 72)

    out = Path(__file__).resolve().parent / "setup_report.txt"
    out.write_text("\n".join(REPORT) + "\n", encoding="utf-8")
    print(f"\nFull report saved -> {out}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
