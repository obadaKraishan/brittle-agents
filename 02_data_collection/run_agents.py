#!/usr/bin/env python3
"""
=============================================================================
 BRITTLE AGENTS | 02_data_collection/run_agents.py
=============================================================================
 Runs the agent loop across the design grid and records full trajectories.

 Protocol (one trial):
   For each user turn of the task, the turn text is appended to the
   conversation and the model acts until it answers without calling a tool
   (turn complete) or the global step budget is exhausted. Tool calls go
   through the fault-injection layer, so exactly one typed fault fires at
   the configured position. Every message, tool call, observation, and the
   fault marker are recorded.

 Trials are keyed by (task, model, fault, position, seed, scaffolding) and
 CHECKPOINTED: interrupt at any time and rerun; completed trials are
 skipped.

 Recovery scoring needs a reference: for every task we also execute the
 benchmark's ground-truth call path on a clean environment and store the
 resulting state, so 03_data_processing can compare the agent's final state
 against it.

 Usage:
   python 02_data_collection/run_agents.py --smoke-test      # 2 tasks, 1 seed
   python 02_data_collection/run_agents.py                   # whole grid
   python 02_data_collection/run_agents.py --model deepseek-r1
   python 02_data_collection/run_agents.py --dry-run         # no API calls

 Requires OPENROUTER_API_KEY in .env (see .env.example).

 Saves (in 02_data_collection/outputs/trajectories/):
   {model}__{fault}__{scaffolding}.jsonl   one line per completed trial
   run_manifest.json                       grid + live counters
   run_log.txt                             condensed log
   reference_states.json                   ground-truth end state per task
=============================================================================
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "01_environment"))

from fault_injection import ToolEnvironment, state_snapshot  # noqa: E402

SYSTEM_PROMPT = (
    "You are an autonomous assistant that completes tasks by calling tools.\n"
    "Work step by step: call a tool, read its result, then decide what to do "
    "next. Call one tool at a time.\n"
    "Tools can fail or behave unexpectedly. If something goes wrong, deal "
    "with it as you see fit and keep working toward the user's goal.\n"
    "When the current request is complete, reply with a short plain-text "
    "summary and no tool call."
)
SCAFFOLD_LINE = (
    "\nAfter every tool call, briefly check whether the result is what you "
    "expected before continuing."
)

LOG = []


def log(line: str = "") -> None:
    print(line, flush=True)
    LOG.append(line)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
CALL_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\((.*)\)$", re.DOTALL)


def build_reference_states(suite) -> dict:
    """Execute each task's ground-truth calls on a clean env; store end state.

    The benchmark ships the reference solution as executable call strings
    (one list per user turn). Running them gives the state an ideal agent
    would reach, so recovery can be scored as state agreement rather than
    by asking a model whether the task looked done.
    """
    refs = {}
    for task in suite:
        env = ToolEnvironment(task, "clean")
        ok, failed, errs = 0, 0, []
        for turn in (task.get("ground_truth") or []):
            calls = turn if isinstance(turn, list) else [turn]
            for c in calls:
                if not isinstance(c, str):
                    continue
                m = CALL_RE.match(c.strip())
                if not m:
                    failed += 1
                    errs.append(f"unparsed: {c[:60]}")
                    continue
                name, argstr = m.group(1), m.group(2)
                if name not in env.registry:
                    failed += 1
                    errs.append(f"unknown tool: {name}")
                    continue
                try:
                    _, meth = env.registry[name]
                    eval(f"_m({argstr})", {"__builtins__": {}},
                         {"_m": meth, "True": True, "False": False,
                          "None": None})
                    ok += 1
                except Exception as e:
                    failed += 1
                    errs.append(f"{name}: {type(e).__name__}")
        refs[task["task_id"]] = {"state": state_snapshot(env),
                                 "calls_ok": ok, "calls_failed": failed,
                                 "errors": errs[:5]}
    return refs


def build_trials(cfg, suite, models, task_limit=None):
    """Enumerate every (task, model, fault, position, seed, scaffolding)."""
    tasks = suite[:task_limit] if task_limit else suite
    scaff_models = set(cfg["agent"].get("scaffolding_models", []))
    trials = []
    for m in models:
        arms = (["off", "on"] if m["name"] in scaff_models else ["off"])
        for scaff in arms:
            for fault in cfg["faults"]:
                positions = (["na"] if fault == "clean"
                             else cfg["fault_positions"])
                for pos in positions:
                    for task in tasks:
                        for seed in range(cfg["agent"]["n_seeds"]):
                            trials.append({
                                "trial_id": (f"{m['name']}|{fault}|{pos}|"
                                             f"{scaff}|{task['task_id']}|"
                                             f"s{seed}"),
                                "model": m, "fault": fault, "position": pos,
                                "scaffolding": scaff, "task": task,
                                "seed": seed,
                            })
    return trials


def load_completed(raw_dir) -> set:
    done = set()
    for f in raw_dir.glob("*.jsonl"):
        for line in f.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("status") == "ok":
                done.add(rec["trial_id"])
    return done


# ---------------------------------------------------------------------------
# model call
# ---------------------------------------------------------------------------
async def chat(client, cfg, model, messages, tools, sem):
    """One chat completion with retry/backoff. Returns (message, usage, err)."""
    exp = cfg["agent"]
    payload = {
        "model": model["model_id"], "messages": messages, "tools": tools,
        "temperature": exp["temperature"],
        "max_tokens": (model.get("reasoning_budget", 0)
                       + 2 * exp["max_output_tokens"]
                       if model.get("reasoning") else exp["max_output_tokens"]),
    }
    if model.get("reasoning"):
        payload["reasoning"] = {"max_tokens": model.get("reasoning_budget",
                                                        2048)}
    last = None
    async with sem:
        for attempt in range(exp["max_retries"] + 1):
            try:
                r = await client.post("/chat/completions", json=payload,
                                      timeout=exp["request_timeout_s"])
                if r.status_code == 200:
                    data = r.json()
                    if data.get("choices"):
                        return (data["choices"][0]["message"],
                                data.get("usage", {}), None)
                    last = f"no_choices: {str(data)[:150]}"
                elif r.status_code in (429, 500, 502, 503, 529):
                    last = f"http_{r.status_code}"
                else:
                    return None, {}, f"http_{r.status_code}: {r.text[:150]}"
            except Exception as e:
                last = repr(e)[:150]
            await asyncio.sleep(min(2 ** attempt + 0.5, 30))
    return None, {}, f"retries_exhausted: {last}"


async def run_trial(client, cfg, trial, sem, dry_run=False):
    """Execute one agent trial end to end."""
    t0 = time.monotonic()
    task, model = trial["task"], trial["model"]
    env = ToolEnvironment(task, trial["fault"],
                          "late" if trial["position"] == "na"
                          else trial["position"], seed=trial["seed"])
    system = SYSTEM_PROMPT + (SCAFFOLD_LINE if trial["scaffolding"] == "on"
                              else "")
    messages = [{"role": "system", "content": system}]
    transcript, usage_tot = [], {"prompt_tokens": 0, "completion_tokens": 0}
    max_steps = cfg["agent"]["max_steps"]
    finish = "completed"

    if dry_run:
        await asyncio.sleep(0.001)
        for i in range(3):
            env.call(env.visible_tools()[i % len(env.visible_tools())], {})
        return {**_meta(trial), "status": "ok", "finish_reason": "dry_run",
                "n_steps": env.step, "n_model_calls": 3,
                "transcript": [], "env_summary": env.summary(),
                "final_state": state_snapshot(env), "usage": usage_tot,
                "latency_s": 0.0}

    for turn_idx, turn in enumerate(task["turns"]):
        user_text = " ".join(m.get("content", "") for m in turn
                             if m.get("role") == "user") or str(turn)
        messages.append({"role": "user", "content": user_text})
        transcript.append({"type": "user", "turn": turn_idx,
                           "content": user_text})

        while True:
            if env.step >= max_steps:
                finish = "step_budget"
                break
            msg, usage, err = await chat(client, cfg, model, messages,
                                         env.tool_specs(), sem)
            if err:
                return {**_meta(trial), "status": "failed", "error": err,
                        "n_steps": env.step, "transcript": transcript,
                        "latency_s": round(time.monotonic() - t0, 1)}
            for k in usage_tot:
                usage_tot[k] += (usage or {}).get(k, 0) or 0

            calls = msg.get("tool_calls") or []
            content = msg.get("content") or ""
            transcript.append({"type": "assistant", "turn": turn_idx,
                               "content": content,
                               "tool_calls": [c.get("function", {}).get("name")
                                              for c in calls]})
            messages.append({k: v for k, v in msg.items()
                             if k in ("role", "content", "tool_calls")}
                            or {"role": "assistant", "content": content})

            if not calls:                       # turn finished
                break

            for c in calls:
                fn = c.get("function", {})
                name = fn.get("name", "")
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                res = env.call(name, args)
                body = (json.dumps(res.get("value"))[:1500] if res["ok"]
                        else res["error"])
                transcript.append({"type": "tool_result", "turn": turn_idx,
                                   "step": env.step, "tool": name,
                                   "args": args, "ok": res["ok"],
                                   "faulted": res.get("faulted", False),
                                   "result": body})
                messages.append({"role": "tool",
                                 "tool_call_id": c.get("id", name),
                                 "content": body})
        if finish == "step_budget":
            break

    return {**_meta(trial), "status": "ok", "finish_reason": finish,
            "n_steps": env.step,
            "n_model_calls": sum(1 for t in transcript
                                 if t["type"] == "assistant"),
            "transcript": transcript, "env_summary": env.summary(),
            "final_state": state_snapshot(env), "usage": usage_tot,
            "latency_s": round(time.monotonic() - t0, 1)}


def _meta(trial):
    m, t = trial["model"], trial["task"]
    return {"trial_id": trial["trial_id"], "model": m["name"],
            "model_id": m["model_id"], "reasoning": m.get("reasoning", False),
            "pair": m.get("pair"), "fault": trial["fault"],
            "position": trial["position"], "scaffolding": trial["scaffolding"],
            "task_id": t["task_id"], "domain": t["domain"],
            "n_turns": t["n_turns"], "seed": trial["seed"],
            "timestamp": datetime.now(timezone.utc).isoformat()}


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------
async def run(cfg, trials, raw_dir, dry_run=False, spend_cap=None):
    """Bounded worker pool.

    Concurrency is applied at the trial level: exactly `concurrency` trials
    run at a time, each to completion.

    Two guards limit wasted calls:
      * fail-fast   -- a model that fails `max_model_failures` times in a row
                       is dropped for the rest of the run (e.g. rate-limited
                       free tiers returning 429).
      * spend cap   -- estimated cost is accumulated from reported token
                       usage; the run stops when the cap is reached.
    """
    import httpx
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    key = os.getenv(cfg["providers"]["openrouter"]["env_key"], "")
    base = cfg["providers"]["openrouter"]["base_url"]
    call_sem = asyncio.Semaphore(cfg["agent"]["concurrency"])
    max_model_failures = cfg["agent"].get("max_model_failures", 8)

    price = {m["name"]: (m.get("price_in", 0), m.get("price_out", 0))
             for m in cfg["models"]}
    files, stats = {}, {"ok": 0, "failed": 0, "skipped": 0, "steps": 0,
                        "budget_hit": 0, "in_tok": 0, "out_tok": 0,
                        "faults_fired": 0, "cost": 0.0}
    consec_fail, dropped = {}, set()
    stop = asyncio.Event()
    queue = asyncio.Queue()
    for t in trials:
        queue.put_nowait(t)

    def fh(rec):
        k = (rec["model"], rec["fault"], rec["scaffolding"])
        if k not in files:
            files[k] = open(raw_dir / f"{k[0]}__{k[1]}__{k[2]}.jsonl", "a",
                            encoding="utf-8")
        return files[k]

    t_start, n = time.monotonic(), len(trials)
    done_count = 0

    async def worker(client):
        nonlocal done_count
        while not stop.is_set():
            try:
                trial = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            name = trial["model"]["name"]
            if name in dropped:
                stats["skipped"] += 1
                queue.task_done()
                continue
            rec = await run_trial(client, cfg, trial, call_sem, dry_run)

            f = fh(rec)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if rec["status"] == "ok":
                consec_fail[name] = 0
                stats["ok"] += 1
                stats["steps"] += rec["n_steps"]
                stats["budget_hit"] += rec["finish_reason"] == "step_budget"
                u = rec.get("usage", {})
                i_tok = u.get("prompt_tokens", 0) or 0
                o_tok = u.get("completion_tokens", 0) or 0
                stats["in_tok"] += i_tok
                stats["out_tok"] += o_tok
                pi, po = price.get(name, (0, 0))
                stats["cost"] += i_tok / 1e6 * pi + o_tok / 1e6 * po
                stats["faults_fired"] += bool(
                    rec.get("env_summary", {}).get("fault_fired"))
            else:
                stats["failed"] += 1
                consec_fail[name] = consec_fail.get(name, 0) + 1
                if (consec_fail[name] >= max_model_failures
                        and name not in dropped):
                    dropped.add(name)
                    print(f"  [!] DROPPING '{name}': "
                          f"{consec_fail[name]} consecutive failures "
                          f"({(rec.get('error') or '')[:60]}). Remaining "
                          f"trials for this model will be skipped.",
                          flush=True)
            done_count += 1
            if spend_cap and stats["cost"] >= spend_cap:
                print(f"  [!] SPEND CAP reached "
                      f"(${stats['cost']:.2f} >= ${spend_cap:.2f}). Stopping; "
                      f"rerun to continue.", flush=True)
                stop.set()
            if done_count % 5 == 0 or done_count == n:
                el = time.monotonic() - t_start
                ms = stats["steps"] / max(stats["ok"], 1)
                rate = done_count / max(el, 1)
                print(f"  [{done_count:>5}/{n}] ok={stats['ok']} "
                      f"failed={stats['failed']} skipped={stats['skipped']} "
                      f"| steps={ms:.1f} cap-hit={stats['budget_hit']} "
                      f"| ${stats['cost']:.2f} "
                      f"| {el / 60:.0f}m elapsed, "
                      f"~{(n - done_count) / max(rate, 1e-6) / 60:.0f}m left",
                      flush=True)
            queue.task_done()

    async with httpx.AsyncClient(
            base_url=base,
            headers={"Authorization": f"Bearer {key}",
                     "HTTP-Referer": "https://localhost",
                     "X-Title": "brittle-agents"}) as client:
        workers = [asyncio.create_task(worker(client))
                   for _ in range(cfg["agent"]["concurrency"])]
        await asyncio.gather(*workers)

    for f in files.values():
        f.close()
    stats["dropped_models"] = sorted(dropped)
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the agent trials")
    ap.add_argument("--smoke-test", action="store_true",
                    help="tiny live run: 2 tasks x 2 faults x 2 models")
    ap.add_argument("--dry-run", action="store_true",
                    help="no API calls; exercises the pipeline only")
    ap.add_argument("--task-limit", type=int, default=None,
                    help="use only the first N tasks of the frozen suite")
    ap.add_argument("--model", type=str, default=None,
                    help="run only this model (config 'name')")
    ap.add_argument("--spend-cap", type=float, default=None,
                    help="stop once estimated spend (USD) reaches this")
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    raw_dir = ROOT / cfg["paths"]["trajectories_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    suite_path = ROOT / cfg["paths"]["task_suite"]
    if not suite_path.exists():
        log("[ERROR] no frozen task suite; run build_task_suite.py first.")
        return 1
    suite = json.loads(suite_path.read_text())
    models = cfg["models"]

    mode = ("DRY-RUN" if args.dry_run else
            "SMOKE-TEST" if args.smoke_test else "COLLECTION")
    log("=" * 74)
    log(" BRITTLE AGENTS -- RUN AGENTS")
    log(f" Timestamp : {datetime.now(timezone.utc).isoformat()}")
    log(f" Mode      : {mode}")
    log("=" * 74)

    # ------------------------------------------------- reference states -----
    ref_path = raw_dir / "reference_states.json"
    if not ref_path.exists():
        log("\n[1/3] Building reference end-states from the benchmark's "
            "ground-truth paths")
        refs = build_reference_states(suite)
        ref_path.write_text(json.dumps(refs, indent=2), encoding="utf-8")
        tot_ok = sum(r["calls_ok"] for r in refs.values())
        tot_bad = sum(r["calls_failed"] for r in refs.values())
        log(f"      {len(refs)} tasks; reference calls executed: {tot_ok} ok, "
            f"{tot_bad} unparsed/failed")
    else:
        log(f"\n[1/3] Reference states already built "
            f"({ref_path.name}); delete to rebuild")

    # ------------------------------------------------------------ trials ----
    if args.smoke_test:
        suite = suite[:2]
        cfg["faults"] = ["clean", "silent_corruption"]
        cfg["agent"]["n_seeds"] = 1
        models = models[:2]
        log(f"\n Smoke scope: 2 tasks, faults={cfg['faults']}, "
            f"models={[m['name'] for m in models]}, 1 seed")
    if args.model:
        models = [m for m in models if m["name"] == args.model]
        if not models:
            log(f"[ERROR] unknown model '{args.model}'")
            return 1

    trials = build_trials(cfg, suite, models, args.task_limit)
    done = load_completed(raw_dir)
    todo = [t for t in trials if t["trial_id"] not in done]
    log(f"\n[2/3] Grid: {len(trials):,} trials "
        f"({len(done):,} already complete, {len(todo):,} to run)")
    if not todo:
        log("      nothing to do \u2713")
        return 0
    if not args.dry_run:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
        if not os.getenv(cfg["providers"]["openrouter"]["env_key"]):
            log("[ERROR] OPENROUTER_API_KEY not set.")
            return 1

    log("")
    cap = args.spend_cap
    if cap:
        log(f"      spend cap for this run: ${cap:.2f}")
    stats = asyncio.run(run(cfg, todo, raw_dir, args.dry_run, cap))

    # ---------------------------------------------------------- manifest ----
    mean_steps = stats["steps"] / max(stats["ok"], 1)
    manifest = {
        "updated": datetime.now(timezone.utc).isoformat(), "mode": mode,
        "grid_total": len(trials), "completed_before": len(done),
        "ran_now": {"ok": stats["ok"], "failed": stats["failed"]},
        "mean_steps_per_trial": round(mean_steps, 2),
        "step_budget_hits": stats["budget_hit"],
        "faults_fired": stats["faults_fired"],
        "tokens": {"input": stats["in_tok"], "output": stats["out_tok"]},
        "estimated_cost_usd": round(stats["cost"], 2),
        "dropped_models": stats.get("dropped_models", []),
        "skipped": stats["skipped"],
        "models": [m["name"] for m in models], "n_tasks": len(suite),
    }
    (raw_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2),
                                               encoding="utf-8")

    log("\n" + "=" * 74)
    log(f" RESULT: {stats['ok']} ok, {stats['failed']} failed")
    log(f" Mean steps per trial : {mean_steps:.2f}")
    log(f" Step-budget hits     : {stats['budget_hit']} "
        f"({stats['budget_hit'] / max(stats['ok'], 1):.0%} of trials)")
    log(f" Faults fired         : {stats['faults_fired']}/{stats['ok']}")
    log(f" Tokens               : {stats['in_tok']:,} in / "
        f"{stats['out_tok']:,} out")
    log(f" Estimated spend      : ${stats['cost']:.2f}   "
        f"(${stats['cost'] / max(stats['ok'], 1):.3f} per completed trial)")
    if stats.get("dropped_models"):
        log(f" Dropped models       : {stats['dropped_models']} "
            f"(too many consecutive failures)")
    if stats["skipped"]:
        log(f" Skipped trials       : {stats['skipped']}")
    log("=" * 74)
    (raw_dir / "run_log.txt").write_text("\n".join(LOG) + "\n",
                                         encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())