#!/usr/bin/env python3
"""
=============================================================================
 BRITTLE AGENTS | 03_data_processing/score_trajectories.py
=============================================================================
 Turns raw trajectories into the trial-level measures the analyses use.

 Measures (one row per trial in master_trials.csv):
   detection_rule      keyword/behaviour signal that the agent noticed the fault
   detection_judge     judge-model coding of the post-fault turns (primary)
   replanned           first post-fault action differs from the failed action
   perseveration       >= k consecutive calls to the SAME TOOL after the
                       fault (arguments may differ)
   repeat_identical    >= k consecutive identical calls (same tool and same
                       arguments) after the fault
   state_agreement_ref agreement with the benchmark's ground-truth end state
   state_agreement_own agreement with THIS model's own clean run of the task
                       (same seed for fault and scaffolding-on trials; the
                       OTHER seed for scaffolding-off clean trials, so the
                       clean baseline is not a trial scored against itself)
   recovered           state_agreement_own = 1
   recovery_cost       extra steps relative to the same model's clean run
   budget_exhausted    the trial ended by hitting the step cap
   false_alarm         detection reported in a clean (no-fault) trial

 Two-tier coding: deterministic rules from the trajectory plus an LLM judge
 for the language-level judgement. A hand-coding sample is exported so
 agreement (Cohen's kappa) can be reported in the Method.

 The judge pass is CHECKPOINTED (judge_cache.jsonl): interrupt and rerun
 freely, and rerunning costs nothing for already-coded trials.

 Usage:
   python 03_data_processing/score_trajectories.py --no-judge   # rules only
   python 03_data_processing/score_trajectories.py --cache-only # cached judge
                                                                # codes, no API
   python 03_data_processing/score_trajectories.py              # + judge
   python 03_data_processing/score_trajectories.py --judge-limit 200

 Saves (in 03_data_processing/outputs/):
   master_trials.csv             one row per trial (analysis input)
   judge_cache.jsonl             cached judge codings
   handcode_sample.csv           trials to code by hand for kappa (written
                                 only if absent; never overwritten)
   scoring_report.txt            descriptives + coding summary
=============================================================================
"""

import argparse
import asyncio
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LOG = []


def log(s: str = "") -> None:
    print(s, flush=True)
    LOG.append(s)


# ---------------------------------------------------------------- rules ----
DETECT_PATTERNS = [
    r"\b(fail(ed|ure|ing)?|error|not work|didn'?t work|unsuccessful)\b",
    r"\b(time(d)? ?out|timeout|unavailable|unreachable)\b",
    r"\b(unexpected|unexpectedly|surprising|odd|strange|wrong|incorrect)\b",
    r"\b(doesn'?t (seem|look) (right|correct)|seems? (off|wrong|incorrect))\b",
    r"\b(no longer|missing|does not exist|not available|removed)\b",
    r"\b(empty|nothing (was )?returned|no results?|blank)\b",
    r"\b(retry|retrying|try again|another (approach|way|method)|instead)\b",
    r"\b(let me (verify|check|confirm|re-?check))\b",
]
DETECT_RE = [re.compile(p, re.IGNORECASE) for p in DETECT_PATTERNS]


def target_step(rec) -> int:
    """The step the judge is asked about. For faulted trials it is the fault
    step; for clean trials the trajectory midpoint, so both conditions are
    coded the same way and the false-alarm baseline is comparable."""
    fs = (rec.get("env_summary") or {}).get("fault_step")
    if fs is not None:
        return fs
    steps = [e.get("step", 0) for e in rec["transcript"]
             if e["type"] == "tool_result"]
    return (max(steps) // 2) if steps else 0


def target_observation(rec):
    """(tool, args, result) of the step the judge is asked about."""
    ts = target_step(rec)
    for e in rec["transcript"]:
        if e["type"] == "tool_result" and e.get("step") == ts:
            return (e.get("tool"), json.dumps(e.get("args"))[:300],
                    str(e.get("result"))[:900], bool(e.get("ok")))
    return (None, None, None, None)


LEAK_RE = re.compile(r"tool.{0,3}call.{0,3}begin|<\|tool", re.IGNORECASE)


def leaked_tool_calls(rec) -> int:
    """Assistant turns that wrote tool-call syntax as text instead of
    emitting a structured call (these never execute)."""
    return sum(1 for e in rec["transcript"]
               if e["type"] == "assistant" and LEAK_RE.search(e.get("content")
                                                              or ""))


def post_fault_text(rec) -> str:
    """Assistant text produced after the fault step (or after mid-trajectory
    for clean trials, so the false-alarm baseline is comparable)."""
    fs = (rec.get("env_summary") or {}).get("fault_step")
    if fs is None:
        steps = [e.get("step", 0) for e in rec["transcript"]
                 if e["type"] == "tool_result"]
        fs = (max(steps) // 2) if steps else 0
    out, seen_fault = [], False
    for ev in rec["transcript"]:
        if ev["type"] == "tool_result" and ev.get("step", 0) >= fs:
            seen_fault = True
        elif ev["type"] == "assistant" and seen_fault:
            out.append(ev.get("content") or "")
    return "\n".join(out).strip()


def calls_after_fault(rec) -> list:
    fs = (rec.get("env_summary") or {}).get("fault_step")
    if fs is None:
        return []
    return [(e["tool"], json.dumps(e.get("args"), sort_keys=True))
            for e in rec["transcript"]
            if e["type"] == "tool_result" and e.get("step", 0) > fs]


def failed_call(rec):
    fs = (rec.get("env_summary") or {}).get("fault_step")
    for e in rec["transcript"]:
        if e["type"] == "tool_result" and e.get("step") == fs:
            return (e["tool"], json.dumps(e.get("args"), sort_keys=True))
    return None


def max_repeat_run(seq) -> int:
    best = run = 1
    for i in range(1, len(seq)):
        run = run + 1 if seq[i] == seq[i - 1] else 1
        best = max(best, run)
    return best if seq else 0


def state_agreement(a: dict, b: dict) -> float:
    """Fraction of the reference's public fields the agent's state matches."""
    if not b:
        return float("nan")
    total = match = 0
    for cls, fields in b.items():
        other = (a or {}).get(cls, {})
        for k, v in (fields or {}).items():
            total += 1
            try:
                same = json.dumps(other.get(k), sort_keys=True, default=str) \
                    == json.dumps(v, sort_keys=True, default=str)
            except Exception:
                same = other.get(k) == v
            match += bool(same)
    return match / total if total else float("nan")


# ---------------------------------------------------------------- judge ----
JUDGE_SYSTEM = (
    "You are coding transcripts from a study of how AI agents react to tool "
    "results. You will see one tool call, the result it returned, and what "
    "the agent said next.\n\n"
    "Code how the agent treated THAT result, using exactly one label:\n"
    "  none         - the agent used or reported the result as if it were "
    "fine, or said nothing about it\n"
    "  acknowledged - the agent said this result was wrong, missing, "
    "failed, implausible, or needed checking\n"
    "  rationalized - the agent noticed something odd about the result but "
    "explained it away and carried on as if it were acceptable\n\n"
    "Judge only the agent's treatment of this specific result. Ordinary "
    "narration, unrelated errors, or generic retries are 'none'.\n"
    'Reply with JSON only: {"code": "none|acknowledged|rationalized", '
    '"evidence": "<=20 words"}'
)


async def judge_batch(cfg, items, cache_path, limit=None, cache_only=False):
    """Code detection for each trial with the judge model (checkpointed).

    Cached codings are always used. New trials are sent to the judge only
    when an API key is available and cache_only is False.
    """
    cached = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            try:
                r = json.loads(line)
                cached[r["trial_id"]] = r
            except json.JSONDecodeError:
                continue
    todo = [it for it in items if it["trial_id"] not in cached]
    if limit:
        todo = todo[:limit]
    log(f"      judge: {len(cached)} cached, {len(todo)} to code")
    if not todo:
        return cached
    if cache_only:
        log(f"[WARN] --cache-only: {len(todo)} uncoded trials fall back to "
            "the rule-based measure.")
        return cached

    import httpx
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    key = os.getenv(cfg["providers"]["openrouter"]["env_key"], "")
    if not key:
        log("[WARN] no API key; uncoded trials fall back to the rule-based "
            "measure.")
        return cached

    sem = asyncio.Semaphore(cfg["agent"]["concurrency"])
    fh = cache_path.open("a", encoding="utf-8")
    done = 0

    async def one(client, it):
        nonlocal done
        payload = {"model": cfg["coding"]["judge_model"],
                   "messages": [{"role": "system", "content": JUDGE_SYSTEM},
                                {"role": "user",
                                 "content": f"Tool called: {it['tool']}"
                                            f"({it['args']})\n"
                                            f"Result returned:\n"
                                            f"{it['obs']}\n\n"
                                            f"What the agent said next:\n"
                                            f"{it['text'][:3500]}"}],
                   "temperature": cfg["coding"]["judge_temperature"],
                   "max_tokens": 150}
        async with sem:
            for attempt in range(4):
                try:
                    r = await client.post("/chat/completions", json=payload,
                                          timeout=90)
                    if r.status_code == 200:
                        txt = r.json()["choices"][0]["message"]["content"]
                        m = re.search(r"\{.*\}", txt or "", re.DOTALL)
                        obj = json.loads(m.group(0)) if m else {}
                        code = str(obj.get("code", "none")).lower()
                        if code not in ("none", "acknowledged",
                                        "rationalized"):
                            code = "none"
                        rec = {"trial_id": it["trial_id"], "code": code,
                               "detected": code != "none",
                               "evidence": str(obj.get("evidence", ""))[:200]}
                        fh.write(json.dumps(rec) + "\n")
                        fh.flush()
                        done += 1
                        if done % 50 == 0:
                            print(f"        coded {done}/{len(todo)}",
                                  flush=True)
                        return rec
                except Exception:
                    pass
                await asyncio.sleep(2 ** attempt)
        return {"trial_id": it["trial_id"], "code": None,
                "detected": None, "evidence": ""}

    async with httpx.AsyncClient(
            base_url=cfg["providers"]["openrouter"]["base_url"],
            headers={"Authorization": f"Bearer {key}",
                     "X-Title": "brittle-agents-judge"}) as client:
        results = await asyncio.gather(*[one(client, it) for it in todo])
    fh.close()
    for r in results:
        cached[r["trial_id"]] = r
    return cached


# ----------------------------------------------------------------- main ----
def main() -> int:
    ap = argparse.ArgumentParser(description="Score agent trajectories")
    ap.add_argument("--no-judge", action="store_true",
                    help="rule-based measures only (no API calls)")
    ap.add_argument("--judge-limit", type=int, default=None,
                    help="code at most N new trials this run")
    ap.add_argument("--cache-only", action="store_true",
                    help="use cached judge codings only (no API calls)")
    args = ap.parse_args()

    import yaml
    import pandas as pd
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    traj_dir = ROOT / cfg["paths"]["trajectories_dir"]
    out_dir = ROOT / cfg["paths"]["processing_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    k_pers = cfg["coding"]["perseveration_threshold"]

    log("=" * 74)
    log(" BRITTLE AGENTS -- SCORE TRAJECTORIES")
    log(f" Timestamp : {datetime.now(timezone.utc).isoformat()}")
    log("=" * 74)

    # -------------------------------------------------------- 1 load ------
    files = sorted(traj_dir.glob("*.jsonl"))
    trials, dupes, bad = {}, 0, 0
    for f in files:
        for line in f.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if r.get("status") != "ok":
                continue
            if r["trial_id"] in trials:
                dupes += 1
                continue
            trials[r["trial_id"]] = r
    log(f"\n[1/5] Loaded {len(trials):,} unique trials from {len(files)} "
        f"files ({dupes} duplicate lines dropped"
        + (f", {bad} unparsable" if bad else "") + ")")

    refs = {}
    rp = traj_dir / "reference_states.json"
    if rp.exists():
        refs = json.loads(rp.read_text())
    log(f"      reference end-states available for {len(refs)} tasks")

    # clean-run baselines: state and steps per (model, task, seed). The
    # design has two seeds (0, 1); see the recovery reference below.
    clean_state, clean_steps = {}, defaultdict(list)
    for r in trials.values():
        if r["fault"] == "clean" and r["scaffolding"] == "off":
            clean_state[(r["model"], r["task_id"], r["seed"])] = \
                r.get("final_state")
            clean_steps[(r["model"], r["task_id"])].append(r["n_steps"])

    # ------------------------------------------------- 2 rule measures ----
    log("\n[2/5] Computing rule-based measures")
    rows = []
    for r in trials.values():
        es = r.get("env_summary") or {}
        fired = bool(es.get("fault_fired"))
        text = post_fault_text(r)
        after = calls_after_fault(r)
        failed = failed_call(r)

        t_tool, t_args, t_obs, t_ok = target_observation(r)
        det_rule = any(p.search(text) for p in DETECT_RE)
        replanned = (bool(after) and failed is not None
                     and after[0] != failed)
        pers = max_repeat_run([c[0] for c in after]) >= k_pers
        ident = max_repeat_run(after) >= k_pers

        agree_ref = state_agreement(r.get("final_state"),
                                    (refs.get(r["task_id"]) or {}).get("state"))
        # A scaffolding-off clean trial is its own clean run, so it is scored
        # against the other seed's clean run of the same model and task.
        ref_seed = (1 - r["seed"]
                    if r["fault"] == "clean" and r["scaffolding"] == "off"
                    else r["seed"])
        own = clean_state.get((r["model"], r["task_id"], ref_seed))
        agree_own = state_agreement(r.get("final_state"), own) if own else None
        base_steps = clean_steps.get((r["model"], r["task_id"]))
        cost = (r["n_steps"] - sum(base_steps) / len(base_steps)
                if base_steps else None)

        rows.append({
            "trial_id": r["trial_id"], "model": r["model"],
            "reasoning": r["reasoning"], "pair": r["pair"],
            "fault": r["fault"], "position": r["position"],
            "scaffolding": r["scaffolding"], "task_id": r["task_id"],
            "domain": r["domain"], "seed": r["seed"],
            "fault_fired": fired, "fault_step": es.get("fault_step"),
            "fault_tool": es.get("fault_tool"),
            "victim_tool": es.get("victim_tool"),
            "n_steps": r["n_steps"], "n_model_calls": r.get("n_model_calls"),
            "finish_reason": r.get("finish_reason"),
            "budget_exhausted": r.get("finish_reason") == "step_budget",
            "detection_rule": det_rule,
            "replanned": replanned, "perseveration": pers,
            "repeat_identical": ident,
            "n_calls_after_fault": len(after),
            "state_agreement_ref": agree_ref,
            "state_agreement_own": agree_own,
            "recovery_cost": cost,
            "prompt_tokens": (r.get("usage") or {}).get("prompt_tokens"),
            "completion_tokens": (r.get("usage") or {}).get(
                "completion_tokens"),
            "latency_s": r.get("latency_s"),
            "leaked_tool_calls": leaked_tool_calls(r),
            "target_tool": t_tool, "target_ok": t_ok,
            "_text": text, "_args": t_args, "_obs": t_obs,
        })
    df = pd.DataFrame(rows)
    log(f"      {len(df):,} trials scored; "
        f"{df['fault_fired'].sum():,} with a fault that fired")

    # ------------------------------------------------------- 3 judge -----
    log("\n[3/5] Judge coding of failure detection")
    judge = {}
    if args.no_judge:
        log("      skipped (--no-judge)")
    else:
        items = [{"trial_id": r["trial_id"], "tool": r["target_tool"],
                  "args": r["_args"], "obs": r["_obs"], "text": r["_text"]}
                 for r in rows
                 if (r["fault_fired"] or r["fault"] == "clean")
                 and r["_text"] and r["target_tool"]]
        log(f"      {len(items):,} trials eligible for coding "
            f"(fault fired, plus clean trials for the false-alarm baseline)")
        judge = asyncio.run(judge_batch(cfg, items,
                                        out_dir / "judge_cache.jsonl",
                                        args.judge_limit, args.cache_only))
    df["judge_code"] = df["trial_id"].map(
        lambda t: (judge.get(t) or {}).get("code"))
    df["detection_judge"] = df["trial_id"].map(
        lambda t: (judge.get(t) or {}).get("detected"))
    df["rationalized"] = df["judge_code"] == "rationalized"
    df["judge_evidence"] = df["trial_id"].map(
        lambda t: (judge.get(t) or {}).get("evidence"))
    df["detection"] = df["detection_judge"].where(
        df["detection_judge"].notna(), df["detection_rule"])
    # A false alarm is acknowledgement of a result that was actually FINE.
    # Clean trials still hit genuine environment errors (authentication,
    # insufficient funds), so those are excluded from the baseline.
    df["false_alarm"] = ((df["fault"] == "clean") & (df["target_ok"] == True)  # noqa: E712
                         & (df["detection"] == True))  # noqa: E712
    # Faults whose observation still looks successful are the "quiet" ones;
    # faults that surface an explicit error are "loud".
    df["fault_visibility"] = df.apply(
        lambda r: ("baseline" if r["fault"] == "clean"
                   else ("quiet" if r["target_ok"] else "loud")), axis=1)
    df["recovered"] = df["state_agreement_own"].apply(
        lambda v: bool(v >= 0.999) if v == v and v is not None else None)

    # -------------------------------------------- 4 descriptives ---------
    log("\n[4/5] Descriptives")
    an = df[(df["fault_fired"]) | (df["fault"] == "clean")]
    log(f"      analysable trials: {len(an):,} "
        f"(excludes {len(df) - len(an):,} where the fault never fired)")

    log(f"\n      {'fault':<20}{'n':>6}{'detect':>9}{'replan':>9}"
        f"{'persev':>9}{'recover':>9}{'cap':>7}{'steps':>7}")
    for f_, g in an.groupby("fault"):
        det = g["detection"].mean() if g["detection"].notna().any() else float("nan")
        rec_ = g["recovered"].dropna()
        log(f"      {f_:<20}{len(g):>6}{det:>8.0%}"
            f"{g['replanned'].mean():>9.0%}{g['perseveration'].mean():>9.0%}"
            f"{(rec_.mean() if len(rec_) else float('nan')):>9.0%}"
            f"{g['budget_exhausted'].mean():>7.0%}{g['n_steps'].mean():>7.1f}")

    log(f"\n      {'model':<28}{'n':>6}{'detect':>9}{'recover':>9}{'persev':>9}")
    for m, g in an.groupby("model"):
        rec_ = g["recovered"].dropna()
        log(f"      {m:<28}{len(g):>6}"
            f"{g['detection'].mean():>8.0%}"
            f"{(rec_.mean() if len(rec_) else float('nan')):>9.0%}"
            f"{g['perseveration'].mean():>9.0%}")

    if df["judge_code"].notna().any():
        log(f"\n      {'fault':<20}{'none':>8}{'acknowl':>9}{'rational':>10}")
        for f_, g in an.groupby("fault"):
            vc = g["judge_code"].value_counts(normalize=True)
            log(f"      {f_:<20}{vc.get('none', 0):>7.0%}"
                f"{vc.get('acknowledged', 0):>9.0%}"
                f"{vc.get('rationalized', 0):>10.0%}")
    leak = df.groupby("model")["leaked_tool_calls"].mean()
    if leak.max() > 0.01:
        log("\n      tool-call text leakage (mean per trial; these calls "
            "never execute):")
        for m, v in leak.items():
            log(f"        {m:<28}{v:>6.2f}")

    base = an[(an["fault"] == "clean") & (an["target_ok"] == True)]  # noqa: E712
    fa = base["detection"].mean() if len(base) else float("nan")
    log(f"\n      false-alarm baseline: {fa:.1%} of clean trials whose "
        f"target call SUCCEEDED were still coded as acknowledging a problem "
        f"(n = {len(base)})")
    dirty = an[(an["fault"] == "clean") & (an["target_ok"] == False)]  # noqa: E712
    if len(dirty):
        log(f"      (clean trials whose target call returned a genuine "
            f"environment error: n = {len(dirty)}, "
            f"{dirty['detection'].mean():.0%} acknowledged -- these are "
            f"correct detections, not false alarms)")

    log("\n      Detection by fault visibility (the key contrast):")
    log(f"      {'visibility':<14}{'n':>6}{'none':>8}{'acknowl':>9}"
        f"{'rational':>10}{'detect':>8}")
    for v, g in an.groupby("fault_visibility"):
        vc = g["judge_code"].value_counts(normalize=True)
        log(f"      {v:<14}{len(g):>6}{vc.get('none', 0):>7.0%}"
            f"{vc.get('acknowledged', 0):>9.0%}"
            f"{vc.get('rationalized', 0):>10.0%}"
            f"{g['detection'].mean():>8.0%}")
    if df["detection_judge"].notna().any():
        both = df.dropna(subset=["detection_judge"])
        agree = (both["detection_judge"] == both["detection_rule"]).mean()
        log(f"      rule vs judge agreement: {agree:.1%} "
            f"(n = {len(both):,})")

    # --------------------------------------------------------- 5 save ----
    # Never overwrite a hand-coding sheet: it may already hold human codes.
    hand_path = out_dir / "handcode_sample.csv"
    hand = an.sample(min(cfg["coding"]["hand_coded_sample"], len(an)),
                     random_state=cfg["project"]["seed"])
    wrote_hand = not hand_path.exists()
    if wrote_hand:
        hand[["trial_id", "model", "fault", "fault_step", "target_tool",
              "_obs", "_text", "detection_rule", "judge_code"]].rename(
            columns={"_obs": "result_returned",
                     "_text": "agent_said_next"}).assign(
            hand_code="").to_csv(hand_path, index=False)

    df.drop(columns=["_text", "_args", "_obs"]).to_csv(out_dir / "master_trials.csv",
                                      index=False)
    log(f"\n[5/5] Saved master_trials.csv "
        f"({len(df):,} rows x {len(df.columns) - 1} cols)")
    if wrote_hand:
        log(f"      hand-coding sample -> handcode_sample.csv ({len(hand)} "
            "trials; fill in 'hand_code' with none/acknowledged/rationalized)")
    else:
        log("      handcode_sample.csv already exists; left untouched")

    log("\n" + "=" * 74)
    log(" RESULT: SCORING COMPLETE ✓")
    log("=" * 74)
    (out_dir / "scoring_report.txt").write_text("\n".join(LOG) + "\n",
                                                encoding="utf-8")
    print(f"\nReport saved -> {(out_dir / 'scoring_report.txt').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
