#!/usr/bin/env python3
"""
=============================================================================
 BRITTLE AGENTS | 01_environment/fault_injection.py
=============================================================================
 THE INSTRUMENT. Wraps a BFCL environment so that tool calls execute
 normally until a chosen step, at which point a typed fault fires exactly
 once. Everything is deterministic: the same (task, fault, position, seed)
 always produces the same fault at the same step.

 Fault types
   clean              no fault (baseline condition)
   timeout            the call raises a timeout after a delay
   silent_corruption  the call succeeds but returns a plausible wrong value
   schema_drift       one parameter is renamed, so the agent's call is
                      rejected with an unknown-argument error
   missing_tool       the tool disappears from the registry mid-task
   empty_result       a well-formed but empty payload comes back

 Two design commitments make the measurement defensible:
   1. The fault fires ONCE, on the first call to an eligible tool at or
      after the target step. Repeat calls afterwards behave normally, so an
      agent that retries can succeed -- recovery is possible by design.
   2. The environment stays consistent. Faults are applied at the call
      boundary; they never leave the underlying state half-mutated
      (silent_corruption alters only the returned value, not the state).

 Usage:
   python 01_environment/fault_injection.py --self-test      # full suite
   python 01_environment/fault_injection.py --demo T001      # show a trace

 Import:
   from fault_injection import ToolEnvironment, FAULT_TYPES
   env = ToolEnvironment(task, fault="timeout", position="late", seed=0)
   spec = env.tool_specs()                # JSON schema for the agent
   result = env.call("mkdir", {"dir_name": "temp"})

 Saves (with --self-test):
   01_environment/outputs/fault_selftest_report.txt
=============================================================================
"""

import argparse
import copy
import importlib
import inspect
import json
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BFCL_PKG = ROOT / "third_party" / "gorilla" / "berkeley-function-call-leaderboard"
if str(BFCL_PKG) not in sys.path:
    sys.path.insert(0, str(BFCL_PKG))

FAULT_TYPES = ["clean", "timeout", "silent_corruption", "schema_drift",
               "missing_tool", "empty_result"]

TIMEOUT_MSG = ("ToolError: request to '{tool}' timed out after 30s "
               "(no response from service)")
SCHEMA_MSG = ("ToolError: {tool}() got an unexpected keyword argument "
              "'{bad}'. Check the current parameter names for this tool.")
MISSING_MSG = ("ToolError: tool '{tool}' is not available in this "
               "environment.")


# ---------------------------------------------------------------------------
# Environment wrapper
# ---------------------------------------------------------------------------
class ToolEnvironment:
    """A task's BFCL environment classes, exposed as one tool registry with
    a deterministic typed fault."""

    def __init__(self, task: dict, fault: str = "clean",
                 position: str = "late", seed: int = 0):
        if fault not in FAULT_TYPES:
            raise ValueError(f"unknown fault '{fault}'")
        self.task = task
        self.fault = fault
        self.position = position
        self.rng = random.Random(f"{task['task_id']}|{fault}|{position}|{seed}")

        self.instances: dict[str, Any] = {}
        self.registry: dict[str, tuple[str, Any]] = {}   # tool -> (cls, method)
        self._build_registry()

        # step bookkeeping
        self.step = 0
        self.fault_fired = False
        self.fault_step = None
        self.fault_tool = None
        self.log: list[dict] = []

        n_turns = task.get("n_turns", 4)
        self.target_step = 1 if position == "early" else max(
            1, n_turns // 2)          # 'late' = midpoint of the reference

        # for missing_tool / schema_drift we must decide the victim up front
        self.victim_tool = None
        if fault in ("missing_tool", "schema_drift"):
            self.victim_tool = self._pick_victim()
        self.hidden_tools = ({self.victim_tool}
                             if fault == "missing_tool" and self.victim_tool
                             else set())

    # ------------------------------------------------------------ registry --
    def _build_registry(self):
        from bfcl_eval.constants.executable_backend_config import (
            CLASS_FILE_PATH_MAPPING)
        for cls_name in self.task["involved_classes"]:
            module = importlib.import_module(CLASS_FILE_PATH_MAPPING[cls_name])
            inst = getattr(module, cls_name)()
            cfg = (self.task.get("initial_config") or {}).get(cls_name)
            if cfg is not None and hasattr(inst, "_load_scenario"):
                inst._load_scenario(copy.deepcopy(cfg))
            self.instances[cls_name] = inst
            for name, meth in inspect.getmembers(inst, inspect.ismethod):
                if name.startswith("_"):
                    continue
                self.registry.setdefault(name, (cls_name, meth))

    def _pick_victim(self):
        """Choose the fault victim deterministically.

        The victim must be a tool the agent actually needs, otherwise the
        fault may never fire and the trial silently becomes a clean run.
        Priority order:
          1. tools named in the task's ground-truth call path, taken from
             the LAST turn backwards so the victim sits late in the work
          2. any tool in the ground-truth path
          3. a mutating tool by name heuristic (fallback)
        """
        gt_tools = []
        for turn in (self.task.get("ground_truth") or []):
            calls = turn if isinstance(turn, list) else [turn]
            for c in calls:
                if not isinstance(c, str):
                    continue
                m = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\(", c.strip())
                if m and m.group(1) in self.registry:
                    gt_tools.append(m.group(1))

        if gt_tools:
            # prefer a tool used at or after the target step position
            idx = min(max(self.target_step - 1, 0), len(gt_tools) - 1)
            pool = gt_tools[idx:] or gt_tools
            # deduplicate, keep order, then choose deterministically
            seen, ordered = set(), []
            for t in pool:
                if t not in seen:
                    seen.add(t)
                    ordered.append(t)
            return self.rng.choice(ordered)

        prefer = [t for t in sorted(self.registry)
                  if any(t.startswith(p) for p in
                         ("post", "send", "create", "add", "mkdir", "cp",
                          "mv", "book", "buy", "sell", "start", "set",
                          "update", "write", "touch", "echo"))]
        pool = prefer or sorted(self.registry)
        return self.rng.choice(pool) if pool else None

    # --------------------------------------------------------------- specs --
    def tool_specs(self) -> list[dict]:
        """OpenAI-style tool schemas for every visible tool."""
        specs = []
        for name in sorted(self.registry):
            if name in self.hidden_tools and self.fault_fired:
                continue        # removed only after the fault fires
            _, meth = self.registry[name]
            sig = inspect.signature(meth)
            props, required = {}, []
            for pname, param in sig.parameters.items():
                if pname == "self":
                    continue
                ann = param.annotation
                jtype = {int: "integer", float: "number", bool: "boolean",
                         list: "array", dict: "object"}.get(ann, "string")
                props[pname] = {"type": jtype}
                if param.default is inspect.Parameter.empty:
                    required.append(pname)
            doc = (inspect.getdoc(meth) or "").split("\n")[0][:180]
            specs.append({"type": "function", "function": {
                "name": name, "description": doc,
                "parameters": {"type": "object", "properties": props,
                               "required": required}}})
        return specs

    def visible_tools(self) -> list[str]:
        return [s["function"]["name"] for s in self.tool_specs()]

    # ---------------------------------------------------------------- call --
    def _should_fire(self, tool: str) -> bool:
        if self.fault == "clean" or self.fault_fired:
            return False
        if self.step < self.target_step:
            return False
        if self.victim_tool is not None:
            return tool == self.victim_tool
        return True

    def call(self, tool: str, args: dict | None = None) -> dict:
        """Execute a tool call. Returns a result envelope:
        {ok: bool, value|error: ..., faulted: bool}"""
        args = args or {}
        self.step += 1
        rec = {"step": self.step, "tool": tool, "args": args,
               "faulted": False}

        # tool removed earlier in this trial
        if tool in self.hidden_tools and self.fault_fired:
            out = {"ok": False, "error": MISSING_MSG.format(tool=tool),
                   "faulted": False}
            rec.update(out)
            self.log.append(rec)
            return out

        if tool not in self.registry:
            out = {"ok": False,
                   "error": f"ToolError: unknown tool '{tool}'. Available: "
                            f"{', '.join(self.visible_tools()[:12])} ...",
                   "faulted": False}
            rec.update(out)
            self.log.append(rec)
            return out

        # ------------------------------------------------------ fault fires --
        if self._should_fire(tool):
            # Value-based faults (corruption, empty payload) only make sense
            # on a call that would otherwise SUCCEED. If the agent's call is
            # malformed we let the real error through and wait for the next
            # eligible call, so the condition stays pure.
            if self.fault in ("silent_corruption", "empty_result"):
                real = self._execute(tool, args)
                if not real["ok"]:
                    rec.update(real)
                    self.log.append(rec)
                    return real
                self.fault_fired = True
                self.fault_step = self.step
                self.fault_tool = tool
                rec["faulted"] = True
                out = ({"ok": True, "value": _corrupt(real["value"], self.rng),
                        "faulted": True}
                       if self.fault == "silent_corruption" else
                       {"ok": True,
                        "value": {} if isinstance(real.get("value"), dict)
                        else [], "faulted": True})
                rec.update(out)
                self.log.append(rec)
                return out

            self.fault_fired = True
            self.fault_step = self.step
            self.fault_tool = tool
            rec["faulted"] = True
            out = self._apply_fault(tool, args)
            rec.update(out)
            self.log.append(rec)
            return out

        # ------------------------------------------------- normal execution --
        out = self._execute(tool, args)
        rec.update(out)
        self.log.append(rec)
        return out

    def _execute(self, tool: str, args: dict) -> dict:
        _, meth = self.registry[tool]
        try:
            value = meth(**args)
            return {"ok": True, "value": _jsonable(value), "faulted": False}
        except TypeError as e:
            return {"ok": False, "error": f"ToolError: {e}", "faulted": False}
        except Exception as e:
            return {"ok": False, "error": f"ToolError: {type(e).__name__}: {e}",
                    "faulted": False}

    def _apply_fault(self, tool: str, args: dict) -> dict:
        f = self.fault
        if f == "timeout":
            return {"ok": False, "error": TIMEOUT_MSG.format(tool=tool),
                    "faulted": True}

        if f == "missing_tool":
            return {"ok": False, "error": MISSING_MSG.format(tool=tool),
                    "faulted": True}

        if f == "schema_drift":
            bad = sorted(args)[0] if args else "argument"
            return {"ok": False, "error": SCHEMA_MSG.format(tool=tool,
                                                            bad=bad),
                    "faulted": True}

        # silent_corruption and empty_result are handled in call(), because
        # they require a successful underlying execution.
        raise AssertionError(f"unhandled fault {f}")

    # -------------------------------------------------------------- status --
    def summary(self) -> dict:
        return {"task_id": self.task["task_id"], "fault": self.fault,
                "position": self.position, "target_step": self.target_step,
                "fault_fired": self.fault_fired, "fault_step": self.fault_step,
                "fault_tool": self.fault_tool, "n_steps": self.step,
                "victim_tool": self.victim_tool}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _jsonable(v):
    try:
        json.dumps(v)
        return v
    except (TypeError, ValueError):
        return str(v)


def _corrupt(value, rng, _depth=0):
    """Return a plausible but wrong version of a tool result.

    Numbers are perturbed, strings altered, list membership and order
    changed, one dict field replaced. The SHAPE is preserved -- nothing in
    the envelope signals a problem, which is what makes the corruption
    silent. The result is verified to differ from the input; if a value
    cannot be perturbed in kind (e.g. None), a plausible substitute is
    used instead.
    """
    def _same(a, b):
        try:
            return json.dumps(a, sort_keys=True, default=str) == \
                   json.dumps(b, sort_keys=True, default=str)
        except Exception:
            return a == b

    if value is None:
        return "unknown"
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        d = max(1, int(abs(value) * rng.uniform(0.2, 0.6)))
        return value + d * rng.choice([-1, 1])
    if isinstance(value, float):
        return round(value * rng.uniform(1.2, 1.6) + 0.01, 4)
    if isinstance(value, str):
        if not value.strip():
            return "unavailable"
        parts = value.split()
        if len(parts) > 1:
            shuffled = parts[:]
            for _ in range(4):
                rng.shuffle(shuffled)
                if shuffled != parts:
                    break
            if shuffled != parts:
                return " ".join(shuffled)
        rev = value[::-1]
        return rev if rev != value else value + "_x"
    if isinstance(value, list):
        if not value:
            return ["<unexpected entry>"]
        out = list(value)
        if len(out) > 1 and rng.random() < 0.7:
            out.pop(rng.randrange(len(out)))          # drop an element
        else:
            i = rng.randrange(len(out))
            out[i] = _corrupt(out[i], rng, _depth + 1)
        if _same(out, value):
            out = out + ["<unexpected entry>"]
        return out
    if isinstance(value, dict):
        if not value:
            return {"status": "unknown"}
        out = dict(value)
        keys = sorted(out, key=str)
        rng.shuffle(keys)
        for k in keys:                                # first key that moves
            cand = _corrupt(out[k], rng, _depth + 1)
            if not _same(cand, out[k]):
                out[k] = cand
                return out
        k = keys[0]                                   # nothing moved: force it
        out[k] = "unknown" if out[k] != "unknown" else "n/a"
        return out
    return value


def zero_arg_tools(env) -> list[str]:
    """Tools whose parameters are all optional -- callable with {}."""
    out = []
    for spec in env.tool_specs():
        if not spec["function"]["parameters"]["required"]:
            out.append(spec["function"]["name"])
    return out


CALL_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\((.*)\)$", re.DOTALL)


def ground_truth_calls(task) -> list[tuple[str, tuple, dict]]:
    """(tool, args, kwargs) for every ground-truth call string of the task."""
    calls = []
    for turn in (task.get("ground_truth") or []):
        for c in (turn if isinstance(turn, list) else [turn]):
            m = CALL_RE.match(c.strip()) if isinstance(c, str) else None
            if not m:
                continue
            try:
                args, kwargs = eval(f"_cap({m.group(2)})",
                                    {"__builtins__": {}},
                                    {"_cap": lambda *a, **k: (a, k),
                                     "True": True, "False": False,
                                     "None": None})
            except SyntaxError:     # a few BFCL strings are not valid Python
                continue
            calls.append((m.group(1), args, kwargs))
    return calls


def replay(task, fault, position="late", seed=0):
    """Run the task's ground-truth calls through the layer."""
    env = ToolEnvironment(task, fault, position, seed=seed)
    results = []
    for name, args, kwargs in ground_truth_calls(task):
        kwargs = copy.deepcopy(kwargs)
        if args and name in env.registry:       # bind positional arguments
            params = [p for p in inspect.signature(
                env.registry[name][1]).parameters if p != "self"]
            kwargs.update(zip(params, copy.deepcopy(args)))
        results.append(env.call(name, kwargs))
    # A victim reached only through an unparsable ground-truth string is
    # probed once directly, so victim-based faults are still exercised.
    if env.victim_tool and not env.fault_fired:
        results.append(env.call(env.victim_tool, {}))
    return env, results


def state_snapshot(env) -> dict:
    """JSON-safe snapshot of every environment instance's public state."""
    snap = {}
    for cls_name, inst in env.instances.items():
        fields = {}
        for k, v in vars(inst).items():
            if k.startswith("_"):
                continue
            try:
                json.dumps(v)
                fields[k] = v
            except (TypeError, ValueError):
                fields[k] = str(v)
        snap[cls_name] = fields
    return snap


def load_suite():
    import yaml
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    p = ROOT / cfg["paths"]["task_suite"]
    if not p.exists():
        raise SystemExit("[ERROR] no frozen task suite; run "
                         "01_environment/build_task_suite.py first.")
    return json.loads(p.read_text()), cfg


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------
def self_test() -> int:
    suite, cfg = load_suite()
    lines, failures = [], 0

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    def assert_(label, cond, detail=""):
        nonlocal failures
        if not cond:
            failures += 1
        out(f"  [{'✓' if cond else '✗'}] {label}"
            + (f"  -- {detail}" if detail else ""))

    out("=" * 74)
    out(" BRITTLE AGENTS -- FAULT INJECTION SELF-TEST")
    out(f" Timestamp : {datetime.now(timezone.utc).isoformat()}")
    out(f" Suite     : {len(suite)} tasks")
    out("=" * 74)

    # Every check runs on every task. Each task is exercised by replaying its
    # ground-truth calls (real tool names and arguments) through the layer.
    def failing(bad):
        return "" if not bad else "failed on " + ", ".join(bad)

    # -- 1. registry & specs --------------------------------------------------
    out("\n[1/6] Registry and tool specs (every task)")
    bad = []
    for t in suite:
        env = ToolEnvironment(t, "clean")
        specs = env.tool_specs()
        if not (specs and len(specs) == len(env.registry)
                and json.dumps(specs)):
            bad.append(t["task_id"])
    assert_(f"every tool has a JSON-serialisable spec "
            f"({len(suite) - len(bad)}/{len(suite)} tasks)", not bad,
            failing(bad))

    # -- 2. clean condition ---------------------------------------------------
    out("\n[2/6] Clean condition never faults (every task)")
    clean_runs, bad = {}, []
    for t in suite:
        env, results = replay(t, "clean")
        clean_runs[t["task_id"]] = (env, results)
        if env.fault_fired or any(r.get("faulted") for r in results):
            bad.append(t["task_id"])
    assert_(f"no fault in a clean replay ({len(suite) - len(bad)}/"
            f"{len(suite)} tasks)", not bad, failing(bad))

    # -- 3. each fault fires exactly once -------------------------------------
    out("\n[3/6] Each fault type fires exactly once, at or after the target "
        "(every task)")
    runs = {}
    for fault in FAULT_TYPES[1:]:
        fired, once, late = [], [], []
        for t in suite:
            env, results = replay(t, fault)
            runs[(t["task_id"], fault)] = (env, results)
            if not env.fault_fired:
                fired.append(t["task_id"])
            if sum(1 for r in env.log if r["faulted"]) != 1:
                once.append(t["task_id"])
            if env.fault_step is not None and env.fault_step < env.target_step:
                late.append(t["task_id"])
        assert_(f"{fault:<18} fires", not fired, failing(fired))
        assert_(f"{fault:<18} fires exactly once", not once, failing(once))
        assert_(f"{fault:<18} fires at or after the target step", not late,
                failing(late))

    # -- 4. fault semantics ---------------------------------------------------
    out("\n[4/6] Fault semantics (every task)")

    def faulted_record(key):
        env, _ = runs[key]
        return next((r for r in env.log if r["faulted"]), None)

    checks = {
        "timeout returns an error mentioning a timeout":
            lambda t: (lambda r: r is not None and not r["ok"]
                       and "timed out" in r["error"])(
                faulted_record((t["task_id"], "timeout"))),
        "schema_drift returns an unknown-argument error":
            lambda t: (lambda r: r is not None and not r["ok"]
                       and "unexpected keyword argument" in r["error"])(
                faulted_record((t["task_id"], "schema_drift"))),
        "missing_tool removes the tool and it stays gone":
            lambda t: (lambda env: env.fault_fired
                       and env.victim_tool not in env.visible_tools()
                       and not env.call(env.victim_tool, {})["ok"])(
                runs[(t["task_id"], "missing_tool")][0]),
        "empty_result succeeds with an empty payload":
            lambda t: (lambda r: r is not None and r["ok"]
                       and r["value"] in ({}, []))(
                faulted_record((t["task_id"], "empty_result"))),
    }
    for label, fn in checks.items():
        bad = [t["task_id"] for t in suite if not fn(t)]
        assert_(f"{label} ({len(suite) - len(bad)}/{len(suite)})", not bad,
                failing(bad))

    # -- 5. silent corruption changes the value, not the shape or state -------
    out("\n[5/6] silent_corruption: value differs, type and state preserved "
        "(every task)")
    n_none = 0
    diff, same_type, state = [], [], []
    for t in suite:
        tid = t["task_id"]
        env, results = runs[(tid, "silent_corruption")]
        cenv, cresults = clean_runs[tid]
        i = next((k for k, r in enumerate(results) if r.get("faulted")), None)
        if i is None:
            diff.append(tid)
            same_type.append(tid)
            state.append(tid)
            continue
        dv, cv = results[i]["value"], cresults[i].get("value")
        n_none += cv is None
        if (json.dumps(dv, sort_keys=True, default=str)
                == json.dumps(cv, sort_keys=True, default=str)):
            diff.append(tid)
        # Documented exception: an empty (None) result becomes "unknown".
        if not (isinstance(dv, type(cv)) or (cv is None and dv == "unknown")):
            same_type.append(tid)
        if (json.loads(json.dumps(state_snapshot(env), default=str))
                != json.loads(json.dumps(state_snapshot(cenv), default=str))):
            state.append(tid)
    n = len(suite)
    assert_(f"corrupted value differs from the clean value "
            f"({n - len(diff)}/{n})", not diff, failing(diff))
    assert_(f"corrupted value keeps its type, None -> 'unknown' "
            f"({n - len(same_type)}/{n})",
            not same_type, failing(same_type))
    out(f"  (the real result was None on {n_none} task(s); "
        f"'unknown' is returned there by design)")
    assert_(f"end state after replay equals the clean replay "
            f"({n - len(state)}/{n})", not state, failing(state))

    # -- 6. determinism -------------------------------------------------------
    out("\n[6/6] Determinism: same inputs -> identical fault behaviour "
        "(every task)")
    for fault in FAULT_TYPES[1:]:
        bad = []
        for t in suite:
            a, ra = replay(t, fault, seed=1)
            b, rb = replay(t, fault, seed=1)
            if (json.dumps([a.summary(), ra], default=str)
                    != json.dumps([b.summary(), rb], default=str)):
                bad.append(t["task_id"])
        assert_(f"{fault:<18} reproducible ({n - len(bad)}/{n})", not bad,
                failing(bad))

    out("\n" + "=" * 74)
    if failures == 0:
        out(" RESULT: SELF-TEST PASSED ✓  -- the instrument behaves as specified")
    else:
        out(f" RESULT: {failures} CHECK(S) FAILED ✗")
    out("=" * 74)

    rep = ROOT / "01_environment" / "outputs" / "fault_selftest_report.txt"
    rep.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport saved -> {rep.relative_to(ROOT)}")
    return 0 if failures == 0 else 1


def demo(task_id: str) -> int:
    suite, _ = load_suite()
    task = next((t for t in suite if t["task_id"] == task_id), None)
    if task is None:
        raise SystemExit(f"[ERROR] no task '{task_id}' in the suite")
    print(f"Task {task['task_id']}  domain={task['domain']}  "
          f"turns={task['n_turns']}")
    print(f"User goal (turn 1): "
          f"{task['turns'][0][0]['content'][:150]}\n")
    for fault in FAULT_TYPES:
        env = ToolEnvironment(task, fault, "late")
        print(f"--- fault={fault} (target step {env.target_step}, "
              f"victim={env.victim_tool}) ---")
        probe_tools = zero_arg_tools(env)
        if env.victim_tool and env.victim_tool not in probe_tools:
            probe_tools.insert(min(2, len(probe_tools)), env.victim_tool)
        shown = 0
        for name in probe_tools:
            if shown >= 5 and env.fault_fired:
                break
            res = env.call(name, {"some_param": "x"}
                           if name == env.victim_tool
                           and env.fault == "schema_drift" else {})
            shown += 1
            flag = " <-- FAULT" if res.get("faulted") else ""
            body = (str(res.get("value"))[:70] if res["ok"]
                    else res["error"][:70])
            print(f"  step {env.step}: {name:<18} "
                  f"{'ok ' if res['ok'] else 'ERR'} {body}{flag}")
        print()
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fault injection instrument")
    ap.add_argument("--self-test", action="store_true",
                    help="run the full verification suite")
    ap.add_argument("--demo", type=str, metavar="TASK_ID",
                    help="print a short trace of every fault on one task")
    a = ap.parse_args()
    if a.demo:
        raise SystemExit(demo(a.demo))
    if a.self_test:
        raise SystemExit(self_test())
    ap.print_help()
