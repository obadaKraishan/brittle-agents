#!/usr/bin/env python3
"""
=============================================================================
 BRITTLE AGENTS | tests/verify_paper_numbers.py
=============================================================================
 Independent check of every number the paper reports. Nothing here imports
 the pipeline: trials are rebuilt from the raw trajectories and the cached
 judge codings, compared field by field with master_trials.csv, and each
 statistic is recomputed and compared with the value printed in the paper
 (the EXPECTED column below, formatted as the paper prints it).

 With --tex, the paper source is also read: every row of its five data
 tables is compared with the recomputed row, and any TBD left in the source
 fails the run.

 No API calls. Exit code 0 when every check passes, 1 otherwise.

 Usage:
   python tests/verify_paper_numbers.py
   python tests/verify_paper_numbers.py --tex path/to/main.tex
=============================================================================
"""

import json
import math
import re
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
TRAJ = ROOT / "02_data_collection/outputs/trajectories"
PROC = ROOT / "03_data_processing/outputs"
FAULTS = ["timeout", "missing_tool", "schema_drift", "silent_corruption"]
ROWS = []

P1 = lambda v: f"{100 * v:.1f}"                                   # noqa: E731
P0 = lambda v: f"{100 * v:.0f}"                                   # noqa: E731
F2 = lambda v: f"{v:.2f}"                                         # noqa: E731
F1 = lambda v: f"{v:.1f}"                                         # noqa: E731
I = lambda v: f"{int(v)}"                                         # noqa: E731,E741
SP1 = lambda v: f"{100 * v:+.1f}"                                 # noqa: E731
SP2 = lambda v: f"{v:+.2f}"                                       # noqa: E731
CI = lambda v: f"[{100 * v[0]:.1f}, {100 * v[1]:.1f}]"            # noqa: E731
SCI = lambda v: f"[{100 * v[0]:+.1f}, {100 * v[1]:+.1f}]"         # noqa: E731


def P3(p):
    return "<.001" if p < .001 else ("1.000" if p >= .9995
                                     else f"{p:.3f}".lstrip("0"))


def check(section, label, expected, value, fmt):
    got = fmt(value)
    ROWS.append(dict(section=section, quantity=label, expected=expected,
                     computed=got, ok=got == expected))


# ======================================================== rebuild trials ===
DETECT = [re.compile(p, re.I) for p in [
    r"\b(fail(ed|ure|ing)?|error|not work|didn'?t work|unsuccessful)\b",
    r"\b(time(d)? ?out|timeout|unavailable|unreachable)\b",
    r"\b(unexpected|unexpectedly|surprising|odd|strange|wrong|incorrect)\b",
    r"\b(doesn'?t (seem|look) (right|correct)|seems? (off|wrong|incorrect))\b",
    r"\b(no longer|missing|does not exist|not available|removed)\b",
    r"\b(empty|nothing (was )?returned|no results?|blank)\b",
    r"\b(retry|retrying|try again|another (approach|way|method)|instead)\b",
    r"\b(let me (verify|check|confirm|re-?check))\b"]]
LEAK = re.compile(r"tool.{0,3}call.{0,3}begin|<\|tool", re.I)


def maxrun(seq):
    best = run = 1 if seq else 0
    for i in range(1, len(seq)):
        run = run + 1 if seq[i] == seq[i - 1] else 1
        best = max(best, run)
    return best


def agree(a, b):
    tot = m = 0
    for cls, fields in (b or {}).items():
        o = (a or {}).get(cls, {})
        for k, v in (fields or {}).items():
            tot += 1
            m += (json.dumps(o.get(k), sort_keys=True, default=str)
                  == json.dumps(v, sort_keys=True, default=str))
    return m / tot if tot else float("nan")


def rebuild():
    recs, failed = {}, 0
    for f in sorted(TRAJ.glob("*.jsonl")):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("status") != "ok":
                failed += 1
            elif r["trial_id"] not in recs:
                recs[r["trial_id"]] = r
    judge = {}
    for line in (PROC / "judge_cache.jsonl").read_text().splitlines():
        j = json.loads(line)
        judge[j["trial_id"]] = j

    cstate, csteps = {}, defaultdict(list)
    for r in recs.values():
        if r["fault"] == "clean" and r["scaffolding"] == "off":
            cstate[(r["model"], r["task_id"], r["seed"])] = r.get("final_state")
            csteps[(r["model"], r["task_id"])].append(r["n_steps"])

    rows = []
    for r in recs.values():
        res = [e for e in r["transcript"] if e["type"] == "tool_result"]
        fs = (r.get("env_summary") or {}).get("fault_step")
        ts = fs if fs is not None else (
            max(e.get("step", 0) for e in res) // 2 if res else 0)
        tgt = next((e for e in res if e.get("step") == ts), None)
        seen, text = False, []
        for e in r["transcript"]:
            if e["type"] == "tool_result" and e.get("step", 0) >= ts:
                seen = True
            elif e["type"] == "assistant" and seen:
                text.append(e.get("content") or "")
        text = "\n".join(text).strip()
        call = lambda e: (e["tool"], json.dumps(e.get("args"),  # noqa: E731
                                                sort_keys=True))
        after = [call(e) for e in res if fs is not None
                 and e.get("step", 0) > fs]
        failedc = next((call(e) for e in res if e.get("step") == fs), None)
        self_clean = r["fault"] == "clean" and r["scaffolding"] == "off"
        ref_seed = 1 - r["seed"] if self_clean else r["seed"]
        own = cstate.get((r["model"], r["task_id"], ref_seed))
        a_own = agree(r.get("final_state"), own) if own else None
        bs = csteps.get((r["model"], r["task_id"]))
        j = judge.get(r["trial_id"]) or {}
        det = j.get("detected")
        if det is None:
            det = any(p.search(text) for p in DETECT)
        rows.append(dict(
            trial_id=r["trial_id"], model=r["model"],
            reasoning=r["reasoning"], pair=r["pair"], fault=r["fault"],
            scaffolding=r["scaffolding"], task_id=r["task_id"],
            seed=r["seed"], fault_fired=bool(r["env_summary"].get(
                "fault_fired")), fault_step=fs, n_steps=r["n_steps"],
            budget_exhausted=float(r.get("finish_reason") == "step_budget"),
            target_ok=bool(tgt.get("ok")) if tgt else None,
            detection=float(det), judge_code=j.get("code"),
            replanned=float(bool(after) and failedc is not None
                            and after[0] != failedc),
            perseveration=float(maxrun([c[0] for c in after]) >= 3),
            repeat_identical=float(maxrun(after) >= 3),
            recovered=(float(a_own >= .999) if a_own is not None
                       and a_own == a_own else np.nan),
            recovery_cost=(r["n_steps"] - sum(bs) / len(bs)
                           if bs else np.nan),
            leaked=sum(1 for e in r["transcript"] if e["type"] == "assistant"
                       and LEAK.search(e.get("content") or "")),
        ))
    return pd.DataFrame(rows), failed


# =============================================================== stats ====
def prop(a, b):
    a, b = a.dropna(), b.dropna()
    p1, p2 = a.mean(), b.mean()
    p = (a.sum() + b.sum()) / (len(a) + len(b))
    z = (p1 - p2) / math.sqrt(p * (1 - p) * (1 / len(a) + 1 / len(b)))
    return dict(z=z, p=2 * (1 - stats.norm.cdf(abs(z))), diff=p1 - p2,
                h=2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2)))


def cboot(df, col, n_boot=2000, seed=42):
    groups = [g[col].dropna().values for _, g in df.groupby("task_id")]
    groups = [g for g in groups if len(g)]
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(groups), len(groups))
        out[i] = np.concatenate([groups[j] for j in idx]).mean()
    return np.percentile(out, 2.5), np.percentile(out, 97.5)


def holm(ps):
    p = np.asarray(ps, float)
    adj, prev = np.empty(len(p)), 0
    for rank, i in enumerate(np.argsort(p)):
        prev = max(prev, min((len(p) - rank) * p[i], 1))
        adj[i] = prev
    return adj


def gee(df, formula):
    y = formula.split("~")[0].strip()
    d = df.dropna(subset=[y]).copy()
    d[y] = d[y].astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return smf.gee(formula, groups="task_id", data=d,
                       family=sm.families.Binomial(),
                       cov_struct=sm.cov_struct.Exchangeable()).fit()


def term(m, name):
    b, se = m.params[name], m.bse[name]
    return dict(OR=math.exp(b), z=b / se, p=m.pvalues[name])


KEYS = ["task_id", "fault", "seed"]


def pairs(df, split, a_val, b_val, col):
    a = df[df[split] == a_val].groupby(KEYS)[col].mean()
    b = df[df[split] == b_val].groupby(KEYS)[col].mean()
    j = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna().reset_index()
    j["d"] = j["a"] - j["b"]
    return j


def matched_gee(df, split, a_val, b_val, col):
    j = pairs(df, split, a_val, b_val, col)
    keep = pd.MultiIndex.from_frame(j[KEYS])
    d = df[df[split].isin([a_val, b_val])]
    d = d[pd.MultiIndex.from_frame(d[KEYS]).isin(keep)].copy()
    d["x"] = (d[split] == a_val).astype(float)
    return term(gee(d, f"{col} ~ x"), "x")


# ======================================================== paper tables ====
def tex_rows(tex, label):
    """Data rows of the tabular that carries \\label{label}, as cell lists."""
    i = tex.index(f"\\label{{{label}}}")
    body = tex[tex.index("\\begin{tabular}", i):tex.index("\\end{tabular}", i)]
    rows = [r.strip() for r in body.split("\\\\")]
    rows = [r.replace("\\hline", "").strip() for r in rows]
    # the header row is the piece that still carries \begin{tabular}
    rows = [r for r in rows if "&" in r and not r.startswith("\\begin")]
    return [[c.strip() for c in r.split("&")] for r in rows]


def normalise(text):
    return [line.strip() for line in text.splitlines() if line.strip()]


def table_source(tex, label):
    """Lines of the whole table environment that carries \\label{label}."""
    i = tex.index(f"\\label{{{label}}}")
    start = max(tex.rfind("\\begin{table}", 0, i),
                tex.rfind("\\begin{table*}", 0, i))
    env = "table*" if tex.startswith("\\begin{table*}", start) else "table"
    end = tex.index(f"\\end{{{env}}}", i) + len(f"\\end{{{env}}}")
    return normalise(tex[start:end])


def pct(v, d=1):
    return f"{100 * v:.{d}f}\\%"


def signed(v, d=1, scale=100):
    return f"${scale * v:+.{d}f}$".replace("+", "+") if v >= 0 \
        else f"${scale * v:.{d}f}$"


def paper_tables(D, A, F, base, loud, quiet, clean, rq2):
    """The paper's data tables, recomputed, formatted as the paper prints."""
    nice = lambda f: f.replace("_", " ")                       # noqa: E731
    vis = {"timeout": "loud", "missing_tool": "loud",
           "schema_drift": "loud", "silent_corruption": "quiet"}
    t = {"tab:faults": [["clean", "baseline", I((D.fault == "clean").sum()),
                         "--", I(len(clean))]]}
    for f_ in FAULTS:
        g = D[D.fault == f_]
        t["tab:faults"].append([nice(f_), vis[f_], I(len(g)),
                                f"{P0(g.fault_fired.mean())}\\%",
                                I(g.fault_fired.sum())])
    t["tab:design"] = []
    for m_, fam, kind in [("claude-haiku-4.5-thinking", "claude", "reasoning"),
                          ("claude-haiku-4.5", "claude", "instruct"),
                          ("deepseek-r1", "deepseek", "reasoning"),
                          ("deepseek-v3", "deepseek", "instruct"),
                          ("qwen3-thinking", "qwen", "reasoning"),
                          ("qwen3-instruct", "qwen", "instruct")]:
        g = D[D.model == m_]
        t["tab:design"].append([m_, fam, kind, I(len(g)),
                                "yes" if (g.scaffolding == "on").any()
                                else "--"])
    t["tab:rq1"] = []
    for name, g in [("no fault (baseline)", base), ("quiet failure", quiet),
                    ("loud failure", loud)] + [
            (nice(f_), F[F.fault == f_]) for f_ in FAULTS]:
        t["tab:rq1"].append([name, I(len(g)), pct(g.detection.mean()),
                             CI(cboot(g, "detection"))])
    t["tab:rq2"] = []
    names = {"detection": "detection", "replanned": "replanning",
             "recovered": "recovery"}
    for col in ["detection", "replanned", "recovered"]:
        for fam, j, g, ph in rq2[col]:
            lo, hi = cboot(j, "d")
            p = "$<$.001" if ph < .001 else P3(ph)
            t["tab:rq2"].append([
                names[col], fam, pct(j.a.mean()), pct(j.b.mean()),
                signed(j.d.mean()),
                f"$[{100 * lo:+.1f}, {100 * hi:+.1f}]$", F2(g["OR"]), p])
    t["tab:rq3"] = []
    for f_ in ["clean"] + FAULTS:
        g = A[A.fault == f_]
        same = "--" if f_ == "clean" else pct(g.perseveration.mean())
        ident = "--" if f_ == "clean" else pct(g.repeat_identical.mean())
        t["tab:rq3"].append([nice(f_), pct(g.recovered.mean()), same, ident,
                             pct(g.budget_exhausted.mean()),
                             signed(g.recovery_cost.mean(), 2, 1)])
    return t


# ================================================================ main ====
def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--tex", type=Path,
                    help="paper source; also compare its tables and check "
                         "that no TBD remains")
    args = ap.parse_args()

    D, failed = rebuild()

    # rebuilt trials must equal the published scored data
    M = pd.read_csv(PROC / "master_trials.csv").set_index("trial_id")
    X = D.set_index("trial_id").loc[M.index]
    for mine, theirs in [("detection", "detection"),
                         ("replanned", "replanned"),
                         ("perseveration", "perseveration"),
                         ("repeat_identical", "repeat_identical"),
                         ("recovered", "recovered"),
                         ("budget_exhausted", "budget_exhausted"),
                         ("target_ok", "target_ok"),
                         ("fault_fired", "fault_fired"),
                         ("n_steps", "n_steps"),
                         ("recovery_cost", "recovery_cost"),
                         ("leaked", "leaked_tool_calls")]:
        a = pd.to_numeric(X[mine].astype(object).map(
            lambda v: np.nan if v is None else float(v)))
        b = pd.to_numeric(M[theirs].astype(object).map(
            lambda v: np.nan if v is None or v != v else float(
                v in (True, "True") if isinstance(v, (bool, str)) else v)))
        check("rebuild", f"rows differing on {theirs}", "0",
              int((~np.isclose(a, b, equal_nan=True)).sum()), I)

    A = D[(D.fault == "clean") | D.fault_fired].copy()
    A["is_fault"] = A.fault != "clean"
    F = A[A.is_fault]
    base = A[(A.fault == "clean") & (A.target_ok == True)]  # noqa: E712
    loud = F[F.target_ok == False]  # noqa: E712
    quiet = F[F.target_ok == True]  # noqa: E712
    clean = A[A.fault == "clean"]

    # ------------------------------------------------- Method section ---
    S = "method"
    log = (ROOT / "01_environment/outputs/suite_summary.txt").read_text()
    check(S, "candidate tasks", "400",
          re.search(r"total candidate tasks: (\d+)", log).group(1), str)
    check(S, "tasks passing the filters", "312",
          re.search(r"eligible: (\d+)", log).group(1), str)
    suite = json.loads((ROOT / "01_environment/outputs/task_suite.json")
                       .read_text())
    check(S, "tasks", "24", len(suite), I)
    check(S, "domain combinations", "16",
          len({t["domain"] for t in suite}), I)
    check(S, "median user turns", "5",
          np.median([t["n_turns"] for t in suite]), lambda v: f"{v:.0f}")
    check(S, "trials", "1920", len(D), I)
    check(S, "analysed trials", "1694", len(A), I)
    check(S, "failed API records among the six models", "0", failed, I)
    check(S, "second-arm trials", "480", (D.scaffolding == "on").sum(), I)

    ref = {t["task_id"]: sum(len(x) if isinstance(x, list) else 1
                             for x in t["ground_truth"]) for t in suite}
    st = F.fault_step
    check(S, "median firing step", "3", st.median(), lambda v: f"{v:.0f}")
    check(S, "median reference calls", "7", np.median(list(ref.values())),
          lambda v: f"{v:.0f}")
    check(S, "faults at step 1 or 2 %", "42.1", (st <= 2).mean(), P1)
    for f_, med in [("timeout", "2"), ("silent_corruption", "2"),
                    ("missing_tool", "5"), ("schema_drift", "5")]:
        check(S, f"median firing step {f_}", med,
              F[F.fault == f_].fault_step.median(), lambda v: f"{v:.0f}")
    none_sub = 0
    for f in TRAJ.glob("*__silent_corruption__*.jsonl"):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if r.get("status") == "ok" and r["env_summary"].get(
                    "fault_fired"):
                e = next(e for e in r["transcript"]
                         if e["type"] == "tool_result" and e.get("faulted"))
                none_sub += str(e.get("result")).strip('" ') == "unknown"
    check(S, "corruption trials with placeholder for an empty result", "2",
          none_sub, I)

    h = pd.read_csv(ROOT / "04_codebook/outputs/handcode_sample_FILLED.csv")
    h = h[["trial_id", "hand_code"]].merge(
        D[["trial_id", "judge_code"]], on="trial_id").dropna()

    def kappa(a, b):
        a, b = pd.Series(list(a)), pd.Series(list(b))
        po = (a == b).mean()
        pe = sum((a == c).mean() * (b == c).mean() for c in set(a) | set(b))
        return (po - pe) / (1 - pe), po
    k3, _ = kappa(h.hand_code, h.judge_code)
    k2, a2 = kappa(h.hand_code != "none", h.judge_code != "none")
    lead = lambda v: f"{v:.2f}".lstrip("0")                   # noqa: E731
    check(S, "trials coded by both", "92", len(h), I)
    check(S, "three-way kappa", ".47", k3, lead)
    check(S, "binary kappa", ".69", k2, lead)
    check(S, "binary agreement %", "85.9", a2, P1)

    dirty = A[(A.fault == "clean") & (A.target_ok == False)]  # noqa: E712
    check(S, "false-alarm baseline %", "26.8", base.detection.mean(), P1)
    check(S, "clean genuine errors acknowledged %", "90",
          dirty.detection.mean(), P0)
    check(S, "cap hit overall %", "27", D.budget_exhausted.mean(), P0)
    check(S, "DeepSeek-R1 leaked tool-call turns per trial", "1.8",
          D[D.model == "deepseek-r1"].leaked.mean(), F1)

    # ----------------------------------------------------------- RQ1 ----
    S = "RQ1"
    for nm, (a, b), d_, z, h_ in [
            ("loud-quiet", (loud, quiet), "32.5", "13.86", "0.80"),
            ("quiet-baseline", (quiet, base), "32.0", "8.36", "0.66"),
            ("loud-baseline", (loud, base), "64.5", "22.77", "1.46")]:
        t = prop(a.detection, b.detection)
        check(S, f"{nm} gap", d_, t["diff"], P1)
        check(S, f"{nm} z", z, t["z"], F2)
        check(S, f"{nm} h", h_, t["h"], F2)
    ps = [prop(F[F.fault == f_].detection, base.detection)["p"]
          for f_ in FAULTS]
    check(S, "every fault vs baseline, max Holm p", "<.001", max(holm(ps)),
          P3)
    check(S, "corruption below every loud fault (points)", "32",
          min(F[F.fault == f_].detection.mean() for f_ in FAULTS[:3])
          - F[F.fault == "silent_corruption"].detection.mean(), P0)
    V = pd.concat([base.assign(vis="baseline"),
                   F.assign(vis=np.where(F.target_ok == True,  # noqa: E712
                                         "quiet", "loud"))])
    V["vis"] = pd.Categorical(V.vis, ["baseline", "quiet", "loud"])
    m = gee(V, "detection ~ C(vis)")
    for lvl, o, z in [("quiet", "3.96", "6.48"), ("loud", "28.61", "12.14")]:
        t = term(m, f"C(vis)[T.{lvl}]")
        check(S, f"GEE {lvl} OR", o, t["OR"], F2)
        check(S, f"GEE {lvl} z", z, t["z"], F2)
    check(S, "loud trials coded 'none' %", "4",
          loud.judge_code.eq("none").mean(), P0)
    check(S, "corruption trials coded 'none' %", "39",
          quiet.judge_code.eq("none").mean(), P0)
    check(S, "corruption trials not detected %", "41",
          1 - quiet.detection.mean(), P0)

    # ----------------------------------------------------------- RQ2 ----
    S = "RQ2"
    F2off = F[F.scaffolding == "off"]
    rq2 = {}
    for col in ["detection", "replanned", "recovered"]:
        res = []
        for fam in ["claude", "deepseek", "qwen"]:
            sub = F2off[F2off.pair == fam]
            res.append((fam, pairs(sub, "reasoning", True, False, col),
                        matched_gee(sub, "reasoning", True, False, col)))
        adj = holm([g["p"] for _, _, g in res])
        rq2[col] = [(f, j, g, a) for (f, j, g), a in zip(res, adj)]
    for col, dd, ci, o, z, p in [
            ("detection", "-9.3", "[-14.3, -5.3]", "0.56", "-4.28", "<.001"),
            ("replanned", "+10.4", "[+5.6, +15.7]", "2.02", "3.52", "<.001"),
            ("recovered", "-2.4", "[-9.3, +4.5]", "0.91", None, ".512")]:
        allp = pd.concat([j for _, j, _, _ in rq2[col]])
        t = term(gee(F2off, f"{col} ~ C(reasoning) + C(fault)"),
                 "C(reasoning)[T.True]")
        check(S, f"{col} pooled pairs", "450", len(allp), I)
        check(S, f"{col} pooled delta", dd, allp.d.mean(), SP1)
        check(S, f"{col} pooled clustered CI", ci, cboot(allp, "d"), SCI)
        check(S, f"{col} pooled GEE OR", o, t["OR"], F2)
        if z:
            check(S, f"{col} pooled GEE z", z, t["z"], F2)
        check(S, f"{col} pooled GEE p", p, t["p"], P3)
    alt = []
    for fam in ["claude", "deepseek", "qwen"]:
        d = F2off[F2off.pair == fam].assign(
            x=lambda x: x.reasoning.astype(float))
        alt.append(term(gee(d, "detection ~ x + C(fault)"), "x")["p"])
    check(S, "Claude detection, all fault trials + fault, p_Holm", ".209",
          holm(alt)[0], P3)

    # ----------------------------------------------------------- RQ3 ----
    S = "RQ3"
    check(S, "clean-vs-clean agreement %", "63.3", clean.recovered.mean(), P1)
    check(S, "clean-vs-clean CI", "[53.1, 73.2]", cboot(clean, "recovered"),
          CI)
    for m_, r in [("claude-haiku-4.5", "84.4"), ("deepseek-r1", "41.7"),
                  ("deepseek-v3", "41.7"), ("qwen3-thinking", "41.7")]:
        check(S, f"clean agreement {m_}", r,
              clean[clean.model == m_].recovered.mean(), P1)
    d = A[A.recovered.notna()].copy()
    d["cond"] = pd.Categorical(d.fault, ["clean"] + FAULTS)
    m0 = gee(d, "recovered ~ C(cond)")
    m1 = gee(d, "recovered ~ C(cond) + C(model)")
    t0 = {f: term(m0, f"C(cond)[T.{f}]") for f in FAULTS}
    t1 = {f: term(m1, f"C(cond)[T.{f}]") for f in FAULTS}
    a0 = dict(zip(FAULTS, holm([t0[f]["p"] for f in FAULTS])))
    a1 = dict(zip(FAULTS, holm([t1[f]["p"] for f in FAULTS])))
    g = F[F.fault == "missing_tool"]
    check(S, "missing tool recovery %", "39.9", g.recovered.mean(), P1)
    check(S, "missing tool CI", "[28.1, 52.4]", cboot(g, "recovered"), CI)
    check(S, "missing tool h", "-0.47",
          prop(g.recovered, clean.recovered)["h"], F2)
    check(S, "missing tool OR", "0.34", t0["missing_tool"]["OR"], F2)
    check(S, "missing tool p_Holm", "<.001", a0["missing_tool"], P3)
    rest = ["timeout", "schema_drift", "silent_corruption"]
    for f_, r in zip(rest, ["59.6", "59.2", "60.4"]):
        check(S, f"recovery {f_} %", r,
              F[F.fault == f_].recovered.mean(), P1)
    check(S, "other faults OR range", "0.80 to 0.88",
          (min(t0[f]["OR"] for f in rest), max(t0[f]["OR"] for f in rest)),
          lambda v: f"{v[0]:.2f} to {v[1]:.2f}")
    check(S, "other faults smallest p_Holm", ".117",
          min(a0[f] for f in rest), P3)
    check(S, "schema drift OR with model", "0.76",
          t1["schema_drift"]["OR"], F2)
    check(S, "schema drift p_Holm with model", ".024", a1["schema_drift"],
          P3)
    check(S, "only missing tool differs from clean (p_Holm < .05)",
          "missing_tool", ",".join(f for f in FAULTS if a0[f] < .05), str)
    check(S, "same tool x3 range", "12.5 to 22.2",
          (F.groupby("fault").perseveration.mean().min(),
           F.groupby("fault").perseveration.mean().max()),
          lambda v: f"{100 * v[0]:.1f} to {100 * v[1]:.1f}")
    check(S, "identical x3 range", "0.8 to 1.8",
          (F.groupby("fault").repeat_identical.mean().min(),
           F.groupby("fault").repeat_identical.mean().max()),
          lambda v: f"{100 * v[0]:.1f} to {100 * v[1]:.1f}")
    fl = F.dropna(subset=["detection", "recovered"])
    t = prop(fl[fl.detection == 1].recovered, fl[fl.detection == 0].recovered)
    check(S, "recovery when noticed %", "55.6",
          fl[fl.detection == 1].recovered.mean(), P1)
    check(S, "recovery when not noticed %", "55.7",
          fl[fl.detection == 0].recovered.mean(), P1)
    check(S, "noticed vs not z", "-0.03", t["z"], F2)
    check(S, "noticed vs not p", ".974", t["p"], P3)
    t = term(gee(fl, "recovered ~ detection + C(fault)"), "detection")
    check(S, "noticing OR (fault fixed)", "1.51", t["OR"], F2)
    check(S, "noticing z (fault fixed)", "2.91", t["z"], F2)
    check(S, "noticing p (fault fixed)", ".004", t["p"], P3)
    t = term(gee(fl, "recovered ~ detection + C(fault) + C(model)"),
             "detection")
    check(S, "noticing OR (fault and model fixed)", "1.29", t["OR"], F2)
    check(S, "noticing p (fault and model fixed)", ".098", t["p"], P3)

    # --------------------------------------------------- scaffolding ----
    S = "secondary"
    arm = sorted(A[A.scaffolding == "on"].model.unique())
    sub = A[A.model.isin(arm)]
    sf = sub[sub.is_fault]
    for m_, dd, tt, p in [("claude-haiku-4.5", "+0.6", "0.30", ".764"),
                          ("claude-haiku-4.5-thinking", "+2.5", "0.94",
                           ".347")]:
        j = pairs(sf[sf.model == m_], "scaffolding", "on", "off",
                  "detection")
        t, pv = stats.ttest_rel(j.a, j.b)
        check(S, f"detection change {m_}", dd, j.d.mean(), SP1)
        check(S, f"detection t {m_}", tt, t, F2)
        check(S, f"detection df {m_}", "161", len(j) - 1, I)
        check(S, f"detection p {m_}", p, pv, P3)
    fa = sub[(sub.fault == "clean") & (sub.target_ok == True)]  # noqa: E712
    for col, pool, lab in [("recovered", sf, "recovery"),
                           ("perseveration", sf, "repetition"),
                           ("detection", fa, "false alarms")]:
        ps = [stats.ttest_rel(*pairs(pool[pool.model == m_], "scaffolding",
                                     "on", "off", col)[["a", "b"]].T.values)
              .pvalue for m_ in arm]
        check(S, f"{lab} unchanged (smallest p)", "> .05", min(ps),
              lambda v: "> .05" if v > .05 else P3(v))
    q = sf[sf.target_ok == True]  # noqa: E712
    check(S, "quiet detection without the line %", "60.6",
          q[q.scaffolding == "off"].detection.mean(), P1)
    check(S, "quiet detection with the line %", "59.6",
          q[q.scaffolding == "on"].detection.mean(), P1)
    sfv = sf.assign(vis=np.where(sf.target_ok == True,  # noqa: E712
                                 "quiet", "loud"))
    t = term(gee(sfv, "detection ~ C(scaffolding) * C(vis)"),
             "C(scaffolding)[T.on]:C(vis)[T.quiet]")
    check(S, "interaction z", "-1.35", t["z"], F2)
    check(S, "interaction p", ".176", t["p"], P3)

    # ------------------------------------------- tables in the paper ---
    if args.tex:
        tex = args.tex.read_text()
        check("paper", "TBD markers left in the paper", "0",
              len(re.findall(r"TBD", tex)), I)
        mine = paper_tables(D, A, F, base, loud, quiet, clean, rq2)
        for label, rows in mine.items():
            printed = tex_rows(tex, label)
            if len(printed) != len(rows):
                check("paper", f"{label} row count", str(len(rows)),
                      len(printed), I)
            for row, prow in zip(rows, printed):
                check("paper", f"{label} | {row[0]} {row[1]}",
                      " & ".join(prow), " & ".join(row), str)
            # the pipeline's table file must equal the paper's table source
            gen = ROOT / "06_figures_tables/outputs/tables" / \
                f"tab_{label.split(':')[1]}.tex"
            if gen.exists():
                check("paper", f"{gen.name} identical to the paper source",
                      "yes", table_source(tex, label)
                      == normalise(gen.read_text()),
                      lambda v: "yes" if v else "no")

    # --------------------------------------------------------- report ---
    R = pd.DataFrame(ROWS)
    bad = R[~R.ok]
    pd.set_option("display.width", 250, "display.max_rows", 1000,
                  "display.max_colwidth", 90)
    print(R.to_string(index=False))
    print(f"\n{len(R) - len(bad)}/{len(R)} checks pass")
    if len(bad):
        print("\nFAILED:\n" + bad.to_string(index=False))
    return 0 if bad.empty else 1


if __name__ == "__main__":
    raise SystemExit(main())
