#!/usr/bin/env python3
"""
=============================================================================
 BRITTLE AGENTS | 06_figures_tables/make_tables.py
=============================================================================
 Writes the paper's data tables from the analysis outputs, each as an
 IEEEtran table (.tex, identical to the paper's source) and as Markdown.

   tab_faults.tex   Table II   conditions, visibility, how often faults fired
   tab_design.tex   Table III  model panel
   tab_rq1.tex      Table IV   detection rates (RQ1)
   tab_rq2.tex      Table V    reasoning vs instruct siblings (RQ2)
   tab_rq3.tex      Table VI   recovery and effort by condition (RQ3)

 Table I (related work) contains no data and is not generated.

 Usage:
   python 06_figures_tables/make_tables.py
=============================================================================
"""

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
AN = ROOT / "05_analysis"
OUT = Path(__file__).resolve().parent / "outputs" / "tables"
FAULTS = ["timeout", "missing_tool", "schema_drift", "silent_corruption"]
HOLM = "p_{\\text{Holm}}"
WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}


# ------------------------------------------------------------ formatting --
def nice(name):
    return name.replace("_", " ")


def pct(v):
    return f"{100 * v:.1f}\\%"


def signed(v, d=1):
    return f"${v:+.{d}f}$" if v >= 0 else f"${v:.{d}f}$"


def p_cell(p):
    if p < .001:
        return "$<$.001"
    return "1.000" if p >= .9995 else f"{p:.3f}".lstrip("0")


def p_math(p, sym="p"):
    return f"{sym} < .001" if p < .001 else f"{sym} = {p:.3f}".replace("0.",
                                                                         ".")


def thousands(n):
    return f"{n:,}".replace(",", "{,}")


def tex_table(label, caption, colspec, tabcolsep, header, blocks, note,
              place="[!t]", star=False):
    """IEEEtran table; blocks are lists of rows separated by \\hline."""
    env = "table*" if star else "table"
    out = [f"\\begin{{{env}}}{place}", "\\centering",
           f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\footnotesize",
           f"\\setlength{{\\tabcolsep}}{{{tabcolsep}}}",
           f"\\begin{{tabular}}{{{colspec}}}", "\\hline",
           " & ".join(header) + " \\\\", "\\hline"]
    for rows in blocks:
        out += [" & ".join(r) + " \\\\" for r in rows]
        out.append("\\hline")
    out += ["\\end{tabular}", f"\\par\\smallskip\\scriptsize {note}",
            f"\\end{{{env}}}"]
    return "\n".join(out) + "\n"


def md_table(header, blocks):
    clean = lambda c: (c.replace("\\%", "%").replace("$<$", "<")  # noqa: E731
                       .replace("$", "").replace("\\Delta", "Δ")
                       .replace("p_{\\text{Holm}}", "p_Holm"))
    rows = [r for b in blocks for r in b]
    out = ["| " + " | ".join(clean(h) for h in header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(clean(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def save(name, tex, md):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.tex").write_text(tex, encoding="utf-8")
    (OUT / f"{name}.md").write_text(md, encoding="utf-8")
    print(f"  {name}.tex / .md")


# ------------------------------------------------------------------ main --
def main():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    df = pd.read_csv(ROOT / cfg["paths"]["processing_dir"]
                     / "master_trials.csv")

    # Table II: conditions and firing rates
    vis = {f: ("quiet" if f == "silent_corruption" else "loud")
           for f in FAULTS}
    n_clean = int((df["fault"] == "clean").sum())
    rows = [["clean", "baseline", str(n_clean), "--", str(n_clean)]]
    for f in FAULTS:
        g = df[df["fault"] == f]
        rows.append([nice(f), vis[f], str(len(g)),
                     f"{100 * g['fault_fired'].mean():.0f}\\%",
                     str(int(g["fault_fired"].sum()))])
    header = ["Condition", "Visibility", "Trials", "Fired", "Analysed"]
    save("tab_faults",
         tex_table("tab:faults", "Conditions, visibility, and how often each "
                   "fault fired.", "llccc", "4pt", header, [rows],
                   "Loud faults surface an explicit error; the quiet fault "
                   "returns a well-formed but incorrect value. Missing tool "
                   "and schema drift fire only when the agent calls the "
                   "affected tool."),
         md_table(header, [rows]))

    # Table III: model panel
    arm = set(cfg["agent"]["scaffolding_models"])
    rows = [[m["name"], m["pair"],
             "reasoning" if m.get("reasoning") else "instruct",
             str(int((df["model"] == m["name"]).sum())),
             "yes" if m["name"] in arm else "--"] for m in cfg["models"]]
    header = ["Model", "Family", "Type", "Trials", "Prompt arm"]
    save("tab_design",
         tex_table("tab:design", "Model panel. Each family contributes a "
                   "matched reasoning and instruct pair.", "lllcc", "4pt",
                   header, [rows],
                   f"{thousands(len(df))} trials over "
                   f"{df['task_id'].nunique()} tasks, "
                   f"{WORDS[len(cfg['faults'])]} conditions, "
                   f"{WORDS[cfg['agent']['n_seeds']]} seeds. The secondary "
                   "scaffolding analysis doubles one pair.", place="[t]"),
         md_table(header, [rows]))

    # Table IV: detection (RQ1)
    r1 = pd.read_csv(AN / "rq1_detection/outputs/rq1_results.csv")
    vis_rows = r1[r1["analysis"] == "visibility"].set_index("condition")
    ft_rows = r1[r1["analysis"] == "fault_type"].set_index("condition")

    def det_row(name, x):
        return [name, str(int(x["n"])), pct(x["rate"]),
                f"[{100 * x['ci_lo']:.1f}, {100 * x['ci_hi']:.1f}]"]
    top = [det_row("no fault (baseline)", vis_rows.loc["baseline (no fault)"]),
           det_row("quiet failure", vis_rows.loc["quiet fault"]),
           det_row("loud failure", vis_rows.loc["loud fault"])]
    bottom = [det_row(nice(f), ft_rows.loc[f]) for f in FAULTS]
    worst = ft_rows["p_holm"].max()
    header = ["Condition", "$n$", "Detected", "95\\% CI"]
    save("tab_rq1",
         tex_table("tab:rq1", "Detection rates with task-clustered bootstrap "
                   "intervals.", "lccc", "4pt", header, [top, bottom],
                   f"All fault conditions differ from the baseline at "
                   f"${p_math(worst)}$ after Holm correction."),
         md_table(header, [top, bottom]))

    # Table V: reasoning vs instruct (RQ2)
    r2 = pd.read_csv(AN / "rq2_reasoning_recovery/outputs/rq2_results.csv")
    j2 = json.loads((AN / "rq2_reasoning_recovery/outputs/rq2_results.json")
                    .read_text())
    label = {"detection": "detection", "replanned": "replanning",
             "recovered": "recovery"}
    rows = []
    for out in ["detection", "replanned", "recovered"]:
        for _, x in r2[r2["outcome"] == out].iterrows():
            rows.append([label[out], x["family"], pct(x["mean_reasoning"]),
                         pct(x["mean_instruct"]),
                         signed(100 * x["mean_diff"]),
                         f"$[{100 * x['ci_lo_task']:+.1f}, "
                         f"{100 * x['ci_hi_task']:+.1f}]$",
                         f"{x['or_gee']:.2f}", p_cell(x["p_holm_gee"])])
    pooled = ", ".join(
        f"{label[k]} {signed(100 * j2[k]['pooled']['mean_diff'])} points "
        f"(${p_math(j2[k]['gee']['C(reasoning)[T.True]']['p'])}$)"
        for k in ["detection", "replanned", "recovered"])
    header = ["Outcome", "Family", "Reasoning", "Instruct",
              "$\\Delta$ (points)", "Clustered 95\\% CI", "OR",
              "$p_{\\text{Holm}}$"]
    save("tab_rq2",
         tex_table("tab:rq2", "Reasoning versus instruct siblings on fault "
                   "trials, paired within family by task, fault, and seed.",
                   "llcccccc", "5pt", header, [rows],
                   "Negative $\\Delta$ means the reasoning variant scored "
                   "lower. OR and $p_{\\text{Holm}}$ from task-clustered "
                   "models on the matched pairs, Holm-corrected across "
                   f"families. Pooled over {j2['detection']['pooled']['n']} "
                   f"pairs: {pooled}.",
                   star=True),
         md_table(header, [rows]))

    # Table VI: recovery and effort (RQ3)
    r3 = pd.read_csv(AN / "rq3_fault_costs/outputs/rq3_results.csv")
    j3 = json.loads((AN / "rq3_fault_costs/outputs/rq3_results.json")
                    .read_text())
    eff = r3[r3["outcome"] == "effort"].set_index("fault")
    rows = []
    for f in ["clean"] + FAULTS:
        e = eff.loc[f]
        rep = (lambda v: "--" if f == "clean" else pct(v))  # noqa: E731
        rows.append([nice(f), pct(e["recovered"]), rep(e["same_tool_repeat"]),
                     rep(e["identical_repeat"]), pct(e["budget_exhausted"]),
                     signed(e["recovery_cost"], 2)])
    sig = {f: v["gee"]["p_holm"] for f, v in j3["recovery"].items()
           if v["gee"]["p_holm"] < .05}
    if len(sig) == 1:
        (f, p), = sig.items()
        which = (f"Only {nice(f)} differs from clean in recovery "
                 f"(task-clustered GEE, "
                 f"${p_math(p, sym=HOLM)}$).")
    else:
        which = ("Faults differing from clean in recovery (task-clustered "
                 "GEE, Holm): " + (", ".join(nice(f) for f in sig) or "none")
                 + ".")
    header = ["Condition", "Recovered", "Same tool", "Identical", "Cap hit",
              "Extra steps"]
    save("tab_rq3",
         tex_table("tab:rq3", "Recovery and effort by condition.", "lccccc",
                   "3pt", header, [rows],
                   "Clean recovery is the agreement of two independent "
                   "fault-free runs. ``Same tool'' and ``Identical'' are "
                   "three or more consecutive calls after the fault, to the "
                   "same tool or with the same tool and arguments; neither "
                   "is defined for clean trials. " + which),
         md_table(header, [rows]))

    print(f"\n Tables -> {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
