#!/usr/bin/env python3
"""
=============================================================================
 BRITTLE AGENTS | 06_figures_tables/make_figures.py
=============================================================================
 The paper's figures. Chart types are chosen per message rather than
 defaulting to grouped bars:

   fig0_method      schematic of the injection layer and the measures
                    (not used in the paper)
   fig1_detection   Fig. 1: detection by visibility, with the response composition
                    and the gap annotated
   fig2_reasoning   Fig. 2: forest plot of paired reasoning-minus-instruct
                    differences with task-clustered intervals; asterisks
                    mark contrasts significant in the task-clustered GEE
                    after Holm correction across families
   fig3_consequences Fig. 3: dumbbell of recovery against the clean-vs-clean
                    baseline, plus repeated calls to the same tool and effort

 Design rules: no in-figure titles, Okabe-Ito colourblind-safe palette,
 task-clustered bootstrap intervals, direct labels, hairline dotted grids,
 Type-42 embedded fonts.

 Usage:  python 06_figures_tables/make_figures.py            # all
         python 06_figures_tables/make_figures.py fig2 fig3  # subset
=============================================================================
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "05_analysis"))

import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from analysis_utils import load_master, cluster_boot_ci, setup_style, grid

OUT = Path(__file__).resolve().parent / "outputs" / "figures"
BLUE, ORANGE, GREEN, PINK = "#0072B2", "#D55E00", "#009E73", "#CC79A7"
GRAY, LIGHT, DARK, RULE = "#8A8F98", "#D8DBE0", "#2B2B2B", "#B9BDC4"


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=600, bbox_inches="tight")
    print(f"  {name}.pdf/.png")


# ===========================================================================
# FIGURE 0 -- method schematic
# ===========================================================================
def fig_method(plt):
    fig, ax = plt.subplots(figsize=(7.1, 2.75))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 46)
    ax.axis("off")

    def box(x, y, w, h, label, sub=None, fc="white", ec=RULE, lw=0.8,
            fs=7.6, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.35,rounding_size=1.2",
                                    fc=fc, ec=ec, lw=lw, zorder=3))
        ax.text(x + w / 2, y + h / 2 + (1.3 if sub else 0), label,
                ha="center", va="center", fontsize=fs, zorder=4,
                fontweight="bold" if bold else "normal", color=DARK)
        if sub:
            ax.text(x + w / 2, y + h / 2 - 2.7, sub, ha="center", va="center",
                    fontsize=6.3, color="#5F646B", zorder=4, linespacing=1.35)

    def arrow(x1, y1, x2, y2, color=DARK, lw=0.9):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     color=color, lw=lw, mutation_scale=8,
                                     zorder=2, shrinkA=1, shrinkB=1))

    # ---- agent loop --------------------------------------------------------
    ax.text(0, 43.2, "AGENT LOOP", fontsize=6.5, color="#5F646B",
            fontweight="bold")
    top, h = 31.5, 9.5
    box(0, top, 15, h, "user turn", "task from\nfrozen suite", fc="#F4F5F7")
    box(19, top, 15, h, "model", "picks a tool\nand arguments")
    box(38, top, 20, h, "injection layer", "one typed fault,\nfired once",
        fc="#FBF0E8", ec=ORANGE, lw=1.2, bold=True)
    box(62, top, 15, h, "environment", "stateful BFCL\nclasses")
    box(81, top, 17, h, "observation", "returned to\nthe model")
    for a, b in ((15, 19), (34, 38), (58, 62), (77, 81)):
        arrow(a, top + h / 2, b, top + h / 2)
    ax.add_patch(FancyArrowPatch((89.5, top), (26.5, top),
                                 connectionstyle="arc3,rad=0.20",
                                 arrowstyle="-|>", color=GRAY, lw=0.8,
                                 mutation_scale=7, zorder=1))
    ax.text(58, 24.4, "repeats until the model answers without a tool call "
                      "(cap: 15 calls)", fontsize=6.3, color="#5F646B",
            ha="center")

    # ---- conditions --------------------------------------------------------
    ax.text(0, 19.8, "CONDITIONS", fontsize=6.5, color="#5F646B",
            fontweight="bold")
    ax.plot([17.5, 65.0], [17.4, 17.4], color=BLUE, lw=1.0)
    ax.text(41.2, 18.4, "loud: an explicit error comes back", fontsize=6.3,
            color=BLUE, ha="center")
    ax.plot([67.0, 85.5], [17.4, 17.4], color=ORANGE, lw=1.0)
    ax.text(76.2, 18.4, "quiet: no error", fontsize=6.3, color=ORANGE,
            ha="center")

    conds = [("clean", "no fault", GRAY, 15.5),
             ("timeout", "error raised", BLUE, 15.5),
             ("missing tool", "tool removed", BLUE, 15.5),
             ("schema drift", "argument rejected", BLUE, 15.5),
             ("silent corruption", "wrong value,\nright shape", ORANGE, 18.5)]
    x = 0
    for name, sub, col, w in conds:
        box(x, 5.0, w, 9.8, name, sub, ec=col, lw=1.0, fs=7.0)
        x += w + 1.6

    # ---- measures ----------------------------------------------------------
    ax.text(88.5, 19.8, "MEASURES", fontsize=6.5, color="#5F646B",
            fontweight="bold", ha="left")
    ax.text(88.5, 15.4, "noticed\nreplanned\nrecovered\nperseverated\n"
                        "extra steps", fontsize=6.6, color=DARK, ha="left",
            va="top", linespacing=1.55)
    return fig


# ===========================================================================
# FIGURE 1 -- detection by visibility
# ===========================================================================
def fig_detection(plt, df):
    base = df[(df["fault"] == "clean") & (df["target_ok"] == True)]  # noqa
    quiet = df[(df["is_fault"]) & (df["target_ok"] == True)]  # noqa
    loud = df[(df["is_fault"]) & (df["target_ok"] == False)]  # noqa
    groups = [("no fault", base, GRAY), ("quiet", quiet, ORANGE),
              ("loud", loud, BLUE)]

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.55),
                             gridspec_kw={"width_ratios": [1.0, 1.15]})

    # -- left: rate with CI, drawn as dots on stems (cleaner than bars) -----
    ax = axes[0]
    for i, (name, g, col) in enumerate(groups):
        m = g["detection"].mean()
        lo, hi = cluster_boot_ci(g, "detection")
        ax.plot([i, i], [0, m], color=LIGHT, lw=6, solid_capstyle="round",
                zorder=2)
        ax.plot([i, i], [lo, hi], color=DARK, lw=1.1, zorder=4)
        ax.plot([i - 0.09, i + 0.09], [lo, lo], color=DARK, lw=1.1, zorder=4)
        ax.plot([i - 0.09, i + 0.09], [hi, hi], color=DARK, lw=1.1, zorder=4)
        ax.plot(i, m, "o", ms=9, color=col, zorder=5,
                markeredgecolor="white", markeredgewidth=1.0)
        ax.text(i, hi + 0.055, f"{100 * m:.0f}%", ha="center", fontsize=9.5,
                fontweight="bold", color=col)
        ax.text(i, 0.028, f"n={len(g)}", ha="center", fontsize=6.3,
                color="#5F646B", zorder=6)
    # gap annotation between quiet and loud
    q = quiet["detection"].mean()
    l = loud["detection"].mean()
    ax.annotate("", xy=(2.34, l), xytext=(2.34, q),
                arrowprops=dict(arrowstyle="<->", color="#5F646B", lw=0.9))
    ax.text(2.42, (q + l) / 2, f"{100 * (l - q):.0f}\npoints", fontsize=6.8,
            color="#5F646B", va="center", linespacing=1.2)
    ax.set_xlim(-0.55, 2.95)
    ax.set_ylim(0, 1.12)
    ax.set_xticks(range(3), [g[0] for g in groups], fontsize=8.5)
    ax.set_ylabel("treated the result as a problem")
    ax.set_yticks([0, .25, .5, .75, 1.0], ["0", "25%", "50%", "75%", "100%"])
    grid(ax)

    # -- right: response composition, horizontal stacked --------------------
    ax = axes[1]
    codes = [("none", "#C9CDD3"), ("acknowledged", BLUE),
             ("rationalized", ORANGE)]
    ypos = [2, 1, 0]
    for y, (name, g, _) in zip(ypos, groups):
        left = 0.0
        for code, col in codes:
            v = g["judge_code"].eq(code).mean()
            ax.barh(y, v, 0.52, left=left, color=col, zorder=3)
            if v > 0.06:
                ax.text(left + v / 2, y, f"{100 * v:.0f}",
                        ha="center", va="center", fontsize=7.4,
                        color="white" if code != "none" else DARK)
            left += v
    ax.set_yticks(ypos, [g[0] for g in groups], fontsize=8.5)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, .25, .5, .75, 1.0], ["0", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("share of trials")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in codes]
    ax.legend(handles, [n for n, _ in codes], frameon=False, fontsize=7.2,
              ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.20),
              handlelength=1.1, columnspacing=1.1, handleheight=0.9)
    ax.grid(axis="x", color="#DDDDDD", lw=0.5, ls=(0, (1, 2)), zorder=0)
    ax.set_axisbelow(True)
    ax.invert_yaxis()
    fig.subplots_adjust(wspace=0.34)
    return fig


# ===========================================================================
# FIGURE 2 -- forest plot of paired differences
# ===========================================================================
def fig_reasoning(plt):
    r2 = pd.read_csv(ROOT / "05_analysis/rq2_reasoning_recovery/outputs/"
                            "rq2_results.csv")
    label = {"detection": "noticed the failure",
             "replanned": "changed its next action",
             "recovered": "reached the clean-run state"}
    cols = {"detection": ORANGE, "replanned": BLUE, "recovered": GRAY}
    order = ["detection", "replanned", "recovered"]
    fams = ["claude", "deepseek", "qwen"]

    fig, ax = plt.subplots(figsize=(7.1, 2.9))
    y, ticks, ticklabels, seps = 0, [], [], []
    for k, out in enumerate(order):
        sub = r2[r2["outcome"] == out].set_index("family")
        for fam in fams:
            if fam not in sub.index:
                continue
            x = sub.loc[fam, "mean_diff"] * 100
            lo = sub.loc[fam, "ci_lo_task"] * 100
            hi = sub.loc[fam, "ci_hi_task"] * 100
            sig = sub.loc[fam, "p_holm_gee"] < .05
            ax.plot([lo, hi], [y, y], color=cols[out], lw=1.3, zorder=3,
                    alpha=1.0 if sig else 0.55)
            ax.plot(x, y, "o", ms=6.5, color=cols[out], zorder=4,
                    markeredgecolor="white", markeredgewidth=0.9,
                    alpha=1.0 if sig else 0.55)
            ax.text(31.5, y, f"{x:+.1f}" + ("*" if sig else ""),
                    va="center", ha="right", fontsize=6.9, color=DARK)
            ticks.append(y)
            ticklabels.append(fam)
            y -= 1
        seps.append(y + 0.5)
        y -= 0.7
    ax.axvline(0, color=DARK, lw=0.9, zorder=2)
    for s in seps[:-1]:
        ax.axhline(s, color="#E6E8EB", lw=0.7, zorder=1)
    ax.set_yticks(ticks, ticklabels, fontsize=7.6)
    ax.set_xlabel("reasoning minus instruct (percentage points, "
                  "95% task-clustered bootstrap CI)")
    ax.set_xlim(-28, 33)
    # outcome label sits above its block, on the left
    idx = 0
    for out in order:
        n = sum(1 for f in fams
                if f in r2[r2["outcome"] == out]["family"].values)
        ax.text(-27.2, ticks[idx] + 0.62, label[out], fontsize=7.3,
                color=cols[out], ha="left", va="center", fontweight="bold")
        idx += n
    ax.text(-27.2, 2.35, "notices less \u2190", fontsize=6.6,
            color="#5F646B", ha="left")
    ax.text(32.5, 2.35, "\u2192 does more", fontsize=6.6, color="#5F646B",
            ha="right")
    ax.grid(axis="x", color="#DDDDDD", lw=0.5, ls=(0, (1, 2)), zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(y + 0.4, 2.9)
    return fig


# ===========================================================================
# FIGURE 3 -- consequences: dumbbell + effort
# ===========================================================================
def fig_consequences(plt, df):
    faults = ["timeout", "schema_drift", "silent_corruption", "missing_tool"]
    nice = {"timeout": "timeout", "schema_drift": "schema drift",
            "silent_corruption": "silent corruption",
            "missing_tool": "missing tool"}
    colmap = {"timeout": BLUE, "missing_tool": PINK, "schema_drift": GREEN,
              "silent_corruption": ORANGE}
    clean = df[df["fault"] == "clean"]["recovered"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.6),
                             gridspec_kw={"width_ratios": [1.25, 1.0]})

    # -- left: dumbbell from the clean baseline down to each fault ---------
    ax = axes[0]
    order = sorted(faults, key=lambda f: df[df["fault"] == f]["recovered"].mean())
    for i, f in enumerate(order):
        g = df[df["fault"] == f]
        m = g["recovered"].mean()
        lo, hi = cluster_boot_ci(g, "recovered")
        ax.plot([m, clean], [i, i], color=LIGHT, lw=2.2, zorder=2,
                solid_capstyle="round")
        ax.plot([lo, hi], [i, i], color=colmap[f], lw=1.2, zorder=3)
        ax.plot(m, i, "o", ms=7.5, color=colmap[f], zorder=4,
                markeredgecolor="white", markeredgewidth=1.0)
        ax.plot(clean, i, "o", ms=5, color=GRAY, zorder=4,
                markeredgecolor="white", markeredgewidth=0.8)
        # label sits left of the interval so it never overlaps the CI
        ax.text(lo - 0.015, i, f"{100 * m:.0f}%", ha="right", va="center",
                fontsize=7.4, color=colmap[f], fontweight="bold")
    ax.set_yticks(range(len(order)), [nice[f] for f in order], fontsize=8)
    ax.set_xlim(0.18, 1.04)
    ax.set_xticks([.4, .6, .8, 1.0], ["40%", "60%", "80%", "100%"])
    ax.set_xlabel("reached the clean-run state")
    ax.text(clean, len(order) - 0.42, f"no fault ({100 * clean:.0f}%)",
            fontsize=6.8, color="#5F646B", ha="center")
    ax.axvline(clean, color=GRAY, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.grid(axis="x", color="#DDDDDD", lw=0.5, ls=(0, (1, 2)), zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(-0.6, len(order) - 0.15)

    # -- right: repeated calls to the same tool, effort cost annotated -----
    # A scatter of these two measures is unreadable because two faults sit at
    # almost the same coordinates, so they are shown on one axis instead.
    ax = axes[1]
    for i, f in enumerate(order):
        g = df[df["fault"] == f]
        pv = g["perseveration"].mean()
        cost = g["recovery_cost"].mean()
        ax.barh(i, pv, 0.5, color=colmap[f], zorder=3)
        ax.text(pv + 0.008, i, f"{100 * pv:.0f}%", va="center",
                fontsize=7.4, color=colmap[f], fontweight="bold")
        ax.text(0.338, i, f"+{cost:.1f}", va="center", ha="right",
                fontsize=7.4, color=DARK)
    ax.set_yticks(range(len(order)), [nice[f] for f in order], fontsize=8)
    ax.set_xlim(0, 0.345)
    ax.set_xticks([0, .1, .2], ["0", "10%", "20%"])
    ax.set_xlabel("repeated calls to the same tool")
    ax.text(0.338, len(order) - 0.42, "extra steps", ha="right",
            fontsize=6.8, color="#5F646B")
    ax.grid(axis="x", color="#DDDDDD", lw=0.5, ls=(0, (1, 2)), zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(-0.6, len(order) - 0.15)
    fig.subplots_adjust(wspace=0.36)
    return fig


def main():
    df, cfg = load_master()
    plt = setup_style()
    figures = {"fig0": ("fig0_method", lambda: fig_method(plt)),
               "fig1": ("fig1_detection", lambda: fig_detection(plt, df)),
               "fig2": ("fig2_reasoning", lambda: fig_reasoning(plt)),
               "fig3": ("fig3_consequences",
                        lambda: fig_consequences(plt, df))}
    for key in (sys.argv[1:] or list(figures)):
        name, make = figures[key]
        save(make(), name)
    print(f"\n Figures -> {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
