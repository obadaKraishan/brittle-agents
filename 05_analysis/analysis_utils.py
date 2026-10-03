#!/usr/bin/env python3
"""
=============================================================================
 BRITTLE AGENTS | 05_analysis/analysis_utils.py
=============================================================================
 Shared helpers for the analysis scripts and the figures: loading,
 binomial GEEs clustered on task, task-clustered bootstrap CIs, matched-pair
 contrasts, effect sizes, report formatting, and figure styling.

 Failure detection is the binary 'detection' (noticed vs not). The three-way
 judge_code is descriptive only; agreement of both with hand-coding is in
 04_codebook/outputs/judge_agreement.txt.
=============================================================================
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RNG_SEED = 42
N_BOOT = 10000


# ----------------------------------------------------------------- data ----
def load_master(analysable_only=True):
    """Load master_trials.csv.

    analysable_only drops fault trials where the fault never fired: for
    victim-based faults the agent may never call the victim tool, and those
    trials are effectively clean runs wearing a fault label.
    """
    import yaml
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    path = ROOT / cfg["paths"]["processing_dir"] / "master_trials.csv"
    if not path.exists():
        raise SystemExit(f"[ERROR] {path} not found -- run the "
                         "03_data_processing pipeline first.")
    df = pd.read_csv(path)
    df["is_fault"] = df["fault"] != "clean"
    if analysable_only:
        keep = (~df["is_fault"]) | (df["fault_fired"])
        df = df[keep].copy()
    for c in ["detection", "replanned", "perseveration", "repeat_identical",
              "recovered",
              "budget_exhausted", "false_alarm", "rationalized"]:
        if c in df:
            df[c] = df[c].astype("float")   # allows NaN + mean()
    return df, cfg


# ----------------------------------------------------------- statistics ----
def boot_ci(x, n_boot=N_BOOT, seed=RNG_SEED, ci=95):
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    b = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    lo, hi = np.percentile(b, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return float(lo), float(hi)


def cluster_boot_ci(df, col, cluster="task_id", n_boot=2000, seed=RNG_SEED):
    """Bootstrap CI resampling clusters (tasks), not rows."""
    groups = [g[col].dropna().values for _, g in df.groupby(cluster)]
    groups = [g for g in groups if len(g)]
    if not groups:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    k = len(groups)
    out = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, k, k)
        out[i] = np.concatenate([groups[j] for j in idx]).mean()
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def prop_test(x1, n1, x2, n2):
    """Two-proportion z-test with Cohen's h."""
    if min(n1, n2) == 0:
        return dict(z=np.nan, p=np.nan, h=np.nan, diff=np.nan)
    p1, p2 = x1 / n1, x2 / n2
    p = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se if se > 0 else np.nan
    pval = 2 * (1 - stats.norm.cdf(abs(z))) if z == z else np.nan
    h = 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))
    return dict(z=float(z), p=float(pval), h=float(h), diff=float(p1 - p2),
                p1=float(p1), p2=float(p2), n1=int(n1), n2=int(n2))


def gee_logit(df, outcome, formula_rhs, group="task_id"):
    """Binomial GEE, outcome ~ rhs, exchangeable correlation within task.

    Returns the fitted model, or None if the outcome does not vary or the
    fit fails. Standard errors are cluster-robust.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    d = df.dropna(subset=[outcome]).copy()
    d[outcome] = d[outcome].astype(float)
    if d[outcome].nunique() < 2 or len(d) < 20:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            m = smf.gee(f"{outcome} ~ {formula_rhs}", groups=group, data=d,
                        family=sm.families.Binomial(),
                        cov_struct=sm.cov_struct.Exchangeable()).fit()
            return m
        except Exception:
            return None


def model_table(m):
    """Rows of (term, beta, se, z, p, odds ratio) from a fitted GEE."""
    if m is None:
        return []
    rows = []
    for term in m.params.index:
        b = m.params[term]
        se = m.bse[term]
        z = b / se if se else np.nan
        p = m.pvalues[term]
        rows.append((term, float(b), float(se), float(z), float(p),
                     float(np.exp(b)) if abs(b) < 100 else float('nan')))
    return rows


def paired_by(df, key_cols, split_col, split_a, split_b, value):
    """Pair rows on key_cols and return aligned (a, b) value arrays."""
    a = (df[df[split_col] == split_a].groupby(key_cols)[value].mean())
    b = (df[df[split_col] == split_b].groupby(key_cols)[value].mean())
    j = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
    return j["a"].values, j["b"].values, len(j)


def paired_frame(df, split_col, split_a, split_b, value,
                 keys=("task_id", "fault", "seed")):
    """Matched pairs as rows: key columns, a, b and d = a - b."""
    keys = list(keys)
    a = df[df[split_col] == split_a].groupby(keys)[value].mean()
    b = df[df[split_col] == split_b].groupby(keys)[value].mean()
    j = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna().reset_index()
    j["d"] = j["a"] - j["b"]
    return j


def paired_gee(df, split_col, split_a, split_b, value,
               keys=("task_id", "fault", "seed")):
    """Task-clustered binomial GEE for a matched contrast.

    Fits value ~ contrast on the trials that form matched pairs, where
    contrast = 1 for split_a and 0 for split_b. Matching on (task, fault,
    seed) balances fault type across the two arms, so no covariate is
    needed. Returns OR, z and p for the contrast, or NaNs if not estimable.
    """
    pairs = paired_frame(df, split_col, split_a, split_b, value, keys)
    idx = pd.MultiIndex.from_frame(pairs[list(keys)])
    d = df[df[split_col].isin([split_a, split_b])]
    d = d[pd.MultiIndex.from_frame(d[list(keys)]).isin(idx)].copy()
    d["contrast"] = (d[split_col] == split_a).astype(float)
    m = gee_logit(d, value, "contrast")
    if m is None or not np.isfinite(m.bse.get("contrast", np.nan)):
        return dict(OR=np.nan, z=np.nan, p=np.nan, n=len(d))
    b, se = m.params["contrast"], m.bse["contrast"]
    return dict(OR=float(np.exp(b)), z=float(b / se),
                p=float(m.pvalues["contrast"]), n=int(m.nobs))


def gee_terms(m):
    """{term: {OR, z, p}} from a fitted GEE, for JSON output."""
    return {term: dict(OR=orr, z=z, p=p)
            for term, b, se, z, p, orr in model_table(m)}


def paired_tests(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    n = len(d)
    if n < 3 or np.allclose(d, 0):
        return dict(n=n, t=np.nan, df=max(n - 1, 0), p=np.nan, dz=np.nan,
                    mean_diff=float(np.mean(d)) if n else np.nan,
                    ci=(np.nan, np.nan), W=np.nan, p_w=np.nan)
    t, p = stats.ttest_rel(a, b)
    try:
        W, pw = stats.wilcoxon(a, b)
    except ValueError:
        W, pw = np.nan, np.nan
    sd = d.std(ddof=1)
    return dict(n=n, t=float(t), df=n - 1, p=float(p),
                dz=float(d.mean() / sd) if sd > 0 else np.nan,
                mean_diff=float(d.mean()), ci=boot_ci(d),
                W=float(W) if W == W else np.nan,
                p_w=float(pw) if pw == pw else np.nan)


def holm(pvals):
    p = np.asarray(pvals, float)
    m = np.sum(~np.isnan(p))
    order = np.argsort(np.where(np.isnan(p), np.inf, p))
    adj = np.full_like(p, np.nan)
    prev = 0.0
    for rank, idx in enumerate(order):
        if np.isnan(p[idx]):
            continue
        val = min((m - rank) * p[idx], 1.0)
        prev = max(prev, val)
        adj[idx] = prev
    return adj


# ----------------------------------------------------------- formatting ----
def fmt_p(p):
    if p is None or p != p:
        return "p = n/a"
    return "p < .001" if p < .001 else f"p = {p:.3f}".replace("0.", ".")


def f2(v, d=2):
    return "n/a" if v is None or v != v else f"{v:.{d}f}"


def pct(v, d=0):
    return "n/a" if v is None or v != v else f"{100 * v:.{d}f}%"


def table(rows, headers, widths=None):
    widths = widths or [max(len(str(h)),
                            max((len(str(r[i])) for r in rows), default=0)) + 2
                        for i, h in enumerate(headers)]
    out = ["".join(str(h).ljust(w) for h, w in zip(headers, widths)),
           "".join("-" * w for w in widths)]
    for r in rows:
        out.append("".join(str(c).ljust(w) for c, w in zip(r, widths)))
    return "\n".join(out)


def apa_prop(label, r):
    return (f"{label}: {pct(r['p1'], 1)} vs {pct(r['p2'], 1)}, "
            f"difference = {pct(r['diff'], 1)} points, "
            f"z = {f2(r['z'])}, {fmt_p(r['p'])}, Cohen's h = {f2(r['h'])} "
            f"(n = {r['n1']} and {r['n2']}).")


def apa_paired(label, r):
    return (f"{label}: mean difference = {f2(r['mean_diff'], 3)} "
            f"(95% bootstrap CI [{f2(r['ci'][0], 3)}, {f2(r['ci'][1], 3)}]), "
            f"t({r['df']}) = {f2(r['t'])}, {fmt_p(r['p'])}, "
            f"Cohen's dz = {f2(r['dz'])}; Wilcoxon W = {f2(r['W'], 1)}, "
            f"{fmt_p(r['p_w'])} (n = {r['n']} pairs).")


def apa_model(m, label="model"):
    if m is None:
        return [f"{label}: model did not converge."]
    out = [f"{label}: binomial GEE with exchangeable working correlation "
           f"clustered on task (n = {int(m.nobs)})."]
    for term, b, se, z, p, orr in model_table(m):
        out.append(f"    {term}: b = {f2(b, 3)} (SE {f2(se, 3)}), "
                   f"OR = {f2(orr)}, z = {f2(z)}, {fmt_p(p)}")
    return out


# -------------------------------------------------------------- figures ----
def setup_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Prefer Liberation Sans (Linux) but fall back cleanly on macOS/Windows
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    fam = next((f for f in ("Liberation Sans", "Helvetica Neue", "Helvetica",
                            "Arial", "DejaVu Sans") if f in have),
               "DejaVu Sans")
    plt.rcParams.update({
        "font.family": [fam, "DejaVu Sans"],   # per-glyph fallback (arrows)
        "font.size": 8.5,
        "axes.labelsize": 9, "axes.linewidth": 0.7,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    return plt


def grid(ax, axis="y"):
    ax.grid(axis=axis, color="#DDDDDD", lw=0.5, ls=(0, (1, 2)), zorder=0)
    ax.set_axisbelow(True)
