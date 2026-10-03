#!/usr/bin/env python3
"""
=============================================================================
 BRITTLE AGENTS | 04_codebook/judge_agreement.py
=============================================================================
 Agreement between the judge model and one human coder on the hand-coded
 sample: Cohen's kappa and percent agreement for the three-way scheme
 (none / acknowledged / rationalized) and the binary scheme used in every
 analysis (none vs treated as a problem).

 Human codes come from outputs/handcode_sample_FILLED.csv (column
 hand_code); judge codes come from master_trials.csv, merged on trial_id.
 A trial the judge did not code (no agent text after the target step) is
 counted as 'none', which is also how its detection is scored; agreement
 on the coded trials only is reported alongside.

 Usage:
   python 04_codebook/judge_agreement.py

 Saves: outputs/judge_agreement.txt, outputs/judge_agreement.json
=============================================================================
"""

import json
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "outputs"
CODES = ["none", "acknowledged", "rationalized"]


def kappa(a, b):
    a, b = pd.Series(list(a)), pd.Series(list(b))
    po = (a == b).mean()
    cats = sorted(set(a) | set(b))
    pe = sum((a == c).mean() * (b == c).mean() for c in cats)
    return float((po - pe) / (1 - pe)), float(po)


def main():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    hand = pd.read_csv(OUT / "handcode_sample_FILLED.csv")[["trial_id",
                                                            "hand_code"]]
    master = pd.read_csv(ROOT / cfg["paths"]["processing_dir"]
                         / "master_trials.csv")[["trial_id", "judge_code"]]
    d = hand.merge(master, on="trial_id", how="left", validate="one_to_one")
    assert d["hand_code"].isin(CODES).all(), "unexpected hand codes"

    lines, res = [], {}
    uncoded = int(d["judge_code"].isna().sum())
    for label, g in [("all trials (uncoded judge = none)",
                      d.assign(judge_code=d["judge_code"].fillna("none"))),
                     ("judge-coded trials only",
                      d.dropna(subset=["judge_code"]))]:
        k3, a3 = kappa(g["hand_code"], g["judge_code"])
        k2, a2 = kappa(g["hand_code"] != "none", g["judge_code"] != "none")
        res[label] = dict(n=len(g), kappa_3way=k3, agree_3way=a3,
                          kappa_binary=k2, agree_binary=a2)
        lines += [f"{label} (n = {len(g)})",
                  f"  three-way: kappa = {k3:.2f}, agreement = {a3:.1%}",
                  f"  binary:    kappa = {k2:.2f}, agreement = {a2:.1%}",
                  "  confusion (rows = human, columns = judge):",
                  "    " + pd.crosstab(g["hand_code"], g["judge_code"])
                  .to_string().replace("\n", "\n    "), ""]
    lines.insert(0, f"Hand-coded trials: {len(d)}; judge did not code "
                    f"{uncoded} of them.\n")
    text = "\n".join(lines)
    print(text)
    (OUT / "judge_agreement.txt").write_text(text + "\n", encoding="utf-8")
    (OUT / "judge_agreement.json").write_text(json.dumps(res, indent=2),
                                              encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
