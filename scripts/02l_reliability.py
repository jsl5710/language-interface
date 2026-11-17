"""Stage 2l: reliability statistics for the validation of the judge materials and of the judge panel.

    python scripts/02l_reliability.py     # results/aggregate/{reviewer_reliability,judge_reliability}.csv/json and paper/tables/*.tex

A. Reviewer reliability: LLM reviewers rate each of the 132 judge examples and probe texts on label correctness and naturalness (y/n).
B. Judge reliability on the real renderings: gpt-oss-120b versus GLM-4.5-Air (few-shot and zero-shot), each setting against the other,
   each judge against the intended type, and parse failure rates.
Reported with percent agreement, Cohen's and Fleiss' kappa, Krippendorff's alpha and Gwet's AC1 (kappa is unstable when one label
dominates), with 95% bootstrap intervals over items (seed 13). A reviewer file named in EXCLUDE is left out of every statistic.
"""
import csv
import glob
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from dlsd.gate_io import JUDGE_DIR, codeswitch_shards, read_rows
from dlsd.io import write_table
from dlsd.reliability import encode, summary
from dlsd.switch_judge import LABELS

ZERO_DIR = JUDGE_DIR.parent / "judge_zero"
REVIEW_DIR = Path("data/review")
EXCLUDE = {"completed_sonnet-5.5"}        # same model family as the assistant that drafted the examples: not independent
REPS = 2000


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def reviewer_tables():
    base = {r["id"]: r for r in csv.DictReader(open(REVIEW_DIR / "judge_examples_review.csv", encoding="utf-8-sig"))}
    ratings = {}
    for f in sorted(glob.glob(str(REVIEW_DIR / "judge_examples_review_*.csv"))):
        name = os.path.basename(f)[len("judge_examples_review_"):-4]
        if name in EXCLUDE:
            continue
        rows = list(csv.DictReader(open(f, encoding="utf-8-sig")))
        cols = list(rows[0])
        lc = next(c for c in cols if c.startswith("label_correct")); rn = next(c for c in cols if c.startswith("reads_natural"))
        ratings[name] = {r["id"]: (r[lc].strip().lower(), r[rn].strip().lower()) for r in rows}
    names = sorted(ratings)
    recs = []
    for q, qi in (("label_correct", 0), ("reads_natural", 1)):
        scopes = {"all": list(base)}
        for lang in ("Spanish", "Hindi", "Swahili"):
            scopes[lang] = [i for i, r in base.items() if r["language"] == lang]
        scopes["few-shot examples"] = [i for i, r in base.items() if r["set"] == "few-shot"]
        scopes["probe texts"] = [i for i, r in base.items() if r["set"] == "probe"]
        for scope, ids in scopes.items():
            m = encode([[ratings[n][i][qi] for n in names] for i in ids], ["y", "n"])
            s = summary(m, 2, REPS)
            n_no = [(m[k] == 1).sum() for k in range(len(m))]
            recs.append({"question": q, "scope": scope, **s, "share_y": float((m == 0).mean()),
                         "items_flagged_by_1plus": int(sum(x >= 1 for x in n_no)), "items_flagged_by_2plus": int(sum(x >= 2 for x in n_no))})
    return pd.DataFrame(recs), names


def judge_tables(cfg):
    judges = [j["name"] for j in cfg["switch_judges"]["panel"]]
    lab = {}
    for mode, d in (("few", JUDGE_DIR), ("zero", ZERO_DIR)):
        for j in judges:
            for shard in codeswitch_shards():
                cond = shard.name[len("rendered_"):-len(".jsonl")]
                for r in read_rows(d / j / f"judge_{cond}.jsonl"):
                    if r["round"] == 1:
                        lab[(mode, j, cond, r["item_id"])] = (r["label"], r["intended"])
    conds = sorted({k[2] for k in lab})
    def cond_meta(c):
        base = "native" if c.startswith("codeswitch_native") else "English"
        t = next(x for x in ("intra", "inter", "extra") if f"_{x}_" in c)
        return base, t, c.rsplit("_", 1)[1].capitalize()
    recs = []
    def add(kind, scope, pairs, a_name, b_name):
        """pairs: list of (labelA, labelB). Agreement statistics for two raters, unparseable replies dropped."""
        ok = [(a, b) for a, b in pairs if a is not None and b is not None]
        m = encode([[a, b] for a, b in ok], LABELS)
        s = summary(m, len(LABELS), REPS) if len(ok) > 1 else {}
        recs.append({"comparison": kind, "scope": scope, "rater_a": a_name, "rater_b": b_name, "n": len(ok),
                     "unparseable_rate": 1 - len(ok) / max(len(pairs), 1), **s})
    for mode in ("few", "zero"):
        pairs_by = defaultdict(list)
        for c in conds:
            base, t, lang = cond_meta(c)
            ids = [k[3] for k in lab if k[0] == mode and k[1] == judges[0] and k[2] == c]
            for i in ids:
                p = (lab[(mode, judges[0], c, i)][0], lab[(mode, judges[1], c, i)][0])
                for sc in ("all", f"{base} {t}", f"{base} base", f"{lang}"):
                    pairs_by[sc].append(p)
        for sc, pairs in pairs_by.items():
            add(f"judge vs judge ({mode}-shot)", sc, pairs, judges[0], judges[1])
    for j in judges:
        pairs_by = defaultdict(list)
        for c in conds:
            base, t, lang = cond_meta(c)
            for i in [k[3] for k in lab if k[0] == "few" and k[1] == j and k[2] == c]:
                p = (lab[("few", j, c, i)][0], lab[("zero", j, c, i)][0])
                for sc in ("all", f"{base} {t}"):
                    pairs_by[sc].append(p)
        for sc, pairs in pairs_by.items():
            add(f"few-shot vs zero-shot ({j})", sc, pairs, "few", "zero")
    acc = []
    for mode in ("few", "zero"):
        for j in judges:
            for sc in ("all", "English intra", "English inter", "English extra", "native intra", "native inter", "native extra"):
                hit = tot = unp = 0
                for (md, jj, c, i), (l, intended) in lab.items():
                    if md != mode or jj != j:
                        continue
                    base, t, lang = cond_meta(c)
                    if sc != "all" and sc != f"{base} {t}":
                        continue
                    tot += 1; unp += l is None; hit += l == intended
                lo, hi = wilson(hit, tot)
                acc.append({"judge": j, "mode": mode, "scope": sc, "n": tot, "matches_intended": hit / tot, "lo": lo, "hi": hi, "unparseable_rate": unp / tot})
    return pd.DataFrame(recs), pd.DataFrame(acc)


def f2(x):
    return "--" if x is None or x != x else f"{x:.2f}"


def latex(rev, jj, acc):
    L = ["% generated by scripts/02l_reliability.py; do not edit by hand",
         "\\begin{tabular}{@{}llrrrrr@{}}", "\\hline", "Question & Scope & Agree & Fleiss $\\kappa$ & Alpha & AC1 [95\\% CI] & Flagged \\\\", "\\hline"]
    for _, r in rev[rev.scope.isin(["all", "Spanish", "Hindi", "Swahili"])].iterrows():
        q = "Label correct" if r.question == "label_correct" else "Reads natural"
        L.append(f"{q} & {r.scope} & {f2(r.percent_agreement)} & {f2(r.fleiss_kappa)} & {f2(r.krippendorff_alpha)} & "
                 f"{f2(r.gwet_ac1)} [{f2(r.gwet_ac1_lo)}, {f2(r.gwet_ac1_hi)}] & {int(r.items_flagged_by_1plus)} \\\\")
    L += ["\\hline", "\\end{tabular}", ""]
    open("paper/tables/reliability_reviewers.tex", "w").write("\n".join(L))
    M = ["% generated by scripts/02l_reliability.py; do not edit by hand",
         "\\begin{tabular}{@{}llrrrrr@{}}", "\\hline", "Setting & Scope & $n$ & Agree & Cohen $\\kappa$ & AC1 & Matches intended (A / B) \\\\", "\\hline"]
    for mode in ("few", "zero"):
        for _, r in jj[(jj.comparison == f"judge vs judge ({mode}-shot)") & jj.scope.isin(["all", "English intra", "English inter", "English extra", "native intra", "native inter", "native extra"])].iterrows():
            a = {j: acc[(acc.judge == j) & (acc["mode"] == mode) & (acc.scope == r.scope)].iloc[0].matches_intended for j in (r.rater_a, r.rater_b)}
            M.append(f"{mode}-shot & {r.scope} & {int(r.n)} & {f2(r.percent_agreement)} & {f2(r.cohen_mean)} & {f2(r.gwet_ac1)} & {f2(a[r.rater_a])} / {f2(a[r.rater_b])} \\\\")
        M.append("\\hline")
    M += ["\\end{tabular}", ""]
    open("paper/tables/reliability_judges.tex", "w").write("\n".join(M))


def main():
    cfg = yaml.safe_load(open("configs/experiment.yaml"))
    rev, names = reviewer_tables()
    jj, acc = judge_tables(cfg)
    write_table(rev, "reviewer_reliability", {"experiment": "both", "reviewers": names, "excluded": sorted(EXCLUDE), "bootstrap_reps": REPS, "seed": 13})
    write_table(pd.concat([jj.assign(table="agreement"), acc.assign(table="accuracy")], ignore_index=True), "judge_reliability",
                {"experiment": "both", "judges": [j["name"] for j in cfg["switch_judges"]["panel"]], "bootstrap_reps": REPS, "seed": 13})
    latex(rev, jj, acc)
    pd.set_option("display.width", 250)
    print(rev[["question", "scope", "percent_agreement", "fleiss_kappa", "krippendorff_alpha", "gwet_ac1", "cohen_mean", "share_y", "items_flagged_by_1plus", "items_flagged_by_2plus"]].round(3).to_string())
    print(jj[jj.scope == "all"][["comparison", "n", "unparseable_rate", "percent_agreement", "cohen_mean", "krippendorff_alpha", "gwet_ac1"]].round(3).to_string())
    print(acc[acc.scope == "all"].round(3).to_string())


if __name__ == "__main__":
    main()
