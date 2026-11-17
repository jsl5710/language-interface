"""Stage 2k: blind human annotation of a stratified sample of code-switched renderings, to validate the judge panel.

    python scripts/02k_annotation_set.py build      # after judges ran: data/review/annotation_<Language>.csv (blind) + data/review/annotation_key.jsonl (private)
    python scripts/02k_annotation_set.py analyze    # after annotators returned data/review/annotation_<Language>_<annotator>.csv

Sample: PER_CELL items from each of the 18 conditions (language x type x base), half drawn from items on which the judge settings
disagree (few vs zero shot, or judges differ) and half at random, so the human labels test the hard cases and the typical ones.
Annotators see only the text and the language, never a judge label or the intended type. Every annotator labels every item.
Analysis: human-human kappa, human majority label, and agreement of each judge and setting with the human majority, overall and
on the disagreement stratum. Harmful rendered text stays local: these files are not committed.
"""
import csv
import functools
import json
import random
import sys
from pathlib import Path

import pandas as pd
import yaml

from dlsd.gate_io import JUDGE_DIR, codeswitch_shards, read_rows
from dlsd.io import write_table
from dlsd.switch_judge import LABELS, cohen_kappa

ZERO_DIR = JUDGE_DIR.parent / "judge_zero"
REV = Path("data/review")
PER_CELL = 6          # 3 hard + 3 random per condition: 108 items, 36 per language


@functools.lru_cache(maxsize=None)
def _labels(path):
    return {r["item_id"]: r["label"] for r in read_rows(path) if r["round"] == 1}


def labels_for(cfg, cond_name, item_id):
    out = {}
    for j in cfg["switch_judges"]["panel"]:
        for mode, d in (("few", JUDGE_DIR), ("zero", ZERO_DIR)):
            out[f"{j['name']}|{mode}"] = _labels(d / j["name"] / f"judge_{cond_name}.jsonl").get(item_id)
    return out


def build(cfg):
    rng = random.Random(13)
    key, per_lang = [], {}
    for shard in codeswitch_shards():
        cond = shard.name[len("rendered_"):-len(".jsonl")]
        rows = [r for r in read_rows(shard) if r.get("round", 1) == 1 and r["rendered"]]
        hard, easy = [], []
        for r in rows:
            labs = labels_for(cfg, cond, r["item_id"])
            (hard if len({v for v in labs.values() if v}) > 1 else easy).append((r, labs))     # any disagreement among the 6 judge runs
        pick = rng.sample(hard, min(PER_CELL // 2, len(hard)))
        rest = [x for x in hard + easy if x not in pick]
        pick += rng.sample(rest, PER_CELL - len(pick))
        for r, labs in pick:
            key.append({"cell": cond, "item_id": r["item_id"], "language": r["language_pair"], "intended": r["switch_type"], "stratum": "hard" if (r, labs) in hard else "random", "judge_labels": labs})
            per_lang.setdefault(r["language_pair"], []).append({"annotation_id": f"{r['language_pair'][:2].lower()}-{len(per_lang.get(r['language_pair'], [])) + 1:03d}", "text": r["rendered"]})
    rng.shuffle(key)
    ids = {}
    for lang, items in per_lang.items():
        rng.shuffle(items)
        with open(REV / f"annotation_{lang}.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["annotation_id", "text", f"label (one of: {', '.join(LABELS)})", "comment"])
            for it in items:
                w.writerow([it["annotation_id"], it["text"], "", ""])
                ids[it["text"]] = it["annotation_id"]
    with open(REV / "annotation_key.jsonl", "w") as f:
        for k in key:
            f.write(json.dumps(k, ensure_ascii=False) + "\n")
    print({l: len(v) for l, v in per_lang.items()}, "items; key ->", REV / "annotation_key.jsonl")


def analyze(cfg):
    """Annotator reliability and judge validity. Annotators are whoever returned annotation_<Language>_<name>.csv (humans or local LLMs)."""
    from dlsd.reliability import encode, qualified_annotators, summary
    key = [json.loads(l) for l in open(REV / "annotation_key.jsonl")]
    texts = {}
    for shard in codeswitch_shards():
        cond = shard.name[len("rendered_"):-len(".jsonl")]
        for r in read_rows(shard):
            texts[(cond, r["item_id"])] = r["rendered"]
    by_text = {texts[(k["cell"], k["item_id"])]: k for k in key}
    judges = [j["name"] for j in cfg["switch_judges"]["panel"]]
    probe_acc, qualified = qualified_annotators(cfg)
    llm_annotators = bool(probe_acc)
    print("annotator probe accuracy:", {k: round(v, 3) for k, v in probe_acc.items()}, "qualified:", qualified)
    recs, rel, ann_acc = [], [], []
    all_rows, names_all = [], None
    for lang in ("Spanish", "Hindi", "Swahili"):
        sheets = {p.stem.split("_")[-1]: pd.read_csv(p, encoding="utf-8-sig", keep_default_na=False) for p in sorted(REV.glob(f"annotation_{lang}_*.csv"))}
        sheets = {k: v for k, v in sheets.items() if k in qualified or not llm_annotators}       # LLM annotators below the probe bar are left out
        if len(sheets) < 2:
            continue
        ann = {a: dict(zip(df["text"], df[df.columns[2]])) for a, df in sheets.items()}
        names = sorted(ann)
        names_all = names
        texts_l = [t for t in ann[names[0]] if t in by_text]
        mat = [[ann[a].get(t) or None for a in names] for t in texts_l]
        all_rows += mat
        m = encode(mat, LABELS)
        rel.append({"language": lang, "annotators": ",".join(names), **summary(m, len(LABELS), 2000), "missing_labels": int((m < 0).sum())})
        for t, row in zip(texts_l, mat):
            k = by_text[t]
            votes = [v for v in row if v]
            top = max(set(votes), key=votes.count) if votes else None
            ok = top is not None and votes.count(top) > len(row) / 2
            for a, v in zip(names, row):
                ann_acc.append({"language": lang, "annotator": a, "label": v, "intended": k["intended"], "matches_intended": v == k["intended"]})
            if not ok:
                continue
            for jm, lab in k["judge_labels"].items():
                recs.append({"text": t, "language": lang, "stratum": k["stratum"], "judge_setting": jm, "annotator_majority": top, "judge": lab, "match": lab == top,
                             "intended": k["intended"], "majority_matches_intended": top == k["intended"]})
            both = all(k["judge_labels"].get(f"{j}|few") == k["intended"] for j in judges)
            recs.append({"text": t, "language": lang, "stratum": k["stratum"], "judge_setting": "gate (both judges, few-shot) says intended", "annotator_majority": top,
                         "judge": k["intended"] if both else "not intended", "match": both == (top == k["intended"]), "intended": k["intended"], "majority_matches_intended": top == k["intended"]})
    if not rel:
        print("no annotator sheets found")
        return
    m = encode(all_rows, LABELS)
    rel.append({"language": "all", "annotators": ",".join(names_all), **summary(m, len(LABELS), 2000), "missing_labels": int((m < 0).sum())})
    df = pd.DataFrame(recs)
    summ = df.groupby(["judge_setting", "stratum"])["match"].mean().unstack().round(3)
    summ["overall"] = df.groupby("judge_setting")["match"].mean().round(3)
    aa = pd.DataFrame(ann_acc).groupby("annotator")["matches_intended"].mean().round(3).rename("matches_intended").reset_index()
    write_table(pd.DataFrame(rel), "annotator_reliability", {"experiment": "both", "annotators": names_all})
    write_table(summ.reset_index(), "judge_vs_annotators", {"experiment": "both", "annotators": names_all})
    write_table(aa, "annotator_vs_intended", {"experiment": "both"})
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rel)[["language", "n_items", "raters", "percent_agreement", "fleiss_kappa", "krippendorff_alpha", "gwet_ac1", "cohen_mean", "missing_labels"]].round(3).to_string())
    print(summ.to_string(), "\n", aa.to_string())
    print("annotator majority equals intended type:", round(df.drop_duplicates("text").majority_matches_intended.mean(), 3), "on", df.text.nunique(), "items")


if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/experiment.yaml"))
    {"build": build, "analyze": analyze}[sys.argv[1]](cfg)
