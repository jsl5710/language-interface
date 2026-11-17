"""Stage 2j: few-shot versus zero-shot judge labels on the real renderings (no ground truth).

    python scripts/02j_shot_agreement.py     # table -> results/aggregate/judge_shot_agreement, sample -> data/review/shot_disagreements.csv

Per judge and condition: how often the two settings give the same label, Cohen's kappa between them, and how often each setting
agrees with the intended type. A stratified sample of the disagreements is exported for researchers to label blind.
"""
import csv
import random

import pandas as pd
import yaml

from dlsd.gate_io import JUDGE_DIR, codeswitch_shards, read_rows
from dlsd.io import write_table
from dlsd.switch_judge import cohen_kappa

ZERO_DIR = JUDGE_DIR.parent / "judge_zero"
PER_JUDGE_SAMPLE = 60


def main():
    cfg = yaml.safe_load(open("configs/experiment.yaml"))
    recs, dis = [], []
    for j in cfg["switch_judges"]["panel"]:
        for shard in codeswitch_shards():
            name = shard.name.replace("rendered_", "judge_")
            few = {r["item_id"]: r for r in read_rows(JUDGE_DIR / j["name"] / name) if r["round"] == 1}
            zero = {r["item_id"]: r for r in read_rows(ZERO_DIR / j["name"] / name) if r["round"] == 1}
            ids = sorted(set(few) & set(zero))
            if not ids:
                continue
            f, z = [few[i]["label"] for i in ids], [zero[i]["label"] for i in ids]
            tgt = [few[i]["intended"] for i in ids]
            recs.append({"judge": j["name"], "condition": name[len("judge_"):-len(".jsonl")], "n": len(ids),
                         "agree": sum(a == b for a, b in zip(f, z)) / len(ids), "kappa": cohen_kappa(f, z),
                         "few_matches_intended": sum(a == t for a, t in zip(f, tgt)) / len(ids),
                         "zero_matches_intended": sum(a == t for a, t in zip(z, tgt)) / len(ids)})
            dis += [{"judge": j["name"], "condition": recs[-1]["condition"], "item_id": i, "few_label": few[i]["label"], "zero_label": zero[i]["label"]}
                    for i, a, b in zip(ids, f, z) if a != b]
    df = pd.DataFrame(recs)
    write_table(df, "judge_shot_agreement", {"experiment": "both", "judges": [j["name"] for j in cfg["switch_judges"]["panel"]]})
    print(df.groupby("judge")[["agree", "kappa", "few_matches_intended", "zero_matches_intended"]].mean().round(3).to_string())
    texts = {}
    for shard in codeswitch_shards():
        for r in read_rows(shard):
            if r.get("round", 1) == 1:
                texts[(shard.name[len("rendered_"):-len(".jsonl")], r["item_id"])] = r["rendered"]
    rng, out = random.Random(13), []
    for j in {d["judge"] for d in dis}:
        mine = [d for d in dis if d["judge"] == j]
        by_cond: dict = {}
        for d in mine:
            by_cond.setdefault(d["condition"], []).append(d)
        per = max(1, PER_JUDGE_SAMPLE // len(by_cond))
        for c, v in by_cond.items():
            out += rng.sample(v, min(per, len(v)))
    with open("data/review/shot_disagreements.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["judge", "condition", "item_id", "text", "few_label", "zero_label", "researcher_label (blind: fill before looking at the two labels)", "comment"])
        rng.shuffle(out)
        for d in out:
            w.writerow([d["judge"], d["condition"], d["item_id"], texts[(d["condition"], d["item_id"])], d["few_label"], d["zero_label"], "", ""])
    print(f"{len(dis)} disagreements; {len(out)} sampled -> data/review/shot_disagreements.csv")


if __name__ == "__main__":
    main()
