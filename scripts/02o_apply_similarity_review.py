"""Stage 2o: turn the annotators' pass or fail decisions on the low-similarity queue into data/review/decisions.jsonl.

    python scripts/02o_apply_similarity_review.py

Rule (configs/experiment.yaml annotators.similarity_pass_rule = majority): an item passes when more than half of the annotators say pass;
an unparseable reply counts as fail, so a tie fails. The merge then marks passed items ok_reviewed. The decisions are LLM decisions and
the reviewer field says so; replace them with bilingual human decisions when available (same file format).
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import yaml

REV = Path("data/review")


def main():
    cfg = yaml.safe_load(open("configs/experiment.yaml"))
    from dlsd.reliability import qualified_annotators
    acc, names = qualified_annotators(cfg)
    print("probe accuracy:", {k: round(v, 3) for k, v in acc.items()}, "-> qualified:", names)
    if len(names) < 3:
        print("fewer than 3 qualified annotators: majority decisions are not reliable; add or replace an annotator")
    votes = defaultdict(dict)
    meta = {}
    for n in names:
        p = REV / f"similarity_{n}.jsonl"
        if not p.exists():
            print("missing", p)
            continue
        for d in map(json.loads, open(p, encoding="utf-8")):
            votes[(d["condition_id"], d["item_id"])][n] = d["decision"]
            meta[(d["condition_id"], d["item_id"])] = d["language"]
    out, n_pass = [], 0
    for (cid, iid), v in sorted(votes.items()):
        passes = sum(x == "pass" for x in v.values())
        dec = "pass" if passes > len(names) / 2 else "fail"
        n_pass += dec == "pass"
        out.append({"condition_id": cid, "item_id": iid, "language": meta[(cid, iid)], "decision": dec,
                    "reviewer": "llm-majority:" + ",".join(names), "votes": v})
    with open(REV / "decisions.jsonl", "w") as f:
        for d in out:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"{len(out)} items: {n_pass} pass, {len(out) - n_pass} fail -> {REV / 'decisions.jsonl'}")
    by_lang = defaultdict(lambda: [0, 0])
    for d in out:
        by_lang[d["language"]][d["decision"] == "pass"] += 1
    print({k: {"fail": v[0], "pass": v[1]} for k, v in by_lang.items()})


if __name__ == "__main__":
    main()
