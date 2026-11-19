"""Stage 1b: matched benign rewrites of the harmful MedSafetyBench items (E1 benign contrast).

    python scripts/01b_benign_match.py --dry-run     # 5 items, prints nothing from the data to the log beyond counts
    python scripts/01b_benign_match.py               # all harmful items, resumable
    python scripts/01b_benign_match.py --spotcheck   # writes the fixed-seed reading sample (20 pairs) to data/benign_match_spotcheck.csv
    python scripts/01b_benign_match.py --emit-items  # writes data/items_benign.jsonl (passed rewrites as new items, source medsafetybench_benign)

Needs data/items.jsonl and the renderer endpoint (configs/experiment.yaml, rendering.code_switch). Failed items are kept with passed=false,
never dropped silently. Everything under data/ stays local (harmful text). Thresholds are provisional: confirm with the author.
"""
import argparse
import csv
import json
import random
from pathlib import Path

import yaml

from dlsd import benign_match as B

OUT = Path("data/benign_match.jsonl")


def load_done() -> dict:
    return {r["item_id"]: r for r in map(json.loads, open(OUT))} if OUT.exists() else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--spotcheck", action="store_true")
    ap.add_argument("--emit-items", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--items", default="data/items.jsonl", help="items file; point at the server copy, do not copy it into the repository")
    args = ap.parse_args()
    items = [json.loads(l) for l in open(args.items)]
    harmful = [it for it in items if it["source"] == "medsafetybench"]
    done = load_done()
    if args.spotcheck:
        rows = [r for r in done.values() if r["passed"]]
        sample = random.Random(args.seed).sample(sorted(rows, key=lambda r: r["item_id"]), min(20, len(rows)))
        src = {it["item_id"]: it["text"] for it in harmful}
        with open("data/benign_match_spotcheck.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["item_id", "original", "benign", "harm_removed(y/n)", "same_topic(y/n)", "same_format(y/n)", "note"])
            for r in sample:
                w.writerow([r["item_id"], src[r["item_id"]], r["benign"], "", "", "", ""])
        print(f"wrote {len(sample)} pairs to read")
        return
    if args.emit_items:
        src = {it["item_id"]: it for it in harmful}
        n = 0
        with open("data/items_benign.jsonl", "w") as f:
            for iid, r in sorted(done.items()):
                if r["passed"]:
                    f.write(json.dumps({"item_id": iid + "-benign", "source": B.SOURCE, "text": r["benign"], "options": None, "answer": None,
                                        "principle": src[iid]["principle"], "matched_to": iid}) + "\n")
                    n += 1
        print(f"{n} benign items ({len(harmful) - n} harmful items have no passed rewrite)")
        return
    from dlsd.backends import Backend
    from dlsd.checks import Similarity
    cfg = yaml.safe_load(open("configs/experiment.yaml"))
    rc = cfg["rendering"]
    backend = Backend(rc["code_switch"]["model"], rc["code_switch"]["base_url"], extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    sim = Similarity(rc["similarity_model"])
    todo = [it for it in harmful if it["item_id"] not in done][: 5 if args.dry_run else None]
    OUT.parent.mkdir(exist_ok=True)
    with open("data/benign_match_dry.jsonl" if args.dry_run else OUT, "a") as f:
        for i, it in enumerate(todo, 1):
            r = {"item_id": it["item_id"], "renderer": rc["code_switch"]["model"], **B.rewrite(it["text"], backend, sim)}
            f.write(json.dumps(r) + "\n"); f.flush()
            if i % 25 == 0:
                print(f"{i}/{len(todo)}", flush=True)
    rows = list(load_done().values()) if not args.dry_run else []
    if rows:
        print(f"passed {sum(r['passed'] for r in rows)} of {len(rows)}")


if __name__ == "__main__":
    main()
