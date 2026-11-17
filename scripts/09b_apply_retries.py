"""Replace empty Experiment 1 replies by their retried replies (results/raw/retry_exp1*.jsonl, from 03e_retry_empty.py).
For each retried (item, condition, model): the old generation row, scored row and classifier output flags are moved to results/raw/archive/
(local only) and the retried generation row takes their place, so 04_score.py and 05_classify.py redo exactly those rows (both are resumable by key).
    python scripts/09b_apply_retries.py [--dry-run]"""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument("--dry-run", action="store_true"); a = ap.parse_args()
raw, scored, arch = Path("results/raw"), Path("results/scored"), Path("results/raw/archive")
retries = {}
for f in sorted(raw.glob("retry_exp1*.jsonl")):
    for l in open(f):
        r = json.loads(l)
        retries[(r["item_id"], r["condition_id"], r["model_id"])] = r
print(f"{len(retries)} retried rows")
if not retries:
    raise SystemExit

def rewrite(path, keyf, replace=None):
    kept, removed = [], []
    for l in open(path):
        r = json.loads(l)
        (removed if keyf(r) in retries else kept).append(r)
    if replace:
        kept += [retries[k] for k in sorted({keyf(r) for r in removed})]
    print(f"{path}: {len(removed)} rows replaced or removed")
    if not a.dry_run:
        arch.mkdir(exist_ok=True)
        with open(arch / f"{path.stem}_replaced_by_retry.jsonl", "a") as f:
            f.writelines(json.dumps(r) + "\n" for r in removed)
        path.write_text("".join(json.dumps(r) + "\n" for r in kept))

rewrite(raw / "generations_exp1.jsonl", lambda r: (r["item_id"], r["condition_id"], r["model_id"]), replace=True)
rewrite(scored / "scored_exp1.jsonl", lambda r: (r["item_id"], r["condition_id"], r["model_id"]))
for f in sorted(scored.glob("flags_*.jsonl")):
    if "gatefailed" not in f.name:
        rewrite(f, lambda r: (r["item_id"], r["condition_id"], r["model_id"]) if r["scope"] == "output" else None)
