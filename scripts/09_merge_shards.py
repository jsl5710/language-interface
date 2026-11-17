"""Concatenate per-job generation shards (results/raw/generations_<exp>[_gatefailed].<shard>.jsonl) into the main file.
Run only when no generation job is running. Drops duplicate (item, condition, model) rows and truncated lines; shards are then moved to results/raw/shards_merged/.
    python scripts/09_merge_shards.py [--dry-run]"""
import argparse, json, shutil
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument("--dry-run", action="store_true"); a = ap.parse_args()
raw = Path("results/raw")
for stem in ["determinism_exp1", "generations_exp1", "generations_exp1_gatefailed", "generations_exp2", "generations_exp2_gatefailed"]:
    shards = sorted(raw.glob(f"{stem}.*.jsonl"))
    if not shards:
        continue
    main = raw / f"{stem}.jsonl"
    seen, rows, bad = set(), [], 0
    for f in ([main] if main.exists() else []) + shards:
        for l in open(f):
            try:
                r = json.loads(l)
            except ValueError:
                bad += 1
                continue
            k = (r["item_id"], r["condition_id"], r["model_id"])
            if k not in seen:
                seen.add(k); rows.append(l if l.endswith("\n") else l + "\n")
    print(f"{stem}: {len(shards)} shards, {len(rows)} unique rows, {bad} truncated lines dropped")
    if not a.dry_run:
        main.write_text("".join(rows))
        (raw / "shards_merged").mkdir(exist_ok=True)
        for f in shards:
            shutil.move(str(f), raw / "shards_merged" / f.name)
