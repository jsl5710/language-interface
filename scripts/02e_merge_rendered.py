"""Stage 2e: merge baseline, dialect, translation and code-switch shards into data/rendered.jsonl (one row per item and condition).

    python scripts/02e_merge_rendered.py            # writes data/rendered.jsonl and results/aggregate/data_quality_log.{csv,json}

Run after the dialect shards are complete and after the LLM shards have been gated. Safe to run at any time: items or conditions that
are not rendered yet appear as render_status "missing". Fails if a baseline row differs from the original text."""
import json

import pandas as pd
import yaml

from dlsd.conditions import build_conditions
from dlsd.io import write_jsonl, write_table
from dlsd.merge import merge, quality_log

if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/experiment.yaml"))
    items = [json.loads(l) for l in open("data/items.jsonl")]
    rows = merge(items, cfg)
    by_id = {it["item_id"]: it for it in items}
    for r in rows:
        if r["condition_axis"] == "baseline":
            assert r["rendered"] == by_id[r["item_id"]]["text"] and r["options"] == by_id[r["item_id"]]["options"], "baseline must equal the original"
    write_jsonl(rows, "data/rendered.jsonl")
    log = pd.DataFrame(quality_log(rows))
    write_table(log, "data_quality_log", {"experiment": "both", "n_items": len(items), "n_conditions": len(build_conditions(cfg))})
    print(log[["condition_id", "n", "share_ok", "n_failed_checks", "n_failed_gate", "n_error", "n_missing"]].to_string())
    print(f"\n{len(rows)} rows written to data/rendered.jsonl")
