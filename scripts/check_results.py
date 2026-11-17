"""Run the section 17 sanity checks on whatever exists and write results/aggregate/sanity_checks.json (and .csv).

    python scripts/check_results.py            # exit code 1 if any check fails; "skipped" means the stage has not run yet
Checks never pass by default: a missing input is reported as skipped."""
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

from dlsd.conditions import build_conditions
from dlsd.gate_io import read_rows
from dlsd.io import write_table
from dlsd.sanity import run_all


def table(name):
    p = Path(f"results/aggregate/{name}.json")
    return json.loads(p.read_text())["records"] if p.exists() else []


if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/experiment.yaml"))
    items = read_rows("data/items.jsonl")
    ctx = {"items": items, "rendered": read_rows("data/rendered.jsonl"), "expected_cids": [c.cid for c in build_conditions(cfg)],
           "quality_log": table("data_quality_log"), "gens1": read_rows("results/raw/generations_exp1.jsonl"),
           "gens2": read_rows("results/raw/generations_exp2.jsonl"), "rerun1": read_rows("results/raw/determinism_exp1.jsonl"),
           "scored": read_rows("results/scored/layered_exp1.jsonl"), "bins": cfg["uncertainty"]["ece_bins"],
           "tables": {n: t for n in ("exp2_uq_by_condition_long", "exp1_all_metrics_long") if (t := table(n))},
           "models": [m for g in cfg["models"].values() for m in g], "mcq_ids": {it["item_id"] for it in items if it["options"] and it["source"] == "mmedbench"},
           "dialect_rows": [r for p in sorted(Path("data/dialect").glob("rendered_*.jsonl")) for r in read_rows(p)],
           "dialect_quality": table("data_quality_dialect"), "commit": cfg["rendering"]["dialect"]["commit"].split()[0]}
    if not ctx["rendered"]:
        sys.exit("data/rendered.jsonl is missing: run scripts/02e_merge_rendered.py first")
    out = run_all(ctx)
    write_table(pd.DataFrame(out), "sanity_checks", {"experiment": "both"})
    for r in out:
        print(f'{r["status"]:8} {r["check"]:24} {r["detail"]}')
    sys.exit(1 if any(r["status"] == "fail" for r in out) else 0)
