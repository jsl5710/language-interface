"""Sanity check 10: re-run the temperature-0 Experiment 1 generation for 20 random rows of ONE served model and store the new rows.
    python scripts/03d_determinism.py --model <id>        # serve it first
Output: results/raw/determinism_exp1[.<shard>].jsonl; check_results.py compares them with the original rows. Fixed seed; local only."""
import argparse
import json
import os
import random
import re

import yaml

from dlsd.backends import Backend
from dlsd.generate import RAW_MODELS, ID_KEYS, gen_exp1, select_rows, shard_path

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()
    cfg = yaml.safe_load(open("configs/experiment.yaml"))
    rows = select_rows([json.loads(l) for l in open("data/rendered.jsonl")], "exp1")
    by_key = {(r["item_id"], r["condition_id"]): r for r in rows}
    orig = [json.loads(l) for l in open("results/raw/generations_exp1.jsonl")]
    orig = [r for r in orig if r["model_id"] == a.model and (r["item_id"], r["condition_id"]) in by_key]
    pick = random.Random(13).sample(orig, a.n)
    m = next(m for g in cfg["models"].values() for m in g if m["id"] == a.model)
    url = re.sub(r":\d+/", f":{os.environ['DLSD_MODEL_PORT']}/", m["base_url"]) if os.environ.get("DLSD_MODEL_PORT") else m["base_url"]
    be, raw = Backend(a.model, url), a.model in RAW_MODELS
    out = shard_path("results/raw/determinism_exp1.jsonl")
    with open(out, "a") as f:
        for o in pick:
            r = by_key[(o["item_id"], o["condition_id"])]
            new = gen_exp1(be, r, cfg["decoding"], raw) | {k: r[k] for k in ID_KEYS} | {"model_id": a.model}
            f.write(json.dumps(new) + "\n")
    print(f"{a.model}: {a.n} rows re-generated -> {out}")
