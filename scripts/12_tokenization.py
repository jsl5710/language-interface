"""Stage 12 (E1): tokenization covariates per prompt and, if activations exist, how much of each condition's projection gap tokenization explains.

    python scripts/12_tokenization.py google/gemma-3-27b-it [--act-dir results/activations/<slug>] [--n-boot 1000]

Writes results/aggregate/e1_tokenization_<slug>.csv/json (condition means of the measures) and, with --act-dir, e1_tokenization_gaps_<slug>
(raw and adjusted gaps in projection, see dlsd.tokenization.adjusted_gaps). Descriptive only. Row text is never written.
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd

from dlsd import mechanism as M
from dlsd import tokenization as T
from dlsd.activations import labelled_rows
from dlsd.io import write_table

BASE = "baseline|SAE"
slug = lambda s: re.sub(r"[^A-Za-z0-9._-]+", "_", s)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--rendered", default="data/rendered.jsonl")
    ap.add_argument("--act-dir")
    ap.add_argument("--n-boot", type=int, default=0)
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    tokenize = lambda t: tok.convert_ids_to_tokens(tok(t, add_special_tokens=False)["input_ids"])
    rows = labelled_rows([json.loads(l) for l in open(a.rendered) if l.strip()])
    tab = T.item_table(rows, tokenize)
    meas = ["n_tokens", "fertility", "premium", "byte_fallback_share", "split3_share"]
    write_table(tab.groupby("condition_id")[meas].mean().reset_index(), f"e1_tokenization_{slug(a.model)}", {"experiment": "E1", "model": a.model})
    if a.act_dir:
        shards = {str(np.load(f)["condition_id"]): np.load(f) for f in sorted(glob.glob(os.path.join(a.act_dir, "*.npz")))}
        layer = int(json.load(open(f"results/aggregate/e1_directions_{os.path.basename(a.act_dir.rstrip('/'))}.json"))["layer"])
        eng = shards[BASE]
        lab = eng["labels"].astype(bool)
        d_en = M.diff_in_means(eng[f"L{layer}"][lab], eng[f"L{layer}"][~lab])
        recs = []
        for cid, z in shards.items():
            for i, y, l in zip(z["item_ids"], M.project(z[f"L{layer}"], d_en), z["labels"]):
                if l == 1:                                     # harmful prompts: does projection (recognition) fall as tokens fragment?
                    recs.append({"item_id": i, "condition_id": cid, "projection": float(y)})
        df = pd.DataFrame(recs).merge(tab, on=["item_id", "condition_id"])
        gaps = T.adjusted_gaps(df, "projection", "premium", "condition_id", BASE, n_boot=a.n_boot, seed=a.seed)
        write_table(gaps, f"e1_tokenization_gaps_{slug(a.model)}", {"experiment": "E1", "model": a.model, "layer": layer, "outcome": "projection of harmful prompts on English direction"})
        print(gaps.to_string())
