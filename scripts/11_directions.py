"""Stage 11 (E1): per-condition projections on the English-derived direction.

    python scripts/11_directions.py results/activations/<model_slug> [--seed 13]

Order matters (docs/EXPERIMENTS_LANGUAGE_INTERFACE.md): the layer is chosen on the English baseline only, from a held-out split, and
recorded before any other condition is summarised. The direction used for every condition is fit on the English baseline half A.
Writes results/aggregate/e1_directions_<model_slug>.{csv,json}. Descriptive only: no significance tests across the grid.
"""
import argparse
import os

import numpy as np
import pandas as pd

from dlsd import mechanism as M
from dlsd.activations import load_shards
from dlsd.io import write_table

BASE = "baseline|SAE"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("act_dir")
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()
    shards = load_shards(a.act_dir)
    assert BASE in shards, f"English baseline {BASE} missing from {a.act_dir}"
    eng = shards[BASE]
    layers = [int(l) for l in eng["layers"]]
    ids, labels = list(eng["item_ids"]), eng["labels"]
    choice = M.pick_layer({l: eng[f"L{l}"] for l in layers}, labels, ids, a.seed)
    L = choice["layer"]
    half_a = set(choice["half_a"])
    in_a = np.array([i in half_a for i in ids])
    acts = eng[f"L{L}"]
    lab = labels.astype(bool)
    d_en = M.diff_in_means(acts[in_a & lab], acts[in_a & ~lab])
    sep_en = M.separation(acts[~in_a & lab], acts[~in_a & ~lab], d_en)      # English reference on the held-out half
    control = M.random_unit(len(d_en), a.seed)
    rows = []
    for cid, z in sorted(shards.items()):
        x, y = z[f"L{L}"], z["labels"].astype(bool)
        if cid == BASE:                                                       # score English on the held-out half only
            keep = np.array([i not in half_a for i in z["item_ids"]])
            x, y = x[keep], y[keep]
        if y.sum() < 2 or (~y).sum() < 2:
            continue
        rows.append({"condition_id": cid, "layer": L, **M.condition_summary(x, y, d_en, sep_en, control)})
    df = pd.DataFrame(rows)
    name = "e1_directions_" + os.path.basename(a.act_dir.rstrip("/"))
    write_table(df, name, {"experiment": "E1", "layer": L, "heldout_auroc_by_layer": {str(k): v for k, v in choice["heldout_auroc"].items()},
                           "seed": a.seed, "reference": BASE})
    print(df[["condition_id", "retention", "auroc_english_direction", "cosine_to_english", "cosine_control"]].to_string())
