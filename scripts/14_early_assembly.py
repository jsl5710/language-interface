"""Stage 14 (E1 stage 3, early-layer assembly): paired similarity of mean-pooled early-layer states, each rendered prompt against the same
item's English prompt (baseline|SAE).

    python scripts/14_early_assembly.py results/activations/<model_slug> [--n-boot 1000]

Early layers are the extracted layers at or below e1_stage_map.early_layer_max_fraction of depth (provisional; pass early layers to
10_extract_activations.py, for example 1,2,4,6, or this stage has nothing to read). Each condition row is reported beside the English
baseline of the same layer: same item, SAE against the English prompt, which is 1.0 by construction when the reference is the SAE prompt
itself (no paraphrase is involved), and a mismatched-item floor (each English prompt against a different English item, seeded).
`normalized` places the condition between floor (0) and same prompt (1). Writes results/aggregate/e1_early_assembly_<slug>.{csv,json}.
"""
import argparse
import os

import numpy as np
import pandas as pd
import yaml

from dlsd import representation as R
from dlsd.activations import load_shards
from dlsd.io import write_table

BASE = "baseline|SAE"


def similarity_table(shards: dict, layers: list, seed: int, n_boot: int, base: str = BASE) -> pd.DataFrame:
    eng = shards[base]
    rows = []
    for l in layers:
        xe = eng[f"M{l}"]
        floor = R.mismatched_floor(xe, seed)
        rows.append({"condition_id": base, "layer": l, "n": len(xe), "cosine": 1.0, "cosine_lo": np.nan, "cosine_hi": np.nan,
                     "floor_mismatched": float(floor.mean()), "normalized": 1.0, "note": "same prompt: 1.0 by construction"})
        for cid, z in sorted(shards.items()):
            if cid == base or f"M{l}" not in z.files:
                continue
            ia, ib, _ = R.align(z["item_ids"], eng["item_ids"])
            if len(ia) < 2:
                continue
            res = R.paired_similarity(z[f"M{l}"][ia], xe[ib], floor, n_boot, seed)
            rows.append({"condition_id": cid, "layer": l, **res, "note": ""})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("act_dir")
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--n-boot", type=int)
    ap.add_argument("--out-dir", default="results/aggregate")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    P, seed = cfg["e1_stage_map"], cfg["seed"]
    shards = load_shards(a.act_dir)
    eng = shards[BASE]
    assert "n_layers" in eng.files, "rerun 10_extract_activations.py without --last-only"
    layers = R.early_layers([int(l) for l in eng["layers"]], int(eng["n_layers"]), P["early_layer_max_fraction"])
    assert layers, f"no extracted layer is at or below {P['early_layer_max_fraction']} of depth; extract early layers first"
    df = similarity_table(shards, layers, seed, P["n_boot"] if a.n_boot is None else a.n_boot)
    name = os.path.basename(a.act_dir.rstrip("/"))
    write_table(df, f"e1_early_assembly_{name}", {"experiment": "E1", "stage": 3, "layers": layers, "reference": BASE, "seed": seed,
                                                    "provisional": ["early_layer_max_fraction"]}, a.out_dir)
    print(df.to_string())
