"""Stage 16 (E1 stage 5, semantic representation): mid-layer mean-pooled states of each condition against English.

    python scripts/16_semantic.py results/activations/<model_slug> [--comprehension data/comprehension.jsonl]
                                  [--backtranslation-dir data/backtranslation --items data/items.jsonl] [--n-boot 1000]

Layer: the extracted layer nearest e1_stage_map.semantic_layer_fraction of depth (provisional). Per condition: paired cosine against the same
item's English prompt with a bootstrap interval and the mismatched-item floor, and linear CKA against English over the same items beside a
CKA floor (English against English with rows mismatched). Comprehension check hook: restatement scores joined by item id and condition,
either from a jsonl of {item_id, condition_id, comprehension} or computed in the scripts/02q_backtranslate.py pattern (chrF++ of the
back-translation against the English item text; resource conditions only, as in 02q). Items at or above e1_stage_map.comprehension_min
(provisional) count as understood; similarity is reported for understood and not-understood items separately, so a fall in similarity
is not read as a failure to understand. Writes results/aggregate/e1_semantic_<slug>.{csv,json}. Row text and row scores are never written.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from dlsd import representation as R
from dlsd.activations import load_shards
from dlsd.io import write_table

BASE = "baseline|SAE"


def semantic_table(shards: dict, layer: int, seed: int, n_boot: int, comp: pd.DataFrame | None, comp_min: float, base: str = BASE) -> pd.DataFrame:
    eng = shards[base]
    xe = eng[f"M{layer}"]
    floor = R.mismatched_floor(xe, seed)
    cka_floor = R.linear_cka(xe, xe[R.derangement(len(xe), seed)])
    cmap = {} if comp is None else {(str(r.item_id), r.condition_id): r.comprehension for r in comp.itertuples()}
    rows = []
    for cid, z in sorted(shards.items()):
        if cid == base or f"M{layer}" not in z.files:
            continue
        ia, ib, common = R.align(z["item_ids"], eng["item_ids"])
        if len(ia) < 3:
            continue
        xc, xr = z[f"M{layer}"][ia], xe[ib]
        row = {"condition_id": cid, "layer": layer, **R.paired_similarity(xc, xr, floor, n_boot, seed),
               "cka": R.linear_cka(xc, xr), "cka_floor_mismatched": cka_floor}
        sc = np.array([cmap.get((i, cid), np.nan) for i in common], float)
        has = ~np.isnan(sc)
        row["n_comprehension"] = int(has.sum())
        if has.any():
            cos = R.rowwise_cosine(xc, xr)
            ok = has & (sc >= comp_min)
            row |= {"comprehension_mean": float(sc[has].mean()), "share_understood": float(ok.sum() / has.sum()),
                    "cosine_understood": float(cos[ok].mean()) if ok.any() else np.nan,
                    "cosine_not_understood": float(cos[has & ~ok].mean()) if (has & ~ok).any() else np.nan}
        rows.append(row)
    return pd.DataFrame(rows)


def load_comprehension(a) -> pd.DataFrame | None:
    if a.comprehension:
        return pd.DataFrame([json.loads(l) for l in open(a.comprehension) if l.strip()])
    if a.backtranslation_dir:
        import sacrebleu
        chrf = sacrebleu.metrics.CHRF(word_order=2)
        items = {str(r["item_id"]): r["text"] for r in (json.loads(l) for l in open(a.items) if l.strip())}
        cfg = yaml.safe_load(open(a.config))
        cond_of = {lang: f"resource|{lvl}|{lang}" for lvl, lang in cfg["conditions"]["languages"].items()}
        bt = [json.loads(l) for f in sorted(Path(a.backtranslation_dir).glob("*.jsonl")) for l in open(f) if l.strip()]
        return R.comprehension_from_backtranslation(bt, items, lambda h, ref: chrf.sentence_score(h, [ref]).score, cond_of)
    return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("act_dir")
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--comprehension")
    ap.add_argument("--backtranslation-dir")
    ap.add_argument("--items", default="data/items.jsonl")
    ap.add_argument("--n-boot", type=int)
    ap.add_argument("--out-dir", default="results/aggregate")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    P, seed = cfg["e1_stage_map"], cfg["seed"]
    shards = load_shards(a.act_dir)
    eng = shards[BASE]
    assert "n_layers" in eng.files, "rerun 10_extract_activations.py without --last-only"
    layer = R.nearest_layer([int(l) for l in eng["layers"]], int(eng["n_layers"]), P["semantic_layer_fraction"])
    df = semantic_table(shards, layer, seed, P["n_boot"] if a.n_boot is None else a.n_boot, load_comprehension(a), P["comprehension_min"])
    name = os.path.basename(a.act_dir.rstrip("/"))
    write_table(df, f"e1_semantic_{name}", {"experiment": "E1", "stage": 5, "layer": layer, "reference": BASE, "seed": seed,
                                             "comprehension_min": P["comprehension_min"],
                                             "provisional": ["semantic_layer_fraction", "comprehension_min"]}, a.out_dir)
    print(df.to_string())
