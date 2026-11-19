"""Stage 17 (E1 stages 6 and 7): harmfulness at the instruction token versus refusal at the post-instruction token, per condition.

    python scripts/17_harm_refusal.py results/activations/<model_slug> [--scored results/scored/scored_exp1.jsonl] [--layer 14]

Positions follow Zhao et al. 2025: harmfulness read at t_inst (shard key I<l>), refusal at t_post-inst (L<l>). Refusal labels are the judge
labels of the existing generations for this model (local row-level files, read here and never written out); e1_stage_map.refused_labels
says which labels count as refused (provisional). Layer: --layer, else the layer chosen on English by 11_directions.py
(results/aggregate/e1_directions_<slug>.json). Writes results/aggregate/e1_harm_refusal_<slug>.{csv,json}: aggregates only.
"""
import argparse
import json
import os

import pandas as pd
import yaml

from dlsd import harm_refusal as H
from dlsd.activations import load_shards
from dlsd.io import write_table

BASE = "baseline|SAE"


def read_scored(paths: list) -> list:
    rows = []
    for p in paths:
        if os.path.exists(p):
            rows += [json.loads(l) for l in open(p) if l.strip()]
    return rows


def pick_layer(a, name: str) -> int:
    if a.layer is not None:
        return a.layer
    return int(json.load(open(os.path.join(a.out_dir, f"e1_directions_{name}.json")))["layer"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("act_dir")
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--scored", nargs="+", default=["results/scored/scored_exp1.jsonl"])
    ap.add_argument("--layer", type=int)
    ap.add_argument("--n-boot", type=int)
    ap.add_argument("--out-dir", default="results/aggregate")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    P, seed = cfg["e1_stage_map"], cfg["seed"]
    shards = load_shards(a.act_dir)
    name = os.path.basename(a.act_dir.rstrip("/"))
    layer = pick_layer(a, name)
    model = str(shards[BASE]["model"])
    labels = H.refusal_labels(read_scored(a.scored), model, P["refused_labels"])
    dirs = H.fit_directions(shards[BASE], layer, labels, BASE, seed, P["min_per_class"])
    nb = P["n_boot"] if a.n_boot is None else a.n_boot
    df = pd.DataFrame([H.condition_rows(z, dirs, labels, cid, BASE, nb, seed, P["min_per_class"]) for cid, z in sorted(shards.items())
                       if f"I{layer}" in z.files])
    meta = {"experiment": "E1", "stages": [6, 7], "layer": layer, "model": model, "seed": seed, "reference": BASE,
            "harm_position": "t_inst (I)", "refusal_position": "t_post-inst (L)", "refused_labels": P["refused_labels"],
            "refusal_direction_fitted": dirs["d_refusal"] is not None, "n_en_refused_half_a": dirs["n_en_refused_a"],
            "n_en_complied_half_a": dirs["n_en_complied_a"], "cosine_harm_refusal": dirs.get("cosine_harm_refusal"),
            "provisional": ["refused_labels", "min_per_class", "n_boot"]}
    write_table(df, f"e1_harm_refusal_{name}", meta, a.out_dir)
    print(df.to_string())
