"""Stage 15 (E1 stage 4, language and variety identification): a seeded, item-grouped, cross-validated linear probe per layer.

    python scripts/15_language_probe.py results/activations/<model_slug> [--positions mean,last]

Two tasks: language (en, es, hi, sw; dialects count as en) and variety (SAE, each dialect, each translation). Trained and scored on the
monolingual conditions with folds grouped by item; shuffled-label control and chance reported beside accuracy, overall and per condition.
Code-switched conditions are scored by the probe fitted on all monolingual rows: share of rows given each label. This measures linear
decodability only; it is not evidence of language-identification routing (a hypothesis in Minionese, not a measured mechanism).
Writes results/aggregate/e1_langprobe_<slug> and e1_langprobe_codeswitch_<slug>.
"""
import argparse
import os

import numpy as np
import yaml

from dlsd import langprobe as LP
from dlsd.activations import load_shards
from dlsd.io import write_table

POS = {"mean": "M", "last": "L", "inst": "I"}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("act_dir")
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--positions", default="mean,last")
    ap.add_argument("--out-dir", default="results/aggregate")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    P, seed = cfg["e1_stage_map"], cfg["seed"]
    shards = load_shards(a.act_dir)
    layers = sorted({int(l) for z in shards.values() for l in z["layers"]})
    results = {}
    for pos in a.positions.split(","):
        for l in layers:
            key = f"{POS[pos]}{l}"
            mono = [(c, z) for c, z in sorted(shards.items()) if key in z.files and LP.condition_labels(c)["monolingual"]]
            cs = [(c, z) for c, z in sorted(shards.items()) if key in z.files and not LP.condition_labels(c)["monolingual"]]
            if len({c for c, _ in mono}) < 2:
                continue
            x = np.concatenate([z[key] for _, z in mono])
            groups = np.concatenate([[str(i) for i in z["item_ids"]] for _, z in mono])
            conds = np.concatenate([[c] * len(z["item_ids"]) for c, z in mono])
            x_cs = np.concatenate([z[key] for _, z in cs]) if cs else None
            conds_cs = np.concatenate([[c] * len(z["item_ids"]) for c, z in cs]) if cs else None
            for task in ("language", "variety"):
                y = np.array([LP.condition_labels(c)[task] for c in conds])
                if len(set(y)) < 2:
                    continue
                results[(task, pos, l)] = LP.probe_layer(x, y, groups, conds, P["probe_folds"], P["probe_C"], P["probe_max_iter"], seed, x_cs, conds_cs)
    acc, csh = LP.tidy(results)
    name = os.path.basename(a.act_dir.rstrip("/"))
    meta = {"experiment": "E1", "stage": 4, "seed": seed, "folds": P["probe_folds"], "C": P["probe_C"], "grouped_by": "item_id",
            "control": "labels shuffled within item", "provisional": ["probe_folds", "probe_C", "probe_max_iter"],
            "caveat": "linear decodability only; not a test of language-identification routing"}
    write_table(acc, f"e1_langprobe_{name}", meta, a.out_dir)
    write_table(csh, f"e1_langprobe_codeswitch_{name}", meta, a.out_dir)
    print(acc[acc.condition_id == "ALL"].to_string())
