"""Stage 18 (E1 stage 8, output): per-item projections joined with the response language and the refusal label; aggregates only.

    python scripts/18_output_join.py results/activations/<model_slug> [--scored results/scored/scored_exp1.jsonl] [--layer 14]

For each harmful prompt of this model: projection on the harmfulness direction (t_inst) and the refusal direction (t_post-inst) from
dlsd.harm_refusal, the judge's refusal label, and the language of the response (dlsd.checks.detect over en, es, hi, sw; blank responses
are "empty"). The join happens in memory; only per-condition aggregates are written (results/aggregate/e1_output_<slug>.{csv,json}).
"""
import argparse
import importlib.util
import os
from pathlib import Path

import pandas as pd
import yaml

from dlsd import harm_refusal as H
from dlsd import langprobe as LP
from dlsd.activations import load_shards
from dlsd.io import write_table

BASE = "baseline|SAE"
_spec = importlib.util.spec_from_file_location("s17", Path(__file__).with_name("17_harm_refusal.py"))
s17 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s17)


def response_language(text, detect) -> str:
    return "empty" if not (text or "").strip() else detect(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("act_dir")
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--scored", nargs="+", default=["results/scored/scored_exp1.jsonl"])
    ap.add_argument("--layer", type=int)
    ap.add_argument("--out-dir", default="results/aggregate")
    a = ap.parse_args()
    from dlsd.checks import detect
    cfg = yaml.safe_load(open(a.config))
    P, seed = cfg["e1_stage_map"], cfg["seed"]
    shards = load_shards(a.act_dir)
    name = os.path.basename(a.act_dir.rstrip("/"))
    layer = s17.pick_layer(a, name)
    model = str(shards[BASE]["model"])
    scored = [r for r in s17.read_scored(a.scored) if r.get("model_id") == model]
    labels = H.refusal_labels(scored, model, P["refused_labels"])
    dirs = H.fit_directions(shards[BASE], layer, labels, BASE, seed, P["min_per_class"])
    proj = pd.concat([H.item_projections(z, dirs, labels, cid) for cid, z in sorted(shards.items()) if f"I{layer}" in z.files], ignore_index=True)
    lang = {(str(r["item_id"]), r["condition_id"]): response_language(r.get("response_text"), detect) for r in scored}
    proj["response_lang"] = [lang.get(k) for k in zip(proj["item_id"], proj["condition_id"])]
    proj = proj.dropna(subset=["response_lang"])
    expected = {c: LP.condition_labels(c)["language"] for c in shards}
    df = H.output_aggregates(proj, expected)
    write_table(df, f"e1_output_{name}", {"experiment": "E1", "stage": 8, "layer": layer, "model": model, "seed": seed,
                                           "language_id": "dlsd.checks.detect (langid over en, es, hi, sw)",
                                           "refused_labels": P["refused_labels"], "provisional": ["refused_labels", "min_per_class"]}, a.out_dir)
    print(df.to_string())
