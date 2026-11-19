"""Stage 13 (E1 stage 2, embedding): per-prompt embedding-norm measures from hidden_states[0] of the user-content tokens.

    python scripts/13_embedding.py results/activations/<model_slug> [--n-boot 1000]

Per prompt: mean token-embedding norm, share of tokens above the e1_stage_map.embedding_norm_percentile of English (baseline|SAE) content
tokens (provisional), and the token count (premium = tokens over the same item's English token count). Per condition: means, and the gap in
mean norm from English raw and adjusted for log premium (dlsd.tokenization.adjusted_gaps), which is the fragmentation-aware summary.
Writes results/aggregate/e1_embedding_<slug>.{csv,json} and e1_embedding_gaps_<slug>. Needs shards written by 10_extract_activations.py
without --last-only. Descriptive only.
"""
import argparse
import os

import numpy as np
import pandas as pd
import yaml

from dlsd import representation as R
from dlsd import tokenization as T
from dlsd.activations import load_shards
from dlsd.io import write_table

BASE = "baseline|SAE"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("act_dir")
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--n-boot", type=int)
    ap.add_argument("--out-dir", default="results/aggregate")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    P, seed = cfg["e1_stage_map"], cfg["seed"]
    n_boot = P["n_boot"] if a.n_boot is None else a.n_boot
    shards = load_shards(a.act_dir)
    assert BASE in shards and "emb_norms" in shards[BASE].files, "English shard with embedding norms needed (rerun 10 without --last-only)"
    norms = {c: R.split_norms(z["emb_norms"], z["emb_offsets"]) for c, z in shards.items() if "emb_norms" in z.files}
    thr = R.norm_threshold(norms[BASE], P["embedding_norm_percentile"])
    recs = []
    for cid, nl in norms.items():
        st = R.embedding_stats(nl, thr)
        st["item_id"] = [str(i) for i in shards[cid]["item_ids"]]
        st["label"] = shards[cid]["labels"]
        st["condition_id"] = cid
        recs.append(st)
    df = pd.concat(recs, ignore_index=True)
    en_len = df[df.condition_id == BASE].set_index("item_id")["n_content_tokens"]
    df["premium"] = df["n_content_tokens"] / df["item_id"].map(en_len)
    df = df[df["n_content_tokens"] > 0]
    by = df.groupby(["condition_id", "label"])[["n_content_tokens", "premium", "mean_norm", "share_above"]].mean().reset_index()
    name = os.path.basename(a.act_dir.rstrip("/"))
    meta = {"experiment": "E1", "stage": 2, "threshold_norm": thr, "percentile": P["embedding_norm_percentile"], "provisional": ["embedding_norm_percentile"],
            "reference": BASE}
    write_table(by, f"e1_embedding_{name}", meta, a.out_dir)
    gaps = T.adjusted_gaps(df.dropna(subset=["premium"]), "mean_norm", "premium", "condition_id", BASE, n_boot=n_boot, seed=seed)
    write_table(gaps, f"e1_embedding_gaps_{name}", meta | {"outcome": "mean content-token embedding norm, all prompts"}, a.out_dir)
    print(by.to_string())
