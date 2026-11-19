"""Stage 10 (E1): residual activations for every rendered prompt, one shard per (model, condition). Resumable.

    python scripts/10_extract_activations.py google/gemma-3-27b-it --layers 10,14,18,22 [--limit-items 20] [--batch-size 8] [--last-only]

Shard keys (backward compatible: L<l> is unchanged, the rest are added):
    L<l>  post-instruction (last prompt) token, as before
    I<l>  last token of the user instruction (Zhao et al. 2025 t_inst)          } default; omitted with --last-only
    M<l>  mean over user-content tokens                                        }
    emb_norms, emb_offsets  per-token embedding norms (hidden_states[0]) of the user content, flat with offsets per prompt
    span_exact, n_layers, capture_version
A shard that exists is skipped if it already holds every requested key; an older last-token-only shard is recomputed and replaced
(written to a temporary file first, so an interrupted run never leaves a partial shard). Stages 13 to 18 read the extended keys.

Needs data/rendered.jsonl (see docs/EXPERIMENTS_LANGUAGE_INTERFACE.md for where it comes from) and a GPU for large models.
Writes results/activations/<model>/<condition>.npz (local only, gitignored; prompts include harmful requests).
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

from dlsd.activations import LABEL_BY_SOURCE, capture, chat_text, has_extended, labelled_rows, last_token_hidden
from dlsd.generate import build_prompt


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--layers", required=True, help="comma-separated block indices, chosen from the English split by 11_directions.py")
    ap.add_argument("--limit-items", type=int)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--rendered", default="data/rendered.jsonl")
    ap.add_argument("--benign-source", default="medsafetybench_benign", choices=["medsafetybench_benign", "mmedbench"],
                    help="benign side of the contrast; mmedbench is multiple choice, so harmful-versus-benign is confounded with format")
    ap.add_argument("--last-only", action="store_true", help="old behaviour: last-token states only (L<l>)")
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    layers = [int(x) for x in a.layers.split(",")]
    rows = labelled_rows([json.loads(l) for l in open(a.rendered) if l.strip()], a.benign_source)
    if a.limit_items:
        keep = sorted({r["item_id"] for r in rows})[: a.limit_items]
        rows = [r for r in rows if r["item_id"] in set(keep)]
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="auto").eval()
    out_dir = Path("results/activations") / slug(a.model)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_cond: dict = {}
    for r in rows:
        by_cond.setdefault(r["condition_id"], []).append(r)
    for cid, rs in sorted(by_cond.items()):
        path = out_dir / f"{slug(cid)}.npz"
        if path.exists():
            with np.load(path) as old:
                done = all(f"L{l}" in old.files for l in layers) if a.last_only else has_extended(old, layers)
            if done:
                continue
        meta = dict(item_ids=np.array([r["item_id"] for r in rs]), labels=np.array([LABEL_BY_SOURCE[r["source"]] for r in rs]),
                    benign_source=a.benign_source, layers=np.array(layers), condition_id=cid, model=a.model)
        if a.last_only:
            acts = last_token_hidden(model, tok, [chat_text(tok, build_prompt(r)) for r in rs], layers, a.batch_size)
            arrays = {f"L{l}": acts[l] for l in layers}
        else:
            c = capture(model, tok, [build_prompt(r) for r in rs], layers, a.batch_size)
            arrays = {f"{p}{l}": c[k][l] for l in layers for p, k in (("L", "last"), ("I", "inst"), ("M", "mean"))}
            arrays |= {"emb_norms": c["emb_norms"], "emb_offsets": c["emb_offsets"], "span_exact": c["span_exact"],
                       "n_layers": model.config.get_text_config().num_hidden_layers, "capture_version": 2}
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, **meta, **arrays)
        tmp.replace(path)
        print(cid, len(rs))
