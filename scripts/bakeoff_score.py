"""Translator bake-off, step 2 (CPU): chrF++ on FLORES, LaBSE similarity and refusal rate on the domain probe, then apply the
selection rule written in bakeoff_translate.py. Writes results/aggregate/translator_bakeoff.{csv,json} and prints the winner."""
import json
from pathlib import Path

import pandas as pd
import sacrebleu

from dlsd.checks import Similarity, looks_like_refusal
from dlsd.io import write_table

SYSTEMS = ["nllb", "madlad", "qwen"]
rows = [json.loads(l) for s in SYSTEMS if Path(f"data/bakeoff/{s}.jsonl").exists() for l in open(f"data/bakeoff/{s}.jsonl", encoding="utf-8")]
df = pd.DataFrame(rows)
sim = Similarity("sentence-transformers/LaBSE")
fl = df[df.set == "flores"]
recs = []
for (system, lang), g in fl.groupby(["system", "language"]):
    recs.append({"system": system, "language": lang, "metric": "chrF++", "value": sacrebleu.corpus_chrf(g.hyp.tolist(), [g.ref.tolist()], word_order=2).score})
pr = df[df.set.str.startswith("probe")].copy()
pr["labse"] = [sim(s, h) for s, h in zip(pr.src, pr.hyp)]
pr["refusal"] = [looks_like_refusal(h, s) for s, h in zip(pr.hyp, pr.src)]
for (system, lang), g in pr.groupby(["system", "language"]):
    recs.append({"system": system, "language": lang, "metric": "probe_labse", "value": float(g.labse.mean())})
    recs.append({"system": system, "language": lang, "metric": "probe_refusals", "value": float(g.refusal.sum())})
for (system, st), g in pr.groupby(["system", "set"]):
    recs.append({"system": system, "language": st, "metric": "probe_labse_by_set", "value": float(g.labse.mean())})
res = pd.DataFrame(recs)
write_table(res, "translator_bakeoff", {"experiment": "both", "n_flores": int(fl.id.nunique()), "n_probe_per_source": 30})
print(res.pivot_table(index=["metric", "language"], columns="system", values="value").round(2).to_string())

m = res.groupby(["system", "metric"]).value.mean().unstack()
best = m["chrF++"].max()
close = m[m["chrF++"] >= best - 1.0]
winner = (close["probe_labse"].sort_values(ascending=False).index[0] if len(close) > 1 else close.index[0])
if len(close) > 1:
    top = close.loc[close.probe_labse >= close.probe_labse.max() - 1e-3]
    if len(top) > 1:
        winner = top["probe_refusals"].sort_values().index[0]
print("\nmean chrF++ per system:\n", m["chrF++"].round(2).to_string(), "\nWINNER by the pre-registered rule:", winner)
