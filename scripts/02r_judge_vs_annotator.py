"""Stage 2r: agreement between each judge and each qualified LLM annotator on the 108-item annotation sample (percent agreement, Cohen's kappa
with a 95% bootstrap interval over items) -> results/aggregate/judge_vs_annotator.csv. Run after 02k analyze."""
import json
import random
from pathlib import Path

import pandas as pd
import yaml

from dlsd.gate_io import codeswitch_shards, read_rows
from dlsd.reliability import qualified_annotators
from dlsd.switch_judge import LABELS, cohen_kappa

REV = Path("data/review")
cfg = yaml.safe_load(open("configs/experiment.yaml"))
_, qualified = qualified_annotators(cfg)
key = [json.loads(l) for l in open(REV / "annotation_key.jsonl")]
texts = {}
for shard in codeswitch_shards():
    cond = shard.name[len("rendered_"):-len(".jsonl")]
    for r in read_rows(shard):
        texts[(cond, r["item_id"])] = r["rendered"]
by_text = {texts[(k["cell"], k["item_id"])]: k for k in key}
pairs = {}
for lang in ("Spanish", "Hindi", "Swahili"):
    for a in qualified:
        p = REV / f"annotation_{lang}_{a}.csv"
        df = pd.read_csv(p, encoding="utf-8-sig", keep_default_na=False)
        for t, lab in zip(df["text"], df[df.columns[2]]):
            if t in by_text and lab:
                k = by_text[t]
                for jname, jl in k["judge_labels"].items():
                    if jl:
                        pairs.setdefault((a, jname), []).append((lab, jl))
rng, out = random.Random(13), []
for (a, j), v in sorted(pairs.items()):
    x, y = [p[0] for p in v], [p[1] for p in v]
    ks = []
    for _ in range(1000):
        idx = [rng.randrange(len(v)) for _ in v]
        kk = cohen_kappa([x[i] for i in idx], [y[i] for i in idx])
        if kk == kk:
            ks.append(kk)
    ks.sort()
    out.append({"annotator": a, "judge_setting": j, "n": len(v), "percent_agreement": sum(p == q for p, q in v) / len(v),
                "cohen_kappa": cohen_kappa(x, y), "kappa_lo": ks[24] if ks else None, "kappa_hi": ks[974] if ks else None})
res = pd.DataFrame(out)
res.to_csv("results/aggregate/judge_vs_annotator.csv", index=False)
print(res.round(3).to_string())
