"""Stage 2i: qualify a judge on the held-out probe set before it labels the shards.

    python scripts/02i_probe_judge.py --judge gpt_oss_120b        # exit code 0 if accuracy >= switch_judges.probe_min_accuracy

The probe (configs/switch_judge_probe.json) has 60 hand-written texts with a known type, 20 per language, none of them in the
few-shot examples. Same blind prompt and decoding as the shard run. Output: data/judge_probe/<judge>.jsonl and a printed table.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import yaml

from dlsd.backends import Backend
from dlsd.switch_judge import LABELS, judge_one

REASONING = {"gpt_oss_120b": "Reasoning: low"}
EXTRA_BODY = {"glm_45_air": {"chat_template_kwargs": {"enable_thinking": False}}}


def backend_for(spec):
    url = spec["base_url"]
    if os.environ.get("DLSD_JUDGE_PORT"):                          # set by slurm/serve_judge.sbatch (one port per job)
        url = re.sub(r":\d+/", f":{os.environ['DLSD_JUDGE_PORT']}/", url)
    return Backend(spec["id"], url, extra_body=EXTRA_BODY.get(spec["name"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", required=True)
    ap.add_argument("--config", default="configs/experiment.yaml")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    sj = cfg["switch_judges"]
    spec = next(j for j in sj["panel"] if j["name"] == a.judge)
    be = backend_for(spec)
    probe = json.load(open("configs/switch_judge_probe.json", encoding="utf-8"))
    out = Path("data/judge_probe"); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for shots in (True, False):                                    # few-shot (the pipeline setting) and the zero-shot ablation
        for lang, items in probe.items():
            for it in items:
                label, raw = judge_one(be, it["text"], lang, f'{it["id"]}|{lang}|probe', system=REASONING.get(a.judge), shots=shots)
                rows.append({**it, "language": lang, "judge": a.judge, "shots": "few" if shots else "zero", "pred": label,
                             "correct": label == it["label"], "raw_tail": raw[-1500:]})
    (out / f"{a.judge}.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    acc = {}
    for mode in ("few", "zero"):
        rs = [r for r in rows if r["shots"] == mode]
        acc[mode] = sum(r["correct"] for r in rs) / len(rs)
        print(f"{a.judge} {mode}-shot probe accuracy {acc[mode]:.3f} ({sum(r['correct'] for r in rs)}/{len(rs)})")
        for lang in probe:
            print(f"  {lang}: " + "  ".join(f"{l} {sum(r['correct'] for r in rs if r['language'] == lang and r['label'] == l)}/{sum(r['language'] == lang and r['label'] == l for r in rs)}" for l in LABELS))
    print(f"few-shot minus zero-shot: {acc['few'] - acc['zero']:+.3f}")
    sys.exit(0 if acc["few"] >= sj["probe_min_accuracy"] else 3)       # qualification uses the few-shot setting


if __name__ == "__main__":
    main()
