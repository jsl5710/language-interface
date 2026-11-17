"""Stage 4: judge labels for Experiment 1 generations (serve the judge first: slurm/serve_judge_exp1.sbatch).

    python scripts/04_score.py                      # results/raw/generations_exp1.jsonl -> results/scored/scored_exp1.jsonl
Resumable. Local only: scored rows contain responses to harmful requests."""
import json
import os
import re
from pathlib import Path

import yaml

from dlsd.backends import Backend
from dlsd.judge import run_scoring

PAIRS = [("results/raw/generations_exp1.jsonl", "results/scored/scored_exp1.jsonl"),
         ("results/raw/generations_exp1_gatefailed.jsonl", "results/scored/scored_exp1_gatefailed.jsonl")]      # sensitivity run: judge-rejected renderings

if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/experiment.yaml"))
    rendered = [json.loads(l) for l in open("data/rendered.jsonl")]
    url = cfg["judge"]["base_url"]
    if os.environ.get("DLSD_JUDGE_PORT"):
        url = re.sub(r":\d+/", f":{os.environ['DLSD_JUDGE_PORT']}/", url)
    be = Backend(cfg["judge"]["model"], url, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    for src, dst in PAIRS:
        if not Path(src).exists():
            continue
        gens = [json.loads(l) for l in open(src) if l.strip()]
        n = run_scoring(gens, rendered, be, dst, judge_id=cfg["judge"]["model"])
        print(f"{src}: scored {n} new rows", flush=True)
