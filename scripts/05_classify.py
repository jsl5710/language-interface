"""Stage 5: safety-classifier flags and the layered outcome (section 5.3).

    python scripts/05_classify.py --classifier llama_guard      # serve it first (slurm/serve_classifier.sbatch); resumable
    python scripts/05_classify.py --merge                       # scored_exp1 + all flag files -> results/scored/layered_exp1.jsonl
Input flags come from an input-only call on the rendered request (one per item and condition, shared by all models); output flags
from an input-plus-response call per generation. Local only."""
import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from dlsd.backends import Backend
from dlsd.classifiers import GuardClassifier
from dlsd.layered import merge_layered

SCORED = Path("results/scored/scored_exp1.jsonl")
VARIANTS = [("", SCORED, lambda r: r["render_ok"]),      # main run: judge-approved renderings
            ("_gatefailed", Path("results/scored/scored_exp1_gatefailed.jsonl"), lambda r: r["render_status"] == "failed_gate")]   # sensitivity run


def read(p):
    return [json.loads(l) for l in open(p) if l.strip()] if Path(p).exists() else []


def classify(cfg, name, workers=16):
    for suffix, scored_path, keep in VARIANTS:
        classify_one(cfg, name, suffix, scored_path, keep, workers)


def classify_one(cfg, name, suffix, scored_path, keep, workers):
    spec = next(c for c in cfg["classifiers"] if c["name"] == name)
    clf = GuardClassifier(name, Backend(spec["id"], re.sub(r":\d+/", f":{os.environ['DLSD_CLF_PORT']}/", spec["base_url"]) if os.environ.get("DLSD_CLF_PORT") else spec["base_url"]))
    path = Path(f"results/scored/flags_{name}{suffix}.jsonl")
    scored, rendered = read(scored_path), read("data/rendered.jsonl")
    done = {(f["scope"], f["item_id"], f["condition_id"], f.get("model_id")) for f in read(path)}
    jobs = [("input", r["item_id"], r["condition_id"], None, r["rendered"], None) for r in rendered if keep(r) and r["rendered"] is not None]
    req = {(r["item_id"], r["condition_id"]): r["rendered"] for r in rendered}
    jobs += [("output", g["item_id"], g["condition_id"], g["model_id"], req[(g["item_id"], g["condition_id"])], g["response_text"]) for g in scored]
    todo = [j for j in jobs if (j[0], j[1], j[2], j[3]) not in done]

    def one(j):
        v = clf.classify_input(j[4]) if j[0] == "input" else clf.classify_output(j[4], j[5])
        return {"classifier": name, "scope": j[0], "item_id": j[1], "condition_id": j[2], "model_id": j[3], "flagged": v.flagged,
                "parsed_ok": v.parsed_ok, "label": v.label, "detail": v.detail}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as out, ThreadPoolExecutor(workers) as ex:
        for row in ex.map(one, todo):
            out.write(json.dumps(row) + "\n")
            out.flush()
    print(f"{name}{suffix}: {len(todo)} new flags", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--classifier")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--config", default="configs/experiment.yaml")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if a.merge:
        for suffix, scored_path, _ in VARIANTS:
            flags = [f for c in cfg["classifiers"] for f in read(f"results/scored/flags_{c['name']}{suffix}.jsonl")]
            rows = merge_layered(read(scored_path), flags)
            Path(f"results/scored/layered_exp1{suffix}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            print(f"{suffix or 'main'}: {len(rows)} layered rows")
    else:
        classify(cfg, a.classifier)
