"""Re-ask Experiment 1 multiple-choice items whose stored reply is EMPTY (one served model). Only the raw-prompt model (Meditron-7B) has any.
Cause: its prompt ends with 'Answer:' and the model often starts with a newline, which the stop sequence '\\n' turned into an empty reply.
Retry: same prompt, temperature 0, 16 tokens, NO stop sequence; the reply is the first non-empty line (docs/DEVIATIONS.md 2026-11-15).
    python scripts/03e_retry_empty.py --model epfl-llm/meditron-7b      # serve it first
Output: results/raw/retry_exp1[.<shard>].jsonl (local only). Apply with scripts/09b_apply_retries.py."""
import argparse
import json
import os
import re

import yaml

from dlsd.backends import Backend
from dlsd.generate import ID_KEYS, build_prompt, is_mcq, select_rows, shard_path
from dlsd.scoring import correct, format_ok, parse_letter

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open("configs/experiment.yaml"))
    rend = {(r["item_id"], r["condition_id"]): r for r in select_rows([json.loads(l) for l in open("data/rendered.jsonl")], "exp1")}
    m = next(m for g in cfg["models"].values() for m in g if m["id"] == a.model)
    url = re.sub(r":\d+/", f":{os.environ['DLSD_MODEL_PORT']}/", m["base_url"]) if os.environ.get("DLSD_MODEL_PORT") else m["base_url"]
    be = Backend(a.model, url)
    todo = [json.loads(l) for l in open("results/raw/generations_exp1.jsonl")]
    todo = [r for r in todo if r["model_id"] == a.model and "parsed_letter" in r and not (r["response_text"] or "").strip()]
    print(f"{a.model}: {len(todo)} empty multiple-choice replies to retry", flush=True)
    with open(shard_path("results/raw/retry_exp1.jsonl"), "a") as out:
        for o in todo:
            r = rend[(o["item_id"], o["condition_id"])]
            assert is_mcq(r)
            text = be.complete_gen(build_prompt(r, True), 0.0, 16)[0].text
            first = next((l.strip() for l in text.splitlines() if l.strip()), "")
            L = parse_letter(first)
            row = {k: o[k] for k in ID_KEYS} | {"model_id": o["model_id"], "model_group": o["model_group"], "prompt_mode": o["prompt_mode"],
                   "response_text": first, "parsed_letter": L, "correct": correct(L, r["answer"]), "format_ok": format_ok(first), "retry_of_empty": True}
            out.write(json.dumps(row) + "\n")
            out.flush()
