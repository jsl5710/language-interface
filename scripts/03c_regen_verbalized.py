"""Re-ask only the verbalised-confidence question for Experiment 2 rows that have no stored raw reply (one served model).
    python scripts/03c_regen_verbalized.py --model meta-llama/Llama-3.1-8B-Instruct      # serve it first
Output: results/raw/verbalized_exp2[.<shard>].jsonl (local only). Resumable."""
import argparse

import yaml

from dlsd.generate import regen_verbalized

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", default="configs/experiment.yaml")
    a = ap.parse_args()
    regen_verbalized(yaml.safe_load(open(a.config)), "data/rendered.jsonl", "results/raw/verbalized_exp2.jsonl", a.model)
