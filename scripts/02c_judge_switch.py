"""Stage 2c: one judge of the panel classifies code-switched renderings blind (no intended type is shown).

    python scripts/02c_judge_switch.py --judge gpt_oss_120b --round 1
    python scripts/02c_judge_switch.py --judge tiny_aya_global --round 1 --language Swahili

Serve the judge first (slurm/serve_judge.sbatch does this). Resumable per (item, round); one output file per shard under
data/judge/<judge>/. Unparseable replies are stored with label null and count as a failed vote at the gate.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import yaml

from dlsd.backends import Backend
from dlsd.gate_io import JUDGE_DIR, codeswitch_shards, read_rows
from dlsd.switch_judge import judge_one

from importlib import import_module

_probe = import_module("02i_probe_judge")
REASONING = _probe.REASONING   # gpt-oss reads its reasoning effort from the system prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", required=True)
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--shots", choices=["few", "zero"], default="few", help="few = worked examples (pipeline setting), zero = ablation into data/judge_zero")
    ap.add_argument("--workers", type=int, default=16, help="concurrent requests; vLLM batches them")
    ap.add_argument("--language", choices=["Spanish", "Hindi", "Swahili"])
    ap.add_argument("--config", default="configs/experiment.yaml")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    spec = next(j for j in cfg["switch_judges"]["panel"] if j["name"] == a.judge)
    be = _probe.backend_for(spec)
    out_dir = (JUDGE_DIR if a.shots == "few" else JUDGE_DIR.parent / "judge_zero") / a.judge
    out_dir.mkdir(parents=True, exist_ok=True)

    for shard in codeswitch_shards():
        rows = [r for r in read_rows(shard) if r.get("round", 1) == a.round and r["rendered"] and (a.language in (None, r["language_pair"]))]
        if not rows:
            continue
        out = out_dir / shard.name.replace("rendered_", "judge_")
        done = {r["item_id"] for r in read_rows(out) if r["round"] == a.round}
        todo = [r for r in rows if r["item_id"] not in done]
        print(f"{shard.name}: {len(done)} done, {len(todo)} to do", flush=True)
        def one(r):
            label, raw = judge_one(be, r["rendered"], r["language_pair"], f'{r["item_id"]}|{r["language_pair"]}|{r["switch_type"]}|{a.round}',
                                   system=REASONING.get(a.judge), shots=a.shots == "few")
            return {"item_id": r["item_id"], "round": a.round, "judge": a.judge, "shots": a.shots, "label": label,
                    "intended": r["switch_type"], "language_pair": r["language_pair"], "raw_tail": raw[-1500:]}
        with open(out, "a") as f, ThreadPoolExecutor(a.workers) as pool:
            for n, rec in enumerate(pool.map(one, todo), 1):              # map keeps item order; a crash loses at most the in-flight chunk
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                if n % 100 == 0:
                    print(f"  {n}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
