"""Stage 3: generate with ONE served model (start it first, e.g. slurm/serve_model.sbatch).

    python scripts/03_generate.py --experiment exp1 --model google/gemma-3-27b-it
    python scripts/03_generate.py --experiment exp2 --model google/gemma-3-27b-it --pilot     # 20 items x 18 conditions
    python scripts/03_generate.py ... --dry-run                                               # 5 items into results/dry_run/
Output: results/raw/generations_<exp>.jsonl (local only, gitignored). Resumable."""
import argparse

import yaml

from dlsd.generate import run

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", choices=["exp1", "exp2"], required=True)
    ap.add_argument("--model", required=True, help="model id from configs/experiment.yaml")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--axes", nargs="*", help="only these condition axes (baseline dialect resource codeswitch codeswitch_native)")
    ap.add_argument("--gate-failed", action="store_true", help="sensitivity run: only renderings the judge panel rejected -> results/raw/generations_<exp>_gatefailed.jsonl")
    ap.add_argument("--out", help="override the output path (diagnostics)")
    ap.add_argument("--config", default="configs/experiment.yaml")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    out = f"results/raw/generations_{a.experiment}.jsonl"
    if a.gate_failed:
        out = f"results/raw/generations_{a.experiment}_gatefailed.jsonl"
    if a.dry_run or a.pilot:
        out = f"results/{'dry_run' if a.dry_run else 'pilot'}/generations_{a.experiment}.jsonl"
    out = a.out or out
    run(cfg, "data/rendered.jsonl", out, a.experiment, a.model, limit_items=5 if a.dry_run else 20 if a.pilot else None, axes=a.axes, statuses=["failed_gate"] if a.gate_failed else None)
