# Resumable pipeline. CPU stages run anywhere; GPU stages are SLURM jobs (see slurm/). Every stage skips work already on disk.
# Usage: make help
PY = PYTHONPATH=src python -W ignore
MODELS = gemma medgemma meditron mistral llama med42
CLASSIFIERS = llama_guard wildguard polyguard qwen3guard

help:
	@echo "make test | items | dialects | merge | check | sbatch-render LANG | sbatch-gen KEY EXP | sbatch-score | sbatch-clf NAME | layered | analyze | confusion"

test:
	python -m pytest -q

items:
	$(PY) scripts/01_sample_items.py

merge:                       # baseline + dialect + gated LLM shards -> data/rendered.jsonl, data_quality_log
	$(PY) scripts/02e_merge_rendered.py

check:                       # section 17 sanity checks
	$(PY) scripts/check_results.py

layered:                     # scored_exp1 + classifier flags -> layered_exp1.jsonl
	$(PY) scripts/05_classify.py --merge

analyze:
	$(PY) scripts/07_analyze.py

confusion:                   # Experiment 2: confidence as a trust flag (TPR, FNR, FPR, TNR at tau 0.8; 0.6 and 0.9 as sensitivity)
	$(PY) scripts/06b_confusion.py

# GPU stages (submit one at a time, in this order):
#   sbatch slurm/serve_render.sbatch --language Spanish|Hindi|Swahili   (render translation + code-switch)
#   sbatch slurm/serve_judge.sbatch gpt_oss_120b|aya_expanse_32b|tiny_aya_global --round 1   then  python scripts/02d_gate_switch.py --round 1
#   sbatch slurm/serve_model.sbatch $(MODELS) exp1   and   exp2
#   sbatch slurm/serve_judge_exp1.sbatch                                (04_score)
#   sbatch slurm/serve_classifier.sbatch $(CLASSIFIERS)
