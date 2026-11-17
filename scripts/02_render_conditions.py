"""Render every item into the 18 conditions.
Dialects: rule-based (Multi-VALUE, dialect_rules.py). Languages: translation. Code-switching: language model.
Still to add: automatic checks, regeneration, refusal logging, resumable writer (docs/EXPERIMENTS.md section 3.2)."""
import json, yaml
from tqdm import tqdm
from dlsd.conditions import build_conditions
from dlsd.render import render_item
from dlsd.backends import Backend
cfg = yaml.safe_load(open("configs/experiment.yaml"))
conds = build_conditions(cfg)
rc = cfg["rendering"]
translator = Backend(rc["translation"]["model"], base_url=rc["translation"]["base_url"])
code_switcher = Backend(rc["code_switch"]["model"], base_url=rc["code_switch"]["base_url"])
items = [json.loads(l) for l in open("data/items.jsonl")]
with open("data/rendered.jsonl", "w") as f:
    for it in tqdm(items):
        for c in conds:
            r = render_item(c, it["text"], cfg, translator, code_switcher)
            f.write(json.dumps({**it, **r, "condition_axis": c.axis, "condition_level": c.level,
                                "language_pair": c.language, "switch_type": c.switch_type}) + "\n")
