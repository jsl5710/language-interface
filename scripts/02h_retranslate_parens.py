"""Stage 2h: re-translate the resource-level items that hold a number in parentheses after a quantity ("101.8°F (38.8°C)").

NLLB dropped such conversions; translator.py now holds them out and restores them. Old rows are archived to
data/archive/resource_before_paren_fix/ and replaced by the new row for the same item (round 1), so the shard stays one row per item.
    python scripts/02h_retranslate_parens.py
"""
import json
import re
import shutil
from pathlib import Path

import yaml

from dlsd.checks import Similarity
from dlsd.conditions import build_conditions
from dlsd.gate_io import read_rows
from dlsd.llm_render import render_translation
from dlsd.translator import NLLBTranslator, strip_value_parens

cfg = yaml.safe_load(open("configs/experiment.yaml"))
rc = cfg["rendering"]
items = {json.loads(l)["item_id"]: json.loads(l) for l in open("data/items.jsonl")}
affected = {i for i, it in items.items() if any(strip_value_parens(t)[1] for t in [it["text"]] + list((it["options"] or {}).values()))}
print(len(affected), "items hold a value in parentheses")
sim, tr = Similarity(rc["similarity_model"]), NLLBTranslator()
arch = Path("data/archive/resource_before_paren_fix"); arch.mkdir(parents=True, exist_ok=True)
for cond in [c for c in build_conditions(cfg) if c.axis == "resource"]:
    path = Path(f"data/llm/rendered_{'_'.join(x for x in (cond.axis, cond.level, cond.language)).lower()}.jsonl")
    if not (arch / path.name).exists():             # the archive keeps the ORIGINAL shard even if this script is run again
        shutil.copy(path, arch / path.name)
    rows = read_rows(arch / path.name)               # always start from the original rows
    before = sum(r["passed"] for r in rows)
    for r in rows:
        if r["item_id"] not in affected:
            continue
        it = items[r["item_id"]]
        o = render_translation(cond, it["text"], it["options"], tr, sim, rc["semantic_similarity_min"])
        r.update({"rendered": o.rendered, "rendered_options": o.rendered_options, "passed": o.passed, "attempts": o.attempts,
                  "attempt_log": o.attempt_log, "checks": o.checks, "paren_fix": True})
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    print(cond.cid, f"passed {before} -> {sum(r['passed'] for r in rows)} of {len(rows)}", flush=True)
