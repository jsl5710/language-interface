import json, yaml, difflib
from importlib import import_module
import sys
sys.path.insert(0, "scripts")
m = import_module("02a_render_dialects")
cfg = yaml.safe_load(open("configs/experiment.yaml"))
items = {i["item_id"]: i for i in map(json.loads, open("data/items.jsonl"))}
stored = {r["item_id"]: r for r in map(json.loads, open("data/dialect/rendered_aave.jsonl"))}
for iid in ["mmedbench-1185"]:
    outs = [m.render_row(items[iid], "AAVE", cfg) for _ in range(3)]
    print(iid, "same within process:", len({o["rendered"] for o in outs}) == 1, "| equals stored:", outs[0]["rendered"] == stored[iid]["rendered"])
    a, b = stored[iid]["rendered"], outs[0]["rendered"]
    print("stored mode/rules:", stored[iid].get("transform_mode"), stored[iid].get("rules_fired")); print("new    mode/rules:", outs[0].get("transform_mode"), outs[0].get("rules_fired"))
    for l in difflib.unified_diff(a.split(". "), b.split(". "), lineterm="", n=0): print(l[:240])
