import json, yaml
from dlsd.sampling import sample_all
cfg = yaml.safe_load(open("configs/experiment.yaml"))
items = sample_all(cfg)
with open("data/items.jsonl", "w") as f:
    for it in items: f.write(json.dumps(it) + "\n")
print(len(items), "items")
