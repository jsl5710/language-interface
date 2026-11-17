"""Fixed-seed item sampling. Same items are reused in every condition.

Every loader returns dicts with: item_id, source, text, options (dict letter -> text, MCQ only), answer (letter, MCQ only),
principle (safety items only). Field mappings were checked against one real row of each dataset (see tests/test_sampling.py).
"""
import csv
import glob
import json
import random
import re
import zipfile
from pathlib import Path

LETTERS = "ABCDEFGHIJ"


def _stratified(items: list, n: int, rng: random.Random, key: str) -> list:
    """Take n items, as evenly as possible across the groups in `key`; remainder goes to the largest groups, in sorted key order."""
    groups: dict = {}
    for it in items:
        groups.setdefault(it.get(key) or "NA", []).append(it)
    keys = sorted(groups)
    base, extra = divmod(n, len(keys))
    out = []
    for i, k in enumerate(keys):
        take = base + (1 if i < extra else 0)
        out += rng.sample(groups[k], min(take, len(groups[k])))
    rng.shuffle(out)
    return out


def _sample(items: list, n: int, seed, stratify_by: str | None = None) -> list:
    rng = random.Random(seed)
    if stratify_by:
        return _stratified(items, n, rng, stratify_by)
    return rng.sample(items, min(n, len(items)))


def load_medsafetybench(spec: dict) -> list:
    """MedSafetyBench (Han et al. 2024): one CSV per AMA principle (1 to 9), column `harmful_medical_request`."""
    files = sorted(glob.glob(str(Path(spec["path"]) / "*.csv")))
    out = []
    for f in files:
        principle = "principle_" + re.search(r"category_(\d+)", f).group(1)
        for r in csv.DictReader(open(f, newline="")):
            idx = r[""] if "" in r else str(len(out))
            out.append({"item_id": f"{spec['name']}-{principle}-{idx}", "source": spec["name"],
                        "text": r["harmful_medical_request"].strip(), "options": None, "answer": None, "principle": principle})
    return out


def load_mmedbench(spec: dict) -> list:
    """MMedBench (Qiu et al. 2024), English test split. Options are a dict A..D; answer is `answer_idx` (a letter)."""
    path = Path(spec["path"])
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            lines = z.read(f"MMedBench/{spec.get('split_dir', 'Test')}/{spec['language']}.jsonl").decode().splitlines()
    else:
        lines = path.read_text().splitlines()
    out = []
    for i, line in enumerate(lines):
        r = json.loads(line)
        opts = r["options"]
        if isinstance(opts, str):
            import ast
            opts = ast.literal_eval(opts)
        out.append({"item_id": f"{spec['name']}-{i}", "source": spec["name"], "text": r["question"].strip(),
                    "options": dict(opts), "answer": r["answer_idx"], "principle": None})
    return out


def load_wmdp(spec: dict) -> list:
    """WMDP (Li et al. 2024), parquet with `question`, `choices` (list of 4) and `answer` (int index)."""
    import pandas as pd
    df = pd.read_parquet(spec["path"])
    out = []
    for i, r in df.iterrows():
        choices = list(r["choices"])
        out.append({"item_id": f"{spec['name']}-{i}", "source": spec["name"], "text": r["question"].strip(),
                    "options": {LETTERS[j]: c for j, c in enumerate(choices)}, "answer": LETTERS[int(r["answer"])], "principle": None})
    return out


LOADERS = {"medsafety_csv_dir": load_medsafetybench, "mmedbench_local": load_mmedbench, "wmdp_parquet": load_wmdp}


def load_source(spec: dict) -> list:
    return LOADERS[spec["loader"]](spec)


def sample_all(cfg: dict) -> list:
    items = []
    for spec in cfg["sources"].values():
        pool = load_source(spec)
        # Seed per source so equal-sized pools do not draw identical index sets.
        items += _sample(pool, cfg["n_items_per_source"], f'{cfg["seed"]}-{spec["name"]}', "principle" if spec.get("stratify_by") else None)
    return items
