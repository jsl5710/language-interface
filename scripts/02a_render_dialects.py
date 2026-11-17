"""Stage 2a: rule-based dialect renderings with Multi-VALUE (no language model). CPU only.

    python scripts/02a_render_dialects.py --dialect "Jamaican English"   # one shard, resumable, appends row by row
    python scripts/02a_render_dialects.py --summarize                    # merge shards, write quality log and spot-check sample
    add --dry-run to process 5 items into data/dry_run/

Each row renders the question stem and, for multiple-choice items, every option text. Letters and the answer label are
never touched (docs/EXPERIMENTS.md section 3.2). Rules fired and the changed flag are recorded per stem and per option,
so a dialect that leaves an item unchanged is visible (an unchanged item makes that condition identical to SAE).
"""
import argparse
import json
import random
import re
import statistics
from pathlib import Path

import pandas as pd
import yaml

from dlsd.dialect_rules import numbers_intact, transform
from dlsd.io import write_table

SHARD_DIR = Path("data/dialect")
NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def numbers_preserved(src: str, out: str) -> bool:
    """Doses, counts and lab values must survive: every number token in the source must appear in the rendering (multiset)."""
    return numbers_intact(src, out)


def render_row(item: dict, dialect: str, cfg: dict) -> dict:
    """Never raises: a failed transform is stored with error set, rendered null, and counted in the quality log."""
    try:
        return _render_row(item, dialect, cfg)
    except Exception as e:  # noqa: BLE001 (we want every failure recorded, whatever its type)
        return {"item_id": item["item_id"], "source": item["source"], "condition_axis": "dialect", "condition_level": dialect,
                "language_pair": None, "switch_type": None, "method": "multivalue", "rendered": None, "rendered_options": None,
                "rules_fired": None, "dialect_changed": None, "numbers_preserved": None, "protected": None,
                "error": f"{type(e).__name__}: {e}"[:200]}


def _render_row(item: dict, dialect: str, cfg: dict) -> dict:
    stem = transform(item["text"], dialect, cfg)
    row = {"item_id": item["item_id"], "source": item["source"], "condition_axis": "dialect", "condition_level": dialect,
           "language_pair": None, "switch_type": None, "method": "multivalue",
           "multivalue_commit": cfg["rendering"]["dialect"]["commit"].split()[0],
           "rendered": stem.text, "rules_fired": stem.rules_fired, "dialect_changed": stem.changed, "protected": stem.protected, "transform_mode": stem.mode,
           "n_sentence_failures": stem.n_sentence_failures, "error": None,
           "rendered_options": None, "option_rules_fired": None, "options_changed": None}
    ok = numbers_preserved(item["text"], stem.text)
    if item["options"]:
        ro, rules, ch = {}, {}, {}
        for letter, text in item["options"].items():
            r = transform(text, dialect, cfg)
            ro[letter], rules[letter], ch[letter] = r.text, r.rules_fired, r.changed
            ok = ok and numbers_preserved(text, r.text)
        row.update(rendered_options=ro, option_rules_fired=rules, options_changed=ch)
    row["numbers_preserved"] = ok
    row["number_check_v2"] = True
    return row


def render(args, cfg):
    items = [json.loads(l) for l in open("data/items.jsonl")]
    out_dir = Path("data/dry_run") if args.dry_run else SHARD_DIR   # dry run: 5 items, separate folder
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        items = items[:5]
    out_path = out_dir / f"rendered_{slug(args.dialect)}.jsonl"
    if out_path.exists() and not args.dry_run:      # earlier failed rows are set aside (kept for the log) and redone
        rows = [json.loads(l) for l in open(out_path)]
        by_id = {it["item_id"]: it for it in items}
        for r in rows:        # rows rendered before the strict number check: redo any that changed a number
            if not r.get("error") and r["rendered"] is not None and not r.get("number_check_v2"):
                it = by_id[r["item_id"]]
                pairs = [(it["text"], r["rendered"])] + [(it["options"][k], v) for k, v in (r["rendered_options"] or {}).items()]
                if not all(numbers_intact(a, b) for a, b in pairs):
                    r["error"] = "NumberChanged: redo with numbers masked"
        bad = [r for r in rows if r.get("error")]
        if bad:
            with open(out_dir / f"errors_{slug(args.dialect)}.jsonl", "a") as ef:
                ef.writelines(json.dumps(r) + "\n" for r in bad)
            out_path.write_text("".join(json.dumps(r) + "\n" for r in rows if not r.get("error")))
    done = {json.loads(l)["item_id"] for l in open(out_path)} if out_path.exists() else set()
    todo = [it for it in items if it["item_id"] not in done]
    print(f"{args.dialect}: {len(done)} done, {len(todo)} to do -> {out_path}", flush=True)
    with open(out_path, "a") as f:
        for n, it in enumerate(todo, 1):
            f.write(json.dumps(render_row(it, args.dialect, cfg)) + "\n")
            f.flush()
            if n % 20 == 0:
                print(f"  {n}/{len(todo)}", flush=True)


def summarize(cfg):
    rows = [json.loads(l) for p in sorted(SHARD_DIR.glob("rendered_*.jsonl")) for l in open(p)]
    df = pd.DataFrame(rows)
    recs = []
    for (d, src), all_g in df.groupby(["condition_level", "source"]):
        g = all_g[all_g["error"].isna()]
        recs.append({"condition_axis": "dialect", "condition_level": d, "source": src, "n": len(all_g), "n_errors": int(len(all_g) - len(g)),
                     "n_protected": int(g["protected"].fillna(False).astype(bool).sum()),
                     "n_sentence_fallback": int((g.get("transform_mode") == "sentence").sum()) if "transform_mode" in g else 0,
                     "n_sentences_unchanged_by_failure": int(g["n_sentence_failures"].fillna(0).sum()) if "n_sentence_failures" in g else 0,
                     "share_stem_unchanged": float((~g["dialect_changed"].astype(bool)).mean()),
                     "mean_rules_fired_stem": float(statistics.mean(len(r) for r in g["rules_fired"])),
                     "share_numbers_preserved": float(g["numbers_preserved"].astype(bool).mean())})
    write_table(pd.DataFrame(recs), "data_quality_dialect",
                {"experiment": "both", "note": "Rule-based Multi-VALUE renderings; unchanged items equal SAE",
                 "multivalue_commit": cfg["rendering"]["dialect"]["commit"].split()[0]})
    # Spot-check sample: 20 renderings per dialect, kept local (contains harmful requests).
    rng = random.Random(cfg["seed"])
    sample = []
    for d, g in df[df["error"].isna()].groupby("condition_level"):
        idx = rng.sample(range(len(g)), min(20, len(g)))
        sample += g.iloc[idx][["item_id", "source", "condition_level", "rendered", "rules_fired", "dialect_changed"]].to_dict("records")
    Path("data/spotcheck_dialect.jsonl").write_text("\n".join(json.dumps(s) for s in sample))
    print(pd.DataFrame(recs).to_string())


def check_determinism(cfg, n=20):
    """Section 17 check 12: re-render n random items per dialect and compare with the stored shard (identical text required)."""
    items = {it["item_id"]: it for it in (json.loads(l) for l in open("data/items.jsonl"))}
    rng, report = random.Random(cfg["seed"]), []
    for d in cfg["conditions"]["dialects"]:
        stored = {r["item_id"]: r for r in (json.loads(l) for l in open(SHARD_DIR / f"rendered_{slug(d)}.jsonl")) if r.get("rendered")}
        pick = rng.sample(sorted(stored), min(n, len(stored)))
        diff = [i for i in pick if render_row(items[i], d, cfg)["rendered"] != stored[i]["rendered"]]
        report.append({"dialect": d, "checked": len(pick), "differing": len(diff), "item_ids": diff})
        print(report[-1], flush=True)
    write_table(pd.DataFrame(report), "dialect_determinism", {"experiment": "both"})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialect")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check-determinism", action="store_true")
    ap.add_argument("--config", default="configs/experiment.yaml")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    check_determinism(cfg) if a.check_determinism else summarize(cfg) if a.summarize else render(a, cfg)
