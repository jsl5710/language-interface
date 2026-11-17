"""Stage 2g: sort failed renderings into false failures (flip to pass), real failures (re-run) and low-similarity cases (review).

    python scripts/02g_triage_failures.py --dir data/pilot_llm      # writes <dir>/triage.jsonl and prints the table

Only renderings that exist are triaged. The checks are recomputed with the current rules (decimal comma, spelled-out 1 to 10,
localised named terms); the stored LaBSE similarity is the marker:
  flip    all hard checks pass now and similarity >= semantic_similarity_min       -> counted as passed (flag `triage_flip`)
  review  checks pass now but similarity < threshold                               -> a person reads it (real translation problem or threshold too strict)
  rerun   still failing a hard check, or no rendering exists                       -> render again
Nothing is flipped by hand: a row is flipped only by the rule above, and the reason is stored.
"""
import argparse
import json
from pathlib import Path

import yaml

from dlsd import checks
from dlsd.gate_io import read_rows


def recheck(row, item, sim_min):
    lang, axis = row["language_pair"], row["condition_axis"]
    sim = (row.get("checks") or {}).get("semantic_sim")
    if axis == "resource":
        src = [item["text"]] + list((item["options"] or {}).values())
        out = " ".join([row["rendered"]] + list((row.get("rendered_options") or {}).values()))
        numbers = all(checks.numbers_preserved(p, out, lang) for p in src)
        missing = checks.missing_terms(src, out, lang)
        return numbers, missing, sim
    numbers = checks.numbers_preserved(item["text"], row["rendered"], lang)
    return numbers, [], sim


def triage(dir_, sim_min):
    items = {json.loads(l)["item_id"]: json.loads(l) for l in open("data/items.jsonl")}
    out = []
    for p in sorted(Path(dir_).glob("rendered_*.jsonl")):
        for r in read_rows(p):
            if r["passed"]:
                continue
            ck = r.get("checks") or {}
            if not r.get("rendered"):
                out.append({"file": p.name, "item_id": r["item_id"], "decision": "rerun", "why": "no rendering (" + (r["attempt_log"][-1] if r["attempt_log"] else "?") + ")"})
                continue
            numbers, missing, sim = recheck(r, items[r["item_id"]], sim_min)
            other_fail = [k for k in ("changed_ok", "langid_ok", "no_repeated_content", "options_letters_ok") if ck.get(k) is False]
            if not numbers or other_fail:
                d, why = "rerun", ("numbers lost" if not numbers else "") + ",".join(other_fail)
            elif sim is not None and sim < sim_min:
                d, why = "review", f"similarity {sim:.3f} below {sim_min}"
            else:
                d, why = "flip", ("localised terms " + ",".join(missing) if missing else "decimal comma / spelled-out number") + (f", sim {sim:.3f}" if sim else "")
            out.append({"file": p.name, "item_id": r["item_id"], "decision": d, "why": why, "similarity": sim})
    export_review_queue(out, dir_)
    (Path(dir_) / "triage.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in out) + "\n")
    return out


def export_review_queue(triaged, dir_):
    """Items below the similarity threshold go to a bilingual reviewer. The reviewer fills `decision` (pass or fail) in
    data/review/decisions.jsonl; merge.py then counts a `pass` as rendered. Until then these rows stay out (status needs_review)."""
    rows = []
    for x in triaged:
        if x["decision"] != "review":
            continue
        r = next(r for r in read_rows(Path(dir_) / x["file"]) if r["item_id"] == x["item_id"])
        item = next(json.loads(l) for l in open("data/items.jsonl") if json.loads(l)["item_id"] == x["item_id"])
        rows.append({"condition_id": f"{r['condition_axis']}|{r['condition_level']}|{r['language_pair']}", "item_id": x["item_id"],
                     "language": r["language_pair"], "similarity": x["similarity"], "original": item["text"], "rendered": r["rendered"],
                     "decision": None, "reviewer": None, "note": None})
    out = Path("data/review"); out.mkdir(parents=True, exist_ok=True)
    (out / "review_queue.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + ("\n" if rows else ""))
    import csv
    with open(out / "review_queue.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["item_id"]); w.writeheader(); w.writerows(rows)
    print(f"review queue: {len(rows)} items -> data/review/review_queue.csv (fill `decision` with pass or fail, save as data/review/decisions.jsonl)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/pilot_llm")
    ap.add_argument("--config", default="configs/experiment.yaml")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    res = triage(a.dir, cfg["rendering"]["semantic_similarity_min"])
    for x in res:
        print(f"{x['decision']:7s} {x['file'][9:-6]:32s} {x['item_id']:16s} {x['why']}")
    print({d: sum(x["decision"] == d for x in res) for d in ("flip", "review", "rerun")})
