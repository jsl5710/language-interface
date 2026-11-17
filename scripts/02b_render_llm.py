"""Stage 2b: translation (resource level) and code-switching renderings with Qwen3-32B served by vLLM.

    python scripts/02b_render_llm.py --language Hindi              # 4 conditions: translation + intra/inter/extra
    python scripts/02b_render_llm.py --language Hindi --dry-run    # 5 items into data/dry_run_llm/
    python scripts/02b_render_llm.py --pilot                       # 20 items x 12 conditions (section 13 step 3)
    python scripts/02b_render_llm.py --language Hindi --round 2    # re-render code-switch items that failed the judge gate in round 1
    python scripts/02b_render_llm.py --summarize                   # data_quality_llm table from the shards

Needs the renderer endpoint from configs/experiment.yaml (rendering.translation.base_url). Resumable per (item, condition);
rows are appended as they finish. Failed items are kept with passed=false and are counted, never dropped silently.
"""
import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd
import yaml

from dlsd.backends import Backend
from dlsd.checks import Similarity
from dlsd.conditions import build_conditions
from dlsd.io import write_table
from dlsd.gate_io import failed_with_feedback, read_rows
from dlsd.conditions import is_codeswitch
from dlsd.llm_render import render_codeswitch, render_codeswitch_native, render_item, render_translation
from dlsd.translator import NLLBTranslator

SHARD_DIR = Path("data/llm")


def cid_slug(cond) -> str:
    return "_".join(x for x in (cond.axis, cond.level, cond.language) if x).lower()


def render(args, cfg):
    rc = cfg["rendering"]
    base_url = rc["translation"]["base_url"]
    if os.environ.get("DLSD_RENDER_PORT"):                       # set by slurm/serve_render.sbatch (one port per job)
        base_url = re.sub(r":\d+/", f":{os.environ['DLSD_RENDER_PORT']}/", base_url)
    backend = Backend(rc["translation"]["model"], base_url,
                      extra_body={"chat_template_kwargs": {"enable_thinking": False}})   # Qwen3 non-thinking mode
    sim = Similarity(rc["similarity_model"])
    translator = NLLBTranslator() if rc.get("translator", {}).get("name") == "nllb-200-3.3B" else None   # bake-off winner; Qwen3 stays the planner
    axes = ("codeswitch", "codeswitch_native") if args.round > 1 else ("resource", "codeswitch", "codeswitch_native")   # only code-switching is gated by the judge panel
    conds = [c for c in build_conditions(cfg) if c.axis in axes and (args.language in (None, c.language))]
    items = [json.loads(l) for l in open("data/items.jsonl")]
    out_dir = SHARD_DIR
    all_items = items
    if args.dry_run or args.pilot:
        out_dir = Path("data/dry_run_llm" if args.dry_run else "data/pilot_llm")
        n = 5 if args.dry_run else 20
        by_src: dict = {}
        for it in items:                      # pilot takes the first items of each source so every source is covered
            by_src.setdefault(it["source"], []).append(it)
        per = -(-n // len(by_src))             # ceiling, then trim to exactly n
        items = [it for v in by_src.values() for it in v[:per]][:n]
    only: dict = {}
    if args.rerun_triaged:                    # re-render just the pilot items that triage marked `rerun`, into a separate folder
        for x in map(json.loads, open(Path(args.rerun_triaged) / "triage.jsonl")):
            if x["decision"] == "rerun":
                only.setdefault(x["file"], set()).add(x["item_id"])
        out_dir = Path(args.rerun_triaged + "_rerun")
        conds = [c for c in conds if f"rendered_{cid_slug(c)}.jsonl" in only]
    out_dir.mkdir(parents=True, exist_ok=True)
    for cond in conds:
        if only:
            items = [it for it in all_items if it["item_id"] in only[f"rendered_{cid_slug(cond)}.jsonl"]]
        path = out_dir / f"rendered_{cid_slug(cond)}.jsonl"
        done = {r["item_id"] for r in read_rows(path) if r.get("round", 1) == args.round}
        feedback = {}
        if args.round > 1:                      # re-render only what failed the gate last round, with the judges' labels as feedback
            feedback = failed_with_feedback(path, args.round)
            items_r = [it for it in items if it["item_id"] in feedback]
        else:
            items_r = items
        todo = [it for it in items_r if it["item_id"] not in done]
        print(f"{cond.cid} round {args.round}: {len(done)} done, {len(todo)} to do", flush=True)
        with open(path, "a") as f:
            for n_done, it in enumerate(todo, 1):
                fb = None
                if feedback:
                    labs = [l for l in feedback[it["item_id"]]["judge_labels"] if l]
                    fb = (f"your previous rewrite was classified as {', '.join(sorted(set(labs))) or 'unclassifiable'}, not "
                          f"{cond.switch_type}. Make the {cond.switch_type}-sentential pattern clear and keep the meaning unchanged.")
                if cond.axis == "codeswitch_native":                             # Qwen3 plans which English pieces stay, NLLB translates the rest
                    o = render_codeswitch_native(cond, it["text"], it["options"], backend, sim, rc["semantic_similarity_min"], rc["max_attempts"], fb, translator)
                elif cond.axis == "codeswitch":                                  # Qwen3 plans, NLLB translates the pieces
                    o = render_codeswitch(cond, it["text"], it["options"], backend, sim, rc["semantic_similarity_min"], rc["max_attempts"], fb, translator)
                elif translator is not None:
                    o = render_translation(cond, it["text"], it["options"], translator, sim, rc["semantic_similarity_min"])
                else:
                    o = render_item(cond, it["text"], it["options"], backend, sim, rc["semantic_similarity_min"], rc["max_attempts"], fb)
                f.write(json.dumps({
                    "item_id": it["item_id"], "source": it["source"], "condition_axis": cond.axis, "condition_level": cond.level,
                    "language_pair": cond.language, "switch_type": cond.switch_type, "method": "llm", "round": args.round,
                    "renderer_model": rc["translation"]["model"], "translator": rc.get("translator", {}).get("name"), "rendered": o.rendered, "rendered_options": o.rendered_options,
                    "passed": o.passed, "attempts": o.attempts, "renderer_refusals": o.refusals,
                    "attempt_log": o.attempt_log, "checks": o.checks, "plan": o.plan, "not_renderable": o.not_renderable}) + "\n")
                f.flush()
                if n_done % 20 == 0:
                    print(f"  {n_done}/{len(todo)}", flush=True)


def summarize(cfg):
    from dlsd.gate_io import latest_by_item
    rows = [r for p in sorted(SHARD_DIR.glob("rendered_*.jsonl")) for r in latest_by_item(read_rows(p)).values()]   # latest round per item
    df = pd.DataFrame(rows)
    recs = []
    for (ax, lv, lang, src), g in df.groupby(["condition_axis", "condition_level", "language_pair", "source"]):
        ck = pd.json_normalize(g["checks"].tolist())
        recs.append({"condition_axis": ax, "condition_level": lv, "language_pair": lang, "switch_type": lv if is_codeswitch(ax) else None,
                     "source": src, "n": len(g), "pass_rate": float(g["passed"].mean()),
                     "mean_attempts": float(g["attempts"].mean()), "not_renderable_rate": float(g["not_renderable"].fillna(False).astype(bool).mean()) if "not_renderable" in g else 0.0, "renderer_refusal_rate": float((g["renderer_refusals"] > 0).mean()),
                     **{f"{c}_rate": float(ck[c].mean()) for c in ("semantic_ok", "key_terms_ok", "langid_ok", "changed_ok", "switch_type_ok") if c in ck}})
    write_table(pd.DataFrame(recs), "data_quality_llm",
                {"experiment": "both", "renderer": cfg["rendering"]["translation"]["model"],
                 "semantic_similarity_min": cfg["rendering"]["semantic_similarity_min"], "max_attempts": cfg["rendering"]["max_attempts"]})
    print(pd.DataFrame(recs).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--language", choices=["Spanish", "Hindi", "Swahili"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--rerun-triaged", metavar="DIR", help="re-render the items marked rerun in DIR/triage.jsonl (see 02g_triage_failures.py)")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--config", default="configs/experiment.yaml")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    summarize(cfg) if a.summarize else render(a, cfg)
