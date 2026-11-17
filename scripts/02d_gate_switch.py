"""Stage 2d: combine the judges' blind labels with the heuristic vote into a pass/fail per code-switched rendering.

    python scripts/02d_gate_switch.py --round 1       # writes data/gate/gate_<condition>.jsonl, prints pass rates
    python scripts/02d_gate_switch.py --summarize     # judge agreement (Cohen's kappa) and pass rates -> results/aggregate

A rendering passes when at least `votes_needed` of the votes (both of the 2 judges) say the observed type equals the intended type. A rendering that
failed the earlier hard checks (no usable text) fails without judging. Failures feed `02b_render_llm.py --round N+1`.
"""
import argparse
import json

import pandas as pd
import yaml

from dlsd.checks import switch_type_ok
from dlsd.gate_io import GATE_DIR, JUDGE_DIR, codeswitch_shards, latest_by_item, read_rows
from dlsd.io import write_table
from dlsd.switch_judge import cohen_kappa, gate


def run_gate(cfg, round_no):
    sj = cfg["switch_judges"]
    GATE_DIR.mkdir(parents=True, exist_ok=True)
    for shard in codeswitch_shards():
        rows = [r for r in read_rows(shard) if r.get("round", 1) == round_no]
        if not rows:
            continue
        labels = {j["name"]: {r["item_id"]: r["label"] for r in read_rows(JUDGE_DIR / j["name"] / shard.name.replace("rendered_", "judge_"))
                              if r["round"] == round_no} for j in sj["panel"]}
        out = []
        for r in rows:
            if not r["rendered"]:       # includes inter items with no natural clause boundary (not_renderable): reported, never re-rendered
                g = {"votes": [], "judge_labels": [None] * len(sj["panel"]), "heuristic_ok": None, "passed": False,
                     "not_renderable": bool(r.get("not_renderable"))}
            else:
                jl = [labels[j["name"]].get(r["item_id"]) for j in sj["panel"]]
                h = switch_type_ok(r["rendered"], r["language_pair"], r["switch_type"]) if sj["heuristic_vote"] else None
                g = gate(r["switch_type"], jl, h, sj["votes_needed"])
            out.append({"item_id": r["item_id"], "round": round_no, "language_pair": r["language_pair"],
                        "switch_type": r["switch_type"], "judges": [j["name"] for j in sj["panel"]], **g})
        path = GATE_DIR / shard.name.replace("rendered_", "gate_")
        keep = [x for x in read_rows(path) if x["round"] != round_no]       # rerunning a round replaces that round only
        path.write_text("\n".join(json.dumps(x) for x in keep + out) + "\n")
        print(f"{shard.name} round {round_no}: {sum(x['passed'] for x in out)}/{len(out)} passed")


def summarize(cfg):
    names = [j["name"] for j in cfg["switch_judges"]["panel"]]
    recs = []
    for shard in codeswitch_shards():
        gate_rows = read_rows(GATE_DIR / shard.name.replace("rendered_", "gate_"))
        if not gate_rows:
            continue
        final = latest_by_item(gate_rows)
        r1 = [g for g in gate_rows if g["round"] == 1]
        cols = [[g["judge_labels"][k] for g in r1] for k in range(len(names))]
        a, b = cols[0], cols[1]
        pairs = {f"kappa_{names[i]}_{names[j]}": cohen_kappa(cols[i], cols[j]) for i in range(len(cols)) for j in range(i + 1, len(cols))}
        lang, st = gate_rows[0]["language_pair"], gate_rows[0]["switch_type"]
        recs.append({"condition_axis": "codeswitch_native" if shard.name.startswith("rendered_codeswitch_native") else "codeswitch", "condition_level": st, "language_pair": lang, "switch_type": st,
                     "n": len(final), "pass_round1": float(sum(g["passed"] for g in r1) / max(len(r1), 1)),
                     "pass_final": float(sum(g["passed"] for g in final.values()) / len(final)),
                     "rounds_used_mean": float(sum(g["round"] for g in final.values()) / len(final)),
                     "judge_unparseable_rate": float(sum(x is None for c in cols for x in c) / max(sum(len(c) for c in cols), 1)),
                     "judge_all_agree": float(sum(len(set(t)) == 1 for t in zip(*cols)) / max(len(a), 1)), **pairs,
                     "judges": ",".join(names)})
    df = pd.DataFrame(recs)
    write_table(df, "data_quality_switch_gate", {"experiment": "both", "judges": cfg["switch_judges"]["panel"],
                                                 "votes_needed": cfg["switch_judges"]["votes_needed"], "heuristic_vote": cfg["switch_judges"]["heuristic_vote"]})
    print(df.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--config", default="configs/experiment.yaml")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    summarize(cfg) if a.summarize else run_gate(cfg, a.round)
