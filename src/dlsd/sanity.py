"""Automatic sanity checks (docs/EXPERIMENTS.md section 17). Each check takes plain rows and returns
{"check", "status": pass|fail|skipped|warn, "detail"}; a check whose inputs do not exist yet is "skipped", never "pass"."""
import json
import random
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np

RATE_METRICS = {"accuracy", "mean_conf", "brier", "sel_acc", "overconf_err", "harmful_compliance_rate", "refusal_rate",
                "partial_rate", "residual_risk_rate", "layered_block_rate", "input_flag_rate", "output_flag_rate", "ece", "auroc"}
KEY = ("condition_axis", "condition_level", "language_pair", "switch_type")


def res(check, status, detail=""):
    return {"check": check, "status": status, "detail": detail}


def _cid(r):
    parts = [r["condition_axis"], r["condition_level"]] + ([r["language_pair"]] if r.get("language_pair") else [])
    return "|".join(parts)


def check_1_item_set(items, rendered, gens: dict):
    ids = {it["item_id"] for it in items}
    by_cond = defaultdict(set)
    for r in rendered:
        by_cond[r["condition_id"]].add(r["item_id"])
    bad = [c for c, s in by_cond.items() if s != ids]
    for name, rows in gens.items():
        if rows:
            by_model = defaultdict(set)
            for r in rows:
                by_model[r["model_id"]].add(r["item_id"])
            bad += [f"{name}:{m}" for m, s in by_model.items() if not s <= ids]
    return res("1_item_set", "fail" if bad else "pass", f"cells with a different item set: {bad[:5]}" if bad else f"{len(ids)} items in every condition")


def check_2_baseline(items, rendered):
    orig = {it["item_id"]: it for it in items}
    bad = [r["item_id"] for r in rendered if r["condition_axis"] == "baseline"
           and (r["rendered"] != orig[r["item_id"]]["text"] or r["options"] != orig[r["item_id"]]["options"])]
    return res("2_baseline_identical", "fail" if bad else "pass", f"{len(bad)} baseline rows differ" if bad else "all baseline rows are byte-identical")


def check_3_condition_count(rendered, expected_cids):
    per_item = defaultdict(list)
    for r in rendered:
        per_item[r["item_id"]].append(r["condition_id"])
    bad = [i for i, c in per_item.items() if sorted(c) != sorted(expected_cids)]
    return res("3_condition_count", "fail" if bad else "pass", f"{len(bad)} items without exactly the {len(expected_cids)} conditions" if bad else f"{len(per_item)} items x {len(expected_cids)} conditions")


def check_4_mcq_integrity(items, rendered):
    orig = {it["item_id"]: it for it in items}
    bad = []
    for r in rendered:
        o = orig[r["item_id"]]
        if o["options"] is None:
            continue
        if r["render_status"] in ("missing", "error") or r["options"] is None:
            continue                                         # unrendered rows are counted by check 5
        if list(r["options"]) != list(o["options"]) or r["answer"] != o["answer"]:
            bad.append((r["item_id"], r["condition_id"]))
    return res("4_mcq_integrity", "fail" if bad else "pass", f"{len(bad)} rows changed letters or answer: {bad[:3]}" if bad else "option letters and answers match the originals")


def check_5_rendering_quality(rendered, quality_log):
    by = defaultdict(list)
    for r in rendered:
        by[r["condition_id"]].append(r["render_ok"])
    over = {c: round(1 - sum(v) / len(v), 3) for c, v in by.items() if 1 - sum(v) / len(v) > 0.15}
    logged = {r["condition_id"] for r in quality_log if r.get("flag_over_15pct_unrendered")}
    unflagged = [c for c in over if c not in logged]
    if unflagged:
        return res("5_rendering_quality", "fail", f"over 15% unrendered and not flagged in data_quality_log: {unflagged}")
    return res("5_rendering_quality", "warn" if over else "pass", f"conditions over 15% unrendered (flagged): {over}" if over else "every condition has at least 85% valid renderings")


from .scoring import extract_letter


def check_6_parsing(gens_exp1, gens_exp2):
    worst = []
    options = {}
    if Path("data/items.jsonl").exists():
        options = {r["item_id"]: r.get("options") for r in (json.loads(l) for l in open("data/items.jsonl") if l.strip())}
    for name, rows in (("exp1", gens_exp1), ("exp2", gens_exp2)):
        cells = defaultdict(list)
        for r in rows:
            if "parsed_letter" in r:
                none = r["parsed_letter"] is None if name == "exp2" or "response_text" not in r else extract_letter(r["response_text"], options.get(r["item_id"])) is None   # exp1: lenient reading (docs/DEVIATIONS.md 2026-11-15)
                cells[(name, r["model_id"], _cid(r))].append(none)
        worst += [(k, round(sum(v) / len(v), 3)) for k, v in cells.items() if sum(v) / len(v) > 0.20]
    if not (gens_exp1 or gens_exp2):
        return res("6_parsing", "skipped", "no generations yet")
    return res("6_parsing", "fail" if worst else "pass", f"unparseable above 20% (stop and ask the author): {worst[:5]}" if worst else "unparseable rate at most 20% in every cell")


def check_7_metric_invariants(tables: dict):
    bad = []
    for name, recs in tables.items():
        for r in recs:
            m, v, n = r.get("metric"), r.get("value"), r.get("n")
            if m in RATE_METRICS and v is not None and not (isinstance(v, float) and np.isnan(v)) and not 0 <= v <= 1:
                bad.append((name, m, v))
            if n is not None and n < 1:
                bad.append((name, "n", n))
    if not tables:
        return res("7_metric_invariants", "skipped", "no aggregate metric tables yet")
    return res("7_metric_invariants", "fail" if bad else "pass", f"{bad[:5]}" if bad else f"{len(tables)} tables checked")


def independent_ece(conf, correct, bins=10):
    """Written separately from dlsd.uq.ece on purpose (used for the hand check, section 17 check 8)."""
    n, total = len(conf), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(conf) if (lo < c <= hi) or (b == 0 and c == 0)]
        if idx:
            total += len(idx) / n * abs(sum(correct[i] for i in idx) / len(idx) - sum(conf[i] for i in idx) / len(idx))
    return total


def check_8_hand_check(gens_exp2, ece_table, bins=10):
    if not gens_exp2 or not ece_table:
        return res("8_hand_check", "skipped", "needs Experiment 2 generations and the ECE table")
    rec = next((r for r in ece_table if r.get("metric") == "ece" and r.get("estimator") == "conf_prob"), None)
    if rec is None:
        return res("8_hand_check", "skipped", "no conf_prob ECE record")
    cell = [g for g in gens_exp2 if g["model_id"] == rec["model_id"] and g["condition_axis"] == rec["condition_axis"]
            and g["condition_level"] == rec["condition_level"] and g.get("language_pair") == rec.get("language_pair")
            and g.get("conf_prob") is not None]
    mine = independent_ece([g["conf_prob"] for g in cell], [float(g["correct"]) for g in cell], bins)
    ok = abs(mine - rec["value"]) < 1e-9
    return res("8_hand_check", "pass" if ok else "fail", f"independent ECE {mine:.6f} vs table {rec['value']:.6f} for {rec['model_id']} {rec['condition_axis']}|{rec['condition_level']}")


def check_9_layered_logic(scored):
    if not scored:
        return res("9_layered_logic", "skipped", "no scored Experiment 1 rows yet")
    bad = []
    for r in scored:
        if r.get("input_flagged") is None and r.get("output_flagged") is None:
            continue
        blocked = bool(r.get("input_flagged")) or bool(r.get("output_flagged"))
        if r.get("layered_blocked") != blocked:
            bad.append(("blocked", r["item_id"]))
        if bool(r.get("residual_risk")) != (r.get("judge_label") == "comply" and not blocked):
            bad.append(("residual", r["item_id"]))
    return res("9_layered_logic", "fail" if bad else "pass", f"{len(bad)} rows violate the layered rule: {bad[:3]}" if bad else f"{len(scored)} rows consistent")


def check_10_determinism(first, rerun, n=20, seed=13):
    if not first or not rerun:
        return res("10_determinism", "skipped", "no temperature-0 re-run file yet")
    key = lambda r: (r["item_id"], r["model_id"], _cid(r))
    a = {key(r): r["response_text"] for r in first if "response_text" in r}
    b = {key(r): r["response_text"] for r in rerun if "response_text" in r}
    common = sorted(set(a) & set(b))
    pick = random.Random(seed).sample(common, min(n, len(common)))
    diff = [k for k in pick if a[k] != b[k]]
    return res("10_determinism", "warn" if diff else "pass", f"{len(diff)} of {len(pick)} re-run rows differ (log the difference; some servers are not bit-deterministic)" if diff else f"{len(pick)} rows identical")


def check_11_row_counts(rendered, gens: dict, models: list, mcq_ids: set):
    if not any(gens.values()):
        return res("11_row_counts", "skipped", "no generations yet")
    ok_rows = [r for r in rendered if r["render_ok"]]
    notes = []
    for name, rows in gens.items():
        if not rows:
            continue
        pool = [r for r in ok_rows if name != "exp2" or r["item_id"] in mcq_ids]
        expected = len(pool) * len(models)
        notes.append(f"{name}: {len(rows)} rows vs {expected} expected")
        if len(rows) != expected:
            return res("11_row_counts", "fail", "; ".join(notes) + " (explain the gap or finish the run)")
    return res("11_row_counts", "pass", "; ".join(notes))


def check_12_dialect(dialect_rows, quality, commit):
    if not dialect_rows:
        return res("12_dialect_coverage", "skipped", "no dialect shards")
    commits = {r.get("multivalue_commit") for r in dialect_rows if r.get("multivalue_commit")}
    if commits != {commit}:
        return res("12_dialect_coverage", "fail", f"pinned commit {commit} vs shards {commits}")
    if not quality:
        return res("12_dialect_coverage", "fail", "data_quality_dialect missing: unchanged share and mean rules fired not recorded")
    return res("12_dialect_coverage", "pass", f"commit {commit[:8]} on every shard; quality table has {len(quality)} cells (determinism re-run: scripts/02a --check-determinism)")


def run_all(ctx: dict) -> list:
    return [check_1_item_set(ctx["items"], ctx["rendered"], {"exp1": ctx["gens1"], "exp2": ctx["gens2"]}),
            check_2_baseline(ctx["items"], ctx["rendered"]),
            check_3_condition_count(ctx["rendered"], ctx["expected_cids"]),
            check_4_mcq_integrity(ctx["items"], ctx["rendered"]),
            check_5_rendering_quality(ctx["rendered"], ctx["quality_log"]),
            check_6_parsing(ctx["gens1"], ctx["gens2"]),
            check_7_metric_invariants(ctx["tables"]),
            check_8_hand_check(ctx["gens2"], ctx["tables"].get("exp2_uq_by_condition_long", []), ctx.get("bins", 10)),
            check_9_layered_logic(ctx["scored"]),
            check_10_determinism(ctx["gens1"], ctx["rerun1"]),
            check_11_row_counts(ctx["rendered"], {"exp1": ctx["gens1"], "exp2": ctx["gens2"]}, ctx["models"], ctx["mcq_ids"]),
            check_12_dialect(ctx["dialect_rows"], ctx["dialect_quality"], ctx["commit"])]
