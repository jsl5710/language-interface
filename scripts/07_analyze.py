"""Stage 7: all tables and figures for the paper (docs/EXPERIMENTS.md section 10) with bootstrap intervals.

    python scripts/07_analyze.py            # reads results/scored/{scored,layered}_exp1.jsonl, results/raw/generations_exp2.jsonl
Writes results/aggregate/*.csv|json and figs/*.pdf. Parts whose inputs do not exist yet are skipped and listed."""
import json
from pathlib import Path

import pandas as pd
import yaml

from dlsd import analysis as A
from dlsd import figures as F
from dlsd.io import write_manifest, write_table


def rd(p):
    p = Path(p)
    return pd.DataFrame([json.loads(l) for l in open(p) if l.strip()]) if p.exists() else pd.DataFrame()


def exp1(cfg, meta, skipped):
    scored, layered = rd("results/scored/scored_exp1.jsonl"), rd("results/scored/layered_exp1.jsonl")
    if scored.empty:
        skipped.append("Experiment 1 (no scored_exp1.jsonl)"); return
    cells = A.cell_table(A.pool_groups(A.exp1_frames(scored, layered)))
    write_table(cells, "exp1_all_metrics_long", {**meta, "experiment": "exp1_safety"})
    write_table(A.worst_case(cells), "exp1_worst_case_gap", {**meta, "experiment": "exp1_safety"})
    gm = A.group_means(cells[cells.model_id.str.startswith("group:") & (cells.model_group != "all")], "harmful_compliance_rate")
    lm = A.group_means(cells[cells.model_id.str.startswith("group:") & (cells.model_group != "all")], "residual_risk_rate")
    write_table(pd.concat([gm, lm.assign(metric="residual_risk_rate")]), "tab_caseresults", {**meta, "note": "mean over levels in each axis group; level_min/level_max give the range"})
    dialects = cfg["conditions"]["dialects"]
    write_table(F.fig_compliance(cells, dialects, "figs/fig_case_compliance.pdf"), "fig_case_compliance_data", meta)
    write_table(F.fig_residual(cells, "figs/fig_case_residual.pdf"), "fig_case_residual_data", meta)
    write_table(F.fig_gap(cells, "figs/fig_case_gap.pdf"), "fig_case_gap_data", meta)
    write_table(A.direction_comparison(cells, ["harmful_compliance_rate", "residual_risk_rate", "false_refusal_rate", "instruction_following_accuracy", "answer_accuracy"]),
                "codeswitch_direction_comparison", {**meta, "note": "English-base vs native-base code-switching vs full translation, per language and switch type; differences are point estimates"})
    car = A.caribbean(cells, ["harmful_compliance_rate", "false_refusal_rate", "instruction_following_accuracy", "residual_risk_rate"])
    write_table(car, "caribbean_jamaican_english", {**meta, "verdict_rule": "larger if above the max of the other four dialect gaps, smaller if below the min, else similar"})


def exp2(cfg, meta, skipped):
    gens = rd("results/raw/generations_exp2.jsonl")
    if gens.empty:
        skipped.append("Experiment 2 (no generations_exp2.jsonl)"); return
    tol = cfg["uncertainty"]["pattern_tolerance"]
    ests = ["conf_prob", "conf_verbalized", "conf_sampling"]
    ax = A.uq_axis_table(gens, ests, tol)
    write_table(ax, "tab_uqpatterns", {**meta, "pattern_tolerance": tol, "rule": "tolerance in probability units; change smaller than the tolerance counts as 'holds'"})
    uq = pd.DataFrame(json.load(open("results/aggregate/exp2_uq_by_condition_wide.json"))["records"]) if Path("results/aggregate/exp2_uq_by_condition_wide.json").exists() else pd.DataFrame()
    if not uq.empty:
        uq["group"] = uq["condition_axis"].map(A.GROUP_OF)
        write_table(uq.groupby(["model_group", "estimator", "group"])["ece"].mean().reset_index(), "tab_uqresults", {**meta, "note": "mean ECE over cells in each axis group"})
    else:
        skipped.append("tab_uqresults (run scripts/06_uq.py first)")
    write_table(F.fig_uq(gens, "conf_prob", "figs/fig_case_uq.pdf"), "fig_uq_data", {**meta, "estimator": "conf_prob"})
    write_table(pd.concat([A.item_stability(gens, e) for e in ests]), "exp2_item_stability", meta)


if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/experiment.yaml"))
    meta = {"bootstrap_resamples": A.N_BOOT, "bootstrap_seed": A.SEED, "min_n_for_interval": A.MIN_N, "experiment": "both"}
    skipped: list = []
    exp1(cfg, meta, skipped)
    exp2(cfg, meta, skipped)
    write_manifest(cfg, {"stage": "07_analyze", "skipped": skipped})
    print("skipped:", skipped or "none")
