"""Stage 6b: confidence as a trust flag (Experiment 2; added after the run, docs/DEVIATIONS.md 2026-11-15).

    python scripts/06b_confusion.py
Reads results/raw/generations_exp2.jsonl and applies the same verbalised-confidence re-parse as 06_uq.py (apply_verbalized with
results/raw/verbalized_exp2*.jsonl). Writes results/aggregate/exp2_confusion_by_condition (long), exp2_confusion_gap_from_sae and
fig_confusion_data. Primary tau = uncertainty.overconfident_threshold; 0.6 and 0.9 are sensitivity rows."""
import glob

import pandas as pd
import yaml

from dlsd.analysis import N_BOOT, SEED
from dlsd.confusion import RATES, confusion_table, figure_data, paired_gaps
from dlsd.io import read_jsonl, to_long, write_manifest, write_table
from dlsd.uq import apply_verbalized

SENSITIVITY_TAUS = [0.6, 0.9]
ESTIMATORS = ["conf_prob", "conf_verbalized", "conf_sampling"]

if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/experiment.yaml"))
    tau0 = cfg["uncertainty"]["overconfident_threshold"]
    taus = [tau0] + SENSITIVITY_TAUS
    df = pd.DataFrame(read_jsonl("results/raw/generations_exp2.jsonl"))
    regen = [r for f in sorted(glob.glob("results/raw/verbalized_exp2*.jsonl")) for r in read_jsonl(f)]
    df, n_missing_raw = apply_verbalized(df, regen)
    print(f"rows without a stored verbalised reply (old parse kept): {n_missing_raw}")
    keys = ["model_group", "model_id", "condition_axis", "condition_level", "language_pair"]
    wide = confusion_table(df, keys, ESTIMATORS, taus, N_BOOT, SEED)
    gaps = paired_gaps(df, ["condition_axis", "condition_level", "language_pair"], ESTIMATORS, taus)
    meta = {"experiment": "exp2_confusion", "positive_class": "correct answer", "flag": "confidence >= tau", "primary_tau": tau0,
            "sensitivity_taus": SENSITIVITY_TAUS, "bootstrap_resamples": N_BOOT, "bootstrap_seed": SEED, "low_n_rule": "n_correct or n_wrong below 10",
            "added": "2026-11-15, after the run; exploratory beyond the section 6.3 measures", "verbalized_rows_without_raw": n_missing_raw,
            "flag_unparseable": "Meditron-7B conf_verbalized (85% of replies unparseable)"}
    id_cols = keys + ["estimator", "tau", "n", "n_correct", "n_wrong", "low_n", "flag_unparseable"]
    long = to_long(wide, id_cols, list(RATES))
    ci = {m: wide[keys + ["estimator", "tau", m + "_ci_low", m + "_ci_high"]].rename(columns={m + "_ci_low": "ci_low", m + "_ci_high": "ci_high"}).assign(metric=m) for m in RATES}
    long = long.merge(pd.concat(ci.values()), on=keys + ["estimator", "tau", "metric"], how="left")
    write_table(long, "exp2_confusion_by_condition", meta)
    write_table(gaps, "exp2_confusion_gap_from_sae", meta | {"pairing": "items with a non-null confidence in both conditions of the same model; joint item resample (analysis.gap_ci)"})
    write_table(figure_data(wide, tau0), "fig_confusion_data", meta | {"note": "mean over the conditions of each axis at the primary tau; incl_flagged False drops the flag_unparseable cells"})
    write_manifest(cfg, {"stage": "06b_confusion"})
    print(f"{len(wide)} cells, {len(gaps)} gap rows, {int(wide['low_n'].sum())} low_n cells")
