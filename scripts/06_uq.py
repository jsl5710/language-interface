import pandas as pd, yaml
from dlsd.uq import ece, brier, auroc, selective_accuracy, overconfident_error_rate
from dlsd.io import read_jsonl, write_table, write_manifest, to_long, METRIC_COLS
from dlsd.analysis import bootstrap_ci
import numpy as np
cfg = yaml.safe_load(open("configs/experiment.yaml")); u = cfg["uncertainty"]
import glob
from dlsd.uq import apply_verbalized
df = pd.DataFrame(read_jsonl("results/raw/generations_exp2.jsonl"))
regen = [r for f in sorted(glob.glob("results/raw/verbalized_exp2*.jsonl")) for r in read_jsonl(f)]
df, n_missing_raw = apply_verbalized(df, regen)      # one parser for every model, applied to the stored raw replies (docs/DEVIATIONS.md 2026-11-14)
print(f"rows without a stored verbalised reply (old parse kept): {n_missing_raw}")
keys = ["model_group", "model_id", "condition_axis", "condition_level", "language_pair"]
rows = []
for est in ["conf_prob", "conf_verbalized", "conf_sampling"]:
    d = df.dropna(subset=[est])
    for k, g in d.groupby(keys, dropna=False):
        c, y = g[est].values.astype(float), g["correct"].astype(int).values
        rec = dict(zip(keys, k)) | {"estimator": est, "n": len(g), "accuracy": y.mean(), "mean_conf": c.mean(),
            "ece": ece(c, y, u["ece_bins"]), "brier": brier(c, y), "auroc": auroc(c, y),
            "sel_acc": selective_accuracy(c, y, u["selective_coverage"]),
            "overconf_err": overconfident_error_rate(c, y, u["overconfident_threshold"])}
        _, lo, hi = bootstrap_ci(np.arange(len(g)), stat=lambda idx: ece(c[idx.astype(int)], y[idx.astype(int)], u["ece_bins"]))
        rec |= {"ece_ci_low": lo, "ece_ci_high": hi}
        rows.append(rec)
wide = pd.DataFrame(rows)
meta = {"experiment": "exp2_uncertainty", "ece_bins": u["ece_bins"], "selective_coverage": u["selective_coverage"],
        "overconfident_threshold": u["overconfident_threshold"], "bootstrap_resamples": 1000, "verbalized_rows_without_raw": n_missing_raw}
write_table(wide, "exp2_uq_by_condition_wide", meta)
write_table(to_long(wide, keys + ["estimator", "n"], METRIC_COLS), "exp2_uq_by_condition_long", meta)
write_manifest(cfg, {"stage": "06_uq"})
