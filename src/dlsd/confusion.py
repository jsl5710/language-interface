"""Confidence as a trust flag (Experiment 2, added after the run; docs/EXPERIMENTS.md 6.3, docs/DEVIATIONS.md 2026-11-15).

Positive class = the answer is correct. Flag = confidence >= tau (primary tau = cfg["uncertainty"]["overconfident_threshold"]; 0.6 and 0.9 are
sensitivity rows, not alternatives to choose from).
    TPR = P(flag | correct)   FNR = 1 - TPR   FPR = P(flag | wrong)   TNR = 1 - FPR
A flag the user trusts is a false positive when the answer is wrong, so FPR is the silent-failure rate among wrong answers and FNR the
share of correct answers that carry no trust flag. Every rate is a ratio of sums over items, so intervals come from resampling items
(the same helpers as 06_uq.py and the paired gaps) and a cell with no correct (or no wrong) answers has an undefined rate (NaN)."""
import numpy as np
import pandas as pd

from .analysis import GROUP_OF, bootstrap_ci, gap_ci, _ratio

RATES = ("tpr", "fnr", "fpr", "tnr")
MIN_CLASS = 10                      # fewer correct or wrong answers than this in a cell: still written, low_n = True


def _parts(conf, correct, tau):
    """Per-item numerators and denominators of the four rates as arrays."""
    conf, y = np.asarray(conf, float), np.asarray(correct, float)
    flag = (conf >= tau).astype(float)
    return {"tpr": (flag * y, y), "fnr": ((1 - flag) * y, y), "fpr": (flag * (1 - y), 1 - y), "tnr": ((1 - flag) * (1 - y), 1 - y)}


def rates(conf, correct, tau) -> dict:
    """The four rates and the class sizes. A rate whose denominator is 0 is NaN."""
    out = {k: float(_ratio(num, den)) if den.sum() > 0 else float("nan") for k, (num, den) in _parts(conf, correct, tau).items()}
    y = np.asarray(correct, float)
    return out | {"n_correct": int(y.sum()), "n_wrong": int(len(y) - y.sum())}


def cell_rates(conf, correct, tau, n_boot=1000, seed=13) -> dict:
    """Rates with 95% item-bootstrap intervals. FNR and TNR intervals are 1 minus the TPR and FPR intervals (identical for percentile
    intervals), so only two statistics are resampled."""
    c, y = np.asarray(conf, float), np.asarray(correct, int)
    out = rates(c, y, tau)
    for m in ("tpr", "fpr"):
        _, lo, hi = bootstrap_ci(np.arange(len(c)), stat=lambda idx, m=m: rates(c[idx.astype(int)], y[idx.astype(int)], tau)[m], n=n_boot, seed=seed)
        out[m + "_ci_low"], out[m + "_ci_high"] = lo, hi
    out["fnr_ci_low"], out["fnr_ci_high"] = 1 - out["tpr_ci_high"], 1 - out["tpr_ci_low"]
    out["tnr_ci_low"], out["tnr_ci_high"] = 1 - out["fpr_ci_high"], 1 - out["fpr_ci_low"]
    out["low_n"] = out["n_correct"] < MIN_CLASS or out["n_wrong"] < MIN_CLASS
    return out


def confusion_table(df: pd.DataFrame, keys: list, estimators: list, taus: list, n_boot=1000, seed=13) -> pd.DataFrame:
    """One row per (cell, estimator, tau); cells and non-null filtering exactly as in 06_uq.py."""
    rows = []
    for est in estimators:
        d = df.dropna(subset=[est])
        for k, g in d.groupby(keys, dropna=False):
            c, y = g[est].values.astype(float), g["correct"].astype(int).values
            for tau in taus:
                rows.append(dict(zip(keys, k)) | {"estimator": est, "tau": tau, "n": len(g)} | cell_rates(c, y, tau, n_boot, seed))
    out = pd.DataFrame(rows)
    out["flag_unparseable"] = (out["model_id"].str.contains("meditron")) & (out["estimator"] == "conf_verbalized")      # 85% of Meditron's verbalised replies do not parse
    return out


def paired_gaps(df: pd.DataFrame, keys: list, estimators: list, taus: list) -> pd.DataFrame:
    """Gap of each rate from SAE (baseline) on the items present, with a non-null confidence, in both conditions of the same model;
    the pairing and the joint item resample are those of analysis.gap_ci."""
    rows = []
    for est in estimators:
        d = df.dropna(subset=[est])
        for mid, g in d.groupby("model_id"):
            sae = g[g["condition_axis"] == "baseline"].set_index("item_id")
            for k, cg in g[g["condition_axis"] != "baseline"].groupby(keys, dropna=False):
                cg = cg.set_index("item_id")
                ids = sorted(set(cg.index) & set(sae.index))
                if not ids:
                    continue
                cc, cy = cg.loc[ids, est].values.astype(float), cg.loc[ids, "correct"].astype(int).values
                sc, sy = sae.loc[ids, est].values.astype(float), sae.loc[ids, "correct"].astype(int).values
                for tau in taus:
                    pc, ps = _parts(cc, cy, tau), _parts(sc, sy, tau)
                    low = min(cy.sum(), len(cy) - cy.sum(), sy.sum(), len(sy) - sy.sum()) < MIN_CLASS
                    for m in RATES:
                        v, lo, hi = gap_ci(*pc[m], *ps[m])
                        rows.append(dict(zip(keys, k)) | {"model_id": mid, "model_group": g["model_group"].iloc[0], "estimator": est, "tau": tau, "metric": m,
                                        "value": float(_ratio(*pc[m])), "sae_value": float(_ratio(*ps[m])), "gap": v, "gap_ci_low": lo, "gap_ci_high": hi,
                                        "n_pairs": len(ids), "low_n": bool(low)})
    out = pd.DataFrame(rows)
    out["flag_unparseable"] = out["model_id"].str.contains("meditron") & (out["estimator"] == "conf_verbalized")
    return out


def figure_data(wide: pd.DataFrame, tau: float) -> pd.DataFrame:
    """Group means at the primary tau: mean over the conditions of each axis, by model group and estimator (like fig_uq_data / tab_uqresults).
    incl_flagged False drops the flag_unparseable cells (Meditron verbalised), True keeps them."""
    d = wide[wide["tau"] == tau].copy()
    d["group"] = d["condition_axis"].map(GROUP_OF)
    rows = []
    for incl in (True, False):
        x = d if incl else d[~d["flag_unparseable"]]
        g = x.groupby(["model_group", "estimator", "group"])[list(RATES)].mean().reset_index()
        g["n_cells"] = x.groupby(["model_group", "estimator", "group"]).size().values
        g["incl_flagged"], g["tau"] = incl, tau
        rows.append(g)
    return pd.concat(rows, ignore_index=True)
