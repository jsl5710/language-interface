"""Sensitivity analysis: do renderings the judge panel rejected behave differently from the ones it confirmed? (appendix)

Strata of a code-switched rendering, from the two judges' votes on the intended type: confirmed (both agree), split (one agrees),
rejected (neither). Main-run rows are all confirmed; the gate-failed run supplies split and rejected rows. For each model and condition
cell that has both kinds of rows, the stratum rate is compared with the confirmed rate in the same cell, and cell differences are averaged
with equal weight (so the comparison does not just reflect that failures cluster in Swahili or in some models). 95% intervals come from
resampling items, and an item brings all its rows along.

Outcomes: comply_rate and refuse_rate on the harmful requests (MedSafetyBench), accuracy on the multiple-choice items.
Differences are descriptive: a rejected rendering differs from a confirmed one in switch type and may differ in other ways."""
import numpy as np
import pandas as pd

from .analysis import _item_answers, _item_options
from .scoring import correct, extract_letter

MIN_N = 10                      # rows per stratum in a cell before the cell counts


def strata_from_gate(gate_rows: list) -> dict:
    """{(condition_id, item_id): 'confirmed' | 'split' | 'rejected'} from the latest gate row of each item."""
    out = {}
    for r in gate_rows:
        v = sum(bool(x) for x in r["votes"])
        out[(r["condition_id"], r["item_id"])] = "confirmed" if v == len(r["votes"]) else "split" if v else "rejected"
    return out


def frame(rows: pd.DataFrame, strata: dict) -> pd.DataFrame:
    """Keep code-switch rows, add the stratum and the three outcome columns."""
    d = rows[rows["condition_axis"].isin(["codeswitch", "codeswitch_native"])].copy()
    d["stratum"] = [strata.get((c, i)) for c, i in zip(d["condition_id"], d["item_id"])]
    d = d.dropna(subset=["stratum"])
    harmful = d["source"] == "medsafetybench"
    d["comply"] = np.where(harmful, (d["judge_label"] == "comply").astype(float), np.nan)
    d["refuse"] = np.where(harmful, (d["judge_label"] == "refuse").astype(float), np.nan)
    ans, opt = _item_answers(), _item_options()            # accuracy uses the same lenient answer-letter rule as the main analysis (docs/DEVIATIONS.md 2026-11-15)
    lenient = ([float(correct(extract_letter(t, opt.get(i)), ans.get(i))) for t, i in zip(d["response_text"], d["item_id"])]
               if "response_text" in d else d["correct"].astype(float).tolist())            # frames without response text (unit tests) use the stored strict value
    d["accuracy"] = np.where(~harmful, lenient, np.nan)
    return d


def _cell_diffs(d: pd.DataFrame, weights: np.ndarray, metric: str, stratum: str):
    """Mean over cells of (stratum rate - confirmed rate), weights = item multiplicities from a bootstrap draw."""
    x = d[metric].to_numpy()
    ok = ~np.isnan(x)
    w = weights * ok
    cells = d["cell"].to_numpy()
    sel = {}
    for s in ("confirmed", stratum):
        m = (d["stratum"].to_numpy() == s)
        sel[s] = (np.bincount(cells, weights=w * m, minlength=cells.max() + 1), np.bincount(cells, weights=np.nan_to_num(x) * w * m, minlength=cells.max() + 1))
    n_c, s_c = sel["confirmed"]
    n_s, s_s = sel[stratum]
    good = (n_c >= MIN_N) & (n_s >= MIN_N)
    if not good.any():
        return np.nan, 0
    return float(np.mean(s_s[good] / n_s[good] - s_c[good] / n_c[good])), int(good.sum())


def compare(d: pd.DataFrame, n_boot: int = 1000, seed: int = 13) -> pd.DataFrame:
    d = d.copy()
    d["cell"] = pd.factorize(d["model_id"] + "|" + d["condition_id"])[0]
    items, inv = np.unique(d["item_id"].to_numpy(), return_inverse=True)
    rng = np.random.default_rng(seed)
    draws = [np.bincount(rng.integers(0, len(items), len(items)), minlength=len(items))[inv].astype(float) for _ in range(n_boot)]
    rows = []
    for metric in ("comply", "refuse", "accuracy"):
        for stratum in ("split", "rejected"):
            est, n_cells = _cell_diffs(d, np.ones(len(d)), metric, stratum)
            boots = np.array([_cell_diffs(d, w, metric, stratum)[0] for w in draws])
            lo, hi = (np.nanpercentile(boots, [2.5, 97.5]) if not np.isnan(boots).all() else (np.nan, np.nan))
            sub = d[d["stratum"] == stratum][metric].dropna()
            conf = d[d["stratum"] == "confirmed"][metric].dropna()
            rows.append({"metric": metric, "stratum": stratum, "n_rows": len(sub), "rate": sub.mean() if len(sub) else np.nan,
                         "confirmed_n": len(conf), "confirmed_rate": conf.mean() if len(conf) else np.nan,
                         "cell_diff": est, "ci_lo": lo, "ci_hi": hi, "n_cells": n_cells})
    return pd.DataFrame(rows)
