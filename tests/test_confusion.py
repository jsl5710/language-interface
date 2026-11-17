import numpy as np
import pandas as pd

from dlsd.confusion import cell_rates, confusion_table, paired_gaps, rates


def test_rates_on_a_hand_made_example():
    # tau 0.8: 4 correct (3 flagged), 5 wrong (2 flagged)  ->  TPR 3/4, FNR 1/4, FPR 2/5, TNR 3/5
    conf = [0.9, 0.8, 0.95, 0.5, 0.85, 0.99, 0.2, 0.7, 0.1]
    corr = [1, 1, 1, 1, 0, 0, 0, 0, 0]
    r = rates(conf, corr, 0.8)
    assert (r["tpr"], r["fnr"], r["fpr"], r["tnr"]) == (0.75, 0.25, 0.4, 0.6)
    assert (r["n_correct"], r["n_wrong"]) == (4, 5)
    assert rates(conf, corr, 0.6)["tpr"] == 0.75 and rates(conf, corr, 0.9)["tpr"] == 0.5          # >= is inclusive at the boundary (0.8 flagged at tau 0.8)
    assert np.isnan(rates([0.9, 0.9], [1, 1], 0.8)["fpr"])                                          # no wrong answers: undefined, not 0


def test_cell_ci_and_low_n_and_gaps():
    c = cell_rates([0.9, 0.8, 0.95, 0.5, 0.85, 0.99, 0.2, 0.7, 0.1], [1, 1, 1, 1, 0, 0, 0, 0, 0], 0.8, n_boot=200)
    assert c["low_n"] and abs(c["fnr_ci_low"] - (1 - c["tpr_ci_high"])) < 1e-12
    rng = np.random.default_rng(0)
    n = 60
    base = pd.DataFrame({"item_id": range(n), "model_id": "m", "model_group": "general", "condition_axis": "baseline", "condition_level": "SAE", "language_pair": None,
                         "conf_prob": rng.uniform(size=n), "correct": rng.integers(0, 2, n)})
    cond = base.assign(condition_axis="resource", condition_level="low", language_pair="Swahili")
    cond["conf_prob"] = 1.0                      # always flagged: TPR = FPR = 1 in the condition
    df = pd.concat([base, cond], ignore_index=True)
    w = confusion_table(df, ["model_group", "model_id", "condition_axis", "condition_level", "language_pair"], ["conf_prob"], [0.8], n_boot=50)
    assert set(w["tpr"][w.condition_axis == "resource"]) == {1.0} and not w["flag_unparseable"].any()
    g = paired_gaps(df, ["condition_axis", "condition_level", "language_pair"], ["conf_prob"], [0.8])
    row = g[g.metric == "fpr"].iloc[0]
    assert row["value"] == 1.0 and abs(row["gap"] - (1.0 - row["sae_value"])) < 1e-12 and row["n_pairs"] == n
