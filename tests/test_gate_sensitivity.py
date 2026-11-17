import numpy as np
import pandas as pd

from dlsd.gate_sensitivity import compare, frame, strata_from_gate


def test_strata():
    g = [{"condition_id": "c", "item_id": "a", "votes": [True, True]}, {"condition_id": "c", "item_id": "b", "votes": [True, False]},
         {"condition_id": "c", "item_id": "d", "votes": [False, False]}]
    assert strata_from_gate(g) == {("c", "a"): "confirmed", ("c", "b"): "split", ("c", "d"): "rejected"}


def test_compare_recovers_a_known_difference():
    rng = np.random.default_rng(0)
    rows, strata = [], {}
    for i in range(300):
        for model in ("m1", "m2"):
            s = "confirmed" if i < 200 else "rejected"
            strata[("codeswitch|intra|Hindi", f"i{i}")] = s
            p = 0.2 if s == "confirmed" else 0.5
            rows.append({"item_id": f"i{i}", "model_id": model, "condition_id": "codeswitch|intra|Hindi", "condition_axis": "codeswitch",
                         "source": "medsafetybench", "judge_label": "comply" if rng.random() < p else "refuse", "correct": None})
    res = compare(frame(pd.DataFrame(rows), strata), n_boot=200)
    r = res[(res.metric == "comply") & (res.stratum == "rejected")].iloc[0]
    assert 0.2 < r.cell_diff < 0.4 and r.ci_lo < r.cell_diff < r.ci_hi
