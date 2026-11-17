"""Analysis on synthetic generations: known rates, paired gaps, patterns, Caribbean verdict, figures."""
import numpy as np
import pandas as pd

from dlsd import analysis as A
from dlsd import figures as F

DIALECTS = ["AAVE", "Indian English", "Nigerian English", "Jamaican English", "White Zimbabwean English"]


def cond_rows():
    rows = [("baseline", "SAE", None, None)] + [("dialect", d, None, None) for d in DIALECTS]
    rows += [("resource", lv, lang, None) for lv, lang in (("high", "Spanish"), ("mid", "Hindi"), ("low", "Swahili"))]
    rows += [("codeswitch", t, lang, t) for t in ("intra", "inter", "extra") for lang in ("Spanish", "Hindi", "Swahili")]
    rows += [("codeswitch_native", t, lang, t) for t in ("intra", "inter", "extra") for lang in ("Spanish", "Hindi", "Swahili")]
    return rows


def make(n=40, comply_rate=None):
    """Two models (one general, one medical), n medsafety items and n mmedbench items in every condition."""
    comply_rate = comply_rate or {}
    rng = np.random.default_rng(0)
    sc, lay = [], []
    for mid, grp in (("g1", "general"), ("m1", "medical")):
        for ax, lv, lang, sw in cond_rows():
            p = comply_rate.get((ax, lv), 0.1)
            for i in range(n):
                comply = i < round(p * n)
                base = {"item_id": f"ms-{i}", "source": "medsafetybench", "condition_axis": ax, "condition_level": lv, "language_pair": lang,
                        "switch_type": sw, "model_id": mid, "model_group": grp, "condition_id": f"{ax}|{lv}",
                        "judge_label": "comply" if comply else "refuse", "correct": None, "format_ok": None}
                sc.append(base)
                blocked = i % 2 == 0
                lay.append({**base, "classifier_name": "clf", "input_flagged": blocked, "output_flagged": False, "layered_blocked": blocked,
                            "residual_risk": comply and not blocked})
                sc.append({**base, "item_id": f"mm-{i}", "source": "mmedbench", "judge_label": "comply", "correct": i % 4 != 0, "format_ok": True})
    return pd.DataFrame(sc), pd.DataFrame(lay)


def test_rates_gaps_and_pooling():
    sc, lay = make(comply_rate={("baseline", "SAE"): 0.1, ("dialect", "AAVE"): 0.3, ("dialect", "Jamaican English"): 0.5})
    cells = A.cell_table(A.pool_groups(A.exp1_frames(sc, lay)))
    g = cells[(cells.model_id == "g1") & (cells.metric == "harmful_compliance_rate") & (cells["mode"] == "internal")]
    assert abs(g[g.condition_level == "SAE"]["value"].iloc[0] - 0.1) < 1e-9
    aave = g[g.condition_level == "AAVE"].iloc[0]
    assert abs(aave.gap_from_sae - 0.2) < 1e-9 and aave.ci_low <= aave.gap_from_sae <= aave.ci_high and aave.n == 40
    acc = cells[(cells.model_id == "g1") & (cells.metric == "answer_accuracy") & (cells.condition_level == "SAE")]
    assert abs(acc["value"].iloc[0] - 0.75) < 1e-9
    assert "group:all" in set(cells.model_id) and "group:general" in set(cells.model_id)
    res = cells[(cells.metric == "residual_risk_rate") & (cells.model_id == "g1") & (cells.condition_level == "SAE")]
    assert 0 <= res["value"].iloc[0] <= 0.1 and res["classifier"].iloc[0] == "clf"


def test_small_cells_get_no_interval_and_ratio_nan():
    v, lo, hi = A.rate_ci(np.array([1.0, 0, 1, 0]), np.ones(4))
    assert v == 0.5 and np.isnan(lo) and np.isnan(hi)
    assert np.isnan(A.rate_ci(np.zeros(20), np.zeros(20))[0])           # nothing flagged: precision is NaN, not 0


def test_caribbean_verdict_rule():
    sc, lay = make(comply_rate={("baseline", "SAE"): 0.1, ("dialect", "AAVE"): 0.2, ("dialect", "Indian English"): 0.2,
                                ("dialect", "Nigerian English"): 0.2, ("dialect", "White Zimbabwean English"): 0.2, ("dialect", "Jamaican English"): 0.6})
    cells = A.cell_table(A.pool_groups(A.exp1_frames(sc, lay)))
    car = A.caribbean(cells, ["harmful_compliance_rate"])
    assert set(car[car.model_id == "g1"]["verdict"]) == {"larger"}


def test_uq_patterns_and_tolerance():
    assert A.uq_pattern(-0.10, -0.08, 0.02) == "calibrated_signal"
    assert A.uq_pattern(-0.10, 0.01, 0.02) == "silent_failure"
    assert A.uq_pattern(0.00, -0.10, 0.02) == "lost_usefulness"
    assert A.uq_pattern(-0.01, -0.01, 0.02) == "no_clear_pattern"       # inside the tolerance


def test_uq_axis_table_and_stability():
    rows = []
    for item in range(12):
        for ax, lv, lang, _ in cond_rows():
            drop = ax != "baseline"
            rows.append({"item_id": f"q{item}", "model_group": "general", "model_id": "g1", "condition_axis": ax, "condition_level": lv, "language_pair": lang,
                         "correct": (item % 2 == 0) if not drop else (item % 4 == 0), "conf_prob": 0.9, "parsed_letter": "A" if item % 3 else "B"})
    g = pd.DataFrame(rows)
    t = A.uq_axis_table(g, ["conf_prob"], 0.02)
    assert set(t.pattern) == {"silent_failure"}                          # accuracy falls, confidence stays at 0.9
    s = A.item_stability(g, "conf_prob")
    assert s.iloc[0].conf_sd_mean < 1e-12 and s.iloc[0].agree_with_sae_mean == 1.0


def test_figures_are_written(tmp_path):
    sc, lay = make(comply_rate={("baseline", "SAE"): 0.1, ("dialect", "AAVE"): 0.3})
    cells = A.cell_table(A.pool_groups(A.exp1_frames(sc, lay)))
    d1 = F.fig_compliance(cells, DIALECTS, tmp_path / "a.pdf")
    d2 = F.fig_residual(cells, tmp_path / "b.pdf")
    d3 = F.fig_gap(cells, tmp_path / "c.pdf")
    assert all((tmp_path / f).stat().st_size > 1000 for f in ("a.pdf", "b.pdf", "c.pdf"))
    assert len(d1) == 15 and abs(d1.iloc[0].internal_pct - 10) < 1e-6 and len(d2) == 5 and len(d3) == 8


def test_direction_comparison_pairs_the_three_renderings_per_language_and_type():
    sc, lay = make(comply_rate={("baseline", "SAE"): 0.1, ("resource", "high"): 0.4, ("codeswitch", "intra"): 0.2, ("codeswitch_native", "intra"): 0.3})
    cells = A.cell_table(A.pool_groups(A.exp1_frames(sc, lay)))
    t = A.direction_comparison(cells, ["harmful_compliance_rate"])
    r = t[(t.model_id == "g1") & (t.language == "Spanish") & (t.switch_type == "intra") & (t["mode"] == "internal")].iloc[0]
    assert abs(r.value_english_base - 0.2) < 1e-9 and abs(r.value_native_base - 0.3) < 1e-9 and abs(r.value_translation - 0.4) < 1e-9
    assert abs(r.native_minus_english_base - 0.1) < 1e-9 and abs(r.gap_native_base - 0.2) < 1e-9
