"""Sanity checks (section 17) on a tiny fixture: they pass on clean data and catch planted defects."""
import copy

from dlsd import sanity
from dlsd.conditions import build_conditions
from dlsd.merge import merge, quality_log

CFG = {"conditions": {"dialects": ["AAVE", "Indian English", "Nigerian English", "Jamaican English", "White Zimbabwean English"],
                      "languages": {"high": "Spanish", "mid": "Hindi", "low": "Swahili"}, "code_switch_types": ["intra", "inter", "extra"]}}
ITEMS = [{"item_id": "a-1", "source": "wmdp_bio", "text": "Which drug?", "options": None, "answer": None},
         {"item_id": "a-2", "source": "mmedbench", "text": "A man has pain.", "options": {"A": "Aspirin", "B": "Insulin"}, "answer": "A"}]
CIDS = [c.cid for c in build_conditions(CFG)]


def ok_rows():
    rows = merge(ITEMS, CFG, "none", "none", "none")
    for r in rows:                                  # pretend every condition rendered fine
        if r["rendered"] is None:
            r.update(rendered=r["item_id"] + " rendered", options=ITEMS[int(r["item_id"][-1]) - 1]["options"], render_ok=True, render_status="ok")
    return rows


def test_clean_fixture_passes_and_missing_inputs_are_skipped():
    rows = ok_rows()
    assert sanity.check_1_item_set(ITEMS, rows, {})["status"] == "pass"
    assert sanity.check_2_baseline(ITEMS, rows)["status"] == "pass"
    assert sanity.check_3_condition_count(rows, CIDS)["status"] == "pass"
    assert sanity.check_4_mcq_integrity(ITEMS, rows)["status"] == "pass"
    assert sanity.check_5_rendering_quality(rows, quality_log(rows))["status"] == "pass"
    for chk in (sanity.check_6_parsing([], []), sanity.check_9_layered_logic([]), sanity.check_10_determinism([], []),
                sanity.check_11_row_counts(rows, {"exp1": []}, ["m"], set()), sanity.check_12_dialect([], [], "x")):
        assert chk["status"] == "skipped"


def test_planted_defects_are_caught():
    rows = ok_rows()
    bad = copy.deepcopy(rows)
    next(r for r in bad if r["condition_axis"] == "baseline")["rendered"] = "changed"
    assert sanity.check_2_baseline(ITEMS, bad)["status"] == "fail"
    bad = copy.deepcopy(rows)
    next(r for r in bad if r["condition_axis"] == "dialect" and r["item_id"] == "a-2")["options"] = {"B": "Insulin", "A": "Aspirin"}
    assert sanity.check_4_mcq_integrity(ITEMS, bad)["status"] == "fail"
    assert sanity.check_3_condition_count(rows[:-1], CIDS)["status"] == "fail"
    bad = copy.deepcopy(rows)
    for r in bad:
        if r["condition_id"] == "dialect|AAVE":
            r.update(render_ok=False, render_status="error")
    assert sanity.check_5_rendering_quality(bad, [])["status"] == "fail"           # over 15% and not flagged
    assert sanity.check_5_rendering_quality(bad, quality_log(bad))["status"] == "warn"


def test_parsing_layered_and_metric_checks():
    gens = [{"model_id": "m", "condition_axis": "baseline", "condition_level": "SAE", "language_pair": None, "parsed_letter": None}] * 3
    assert sanity.check_6_parsing(gens, [])["status"] == "fail"
    scored = [{"item_id": "x", "input_flagged": True, "output_flagged": False, "layered_blocked": True, "judge_label": "comply", "residual_risk": False}]
    assert sanity.check_9_layered_logic(scored)["status"] == "pass"
    scored[0]["residual_risk"] = True
    assert sanity.check_9_layered_logic(scored)["status"] == "fail"
    assert sanity.check_7_metric_invariants({"t": [{"metric": "ece", "value": 1.2, "n": 5}]})["status"] == "fail"


def test_independent_ece_matches_library_ece():
    from dlsd.uq import ece
    conf = [0.05, 0.4, 0.55, 0.9, 0.95, 1.0, 0.0, 0.71]
    corr = [0, 1, 1, 1, 0, 1, 0, 1]
    assert abs(sanity.independent_ece(conf, corr) - ece(conf, corr)) < 1e-12
