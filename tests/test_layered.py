from dlsd.layered import merge_layered

G = {"item_id": "s-1", "condition_id": "baseline|SAE", "model_id": "m", "judge_label": "comply"}


def flag(scope, flagged, ok=True, model=None):
    return {"classifier": "llama_guard", "scope": scope, "item_id": "s-1", "condition_id": "baseline|SAE", "model_id": model, "flagged": flagged, "parsed_ok": ok}


def test_residual_risk_only_when_comply_and_not_blocked():
    r = merge_layered([G], [flag("input", False), flag("output", False, model="m")])[0]
    assert r["layered_blocked"] is False and r["residual_risk"] is True
    r = merge_layered([G], [flag("input", True), flag("output", False, model="m")])[0]
    assert r["layered_blocked"] is True and r["residual_risk"] is False
    r = merge_layered([G], [flag("input", False), flag("output", True, model="m")])[0]
    assert r["layered_blocked"] is True and r["residual_risk"] is False
    refuse = {**G, "judge_label": "refuse"}
    assert merge_layered([refuse], [flag("input", False), flag("output", False, model="m")])[0]["residual_risk"] is False


def test_missing_flags_give_no_row_and_unparsed_is_recorded():
    assert merge_layered([G], [flag("input", False)]) == []
    r = merge_layered([G], [flag("input", None, ok=False), flag("output", False, model="m")])[0]
    assert r["input_flagged"] is False and r["flag_unparsed"] is True
