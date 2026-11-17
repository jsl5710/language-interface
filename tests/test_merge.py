"""Merge of shards into one row per (item, condition), using tiny fixtures."""
import json

import yaml

from dlsd.conditions import build_conditions
from dlsd.merge import llm_slug, merge, quality_log

CFG = {"conditions": {"dialects": ["AAVE", "Indian English", "Nigerian English", "Jamaican English", "White Zimbabwean English"],
                      "languages": {"high": "Spanish", "mid": "Hindi", "low": "Swahili"}, "code_switch_types": ["intra", "inter", "extra"]}}
ITEMS = [{"item_id": "a-1", "source": "wmdp_bio", "text": "Which drug?", "options": None, "answer": None},
         {"item_id": "a-2", "source": "mmedbench", "text": "A man has pain.", "options": {"A": "Aspirin", "B": "Insulin"}, "answer": "A"}]


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_every_item_gets_eighteen_rows_and_missing_is_flagged(tmp_path):
    rows = merge(ITEMS, CFG, tmp_path / "d", tmp_path / "l", tmp_path / "g")
    assert len(rows) == 36 and {r["condition_id"] for r in rows} == {c.cid for c in build_conditions(CFG)}
    base = [r for r in rows if r["condition_axis"] == "baseline"]
    assert all(r["render_ok"] for r in base) and base[1]["options"] == ITEMS[1]["options"]
    assert {r["render_status"] for r in rows if r["condition_axis"] != "baseline"} == {"missing"}


def test_dialect_error_gate_failure_and_pass(tmp_path):
    write(tmp_path / "d" / "rendered_aave.jsonl", [
        {"item_id": "a-1", "rendered": "Which drug?!", "rendered_options": None},
        {"item_id": "a-2", "rendered": None, "error": "ValueError", "rendered_options": None}])
    cs = [c for c in build_conditions(CFG) if c.axis == "codeswitch" and c.switch_type == "intra" and c.language == "Spanish"][0]
    f = f"rendered_{llm_slug(cs)}.jsonl"
    write(tmp_path / "l" / f, [{"item_id": "a-1", "rendered": "Which droga?", "rendered_options": None, "passed": True, "round": 1, "translator": "nllb-200-3.3B"},
                               {"item_id": "a-2", "rendered": "A hombre pain.", "rendered_options": {"A": "Aspirin", "B": "Insulin"}, "passed": True, "round": 1}])
    write(tmp_path / "g" / f.replace("rendered_", "gate_"), [{"item_id": "a-1", "round": 1, "passed": True}, {"item_id": "a-2", "round": 1, "passed": False}])
    rows = merge(ITEMS, CFG, tmp_path / "d", tmp_path / "l", tmp_path / "g")
    get = lambda item, cid: next(r for r in rows if r["item_id"] == item and r["condition_id"] == cid)
    assert get("a-1", "dialect|AAVE")["render_ok"] and get("a-2", "dialect|AAVE")["render_status"] == "error"
    assert get("a-1", cs.cid)["render_ok"] and get("a-1", cs.cid)["method"] == "nllb-200-3.3B"
    assert get("a-2", cs.cid)["render_status"] == "failed_gate" and not get("a-2", cs.cid)["render_ok"]
    log = {r["condition_id"]: r for r in quality_log(rows)}
    assert log[cs.cid]["share_ok"] == 0.5 and log["dialect|AAVE"]["n_error"] == 1


def test_below_threshold_rendering_waits_for_the_reviewer(tmp_path):
    hi = [c for c in build_conditions(CFG) if c.axis == "resource" and c.language == "Hindi"][0]
    f = f"rendered_{llm_slug(hi)}.jsonl"
    low = {"semantic_sim": 0.77, "semantic_ok": False, "key_terms_ok": True, "langid_ok": True, "changed_ok": True}
    write(tmp_path / "l" / f, [{"item_id": "a-1", "rendered": "x", "rendered_options": None, "passed": False, "round": 1, "checks": low},
                               {"item_id": "a-2", "rendered": "y", "rendered_options": {"A": "a", "B": "b"}, "passed": False, "round": 1, "checks": low}])
    status = lambda rows, i: next(r for r in rows if r["item_id"] == i and r["condition_id"] == hi.cid)["render_status"]
    rows = merge(ITEMS, CFG, tmp_path / "d", tmp_path / "l", tmp_path / "g", tmp_path / "none.jsonl")
    assert status(rows, "a-1") == "needs_review"
    write(tmp_path / "dec.jsonl", [{"condition_id": hi.cid, "item_id": "a-1", "decision": "pass"}, {"condition_id": hi.cid, "item_id": "a-2", "decision": "fail"}])
    rows = merge(ITEMS, CFG, tmp_path / "d", tmp_path / "l", tmp_path / "g", tmp_path / "dec.jsonl")
    assert status(rows, "a-1") == "ok_reviewed" and status(rows, "a-2") == "failed_review"
    assert next(r for r in rows if r["item_id"] == "a-1" and r["condition_id"] == hi.cid)["render_ok"]
