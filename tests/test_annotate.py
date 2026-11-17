import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import importlib

ann = importlib.import_module("02m_llm_annotate")


def test_parse_takes_last_valid_json_and_rejects_bad_labels():
    assert ann.parse('{"label": "intra", "comment": "x"}', "label", ann.LABELS) == ("intra", "x")
    assert ann.parse('{"label": "bilingual"} {"label": "extra", "comment": "tag"}', "label", ann.LABELS)[0] == "extra"
    assert ann.parse("no json", "decision", ("pass", "fail")) == (None, "")


def test_prompts_are_blind_and_mark_text_as_data():
    p = ann.switch_prompt("Hello, how are you?", "Hindi")
    assert "intended" not in p.lower() and "not instructions" in p and "Hello, how are you?" in p
    q = ann.similarity_prompt("A", "B", "Swahili")
    assert "not instructions" in q and "pass" in q and "fail" in q
