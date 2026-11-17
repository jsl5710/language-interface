import json

from dlsd import gate_io
from dlsd.backends import Gen
from dlsd.switch_judge import LABELS, build_prompt, cohen_kappa, gate, judge_one, label_order, parse_label


def test_label_order_is_a_deterministic_permutation():
    assert sorted(label_order("k")) == sorted(LABELS) and label_order("k") == label_order("k")
    assert len({tuple(label_order(f"item{i}")) for i in range(30)}) > 1      # order varies across items


def test_prompt_is_blind_and_treats_text_as_data():
    p = build_prompt("The patient has chest pain. Necesita una radiografía hoy.", "Spanish", "k")
    assert "intended" not in p.lower() and "requested" not in p.lower()
    assert "not instructions" in p and "Poplack" in p and "single_language" in p


def test_parse_label():
    assert parse_label('Sure. {"label": "intra"}') == "intra"
    assert parse_label('{"label": "bilingual"}') is None and parse_label("intra") is None


def test_gate_majority_and_unparseable_counts_as_fail():
    assert gate("intra", ["intra", "intra"], False)["passed"]            # 2 of 3
    assert gate("intra", ["intra", "inter"], True)["passed"]             # judge 1 + heuristic
    assert not gate("intra", ["inter", "inter"], True)["passed"]         # 1 of 3
    assert not gate("intra", [None, "intra"], False)["passed"]           # null is a failed vote, not skipped


def test_kappa():
    assert cohen_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == 1.0
    assert abs(cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"])) < 1e-9


class Fake:
    def chat(self, prompt, **kw):
        return [Gen('{"label": "extra"}')]


def test_judge_one_passes_system_prompt():
    assert judge_one(Fake(), "x", "Hindi", "k", system="Reasoning: low")[0] == "extra"


def test_failed_with_feedback_reads_only_previous_round(tmp_path, monkeypatch):
    monkeypatch.setattr(gate_io, "GATE_DIR", tmp_path)
    rows = [{"item_id": "a", "round": 1, "passed": False, "judge_labels": ["inter", "inter"]},
            {"item_id": "b", "round": 1, "passed": True, "judge_labels": ["intra", "intra"]},
            {"item_id": "c", "round": 2, "passed": False, "judge_labels": [None, None]}]
    (tmp_path / "gate_codeswitch_intra_spanish.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    shard = gate_io.Path("data/llm/rendered_codeswitch_intra_spanish.jsonl")
    assert list(gate_io.failed_with_feedback(shard, 2)) == ["a"]


def test_judge_examples_cover_every_label_in_both_directions():
    from dlsd.switch_judge import JUDGE_EXAMPLES
    for lang, ex in JUDGE_EXAMPLES.items():
        assert {(e["label"], e["base"]) for e in ex} == {(l, b) for l in LABELS for b in ("EN", "NATIVE")}, lang
        assert len({e["id"] for e in ex}) == len(ex)
        assert all(e["explanation"] and "\u2014" not in e["explanation"] for e in ex)
    p = build_prompt("x", "Hindi", "k")
    assert JUDGE_EXAMPLES["Hindi"][0]["text"] in p and JUDGE_EXAMPLES["Spanish"][0]["text"] not in p


def test_parse_label_takes_the_final_json_object():
    assert parse_label('{"explanation": "switch inside the clause", "label": "intra"}') == "intra"
    assert parse_label('{"label": "extra"} then {"explanation": "x", "label": "inter"}') == "inter"


def test_zero_shot_prompt_drops_only_the_examples():
    few, zero = build_prompt("x", "Hindi", "k"), build_prompt("x", "Hindi", "k", shots=False)
    assert "Worked examples" in few and "Worked examples" not in zero
    assert zero.split("Now classify")[0].startswith("You are classifying") and "Now classify" in zero
