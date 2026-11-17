import numpy as np
from dlsd.uq import ece, brier, auroc, selective_accuracy, overconfident_error_rate, sampling_confidence, parse_verbalized
from dlsd.scoring import parse_letter, format_ok, correct, mcq_prompt
from dlsd.backends import letter_probs
from dlsd.conditions import build_conditions

def test_ece_perfect_and_overconfident():
    assert ece([1.0, 1.0, 0.0, 0.0], [1, 1, 0, 0]) < 1e-9
    assert abs(ece([0.9] * 10, [1] * 5 + [0] * 5) - 0.4) < 1e-9

def test_brier_auroc_selective():
    assert brier([1, 0], [1, 0]) == 0
    assert auroc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    assert selective_accuracy([0.9, 0.8, 0.1], [1, 1, 0], 2 / 3) == 1.0
    assert overconfident_error_rate([0.9, 0.9, 0.5], [0, 1, 0], 0.8) == 1 / 3

def test_sampling_conf():
    m, c, e = sampling_confidence(["A", "A", "B", "A"])
    assert m == "A" and c == 0.75 and e > 0

def test_verbalized_and_letters():
    assert parse_verbalized("B\n85") == ("B", 0.85)
    assert parse_verbalized("B") == (None, None)
    assert parse_verbalized("B. Advanced sleep phase disorder\n80") == ("B", 0.8)      # letter with option text (Med42, Llama)
    assert parse_verbalized("C. Difenhidramina\n\n80\n\nLa respuesta") == ("C", 0.8)
    assert parse_verbalized("B. 80\n\nExplanation: x") == ("B", 0.8)
    assert parse_verbalized("**B**\n**Confidence: 90%**") == ("B", 0.9)
    assert parse_verbalized("B. Vitamin B12 deficiency\n70") == ("B", 0.7)       # digits inside option text are not the confidence
    assert parse_verbalized("A patient presents\n90") == (None, None)
    assert parse_verbalized("B\n150") == (None, None)
    assert parse_letter(" (C). ") == "C" and not format_ok("The answer is C")
    assert correct("B", 1) and correct("B", "b")
    assert "A. x" in mcq_prompt("q", ["x", "y"])

def test_lenient_letter_extraction():
    from dlsd.scoring import extract_letter
    for text, want in [("D. All of the above", "D"), (" Option C", "C"), (" Option A is correct.", "A"), ("(B)", "B"), ("C. Anthrax (B. anthracis)", "C"),
                       ("**B**", "B"), ("The answer is C", "C"), ("", None), ("A patient presents with", None), ("Because B", None)]:
        assert extract_letter(text) == want, text
    assert not format_ok("D. All of the above") and format_ok("D")      # instruction following stays strict
    opts = {"A": "Spironolactone", "B": "Eplerenone", "C": "Hereditary angioedema (HAE)", "D": "Splenic sequestration crisis"}
    assert extract_letter(" C Hereditary angioedema (HAE) is a rare", opts) == "C"                  # letter + option text, validated against the item
    assert extract_letter("Eplerenone is the best choice", opts) == "B"                             # option text alone
    assert extract_letter("A patient presents with", opts) is None and extract_letter("Because B", opts) is None
    assert extract_letter(" D Hereditary angioedema", opts) is None                                  # letter and text disagree: not guessed


def test_letter_probs():
    p = letter_probs([("A", np.log(0.6)), (" B", np.log(0.2)), ("The", np.log(0.1))])
    assert abs(p["A"] - 0.75) < 1e-9

def test_18_conditions():
    cfg = {"conditions": {"dialects": list("12345"), "languages": {"high": "Spanish", "mid": "Hindi", "low": "Swahili"},
                          "code_switch_types": ["intra", "inter", "extra"]}}
    assert len(build_conditions(cfg)) == 18      # extension off

def test_write_table_and_manifest(tmp_path):
    import json, pandas as pd
    from dlsd.io import write_table, write_manifest, to_long
    df = pd.DataFrame({"model_id": ["m"], "ece": [0.1], "accuracy": [0.5]})
    write_table(df, "t", {"experiment": "x"}, out_dir=tmp_path)
    j = json.loads((tmp_path / "t.json").read_text())
    assert j["n_records"] == 1 and j["records"][0]["ece"] == 0.1 and j["experiment"] == "x"
    assert (tmp_path / "t.csv").read_text().splitlines()[0] == "model_id,ece,accuracy"
    long = to_long(df, ["model_id"], ["ece", "accuracy"])
    assert set(long.metric) == {"ece", "accuracy"}
    m = write_manifest({"seed": 1}, {"stage": "t"}, path=tmp_path / "m.json")
    assert m["seed"] == 1
