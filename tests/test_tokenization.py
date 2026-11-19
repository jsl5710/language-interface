import numpy as np
import pandas as pd
from dlsd import tokenization as T


def char_tok(text):            # toy tokenizer: one token per 3 characters, with some byte-fallback tokens
    return [text[i:i + 3] for i in range(0, len(text), 3)]


def test_premium_and_fertility():
    assert T.premium("abcdefghi", "abcdef", char_tok) == 1.5
    assert T.fertility("abcdefghi jkl", char_tok) == 2.5       # 4 tokens over 2 words


def test_byte_and_split_shares():
    tok = lambda t: ["<0x41>", "a", "<unk>", "b"] if t == "x" else ["a"]
    assert T.byte_fallback_share("x", tok) == 0.5
    assert T.split_word_share("x y", tok, k=3) == 0.5


def _data(effect_of_condition, slope, n=300, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for cond, shift in [("SAE", 0.0), ("dialect", 0.4), ("swahili", 1.0)]:
        for i in range(n):
            lx = rng.normal(shift, 0.3)
            y = slope * lx + effect_of_condition.get(cond, 0.0) + rng.normal(0, 0.1)
            rows.append({"item_id": f"i{i}", "condition": cond, "premium": float(np.exp(lx)), "y": y})
    return pd.DataFrame(rows)


def test_gaps_fully_explained_by_tokenization():
    r = T.adjusted_gaps(_data({}, slope=-0.5), "y", "premium", "condition", "SAE").set_index("condition")
    assert r.loc["swahili", "raw_gap"] < -0.3
    assert abs(r.loc["swahili", "adjusted_gap"]) < 0.06 and abs(r.loc["dialect", "adjusted_gap"]) < 0.06
    assert abs(r["slope_per_log_unit"].iloc[0] + 0.5) < 0.1


def test_gaps_with_residual_condition_effect():
    r = T.adjusted_gaps(_data({"swahili": -0.6}, slope=-0.5), "y", "premium", "condition", "SAE").set_index("condition")
    assert r.loc["swahili", "adjusted_gap"] < -0.45           # most of the gap is not tokenization
    assert r.loc["swahili", "share_explained"] < 0.5


def test_bootstrap_intervals_cover_estimate():
    r = T.adjusted_gaps(_data({"swahili": -0.6}, slope=-0.5, n=120), "y", "premium", "condition", "SAE", n_boot=40).set_index("condition")
    s = r.loc["swahili"]
    assert s["adj_lo"] <= s["adjusted_gap"] <= s["adj_hi"]


def test_item_table_uses_same_item_baseline():
    rows = [{"item_id": "a", "condition_id": "baseline|SAE", "rendered": "abcdef"}, {"item_id": "a", "condition_id": "dialect|X", "rendered": "abcdefghi"},
            {"item_id": "b", "condition_id": "dialect|X", "rendered": "zzz"}, {"item_id": "a", "condition_id": "dialect|Y", "rendered": None}]
    t = T.item_table(rows, char_tok)
    assert list(t["condition_id"]) == ["baseline|SAE", "dialect|X"] and t.loc[1, "premium"] == 1.5 and t.loc[0, "premium"] == 1.0
