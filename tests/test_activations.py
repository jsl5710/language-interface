import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
from dlsd import activations as A
from dlsd import mechanism as M


class StubTok:
    """Character-level tokenizer with a chat template, enough to run a tiny random model offline."""
    pad_token, eos_token, padding_side = None, "\x00", "left"

    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        return "U:" + messages[0]["content"] + "\nA:"

    def __call__(self, texts, return_tensors, padding, truncation, max_length, add_special_tokens):
        ids = [[(ord(c) % 90) + 5 for c in t][:max_length] for t in texts]
        n = max(map(len, ids))
        x = torch.zeros(len(ids), n, dtype=torch.long)
        m = torch.zeros(len(ids), n, dtype=torch.long)
        for i, s in enumerate(ids):
            x[i, n - len(s):] = torch.tensor(s)         # left padding
            m[i, n - len(s):] = 1
        return {"input_ids": x, "attention_mask": m}


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(vocab_size=100, hidden_size=32, intermediate_size=64, num_hidden_layers=3, num_attention_heads=4,
                                   num_key_value_heads=4, max_position_embeddings=256)
    return transformers.LlamaForCausalLM(cfg).eval(), StubTok()


def test_batched_equals_single(tiny):
    model, tok = tiny
    texts = [A.chat_text(tok, t) for t in ["hi", "a much longer prompt here", "mid length"]]
    batched = A.last_token_hidden(model, tok, texts, [1, 3], batch_size=3)
    for i, t in enumerate(texts):
        single = A.last_token_hidden(model, tok, [t], [1, 3], batch_size=1)
        for l in (1, 3):
            assert np.allclose(batched[l][i], single[l][0], atol=1e-4)


def test_shapes(tiny):
    model, tok = tiny
    out = A.last_token_hidden(model, tok, [A.chat_text(tok, "x")] * 5, [0, 2], batch_size=2)
    assert out[0].shape == (5, 32) and out[2].shape == (5, 32)


def test_labelled_rows_filters():
    rows = [{"render_ok": True, "rendered": "a", "source": "medsafetybench"}, {"render_ok": True, "rendered": "a", "source": "wmdp_bio"},
            {"render_ok": False, "rendered": "a", "source": "mmedbench"}, {"render_ok": True, "rendered": None, "source": "mmedbench"}]
    assert len(A.labelled_rows(rows)) == 1


def test_pick_layer_and_condition_summary():
    rng = np.random.default_rng(0)
    n, d = 120, 24
    labels = np.array([1, 0] * (n // 2))
    ids = [f"i{k}" for k in range(n)]
    u = np.zeros(d); u[0] = 1
    good = rng.standard_normal((n, d)) * 0.3 + np.outer(labels * 2.0, u)
    bad = rng.standard_normal((n, d)) * 0.3
    res = M.pick_layer({1: bad, 2: good}, labels, ids, seed=1)
    assert res["layer"] == 2 and res["heldout_auroc"][2] > 0.95 > res["heldout_auroc"][1] + 0.2
    de = M.diff_in_means(good[labels == 1], good[labels == 0])
    s = M.condition_summary(good, labels, de, M.separation(good[labels == 1], good[labels == 0], de), M.random_unit(d, 0))
    assert abs(s["retention"] - 1) < 1e-9 and s["cosine_to_english"] > 0.99 and abs(s["cosine_control"]) < 0.6
