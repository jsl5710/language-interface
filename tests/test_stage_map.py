"""E1 stage map (stages 2 to 8): extended capture on a tiny random Llama with a stub tokenizer, the pure measures on synthetic arrays,
and the scripts 13 to 18 run end to end on synthetic shards in a temporary directory."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dlsd import harm_refusal as H
from dlsd import langprobe as LP
from dlsd import representation as R
from dlsd.activations import has_extended, load_shards

ROOT = Path(__file__).resolve().parents[1]


# ---------- capture (needs torch and transformers) ----------

@pytest.fixture(scope="module")
def tiny():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from test_activations import StubTok
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(vocab_size=100, hidden_size=32, intermediate_size=64, num_hidden_layers=3, num_attention_heads=4,
                                   num_key_value_heads=4, max_position_embeddings=256)
    return transformers.LlamaForCausalLM(cfg).eval(), StubTok()


def _manual(model, tok, user_text):
    import torch
    from dlsd.activations import chat_text
    text = chat_text(tok, user_text)
    enc = tok([text], return_tensors="pt", padding=True, truncation=True, max_length=1024, add_special_tokens=False)
    with torch.no_grad():
        hs = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], output_hidden_states=True).hidden_states
    return hs, 2, 2 + len(user_text)            # StubTok template "U:" + text + "\nA:", one token per character


def test_capture_positions_and_backward_compat(tiny):
    from dlsd import activations as A
    model, tok = tiny
    users = ["hi", "a much longer prompt here", "mid length"]
    c = A.capture(model, tok, users, [0, 2], batch_size=3)
    old = A.last_token_hidden(model, tok, [A.chat_text(tok, u) for u in users], [0, 2], batch_size=3)
    for l in (0, 2):
        assert np.allclose(c["last"][l], old[l], atol=1e-4)                 # L<l> unchanged
    for i, u in enumerate(users):
        hs, s, e = _manual(model, tok, u)
        for l in (0, 2):
            assert np.allclose(c["inst"][l][i], hs[l][0, e - 1].numpy(), atol=1e-4)
            assert np.allclose(c["mean"][l][i], hs[l][0, s:e].mean(0).numpy(), atol=1e-4)
        n = c["emb_norms"][c["emb_offsets"][i]:c["emb_offsets"][i + 1]]
        assert len(n) == len(u) and np.allclose(n, hs[0][0, s:e].norm(dim=-1).numpy(), atol=1e-4)
    assert c["span_exact"].all()


def test_capture_batched_equals_single(tiny):
    from dlsd import activations as A
    model, tok = tiny
    users = ["short", "a somewhat longer user message", "x"]
    b = A.capture(model, tok, users, [1, 3], batch_size=3)
    for i, u in enumerate(users):
        s = A.capture(model, tok, [u], [1, 3], batch_size=1)
        for k in ("last", "inst", "mean"):
            for l in (1, 3):
                assert np.allclose(b[k][l][i], s[k][l][0], atol=1e-4)


def test_content_span_stub(tiny):
    from dlsd.activations import content_span
    _, tok = tiny
    sp = content_span(tok, "hello")
    assert sp == {"start": 2, "end": 7, "n": 10, "exact": True}


def test_has_extended_and_load_shards(tmp_path):
    np.savez(tmp_path / "a.npz", condition_id="baseline|SAE", L1=np.zeros((2, 3)))
    np.savez(tmp_path / "b.tmp.npz", condition_id="dialect|AAVE", L1=np.zeros((2, 3)))
    sh = load_shards(str(tmp_path))
    assert list(sh) == ["baseline|SAE"] and not has_extended(sh["baseline|SAE"], [1])
    assert has_extended({"L1": 0, "I1": 0, "M1": 0, "emb_norms": 0, "emb_offsets": 0}, [1])


# ---------- representation ----------

def test_derangement_and_cka():
    for n in (2, 3, 10, 51):
        p = R.derangement(n, seed=n)
        assert sorted(p) == list(range(n)) and (p != np.arange(n)).all()
    rng = np.random.default_rng(0)
    x = rng.standard_normal((40, 8))
    q, _ = np.linalg.qr(rng.standard_normal((8, 8)))
    assert abs(R.linear_cka(x, 3 * x @ q) - 1) < 1e-9
    assert R.linear_cka(x, rng.standard_normal((40, 8))) < 0.5
    assert abs(R.normalized(0.6, 0.2) - 0.5) < 1e-12


def test_embedding_stats_and_layers():
    nl = [np.array([1.0, 2.0, 10.0]), np.array([1.0])]
    thr = R.norm_threshold(nl, 50)
    st = R.embedding_stats(nl, thr)
    assert thr == 1.5 and st["n_content_tokens"].tolist() == [3, 1]
    assert abs(st["share_above"][0] - 2 / 3) < 1e-12 and st["share_above"][1] == 0
    assert R.early_layers([1, 2, 8, 16, 24], 32, 0.25) == [1, 2, 8]
    assert R.nearest_layer([4, 14, 18, 28], 32, 0.5) == 14


def test_paired_similarity_floor():
    rng = np.random.default_rng(1)
    xe = rng.standard_normal((30, 16))
    xc = xe + 0.1 * rng.standard_normal((30, 16))
    res = R.paired_similarity(xc, xe, R.mismatched_floor(xe, 0), n_boot=200, seed=0)
    assert res["cosine"] > 0.95 and res["floor_mismatched"] < 0.5 and res["cosine_lo"] <= res["cosine"] <= res["cosine_hi"]


def test_comprehension_hook():
    bt = [{"item_id": "a", "language": "Swahili", "back": "x"}, {"item_id": "zz", "language": "Swahili", "back": "y"}]
    df = R.comprehension_from_backtranslation(bt, {"a": "ref"}, lambda h, r: 55.0, {"Swahili": "resource|low|Swahili"})
    assert df.to_dict("records") == [{"item_id": "a", "condition_id": "resource|low|Swahili", "comprehension": 55.0}]


# ---------- language probe ----------

def test_condition_labels():
    assert LP.condition_labels("baseline|SAE")["language"] == "en"
    assert LP.condition_labels("dialect|AAVE") == {"axis": "dialect", "monolingual": True, "language": "en", "variety": "AAVE", "partner": None}
    assert LP.condition_labels("resource|low|Swahili")["language"] == "sw"
    cs = LP.condition_labels("codeswitch_native|intra|Hindi")
    assert not cs["monolingual"] and cs["partner"] == "hi"


def test_group_folds_and_shuffle():
    groups = [f"i{k // 3}" for k in range(30)]
    f = LP.group_folds(groups, 5, seed=0)
    for g in set(groups):
        assert len({f[i] for i in range(30) if groups[i] == g}) == 1
    y = np.array(["a", "b", "c"] * 10)
    ys = LP.shuffle_within_groups(y, groups, 0)
    for g in set(groups):
        idx = [i for i in range(30) if groups[i] == g]
        assert sorted(ys[idx]) == sorted(y[idx])
    assert (LP.group_folds(groups, 5, 0) == f).all()


def _probe_data(seed=0, n_items=40, d=12):
    rng = np.random.default_rng(seed)
    conds = ["baseline|SAE", "dialect|AAVE", "resource|high|Spanish", "resource|low|Swahili"]
    lang_vec = {"en": 0, "es": 1, "sw": 2}
    X, y, g, c = [], [], [], []
    for cid in conds:
        lab = LP.condition_labels(cid)["language"]
        for i in range(n_items):
            v = rng.standard_normal(d)
            v[lang_vec[lab]] += 4
            X.append(v); y.append(lab); g.append(f"i{i}"); c.append(cid)
    return np.array(X), np.array(y), np.array(g), np.array(c)


def test_probe_beats_shuffled_control():
    X, y, g, c = _probe_data()
    cs = np.random.default_rng(5).standard_normal((10, X.shape[1]))
    cs[:, 1] += 4
    r = LP.probe_layer(X, y, g, c, k=5, C=1.0, max_iter=2000, seed=13, x_cs=cs, conds_cs=np.array(["codeswitch|intra|Spanish"] * 10))
    assert r["accuracy"] > 0.95 and r["accuracy_shuffled"] < 0.7 and abs(r["chance"] - 1 / 3) < 1e-12 and r["majority"] == 0.5
    assert r["cs_shares"]["codeswitch|intra|Spanish"]["es"] > 0.8
    acc, csh = LP.tidy({("language", "mean", 2): r})
    assert set(acc["condition_id"]) == {"ALL", *set(c)} and abs(csh["share"].sum() - 1) < 1e-9
    r2 = LP.probe_layer(X, y, g, c, k=5, C=1.0, max_iter=2000, seed=13)
    assert r2["accuracy"] == r["accuracy"]                                  # seeded and deterministic


# ---------- harmfulness versus refusal, output ----------

def _shard(rng, ids, labels, refuse_shift, harm_shift, d=10, layer=4, cid="baseline|SAE"):
    n = len(ids)
    I = rng.standard_normal((n, d)) * 0.3
    L = rng.standard_normal((n, d)) * 0.3
    I[:, 0] += harm_shift * labels
    L[:, 1] += refuse_shift
    return {"item_ids": np.array(ids), "labels": np.array(labels), f"I{layer}": I, f"L{layer}": L, f"M{layer}": I.copy(),
            "layers": np.array([layer]), "condition_id": cid, "model": "m"}


def test_harm_refusal_separation():
    rng = np.random.default_rng(0)
    n = 80
    ids = [f"h{k}" for k in range(n)] + [f"b{k}" for k in range(n)]
    labels = np.array([1] * n + [0] * n)
    refused = np.array([k % 2 for k in range(n)] + [0] * n)            # half the English harmful prompts refused
    eng = _shard(rng, ids, labels, refuse_shift=2.0 * refused, harm_shift=3.0)
    rows = [{"model_id": "m", "item_id": i, "condition_id": "baseline|SAE", "judge_label": "refuse" if r else "comply"}
            for i, r, l in zip(ids, refused, labels) if l]
    # dialect: harmfulness kept, nothing refused and refusal projection at the complied level
    dia = _shard(rng, ids, labels, refuse_shift=np.zeros(2 * n), harm_shift=3.0, cid="dialect|AAVE")
    rows += [{"model_id": "m", "item_id": i, "condition_id": "dialect|AAVE", "judge_label": "comply"} for i in ids[:n]]
    rows += [{"model_id": "other", "item_id": "h0", "condition_id": "dialect|AAVE", "judge_label": "refuse"}]
    lab = H.refusal_labels(rows, "m")
    assert lab[("h1", "baseline|SAE")] == 1 and lab[("h0", "dialect|AAVE")] == 0
    dirs = H.fit_directions(eng, 4, lab, "baseline|SAE", seed=13, min_per_class=10)
    assert dirs["d_refusal"] is not None and abs(dirs["d_harm"][0]) > 0.9 and abs(dirs["d_refusal"][1]) > 0.9
    e = H.condition_rows(eng, dirs, lab, "baseline|SAE", "baseline|SAE", 100, 13, 10)
    d = H.condition_rows(dia, dirs, lab, "dialect|AAVE", "baseline|SAE", 100, 13, 10)
    assert abs(e["harm_retention"] - 1) < 1e-9 and 0.3 < e["refusal_position"] < 0.7 and e["refusal_auroc_within"] > 0.95
    assert 0.8 < d["harm_retention"] < 1.2 and abs(d["refusal_position"]) < 0.15 and d["refusal_rate"] == 0.0
    proj = pd.concat([H.item_projections(z, dirs, lab, c) for c, z in (("baseline|SAE", eng), ("dialect|AAVE", dia))])
    proj["response_lang"] = ["en"] * (len(proj) - 3) + ["sw", "empty", "en"]
    agg = H.output_aggregates(proj, {"baseline|SAE": "en", "dialect|AAVE": "en"})
    assert set(agg["condition_id"]) == {"baseline|SAE", "dialect|AAVE"} and "proj_refusal_refused" in agg
    assert abs(agg.set_index("condition_id").loc["baseline|SAE", "share_response_en"] - 1) < 1e-12


def test_refusal_direction_skipped_when_too_few():
    rng = np.random.default_rng(0)
    ids = [f"h{k}" for k in range(20)] + [f"b{k}" for k in range(20)]
    labels = np.array([1] * 20 + [0] * 20)
    eng = _shard(rng, ids, labels, refuse_shift=np.zeros(40), harm_shift=3.0)
    dirs = H.fit_directions(eng, 4, {}, "baseline|SAE", 13, 10)
    assert dirs["d_refusal"] is None
    row = H.condition_rows(eng, dirs, {}, "baseline|SAE", "baseline|SAE", 0, 13, 10)
    assert "refusal_position" not in row and np.isnan(row["refusal_rate"])


# ---------- scripts 13 to 18, end to end on synthetic shards ----------

def _write_synthetic(tmp: Path):
    rng = np.random.default_rng(3)
    conds = ["baseline|SAE", "dialect|AAVE", "resource|low|Swahili", "codeswitch|intra|Swahili"]
    n, d, layers = 60, 12, [1, 2, 4]
    ids = [f"h{k}" for k in range(n)] + [f"b{k}" for k in range(n)]
    labels = np.array([1] * n + [0] * n)
    base = rng.standard_normal((2 * n, d))
    act = tmp / "results/activations/m"
    act.mkdir(parents=True)
    scored = []
    for ci, cid in enumerate(conds):
        arrays = {}
        for l in layers:
            noise = 0.2 * ci * rng.standard_normal((2 * n, d))
            shift = np.zeros(d); shift[3 + ci] = 2.0
            for p in "LIM":
                x = base + noise + shift
                x[:, 0] += 2.0 * labels
                arrays[f"{p}{l}"] = x
        refused = (rng.random(2 * n) < (0.6 if ci == 0 else 0.3)) & labels.astype(bool)
        arrays["L4"][:, 1] += 2.0 * refused
        lens = rng.integers(3, 9 + 3 * ci, 2 * n)
        np.savez(act / f"c{ci}.npz", item_ids=np.array(ids), labels=labels, layers=np.array(layers), condition_id=cid, model="m",
                 benign_source="medsafetybench_benign", emb_norms=rng.random(lens.sum()).astype(np.float32) + ci * 0.1,
                 emb_offsets=np.concatenate([[0], np.cumsum(lens)]), span_exact=np.ones(2 * n, bool), n_layers=8, capture_version=2, **arrays)
        for i, r, l in zip(ids, refused, labels):
            if l:
                scored.append({"model_id": "m", "item_id": i, "condition_id": cid, "judge_label": "refuse" if r else "comply",
                               "response_text": "I cannot help with that request." if r else ("Habari, hii ni jibu." if ci >= 2 else "Here is an answer.")})
    (tmp / "results/scored").mkdir(parents=True)
    with open(tmp / "results/scored/scored_exp1.jsonl", "w") as f:
        for r in scored:
            f.write(json.dumps(r) + "\n")
    (tmp / "configs").mkdir()
    (tmp / "configs/experiment.yaml").write_text((ROOT / "configs/experiment.yaml").read_text())
    with open(tmp / "comp.jsonl", "w") as f:
        for k, i in enumerate(ids):
            f.write(json.dumps({"item_id": i, "condition_id": "resource|low|Swahili", "comprehension": 30.0 + k}) + "\n")


def _run(tmp, script, *args):
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / script), "results/activations/m", *args], cwd=tmp, env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    return r


def test_scripts_13_to_18_end_to_end(tmp_path):
    pytest.importorskip("sklearn")
    _write_synthetic(tmp_path)
    agg = tmp_path / "results/aggregate"
    _run(tmp_path, "13_embedding.py", "--n-boot", "20")
    emb = pd.read_csv(agg / "e1_embedding_m.csv")
    assert len(emb) == 8 and (agg / "e1_embedding_gaps_m.csv").exists()
    _run(tmp_path, "14_early_assembly.py", "--n-boot", "20")
    ea = pd.read_csv(agg / "e1_early_assembly_m.csv")
    assert set(ea["layer"]) == {1, 2} and (ea.loc[ea.condition_id == "baseline|SAE", "cosine"] == 1.0).all()
    _run(tmp_path, "15_language_probe.py")
    lp = pd.read_csv(agg / "e1_langprobe_m.csv")
    allv = lp[(lp.condition_id == "ALL") & (lp.task == "variety")]
    assert (allv["accuracy"] > allv["accuracy_shuffled"]).all()
    assert (agg / "e1_langprobe_codeswitch_m.csv").exists()
    _run(tmp_path, "16_semantic.py", "--comprehension", "comp.jsonl", "--n-boot", "20")
    se = pd.read_csv(agg / "e1_semantic_m.csv").set_index("condition_id")
    assert se.loc["resource|low|Swahili", "n_comprehension"] == 120 and se["layer"].iloc[0] == 4
    _run(tmp_path, "17_harm_refusal.py", "--layer", "4", "--n-boot", "20")
    hr = json.loads((agg / "e1_harm_refusal_m.json").read_text())
    assert hr["refusal_direction_fitted"] and len(hr["records"]) == 4
    _run(tmp_path, "18_output_join.py", "--layer", "4")
    out = pd.read_csv(agg / "e1_output_m.csv")
    assert len(out) == 4 and "item_id" not in out.columns and "response_text" not in out.columns
