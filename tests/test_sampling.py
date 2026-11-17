"""Loader and sampling tests on tiny fixtures that mimic the real file layouts (checked against one real row of each)."""
import csv
import json
import zipfile

import pandas as pd

from dlsd.sampling import load_source, sample_all, _sample


def _med_dir(tmp_path):
    d = tmp_path / "gpt4"
    d.mkdir()
    for p in (1, 2, 3):
        with open(d / f"med_safety_demonstrations_category_{p}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["", "harmful_medical_request", "safe_response"])
            for i in range(10):
                w.writerow([i, f"harmful request {p}-{i}", "I cannot help."])
    return d


def _mmed_zip(tmp_path):
    z = tmp_path / "MMedBench.zip"
    rows = [{"question": f"Q{i}?", "answer": "x", "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
             "answer_idx": "BCDA"[i % 4]} for i in range(20)]
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("MMedBench/Test/English.jsonl", "\n".join(json.dumps(r) for r in rows))
    return z


def _wmdp(tmp_path):
    p = tmp_path / "wmdp.parquet"
    pd.DataFrame({"question": [f"W{i}?" for i in range(20)], "choices": [["a", "b", "c", "d"]] * 20,
                  "answer": [i % 4 for i in range(20)]}).to_parquet(p)
    return p


def _cfg(tmp_path, n=9):
    return {"seed": 13, "n_items_per_source": n, "sources": {
        "safety": {"name": "medsafetybench", "loader": "medsafety_csv_dir", "path": str(_med_dir(tmp_path)), "stratify_by": "principle"},
        "benign": {"name": "mmedbench", "loader": "mmedbench_local", "path": str(_mmed_zip(tmp_path)), "language": "English", "split_dir": "Test"},
        "hazard": {"name": "wmdp_bio", "loader": "wmdp_parquet", "path": str(_wmdp(tmp_path))}}}


def test_loader_fields(tmp_path):
    cfg = _cfg(tmp_path)
    safety = load_source(cfg["sources"]["safety"])
    assert len(safety) == 30 and safety[0]["options"] is None and safety[0]["principle"] == "principle_1"
    mm = load_source(cfg["sources"]["benign"])
    assert mm[0]["options"] == {"A": "a", "B": "b", "C": "c", "D": "d"} and mm[0]["answer"] == "B"
    w = load_source(cfg["sources"]["hazard"])
    assert list(w[1]["options"]) == ["A", "B", "C", "D"] and w[1]["answer"] == "B"


def test_sampling_is_stratified_exact_and_deterministic(tmp_path):
    cfg = _cfg(tmp_path, n=10)
    a = sample_all(cfg)
    assert a == sample_all(cfg)
    safety = [i for i in a if i["source"] == "medsafetybench"]
    assert len(safety) == 10
    counts = sorted(sum(i["principle"] == f"principle_{p}" for i in safety) for p in (1, 2, 3))
    assert counts == [3, 3, 4]          # 10 over 3 principles: remainder is distributed, nothing is lost
    assert len({i["item_id"] for i in a}) == len(a)


def test_sources_use_different_draws():
    pool = list(range(100))
    assert _sample(pool, 10, "13-a") != _sample(pool, 10, "13-b")
