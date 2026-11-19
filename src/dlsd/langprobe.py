"""Stage 4 of E1: a linear probe for language or variety identity, per layer (docs/EXPERIMENTS_LANGUAGE_INTERFACE.md, stage map).

This measures whether language or variety is linearly decodable from a layer's states. It does not test a routing mechanism: Minionese
(Ifebi, Kong, Mehrotra 2026) hypothesizes that code-switching disrupts language-identification routing and leaves testing that to future
work, so decodability here is reported as decodability only.

Design: train on the monolingual conditions (SAE, the dialects, the translations); folds are grouped by item, so no item is in both train
and test; out-of-fold accuracy per layer, overall and per condition, with a shuffled-label control (labels permuted within each item, so
every item keeps its set of labels), chance (1 / number of classes) and the majority-class rate (classes are unbalanced: in the language task the
SAE and dialect conditions are all English). Code-switched conditions have no single correct label: the probe
fitted on all monolingual rows assigns them a label, and the share given to English varieties versus the partner language is reported.
"""
import numpy as np
import pandas as pd

LANG = {"English": "en", "Spanish": "es", "Hindi": "hi", "Swahili": "sw"}


def condition_labels(cid: str) -> dict:
    """Language and variety label for a condition id (axis|level[|language])."""
    parts = cid.split("|")
    axis = parts[0]
    if axis == "baseline":
        return {"axis": axis, "monolingual": True, "language": "en", "variety": parts[1], "partner": None}
    if axis == "dialect":
        return {"axis": axis, "monolingual": True, "language": "en", "variety": parts[1], "partner": None}
    if axis == "resource":
        lang = LANG.get(parts[2], parts[2])
        return {"axis": axis, "monolingual": True, "language": lang, "variety": parts[2], "partner": lang}
    lang = LANG.get(parts[2], parts[2])                          # codeswitch or codeswitch_native: two languages present
    return {"axis": axis, "monolingual": False, "language": None, "variety": None, "partner": lang}


def group_folds(groups, k: int, seed: int) -> np.ndarray:
    """Fold index per row; all rows of a group share a fold. Seeded."""
    uniq = sorted({str(g) for g in groups})
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    fold_of = {g: i % k for i, g in enumerate(uniq)}
    return np.array([fold_of[str(g)] for g in groups])


def shuffle_within_groups(y, groups, seed: int) -> np.ndarray:
    y = np.asarray(y, dtype=object).copy()
    rng = np.random.default_rng(seed)
    groups = np.asarray([str(g) for g in groups])
    for g in sorted(set(groups)):
        idx = np.where(groups == g)[0]
        y[idx] = y[rng.permutation(idx)]
    return y


def _model(C: float, max_iter: int, seed: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=max_iter, random_state=seed))


def oof_predict(x: np.ndarray, y, groups, k: int, C: float, max_iter: int, seed: int) -> np.ndarray:
    folds = group_folds(groups, k, seed)
    y = np.asarray(y, dtype=object)
    pred = np.empty(len(y), dtype=object)
    for f in range(k):
        te = folds == f
        if not te.any():
            continue
        m = _model(C, max_iter, seed).fit(x[~te], y[~te].astype(str))
        pred[te] = m.predict(x[te])
    return pred


def probe_layer(x_mono: np.ndarray, y, groups, conds, k: int, C: float, max_iter: int, seed: int,
                x_cs: np.ndarray | None = None, conds_cs=None) -> dict:
    """Out-of-fold accuracy (overall, per condition), shuffled control, chance; and label shares for code-switched rows."""
    y = np.asarray(y, dtype=object).astype(str)
    conds = np.asarray(conds)
    pred = oof_predict(x_mono, y, groups, k, C, max_iter, seed)
    y_sh = shuffle_within_groups(y, groups, seed).astype(str)
    pred_sh = oof_predict(x_mono, y_sh, groups, k, C, max_iter, seed)
    classes = sorted(set(y))
    per_cond = {c: {"accuracy": float((pred[conds == c] == y[conds == c]).mean()),
                    "accuracy_shuffled": float((pred_sh[conds == c] == y_sh[conds == c]).mean()), "n": int((conds == c).sum())}
                for c in sorted(set(conds))}
    res = {"accuracy": float((pred == y).mean()), "accuracy_shuffled": float((pred_sh == y_sh).mean()),
           "chance": 1.0 / len(classes), "majority": float(max((y == c).mean() for c in classes)), "n_classes": len(classes), "per_condition": per_cond, "cs_shares": {}}
    if x_cs is not None and len(x_cs):
        full = _model(C, max_iter, seed).fit(x_mono, y)
        pc = full.predict(x_cs)
        conds_cs = np.asarray(conds_cs)
        for c in sorted(set(conds_cs)):
            p = pc[conds_cs == c]
            res["cs_shares"][c] = {lab: float((p == lab).mean()) for lab in classes}
    return res


def tidy(results: dict) -> tuple:
    """results[(task, position, layer)] = probe_layer output -> (accuracy table, code-switch share table)."""
    acc, cs = [], []
    for (task, pos, layer), r in sorted(results.items()):
        acc.append({"task": task, "position": pos, "layer": layer, "condition_id": "ALL", "n": sum(v["n"] for v in r["per_condition"].values()),
                    "accuracy": r["accuracy"], "accuracy_shuffled": r["accuracy_shuffled"], "chance": r["chance"],
                    "majority": r["majority"]})
        for c, v in r["per_condition"].items():
            acc.append({"task": task, "position": pos, "layer": layer, "condition_id": c, "n": v["n"], "accuracy": v["accuracy"],
                        "accuracy_shuffled": v["accuracy_shuffled"], "chance": r["chance"], "majority": r["majority"]})
        for c, shares in r["cs_shares"].items():
            for lab, s in shares.items():
                cs.append({"task": task, "position": pos, "layer": layer, "condition_id": c, "predicted_label": lab, "share": s})
    return pd.DataFrame(acc), pd.DataFrame(cs)
