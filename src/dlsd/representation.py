"""Representation measures for E1 stages 2, 3 and 5 (docs/EXPERIMENTS_LANGUAGE_INTERFACE.md, stage map). Pure numpy and pandas.

Stage 2 (embedding): per-prompt mean of the embedding norms of the user-content tokens (hidden_states[0]) and the share of tokens above a
percentile of the English reference distribution. Token count per prompt also gives a premium against the item's English prompt, so the
condition gap in mean norm can be adjusted for fragmentation with dlsd.tokenization.adjusted_gaps.
Stage 3 (early-layer assembly) and stage 5 (semantic representation): paired similarity of mean-pooled states between each rendered
prompt and the same item's English prompt, next to a mismatched-item floor (each English prompt against a different English item), so
a similarity can be read against "unrelated item" (floor) and "same English prompt" (1.0 by construction). Linear CKA (Kornblith et al. 2019)
compares whole condition matrices against English over the same items.
All functions are descriptive. Nothing here says which layer is "early" or "mid": those are config keys (e1_stage_map), provisional.
"""
import numpy as np
import pandas as pd


def split_norms(emb_norms: np.ndarray, offsets: np.ndarray) -> list:
    return [emb_norms[offsets[i]:offsets[i + 1]] for i in range(len(offsets) - 1)]


def norm_threshold(reference_norms: list, percentile: float) -> float:
    flat = np.concatenate([np.asarray(x, float) for x in reference_norms if len(x)])
    return float(np.percentile(flat, percentile))


def embedding_stats(norm_lists: list, threshold: float) -> pd.DataFrame:
    rows = []
    for n in norm_lists:
        n = np.asarray(n, float)
        rows.append({"n_content_tokens": len(n), "mean_norm": float(n.mean()) if len(n) else np.nan,
                     "share_above": float((n > threshold).mean()) if len(n) else np.nan})
    return pd.DataFrame(rows)


def rowwise_cosine(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    nx, ny = np.linalg.norm(x, axis=1), np.linalg.norm(y, axis=1)
    return (x * y).sum(1) / np.maximum(nx * ny, 1e-12)


def derangement(n: int, seed: int) -> np.ndarray:
    """Seeded permutation with no fixed point (n >= 2): row i is paired with a different item."""
    if n < 2:
        raise ValueError("need at least two items for a mismatched pairing")
    rng = np.random.default_rng(seed)
    p = rng.permutation(n)
    return np.roll(p, 1)[np.argsort(p)]          # cycle through a random order: no element maps to itself


def mismatched_floor(x: np.ndarray, seed: int) -> np.ndarray:
    """Cosine of each reference row with a different reference row (unrelated-item floor)."""
    return rowwise_cosine(x, x[derangement(len(x), seed)])


def normalized(sim: float, floor: float, ceiling: float = 1.0) -> float:
    """Where a similarity sits between the mismatched-item floor (0) and the same-prompt ceiling (1)."""
    return float((sim - floor) / (ceiling - floor)) if ceiling != floor else float("nan")


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear CKA between two (n, d) matrices with rows matched (Kornblith et al. 2019). 1 = same up to rotation and scale."""
    x = x - x.mean(0)
    y = y - y.mean(0)
    hsic = np.linalg.norm(y.T @ x) ** 2
    den = np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y)
    return float(hsic / den) if den > 0 else float("nan")


def align(ids_a, ids_b) -> tuple:
    """Index arrays (ia, ib) of item ids present in both, in sorted id order."""
    pa = {str(i): k for k, i in enumerate(ids_a)}
    pb = {str(i): k for k, i in enumerate(ids_b)}
    common = sorted(set(pa) & set(pb))
    return np.array([pa[i] for i in common], dtype=int), np.array([pb[i] for i in common], dtype=int), common


def early_layers(layers: list, n_layers: int, max_fraction: float) -> list:
    return [l for l in layers if l <= max_fraction * n_layers]


def nearest_layer(layers: list, n_layers: int, fraction: float) -> int:
    target = fraction * n_layers
    return min(layers, key=lambda l: (abs(l - target), l))


def boot_ci(values, n_boot: int, seed: int, alpha: float = 0.05) -> tuple:
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    if n_boot <= 0 or len(v) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    m = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(1)
    return float(np.quantile(m, alpha / 2)), float(np.quantile(m, 1 - alpha / 2))


def paired_similarity(x_cond: np.ndarray, x_ref: np.ndarray, floor_values: np.ndarray, n_boot: int = 0, seed: int = 13) -> dict:
    """Paired cosine (rows matched by item) with its interval, the floor and the normalized position between floor and 1."""
    c = rowwise_cosine(x_cond, x_ref)
    fl = float(np.mean(floor_values))
    lo, hi = boot_ci(c, n_boot, seed)
    return {"n": len(c), "cosine": float(c.mean()), "cosine_lo": lo, "cosine_hi": hi, "floor_mismatched": fl,
            "normalized": normalized(float(c.mean()), fl)}


def comprehension_from_backtranslation(bt_rows: list, items: dict, score_fn, condition_of_language: dict) -> pd.DataFrame:
    """Restatement scores in the scripts/02q pattern: rows {item_id, language, back}; score_fn(hypothesis, reference) -> float
    (chrF++ in 02q). Returns item_id, condition_id, comprehension."""
    out = []
    for r in bt_rows:
        if r["item_id"] in items and r.get("language") in condition_of_language:
            out.append({"item_id": r["item_id"], "condition_id": condition_of_language[r["language"]],
                        "comprehension": float(score_fn(r["back"], items[r["item_id"]]))})
    return pd.DataFrame(out, columns=["item_id", "condition_id", "comprehension"])
