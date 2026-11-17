"""Direction arithmetic for the refusal-direction experiment (spec: docs/EXPERIMENTS_LANGUAGE_INTERFACE.md, E1).

Pure numpy. Activation extraction from a model lives elsewhere; these functions take arrays of shape (n_prompts, d_model)
taken at one layer and one token position, so they can be tested without a GPU.
"""
import numpy as np


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("zero vector has no direction")
    return v / n


def diff_in_means(positive: np.ndarray, negative: np.ndarray) -> np.ndarray:
    """Unit direction from the mean of `negative` to the mean of `positive` (difference in means, as in Arditi et al. 2024)."""
    return unit(positive.mean(axis=0) - negative.mean(axis=0))


def project(acts: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Scalar projection of each row onto a unit direction."""
    return acts @ unit(direction)


def separation(positive: np.ndarray, negative: np.ndarray, direction: np.ndarray) -> float:
    """Mean projection of `positive` minus mean projection of `negative` on `direction`."""
    return float(project(positive, direction).mean() - project(negative, direction).mean())


def retention(sep_condition: float, sep_english: float) -> float:
    """Share of the English harmful-versus-benign separation that a condition keeps (1.0 = all, 0.0 = none)."""
    if sep_english == 0:
        raise ValueError("English separation is zero")
    return sep_condition / sep_english


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(unit(a) @ unit(b))


def random_unit(d: int, seed: int) -> np.ndarray:
    """Matched control direction (Zhang 2026: a raw cosine is uninformative unless a control is compared)."""
    return unit(np.random.default_rng(seed).standard_normal(d))


def ablate(acts: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Remove the component of each row along the unit direction."""
    u = unit(direction)
    return acts - np.outer(acts @ u, u)


def auroc(scores_pos: np.ndarray, scores_neg: np.ndarray) -> float:
    """Probability that a random positive scores above a random negative (ties count half)."""
    p = np.asarray(scores_pos, dtype=float)[:, None]
    n = np.asarray(scores_neg, dtype=float)[None, :]
    return float(((p > n).mean()) + 0.5 * ((p == n).mean()))


def pick_layer(acts_by_layer: dict, labels: np.ndarray, item_ids: list, seed: int) -> dict:
    """Choose the layer on English data only, before any other condition is looked at.

    Items are split in half by a seeded shuffle of item ids. For each layer the direction comes from half A and is scored by the
    AUROC of its projection on half B (harmful versus benign). Returns the chosen layer, the table of AUROCs and the split.
    """
    ids = sorted(set(item_ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    half_a = set(ids[: len(ids) // 2])
    in_a = np.array([i in half_a for i in item_ids])
    labels = np.asarray(labels).astype(bool)
    table = {}
    for layer, acts in acts_by_layer.items():
        d = diff_in_means(acts[in_a & labels], acts[in_a & ~labels])
        s = project(acts[~in_a], d)
        table[layer] = auroc(s[labels[~in_a]], s[~labels[~in_a]])
    best = max(table, key=lambda k: (table[k], -k))
    return {"layer": best, "heldout_auroc": table, "half_a": sorted(half_a)}


def condition_summary(acts: np.ndarray, labels: np.ndarray, d_english: np.ndarray, sep_english: float, control: np.ndarray) -> dict:
    """Projection summary for one condition against the English direction and a matched random control."""
    labels = np.asarray(labels).astype(bool)
    pos, neg = acts[labels], acts[~labels]
    d_cond = diff_in_means(pos, neg)
    sep = separation(pos, neg, d_english)
    return {
        "n_harmful": int(labels.sum()), "n_benign": int((~labels).sum()),
        "proj_harmful": float(project(pos, d_english).mean()), "proj_benign": float(project(neg, d_english).mean()),
        "separation": sep, "retention": retention(sep, sep_english),
        "auroc_english_direction": auroc(project(pos, d_english), project(neg, d_english)),
        "cosine_to_english": cosine(d_cond, d_english), "cosine_control": cosine(d_cond, control),
        "retention_control": retention(separation(pos, neg, control), sep_english),
    }
