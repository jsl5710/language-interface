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
