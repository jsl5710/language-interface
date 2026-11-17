import numpy as np
import pytest
from dlsd import mechanism as m


def toy(shift, n=200, d=16, seed=0):
    rng = np.random.default_rng(seed)
    u = np.zeros(d); u[0] = 1.0
    harmful = rng.standard_normal((n, d)) * 0.1 + shift * u
    benign = rng.standard_normal((n, d)) * 0.1
    return harmful, benign, u


def test_direction_recovers_axis():
    h, b, u = toy(2.0)
    assert m.cosine(m.diff_in_means(h, b), u) > 0.99


def test_unit_length():
    h, b, _ = toy(2.0)
    assert np.isclose(np.linalg.norm(m.diff_in_means(h, b)), 1.0)


def test_retention_half_signal():
    h, b, _ = toy(2.0, seed=1)
    d = m.diff_in_means(h, b)
    h2, b2, _ = toy(1.0, seed=2)
    r = m.retention(m.separation(h2, b2, d), m.separation(h, b, d))
    assert 0.45 < r < 0.55


def test_random_control_is_near_orthogonal_in_high_dim():
    h, b, _ = toy(2.0, d=4096)
    d = m.diff_in_means(h, b)
    assert abs(m.cosine(d, m.random_unit(4096, 3))) < 0.1


def test_ablate_removes_component():
    h, b, _ = toy(2.0)
    d = m.diff_in_means(h, b)
    assert np.allclose(m.project(m.ablate(h, d), d), 0, atol=1e-9)


def test_zero_vector_and_zero_english_rejected():
    with pytest.raises(ValueError):
        m.unit(np.zeros(3))
    with pytest.raises(ValueError):
        m.retention(1.0, 0.0)
