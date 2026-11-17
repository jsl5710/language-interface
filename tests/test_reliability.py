import numpy as np

from dlsd.reliability import fleiss_kappa, gwet_ac1, krippendorff_alpha, pairwise_cohen, percent_agreement


def test_perfect_agreement():
    m = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [0, 0, 0]])
    assert percent_agreement(m, 3) == 1 and fleiss_kappa(m, 3) == 1 and krippendorff_alpha(m, 3) == 1 and gwet_ac1(m, 3) == 1


def test_fleiss_textbook_example():
    # Wikipedia worked example: 10 items, 14 raters, 5 categories, kappa = 0.210
    counts = [[0, 0, 0, 0, 14], [0, 2, 6, 4, 2], [0, 0, 3, 5, 6], [0, 3, 9, 2, 0], [2, 2, 8, 1, 1],
              [7, 7, 0, 0, 0], [3, 2, 6, 3, 0], [2, 5, 3, 2, 2], [6, 5, 2, 1, 0], [0, 2, 2, 3, 7]]
    m = np.array([[c for c, n in enumerate(row) for _ in range(n)] for row in counts])
    assert abs(fleiss_kappa(m, 5) - 0.210) < 0.001


def test_prevalence_paradox_and_missing():
    m = np.array([[0, 0, 0, 0]] * 97 + [[0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]])
    assert percent_agreement(m, 2) > 0.97 and fleiss_kappa(m, 2) < 0.5 < gwet_ac1(m, 2)      # kappa punishes skewed data, AC1 does not
    mm = np.array([[0, 0, -1], [1, 1, 1], [0, 1, 0], [1, 1, -1]])
    assert 0 < krippendorff_alpha(mm, 2) < 1


def test_pairwise_cohen_known():
    m = np.array([[0, 0], [0, 0], [1, 1], [1, 0]])        # po=.75, pe=.5 -> kappa .5
    assert abs(pairwise_cohen(m, 2)[0] - 0.5) < 1e-9
