import numpy as np
import pytest

from tests.helpers import X, make_pilot_tree, make_split_tree
from viz_tree.contributions import get_split_and_lin_contributions_test


@pytest.mark.parametrize("row", range(2))
@pytest.mark.parametrize("make_tree", [make_pilot_tree, make_split_tree])
def test_test_contributions_match_train_contributions(make_tree, row):
    # Covers both the Pilot code path and the recursive non-Pilot code path
    tree = make_tree()
    split, lin = get_split_and_lin_contributions_test(tree, X[row])
    np.testing.assert_allclose(split, tree.split_contributions[row])
    np.testing.assert_allclose(lin, tree.linear_contributions[row])


def test_pilot_contributions_row_0(pilot_tree):
    # split: Pcon on x0 = left_avg - weighted_avg = 1 - 0
    #        Blin on x2 = mean(2*x2 over rows 0, 2) - (0.2*2 + 0.1*2) / 4 = 0.2 - 0.15
    # linear: LinearNode on x2 = x2 - mean(x2) = 0 - 0.5, Blin left on x2 = 2*0 - 0.2
    np.testing.assert_allclose(pilot_tree.split_contributions[0], [1, 0, 0.05])
    np.testing.assert_allclose(pilot_tree.linear_contributions[0], [0, 0, -0.7])


def test_unused_features_have_no_contributions(pilot_tree, split_tree):
    # x1 is never used in the Pilot tree; the split tree never splits on x2 and no leaf uses x1
    assert not pilot_tree.split_contributions[:, 1].any()
    assert not pilot_tree.linear_contributions[:, 1].any()
    assert not split_tree.split_contributions[:, 2].any()
    assert not split_tree.linear_contributions[:, 1].any()
