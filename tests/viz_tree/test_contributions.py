import numpy as np
import pytest

from tests.helpers import X, make_pilot_tree, make_split_tree


@pytest.mark.parametrize("row", range(2))
@pytest.mark.parametrize("make_tree", [make_pilot_tree, make_split_tree])
def test_test_contributions_match_train_contributions(make_tree, row):
    tree = make_tree()
    split, lin = tree.get_contributions(X[row:row + 1])
    np.testing.assert_allclose(split[0], tree.split_contributions[row])
    np.testing.assert_allclose(lin[0], tree.linear_contributions[row])


@pytest.mark.parametrize("make_tree", [make_pilot_tree, make_split_tree])
def test_contributions_add_up_to_prediction(make_tree):
    # mean prediction + split contributions + linear contributions = prediction
    tree = make_tree()
    total = np.mean(tree.y_hat) + tree.split_contributions.sum(axis=1) + tree.linear_contributions.sum(axis=1)
    np.testing.assert_allclose(total, tree.y_hat)


def test_pilot_contributions_row_0(pilot_tree):
    # leaf averages: node4 = 1.3, node5 = 2.05, node3 = -0.525 -> Blin node2 = 1.675, root = 0.575
    # split: Pcon on x0 = 1.675 - 0.575, Blin on x2 = 1.3 - 1.675
    # linear: leaf node4 3*x2 on x2 = 3 * (0 - mean(x2 over rows 0, 2))
    np.testing.assert_allclose(pilot_tree.split_contributions[0], [1.1, 0, -0.375])
    np.testing.assert_allclose(pilot_tree.linear_contributions[0], [0, 0, -0.3])


def test_unused_features_have_no_contributions(pilot_tree, split_tree):
    # x1 is never used in the Pilot tree; the split tree never splits on x2 and no leaf uses x1
    assert not pilot_tree.split_contributions[:, 1].any()
    assert not pilot_tree.linear_contributions[:, 1].any()
    assert not split_tree.split_contributions[:, 2].any()
    assert not split_tree.linear_contributions[:, 1].any()
