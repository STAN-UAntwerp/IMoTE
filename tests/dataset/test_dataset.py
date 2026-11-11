import numpy as np
import pytest

from imote.dataset.dataset import Dataset


def test_defaults():
    ds = Dataset(np.zeros((4, 2)), np.zeros((4, 1)))
    assert (ds.n_samples, ds.n_features) == (4, 2)
    assert ds.feature_names == ["X0", "X1"]
    assert ds.target_name == "y"
    assert ds.y.shape == (4,)


@pytest.mark.parametrize("X, y, feature_names", [
    (np.zeros(4), np.zeros(4), None),
    (np.zeros((4, 2)), np.zeros(3), None),
    (np.zeros((4, 2)), np.zeros(4), ["a"]),
])
def test_invalid_input_raises(X, y, feature_names):
    with pytest.raises(ValueError):
        Dataset(X, y, feature_names=feature_names)


def test_categorical_auto_detection():
    X = np.column_stack([np.arange(20) % 3, np.arange(20)])
    assert Dataset(X, np.zeros(20), cat_unique_threshold=8).cat_ids == [0]
    assert Dataset(X[:, 1:], np.zeros(20), cat_unique_threshold=8).cat_ids == [-1]


def test_task_and_classes():
    ds = Dataset(np.zeros((4, 1)), [0, 1, 2, 0], task="classification", classes=["a", "b", "c"])
    assert ds.task == "classification" and ds.classes == ["a", "b", "c"]
    assert Dataset(np.zeros((4, 1)), np.zeros(4)).task == "regression"
