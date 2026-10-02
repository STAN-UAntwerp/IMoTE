import numpy as np
import pandas as pd
import pytest

from dataset.dataset import Dataset


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame({"a": [1, 2, 3], "target": [10, 20, 30], "b": [4, 5, 6]}).to_csv(path, index=False)
    return path


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


@pytest.mark.parametrize("target_col, y, feature_names", [
    ("target", [10, 20, 30], ["a", "b"]),
    (0, [1, 2, 3], ["target", "b"]),
    ("0", [1, 2, 3], ["target", "b"]),
    (-1, [4, 5, 6], ["a", "target"]),
])
def test_from_csv_target(csv_path, target_col, y, feature_names):
    ds = Dataset.from_csv(str(csv_path), target_col)
    np.testing.assert_array_equal(ds.y, y)
    assert ds.feature_names == feature_names
    assert ds.X.shape == (3, 2)


@pytest.mark.parametrize("target_col", ["missing", 3])
def test_from_csv_unknown_target_raises(csv_path, target_col):
    with pytest.raises(ValueError):
        Dataset.from_csv(str(csv_path), target_col)


def test_from_csv_non_numeric_raises(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame({"a": ["x", "y"], "target": [1, 2]}).to_csv(path, index=False)
    with pytest.raises(TypeError):
        Dataset.from_csv(str(path), "target")


def test_from_pmlb(monkeypatch):
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "c": [0.0, 1.0], "target": [5.0, 6.0]})
    monkeypatch.setattr("pmlb.fetch_data", lambda name: df)
    ds = Dataset.from_pmlb("1199_BNG_echoMonths")
    np.testing.assert_array_equal(ds.y, [5.0, 6.0])
    assert ds.feature_names == ["a", "b", "c"]
    assert ds.name == "1199_BNG_echoMonths"
    np.testing.assert_array_equal(ds.cat_ids, [0, 2, 8])  # from PMLB_DATASETS_CAT_IDS
