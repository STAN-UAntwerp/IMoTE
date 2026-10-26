import numpy as np
import pandas as pd
import pytest

from imote.dataset.dataset import Dataset, fetch_pmlb


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


PMLB_DF = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "c": [0.0, 1.0], "target": [5.0, 6.0]})


@pytest.fixture
def downloads(monkeypatch):
    names = []

    def fake_fetch_data(name):
        names.append(name)
        return PMLB_DF.copy()

    monkeypatch.setattr("pmlb.fetch_data", fake_fetch_data)
    return names


def test_from_pmlb(downloads, tmp_path):
    ds = Dataset.from_pmlb("1199_BNG_echoMonths", cache_dir=tmp_path)
    np.testing.assert_array_equal(ds.y, [5.0, 6.0])
    assert ds.feature_names == ["a", "b", "c"]
    assert ds.name == "1199_BNG_echoMonths"
    np.testing.assert_array_equal(ds.cat_ids, [0, 2, 8])  # from PMLB_DATASETS_CAT_IDS


def test_fetch_pmlb_keeps_exact_values(monkeypatch, tmp_path):
    df = pd.DataFrame({"a": [0.1 + 0.2, 1 / 3, 1e-300], "target": [2 / 7, -1e10 / 3, 5.0]})
    monkeypatch.setattr("pmlb.fetch_data", lambda name: df)
    fetch_pmlb("exact", tmp_path)
    np.testing.assert_array_equal(fetch_pmlb("exact", tmp_path).to_numpy(), df.to_numpy())


def test_fetch_pmlb_without_connection_saves_nothing(monkeypatch, tmp_path):
    def offline(name):
        raise OSError("no connection")

    monkeypatch.setattr("pmlb.fetch_data", offline)
    with pytest.raises(ConnectionError):
        fetch_pmlb("547_no2", tmp_path)
