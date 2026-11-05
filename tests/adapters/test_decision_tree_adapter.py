import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from imote.adapters.decision_tree_adapter import DecisionTreeClassifierAdapter, DecisionTreeRegressorAdapter
from imote.callbacks.c_new_tree_callbacks import fit_new_tree
from imote.dataset.dataset import Dataset
from imote.nodes.leaf_node import LeafNode
from imote.nodes.split_node import SplitNode
from imote.viz_tree.viz_tree import VizTree

RNG = np.random.default_rng(0)
X = RNG.normal(size=(300, 4))
y_reg = X[:, 0] + np.sign(X[:, 1]) + RNG.normal(scale=0.1, size=300)
y_class = (X[:, 0] + RNG.normal(scale=0.5, size=300) > 0).astype(float)


@pytest.mark.parametrize("adapter, model_cls, y", [
    (DecisionTreeRegressorAdapter, DecisionTreeRegressor, y_reg),
    (DecisionTreeClassifierAdapter, DecisionTreeClassifier, y_class),
])
def test_tree_matches_sklearn(adapter, model_cls, y):
    model = model_cls(max_depth=3, min_samples_leaf=5, random_state=0).fit(X, y)
    tree = VizTree.from_model(adapter, X, y, model)

    assert tree.task == adapter.task
    assert len(tree.nodes) == model.tree_.node_count
    assert sum(isinstance(n, LeafNode) for n in tree.nodes) == model.get_n_leaves()
    assert all(isinstance(n, (LeafNode, SplitNode)) for n in tree.nodes)
    np.testing.assert_allclose(tree.predict(X), tree.y_hat)
    VizTree.from_dict(tree.to_dict())


def test_regression_residuals_are_rss_of_the_node():
    model = DecisionTreeRegressor(max_depth=2, random_state=0).fit(X, y_reg)
    tree = VizTree.from_model(DecisionTreeRegressorAdapter, X, y_reg, model)
    root = tree.root_node
    assert root.rss == pytest.approx(np.sum((y_reg - y_reg.mean()) ** 2))
    assert root.log_odds is None


def test_classification_log_odds_of_a_pure_leaf_are_finite():
    model = DecisionTreeClassifier(random_state=0).fit(X, y_class)
    tree = VizTree.from_model(DecisionTreeClassifierAdapter, X, y_class, model)
    leaf = next(n for n in tree.nodes if isinstance(n, LeafNode))
    assert np.isfinite(leaf.log_odds).all()
    assert np.isfinite(leaf.y_res).all()
    assert ((tree.y_hat > 0) == y_class).all()


@pytest.fixture
def load_dataset(monkeypatch):
    def use(y):
        dataset = Dataset(X, y)
        monkeypatch.setattr("imote.callbacks.c_new_tree_callbacks.load_dataset", lambda _: dataset)
    return use


@pytest.mark.parametrize("method_name, task, y", [
    ("DecisionTreeRegressor", "regression", y_reg),
    ("DecisionTreeClassifier", "classification", y_class + 1),  # classes don't have to be 0/1
])
def test_fit_new_tree(load_dataset, method_name, task, y):
    load_dataset(y)
    tree, _ = fit_new_tree("data", method_name, 3, 15, 20, 5)
    assert tree.task == task
    assert tree.get_depth() <= 3


def test_fit_classifier_needs_binary_target(load_dataset):
    load_dataset(np.arange(300) % 3)
    with pytest.raises(ValueError, match="binary target"):
        fit_new_tree("data", "DecisionTreeClassifier", 3, 15, 20, 5)
