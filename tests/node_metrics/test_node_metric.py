from types import SimpleNamespace

import numpy as np
import pytest

from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier

from imote.adapters.decision_tree_adapter import DecisionTreeClassifierAdapter
from imote.config import DEFAULT_NODE_METRICS
from imote.nodes.leaf_node import LeafNode
from imote.nodes.node_model import LinearNodeModel
from imote.node_metrics.node_metric import NODE_METRICS_REGISTRY, BaseNodeMetric, metrics_for_task
from imote.viz_tree.viz_tree import VizTree
from tests.helpers import X, make_classification_tree, make_pilot_tree, y, y_cls


@pytest.mark.parametrize("node_id", range(6))
@pytest.mark.parametrize("name", metrics_for_task("regression"))
def test_every_metric_runs_on_every_node(name, node_id):
    tree = make_pilot_tree()
    out = NODE_METRICS_REGISTRY[name].run(tree.nodes[node_id], y)
    assert isinstance(out, str) and out


@pytest.mark.parametrize("node_id", range(3))
@pytest.mark.parametrize("name", metrics_for_task("classification"))
def test_every_classification_metric_runs_on_every_node(name, node_id):
    tree = make_classification_tree()
    out = NODE_METRICS_REGISTRY[name].run(tree.nodes[node_id], y_cls)
    assert isinstance(out, str) and out


def test_metrics_for_task():
    regression, classification = metrics_for_task("regression"), metrics_for_task("classification")
    assert "R2" in regression and "R2" not in classification
    assert "AUROC (of subtree)" in classification and "AUROC (of subtree)" not in regression
    assert "# Samples" in regression and "# Samples" in classification


# node1: Pcon, all 8 rows, y_res = y = [1, 3, 1.5, 3.5, -1, 0, -0.5, 0.5]
#   RSS = sum(y_res^2) = 26, MAE = 11 / 8, MSE = 26 / 8, R2 = 1 - 26 / 18 (mean(y) = 1)
# node3: Leaf, rows 4-7, y_res = [-1, 0, -0.5, 0.5]
#   RSS = 1.5, MAE = 2 / 4, MSE = 1.5 / 4, R2 = 1 - 1.5 / 1.25 (mean = -0.25)
@pytest.mark.parametrize("name, node_id, expected", [
    ("# Samples", 1, "8"),
    ("# Samples", 3, "4"),
    ("RSS", 1, "26"),
    ("RSS", 3, "1.5"),
    ("MAE", 1, "1.375"),
    ("MAE", 3, "0.5"),
    ("MSE", 1, "3.25"),
    ("MSE", 3, "0.375"),
    ("R2", 1, "-0.444444"),
    ("R2", 3, "-0.2"),
    ("Split Balance", 1, "0.5"),
    ("Split Balance", 3, "-"),
])
def test_metric_values(name, node_id, expected):
    tree = make_pilot_tree()
    assert NODE_METRICS_REGISTRY[name].run(tree.nodes[node_id], y) == expected


# Metrics use the log-odds before the node, not the predictions of the whole tree.
# node0: log_odds 0, y = [0, 1, 1, 1, 0, 0, 0, 1]: every row predicted class 0, p = 0.5
# node1: log_odds [-1, 1, 2, 0.5], y = [0, 1, 1, 1]: all correct, the negative ranks lowest
# node2: log_odds [-2, 1, -1, 0.5], y = [0, 0, 0, 1]: row 5 wrong, the positive beats 2 of 3 negatives
#   log-loss = mean(log(1 + e^f) - y * f)
@pytest.mark.parametrize("name, node_id, expected", [
    ("Class 0 count", 0, "4"),
    ("Class 1 count", 0, "4"),
    ("Class 0 count", 1, "1"),
    ("Class 1 count", 1, "3"),
    ("Class 0 fraction", 1, "0.25"),
    ("Mean Probability", 0, "0.5"),
    ("Accuracy", 0, "0.5"),
    ("Accuracy", 1, "1"),
    ("Accuracy", 2, "0.75"),
    ("Log-loss", 0, f"{np.log(2):.6g}"),
    ("Log-loss", 2, f"{np.mean(np.logaddexp(0, [-2, 1, -1, 0.5]) - [0, 0, 0, 0.5]):.6g}"),
    ("Class 1 fraction", 0, "0.5"),
    ("Class 1 fraction", 1, "0.75"),
    ("Gini", 0, "0.5"),
    ("Gini", 1, "0.375"),
    ("Entropy", 0, "1"),
    ("Entropy", 1, "0.811278"),
    ("Brier score", 0, "0.25"),
    ("Split gini decrease", 0, "0.125"),
    ("Split gini decrease", 1, "-"),
    ("Split information gain", 0, "0.188722"),
    ("Split information gain", 2, "-"),
])
def test_classification_metric_values(name, node_id, expected):
    tree = make_classification_tree()
    assert NODE_METRICS_REGISTRY[name].run(tree.nodes[node_id], y_cls) == expected


# Subtree metrics use the leaves below the node.
# node0: log_odds [-1, 1, 2, 0.5, -2, 1, -1, 0.5], y = [0, 1, 1, 1, 0, 0, 0, 1]: only row 5 wrong, 5 predicted class 1
#   the positives beat 13.5 of the 16 positive-negative pairs, one tie
# node1, node2: leaves, same as their own log_odds
@pytest.mark.parametrize("name, node_id, expected", [
    ("Accuracy (of subtree)", 0, "0.875"),
    ("Accuracy (of subtree)", 1, "1"),
    ("Accuracy (of subtree)", 2, "0.75"),
    ("Precision (of subtree)", 0, "0.8"),
    ("Precision (of subtree)", 2, "0.5"),
    ("Recall (of subtree)", 0, "1"),
    ("Recall (of subtree)", 2, "1"),
    ("F1 (of subtree)", 0, "0.888889"),
    ("F1 (of subtree)", 2, "0.666667"),
    ("AUROC (of subtree)", 0, "0.84375"),
    ("AUROC (of subtree)", 1, "1"),
    ("AUROC (of subtree)", 2, "0.666667"),
])
def test_subtree_metric_values(name, node_id, expected):
    tree = make_classification_tree()
    assert NODE_METRICS_REGISTRY[name].run(tree.nodes[node_id], y_cls) == expected


def test_subtree_metrics_still_see_the_leaves_of_a_collapsed_subtree():
    tree = make_classification_tree()
    tree.collapse(tree.root_node)
    assert NODE_METRICS_REGISTRY["Accuracy (of subtree)"].run(tree.root_node, y_cls) == "0.875"


def test_subtree_metrics_of_the_root_are_those_of_the_decision_tree():
    rng = np.random.default_rng(0)
    X_tree = rng.normal(size=(200, 3))
    y_tree = (X_tree[:, 0] + rng.normal(size=200) > 0).astype(float)
    model = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X_tree, y_tree)
    tree = VizTree.from_model(DecisionTreeClassifierAdapter, X_tree, y_tree, model)
    proba = model.predict_proba(X_tree)[:, 1]

    root = tree.root_node
    assert NODE_METRICS_REGISTRY["Accuracy (of subtree)"].compute(root, y_tree) == pytest.approx(model.score(X_tree, y_tree))
    assert NODE_METRICS_REGISTRY["AUROC (of subtree)"].compute(root, y_tree) == pytest.approx(roc_auc_score(y_tree, proba))
    assert NODE_METRICS_REGISTRY["AUPRC (of subtree)"].compute(root, y_tree) == pytest.approx(
        average_precision_score(y_tree, proba))


@pytest.mark.parametrize("name", ["AUROC (of subtree)", "AUPRC (of subtree)"])
def test_ranking_metrics_of_a_leaf_with_one_score(name):
    node = LeafNode(np.ones(4, dtype=bool), np.zeros(4), 0.0, LinearNodeModel(np.zeros(3), 1.0))
    node.log_odds = np.full(4, 1.0)
    assert NODE_METRICS_REGISTRY[name].compute(node, np.array([0, 1, 0, 1])) == "-"


@pytest.mark.parametrize("criterion, metric, gain", [("gini", "Gini", "Split gini decrease"), ("entropy", "Entropy", "Split information gain")])
def test_impurity_metrics_match_sklearn(criterion, metric, gain):
    rng = np.random.default_rng(0)
    X_tree = rng.normal(size=(200, 3))
    y_tree = (X_tree[:, 0] + rng.normal(size=200) > 0).astype(float)
    model = DecisionTreeClassifier(criterion=criterion, max_depth=3, random_state=0).fit(X_tree, y_tree)
    tree = VizTree.from_model(DecisionTreeClassifierAdapter, X_tree, y_tree, model)

    node_ids = [0, model.tree_.children_left[0], model.tree_.children_right[0]]
    impurity, n = model.tree_.impurity[node_ids], model.tree_.n_node_samples[node_ids]
    root = tree.root_node
    assert NODE_METRICS_REGISTRY[metric].compute(root, y_tree) == pytest.approx(impurity[0])
    assert NODE_METRICS_REGISTRY[gain].compute(root, y_tree) == pytest.approx(
        impurity[0] - (n[1] * impurity[1] + n[2] * impurity[2]) / n[0])


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_default_metrics_are_registered(task):
    assert set(DEFAULT_NODE_METRICS[task]) <= set(metrics_for_task(task))


@pytest.mark.parametrize("name", ["Precision (of subtree)", "Recall (of subtree)", "F1 (of subtree)"])
def test_label_metrics_are_nan_without_samples(name):
    node = LeafNode(np.zeros(4, dtype=bool), np.array([]), 0.0, LinearNodeModel(np.zeros(3), 1.0))
    node.log_odds = np.array([])
    assert np.isnan(NODE_METRICS_REGISTRY[name].compute(node, np.array([0, 1, 0, 1])))


@pytest.mark.parametrize("name", ["Precision (of subtree)", "Recall (of subtree)", "F1 (of subtree)"])
def test_label_metrics_are_nan_for_all_correct_class_0_node(name):
    node = LeafNode(np.ones(3, dtype=bool), np.zeros(3), 0.0, LinearNodeModel(np.zeros(3), 1.0))
    node.log_odds = np.full(3, -2.0)
    assert np.isnan(NODE_METRICS_REGISTRY[name].compute(node, np.zeros(3)))


@pytest.mark.parametrize("name, expected", [("Mean Probability", 0.0), ("Brier score", 0.0)])
def test_probability_metrics_do_not_overflow(name, expected):
    node = SimpleNamespace(indices=np.ones(2, dtype=bool), log_odds=np.full(2, -1000.0))
    with np.errstate(over="raise"):
        assert NODE_METRICS_REGISTRY[name].compute(node, np.zeros(2)) == pytest.approx(expected)


@pytest.mark.parametrize("value, expected", [
    (True, "True"),
    (np.int64(5), "5"),
    (0.0, "0"),
    (3.0, "3"),
    (1 / 3, "0.333333"),
])
def test_format(value, expected):
    assert BaseNodeMetric.format(value) == expected
