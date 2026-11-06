import numpy as np
import pytest

from sklearn.tree import DecisionTreeClassifier

from imote.adapters.decision_tree_adapter import DecisionTreeClassifierAdapter
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
    assert "AUROC" in classification and "AUROC" not in regression
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
    ("Class counts", 0, "4 / 4"),
    ("Class counts", 1, "1 / 3"),
    ("Mean Probability", 0, "0.5"),
    ("Accuracy", 0, "0.5"),
    ("Accuracy", 1, "1"),
    ("Accuracy", 2, "0.75"),
    ("Log-loss", 0, f"{np.log(2):.6g}"),
    ("Log-loss", 2, f"{np.mean(np.logaddexp(0, [-2, 1, -1, 0.5]) - [0, 0, 0, 0.5]):.6g}"),
    ("AUROC", 0, "0.5"),
    ("AUROC", 1, "1"),
    ("AUROC", 2, "0.666667"),
    ("Class 1 fraction", 0, "0.5"),
    ("Class 1 fraction", 1, "0.75"),
    ("Gini", 0, "0.5"),
    ("Gini", 1, "0.375"),
    ("Entropy", 0, "1"),
    ("Entropy", 1, "0.811278"),
    ("Brier score", 0, "0.25"),
    ("Precision", 0, "0"),
    ("Precision", 2, "0.5"),
    ("Recall", 1, "1"),
    ("Recall", 2, "1"),
    ("F1", 0, "0"),
    ("F1", 2, "0.666667"),
    ("Gini gain", 0, "0.125"),
    ("Gini gain", 1, "-"),
    ("Information gain", 0, "0.188722"),
    ("Information gain", 2, "-"),
])
def test_classification_metric_values(name, node_id, expected):
    tree = make_classification_tree()
    assert NODE_METRICS_REGISTRY[name].run(tree.nodes[node_id], y_cls) == expected


@pytest.mark.parametrize("criterion, metric, gain", [("gini", "Gini", "Gini gain"), ("entropy", "Entropy", "Information gain")])
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


@pytest.mark.parametrize("value, expected", [
    (True, "True"),
    (np.int64(5), "5"),
    (0.0, "0"),
    (3.0, "3"),
    (1 / 3, "0.333333"),
])
def test_format(value, expected):
    assert BaseNodeMetric.format(value) == expected
