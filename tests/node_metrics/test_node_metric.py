import numpy as np
import pytest

from imote.node_metrics.node_metric import NODE_METRICS_REGISTRY, BaseNodeMetric
from tests.helpers import X, make_pilot_tree, y


@pytest.mark.parametrize("node_id", range(6))
@pytest.mark.parametrize("name", NODE_METRICS_REGISTRY)
def test_every_metric_runs_on_every_node(name, node_id):
    tree = make_pilot_tree()
    out = NODE_METRICS_REGISTRY[name].run(tree.nodes[node_id], X, y, tree.y_hat)
    assert isinstance(out, str) and out


# node1: Pcon, all 8 rows, y_res = y, y - y_hat = [0, 1, -0.1, 1.4, -0.1, 0.2, 0.2, 0.8]
#   RSS = sum(y^2) = 26, MAE = 3.8 / 8, MSE = 3.7 / 8, R2 = 1 - 3.7 / 18 (mean(y) = 1)
# node3: Leaf, rows 4-7, y_res = [-1, 0, -0.5, 0.5], y - y_hat = [-0.1, 0.2, 0.2, 0.8]
#   RSS = 1.5, MAE = 1.3 / 4, MSE = 0.73 / 4, R2 = 1 - 0.73 / 1.25 (mean = -0.25)
@pytest.mark.parametrize("name, node_id, expected", [
    ("# Samples", 1, "8"),
    ("# Samples", 3, "4"),
    ("RSS", 1, "26"),
    ("RSS", 3, "1.5"),
    ("MAE", 1, "0.475"),
    ("MAE", 3, "0.325"),
    ("MSE", 1, "0.4625"),
    ("MSE", 3, "0.1825"),
    ("R2", 1, "0.794444"),
    ("R2", 3, "0.416"),
    ("Split Balance", 1, "0.5"),
    ("Split Balance", 3, "-"),
])
def test_metric_values(name, node_id, expected):
    tree = make_pilot_tree()
    assert NODE_METRICS_REGISTRY[name].run(tree.nodes[node_id], X, y, tree.y_hat) == expected


@pytest.mark.parametrize("value, expected", [
    (True, "True"),
    (np.int64(5), "5"),
    (0.0, "0"),
    (3.0, "3"),
    (1 / 3, "0.333333"),
])
def test_format(value, expected):
    assert BaseNodeMetric.format(value) == expected
