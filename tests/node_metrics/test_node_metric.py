import numpy as np
import pytest

from imote.node_metrics.node_metric import NODE_METRICS_REGISTRY, BaseNodeMetric
from tests.helpers import X, make_pilot_tree, y


@pytest.mark.parametrize("node_id", range(6))
@pytest.mark.parametrize("name", NODE_METRICS_REGISTRY)
def test_every_metric_runs_on_every_node(name, node_id):
    tree = make_pilot_tree()
    out = NODE_METRICS_REGISTRY[name].run(tree.nodes[node_id], y)
    assert isinstance(out, str) and out


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


@pytest.mark.parametrize("value, expected", [
    (True, "True"),
    (np.int64(5), "5"),
    (0.0, "0"),
    (3.0, "3"),
    (1 / 3, "0.333333"),
])
def test_format(value, expected):
    assert BaseNodeMetric.format(value) == expected
