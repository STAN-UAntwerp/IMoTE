import numpy as np

from imote.nodes.internal_node import LinearNode
from imote.nodes.leaf_node import LeafNode
from imote.nodes.node_model import ConstantNodeModel, LinearNodeModel, NoneNodeModel, SimpleLinearNodeModel
from imote.nodes.split_node import BlinNode, PconNode, SplitCNode, SplitNode
from imote.viz_tree.viz_tree import VizTree

# 8 samples x 3 features: x0 = 0..7 (numeric), x1 in {0,1,2} (categorical), x2 in [0,1] (numeric)
X = np.array([[0, 0, 0.0], [1, 1, 1.0], [2, 2, 0.2], [3, 0, 0.9],
              [4, 1, 0.1], [5, 2, 0.8], [6, 0, 0.3], [7, 1, 0.7]], dtype=float)
y = np.array([1.0, 3.0, 1.5, 3.5, -1.0, 0.0, -0.5, 0.5])


def mask(rows):
    m = np.zeros(len(X), dtype=bool)
    m[list(rows)] = True
    return m


def leaf(rows, coefs, intercept):
    m = mask(rows)
    return LeafNode(m, y[m], LinearNodeModel(np.array(coefs, dtype=float), intercept))


def make_pilot_tree() -> VizTree:
    """PILOT-style tree:
        node0 LinearNode(x2: +1*x2)
        └── node1 PconNode(x0 <= 3.5: left +1 / right -1)
            ├── node2 BlinNode(x2 <= 0.5: left 2*x2 / right -2*x2 + 2)   rows 0-3
            │   ├── node4 Leaf 3*x2 + 1                                   rows 0, 2
            │   └── node5 Leaf -x2 + 3                                    rows 1, 3
            └── node3 Leaf x2 - 1                                         rows 4-7
    Leaf models equal the sum of the models on their path, like the Pilot adapter builds them.
    """
    blin = BlinNode(mask(range(4)), y[:4], 2, 0.5,
                    leaf([0, 2], [0, 0, 3], 1.0), leaf([1, 3], [0, 0, -1], 3.0),
                    SimpleLinearNodeModel(2, 2.0, 0.0), SimpleLinearNodeModel(2, -2.0, 2.0))
    pcon = PconNode(mask(range(8)), y, 0, 3.5, blin, leaf(range(4, 8), [0, 0, 1], -1.0),
                    ConstantNodeModel(1.0), ConstantNodeModel(-1.0))
    root = LinearNode(mask(range(8)), y, 2, SimpleLinearNodeModel(2, 1.0, 0.0), pcon)
    return VizTree(root, X, y, None, tree_id="pilot")


def make_split_tree() -> VizTree:
    """Partykit / M5-style tree (models only in the leaves):
        node0 SplitNode(x0 <= 3.5)
        ├── node1 SplitCNode(x1 in [0])                rows 0-3
        │   ├── node3 Leaf x0 + 1                      rows 0, 3
        │   └── node4 Leaf 2 (all-zero coefficients)   rows 1, 2
        └── node2 Leaf 2*x2                            rows 4-7
    """
    sc = SplitCNode(mask(range(4)), y[:4], 1, [0],
                    leaf([0, 3], [1, 0, 0], 1.0), leaf([1, 2], [0, 0, 0], 2.0),
                    NoneNodeModel(), NoneNodeModel())
    root = SplitNode(mask(range(8)), y, 0, 3.5, sc, leaf(range(4, 8), [0, 0, 2], 0.0),
                     NoneNodeModel(), NoneNodeModel())
    return VizTree(root, X, y, None, tree_id="split")


def make_single_leaf_tree() -> VizTree:
    return VizTree(leaf(range(8), [0, 0, 1], 0.0), X, y, None)


y_cls = np.array([0, 1, 1, 1, 0, 0, 0, 1], dtype=float)


def make_classification_tree() -> VizTree:
    """bcPilot-style tree, log_odds are the log-odds of the node's samples before the node:
        node0 PconNode(x0 <= 3.5: left +1 / right -1)   log_odds 0 for all rows
        ├── node1 Leaf   rows 0-3, y = [0, 1, 1, 1], log_odds [-1, 1, 2, 0.5]
        └── node2 Leaf   rows 4-7, y = [0, 0, 0, 1], log_odds [-2, 1, -1, 0.5]
    """
    left = LeafNode(mask(range(4)), y_cls[:4], LinearNodeModel(np.zeros(3), 1.0))
    left.log_odds = np.array([-1.0, 1.0, 2.0, 0.5])
    right = LeafNode(mask(range(4, 8)), y_cls[4:], LinearNodeModel(np.zeros(3), -1.0))
    right.log_odds = np.array([-2.0, 1.0, -1.0, 0.5])
    root = PconNode(mask(range(8)), y_cls, 0, 3.5, left, right, ConstantNodeModel(1.0), ConstantNodeModel(-1.0))
    root.log_odds = np.zeros(8)
    return VizTree(root, X, y_cls, None, tree_id="classification", task="classification")
