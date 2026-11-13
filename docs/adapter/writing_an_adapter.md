# Writing your own adapter

This page shows how an adapter for a made-up algorithm *MyTree* that exports its trees as JSON is made. Every node is
either a split or a leaf with a linear model, and the children of a split are nested inside it:

```json
{
  "feature_names": ["x1", "x2", "x3"],
  "root": {
    "feature": "x1", "threshold": 0.5,
    "left": {
      "feature": "x2", "threshold": 0.3,
      "left":  {"intercept": 1.2, "coefficients": {"x3": 0.3}},
      "right": {"intercept": 0.8, "coefficients": {"x1": 2.0, "x3": -0.5}}
    },
    "right": {"intercept": -0.4, "coefficients": {"x2": 1.1, "x3": -0.2}}
  }
}
```

This tree has two levels of splits:

```
x1 <= 0.5
├── yes: x2 <= 0.3
│   ├── yes: leaf  1.2 + 0.3·x3
│   └── no:  leaf  0.8 + 2.0·x1 - 0.5·x3
└── no:  leaf  -0.4 + 1.1·x2 - 0.2·x3
```

## 1. Create the adapter file

Create `imote/adapters/my_tree_adapter.py`, subclass `BaseAdapter` and register it under a name:

```python
import json
import numpy as np
from imote.adapters.base_adapter import BaseAdapter, register_adapter
from imote.adapters.adapter_utils import split_by_threshold
from imote.nodes.base_node import BaseNode
from imote.nodes.leaf_node import LeafNode
from imote.nodes.split_node import SplitNode
from imote.nodes.node_model import LinearNodeModel, NoneNodeModel


@register_adapter("MyTree")
class MyTreeAdapter(BaseAdapter):
    """Adapter for the JSON export of MyTree."""
```

## 2. Implement `load_model`

```python
    @staticmethod
    def load_model(model_path: str) -> dict:
        with open(model_path) as f:
            return json.load(f)
```

## 3. Implement `build_root_node`

The tree is built recursively. `build_root_node` starts at the root with a mask that selects all training samples.
`_build_recursive` then converts one JSON node:

- a **leaf** becomes a `LeafNode` with its linear model, which ends the recursion,
- a **split** divides its samples over two masks with `split_by_threshold`, which sends the samples with
  `x <= threshold` to the left, then calls itself for both children and returns a `SplitNode` that links them.

```python
    @staticmethod
    def build_root_node(X_train, y_train, model) -> BaseNode:
        root_indices = np.ones(X_train.shape[0], dtype=bool)  # all samples reach the root
        return MyTreeAdapter._build_recursive(model["root"], model["feature_names"], X_train, y_train, root_indices)

    @staticmethod
    def _build_recursive(node, feature_names, X_train, y_train, current_indices) -> BaseNode:
        y_res = y_train[current_indices]  # models only in the leaves: y_res are the y values

        if "threshold" not in node:  # leaf
            coefficients = np.array([node["coefficients"].get(name, 0.0) for name in feature_names])
            node_model = LinearNodeModel(coefficients, node["intercept"])
            return LeafNode(indices=current_indices, y_res=y_res, node_model=node_model)

        pivot_idx = feature_names.index(node["feature"])
        pivot_value = node["threshold"]
        left_indices, right_indices = split_by_threshold(X_train, current_indices, pivot_idx, pivot_value)
        return SplitNode(
            indices=current_indices,
            y_res=y_res,
            pivot_idx=pivot_idx,
            pivot_value=pivot_value,
            left_child=MyTreeAdapter._build_recursive(node["left"], feature_names, X_train, y_train, left_indices),
            right_child=MyTreeAdapter._build_recursive(node["right"], feature_names, X_train, y_train, right_indices),
            left_model=NoneNodeModel(),
            right_model=NoneNodeModel(),
        )
```

For the example tree and a dataset of 100 samples, the calls go as follows. Every call receives a smaller mask, until
a leaf is reached:

```
_build_recursive(root,       100 samples)  → SplitNode on x1
├── _build_recursive(left,    60 samples)  → SplitNode on x2
│   ├── _build_recursive(left.left,  25 samples)  → LeafNode
│   └── _build_recursive(left.right, 35 samples)  → LeafNode
└── _build_recursive(right,   40 samples)  → LeafNode
```

The children are built before their parent, so when `SplitNode` is created, both of its subtrees are already
complete. The `SplitNode` returned by the first call is the root of the whole tree.

Important points:

- Map every feature name to its **column index** in `X_train`, both for `pivot_idx` and for the coefficients of
  `LinearNodeModel`.
- If your algorithm uses `x < threshold` or sends the smaller values to the right, convert the split, otherwise the
  samples end up in the wrong node.

## 4. (Optional) implement `predict`

If `load_model` returns a model object that can still predict, implement `predict` to use it:

```python
    @staticmethod
    def predict(X, model) -> np.ndarray:
        return model.predict(X)
```

MyTree only exports a JSON file, so there is nothing to call and `predict` is not implemented.

## 5. Register the adapter

Import the adapter in `imote/adapters/__init__.py`, so that the `@register_adapter` decorator runs when IMoTE starts:

```python
from .my_tree_adapter import MyTreeAdapter
```

The adapter now appears in the adapter dropdown of the [New Tree](../user-guide/c-new-tree.md) card.

## 6. Load a tree

In the New Tree card, open **Load a tree with an adapter**, select the adapter and the dataset the model was fitted
on, enter the path to the model file and click **Load with Adapter**.

While developing an adapter, it is convenient to build the tree in Python and compare its predictions with those of
the original implementation:

```python
from imote import VizTree

model = MyTreeAdapter.load_model("my_tree.json")
viz_tree = VizTree.from_model(MyTreeAdapter, X_train, y_train, model, feature_names=["x1", "x2", "x3"])
assert np.allclose(viz_tree.y_hat, expected_predictions)
```
