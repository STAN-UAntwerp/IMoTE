# Adapters

IMoTE can fit linear model trees itself with PILOT and M5 (see [New Tree](../user-guide/c-new-tree.md)), but many
other algorithms and implementations exist, often in other languages and each with its own output format and tree
structure. To support all of them without changing the application, IMoTE uses adapters:

- Internally, every tree is converted to one common representation, a [`VizTree`](../reference/viz_tree.md) object.
  All functionality of the application (drawing the graph, node plots, path highlighting, prediction plots,
  collapsing, pruning, saving and loading) is implemented on top of this representation only.
- An **adapter** is a small class that reads a fitted model in its native format and translates it into a `VizTree` 
  object.

Supporting a new type of linear model tree therefore only requires writing one adapter.

```
 fitted model           adapter                  VizTree                   IMoTE
(native format) ──▶   load_model()    ──▶ root_node (BaseNode tree) ──▶ graph, plots,
                    build_root_node()      X_train, y_train, y_hat       saving, ...
                    
```

The PILOT and M5 trees fitted in the application go through the same adapters: after fitting, the model object is
handed directly to the `Pilot` or `M5` adapter to build the `VizTree` (see
[PILOT and M5](#the-pilot-and-m5-adapters) below). Adapters that implement `load_model` can also load a model from a
file, and are listed in the adapter dropdown of the [New Tree](../user-guide/c-new-tree.md) card.

## The VizTree object

To write an adapter you need to know what it has to produce. A `VizTree` object represents a tree as a linked structure of
node objects, with the root node stored in the `root_node` attribute. Besides the tree itself it stores:

- the training data `X_train` and `y_train`,
- the predictions of the tree on the training data, `y_hat`,
- optionally the `feature_names` and `target_name`,
- the `task`, `"regression"` or `"classification"`.

When a `VizTree` object is created, it also computes the split and linear contributions of every training observation,
which are used for the (global) prediction plots.

### Nodes

All nodes inherit from the abstract class [`BaseNode`](../reference/nodes/base_node.md). Each node may have one or more
children, which represent the corresponding subtree. Every node, whatever its type, stores:

| Attribute  | Meaning                                                                                                                                                                                                    |
|------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `indices`  | Boolean mask over the rows of `X_train` (length `n_samples`) that is `True` for the training samples that reach this node, i.e. the data used to fit this node and its descendants.                        |
| `y_res`    | The target values used for training at this node, for the samples selected by `indices`. Their meaning depends on the algorithm (see below).                                                               |
| `log_odds` | Classification trees only: log-odds of the samples at this node from the model *before* this node. Left at `None` for regression trees.                                                                   |

The interpretation of `y_res` depends on how the algorithm fits the tree:

- In **PILOT**, the tree is fitted greedily and sequentially, every node on the residuals of its ancestors. `y_res`
  are those residuals: the target minus the predictions of all models on the path above the node.
- In **M5** (and partykit's `lmtree`), the models are only fitted in the leaves, so `y_res` are simply the `y` values
  of the training samples that reach the node.

In addition, each node type stores its own specific information:

| Node type                                                           | Meaning                                                                                                                                                                                                                      | Specific arguments                                                                                   |
|---------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| [`LeafNode`](../reference/nodes/leaf_node.md)                       | Terminal node containing the model that produces the final prediction: a [`LinearNodeModel`](../reference/nodes/node_model.md) (multiple linear regression) or a [`ConstantNodeModel`](../reference/nodes/node_model.md). | `node_model`                                                                                         |
| [`LinearNode`](../reference/nodes/internal_node.md)                     | Internal node with a single child that applies a one-dimensional linear model on one feature before passing all observations to its child.                                                                                 | `pivot_idx`, `linear_model` (a `SimpleLinearNodeModel`), `child`                                     |
| [`SplitNode`](../reference/nodes/split_node.md)                     | Internal node with two children that routes observations based on a split on one feature (`pivot_idx`). Observations with `x <= pivot_value` go to the left child. No model is applied on either side.                     | `pivot_idx`, `pivot_value`, `left_child`, `right_child`, `left_model`, `right_model` (`NoneNodeModel`) |
| &nbsp;&nbsp;&nbsp;&nbsp;[`SplitCNode`](../reference/nodes/split_node.md) | Categorical version of `SplitNode`: `pivot_value` is the list of categories sent to the left child.                                                                                                                       | as `SplitNode`                                                                                       |
| &nbsp;&nbsp;&nbsp;&nbsp;[`PconNode`](../reference/nodes/split_node.md)   | Piecewise constant split, with a constant model on each side.                                                                                                                                                         | child models are `ConstantNodeModel`                                                                  |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[`PconcNode`](../reference/nodes/split_node.md) | Categorical version of `PconNode`.                                                                                                                                                             | child models are `ConstantNodeModel`                                                                  |
| &nbsp;&nbsp;&nbsp;&nbsp;[`PlinNode`](../reference/nodes/split_node.md)   | Piecewise linear split, with a one-dimensional linear model on each side.                                                                                                                                             | child models are `SimpleLinearNodeModel`                                                              |
| &nbsp;&nbsp;&nbsp;&nbsp;[`BlinNode`](../reference/nodes/split_node.md)   | Broken linear split: a `PlinNode` whose two linear models are continuous at the split point.                                                                                                                          | child models are `SimpleLinearNodeModel`                                                              |

Indented types are subclasses. A plain `SplitNode` is what you need for most algorithms that only fit
models in the leaves, such as M5 and `lmtree`. The `LinearNode`, `PconNode`, `PlinNode` and `BlinNode` types
correspond to the node types of PILOT, which fits a model at every node during tree construction.

Two more node types are used internally by the application and should not be created by an adapter: a
`CollapsedNode` replaces a collapsed subtree in the graph and a `NoneNode` acts as a placeholder for an empty child.

Each node type defines its own labels and a short description, which are shown in the graph and in the node
information card. If an algorithm has a structure that none of these types can express, a new node type can be added
without changing the rest of the application. See the [reference](../reference/reference_overview.md) for the full
documentation of all node types.

### Node models

The models attached to nodes are [`NodeModel`](../reference/nodes/node_model.md) objects:

| Model                   | Constructor                                   | Used for                                                                                         |
|-------------------------|-----------------------------------------------|--------------------------------------------------------------------------------------------------|
| `LinearNodeModel`       | `LinearNodeModel(coefficients, intercept)`    | Leaves. `coefficients` has one entry per column of `X_train`, `0.0` for unused features.         |
| `ConstantNodeModel`     | `ConstantNodeModel(value)`                    | Leaves, and the sides of `PconNode`/`PconcNode`.                                                 |
| `SimpleLinearNodeModel` | `SimpleLinearNodeModel(idx, coefficient, intercept)` | `LinearNode`, and the sides of `PlinNode`/`BlinNode`: a linear model on the single feature `idx`. |

When `VizTree` object predicts by traversing the tree, it routes each observation down to its leaf and uses **only the model
of that leaf**. The leaf model must therefore be the complete prediction model for the observations in that leaf.

For trees with models only in the leaves (M5, `lmtree`) this is automatic. For additive trees like PILOT, the adapter has to add up the
intercepts and coefficients of all models on the path and store that sum in the `LinearNodeModel` of the leaf.

This additive structure is what allows IMoTE to prune a PILOT tree without refitting it: the root of a pruned
subtree is replaced by a new `LeafNode` whose model is the sum of all models on the path from the root of the tree to
that node. The part of the tree that is kept stays unchanged, which makes interactive pruning cheap. This is not
possible for M5 trees, where the internal nodes have no models.

## The adapter interface

An adapter is a class that inherits from [`BaseAdapter`](../reference/adapters/base_adapter.md) and is registered
under a name with the `@register_adapter("YourFormat")` decorator. All methods are static methods (or class methods),
an adapter is never instantiated.

```python
class BaseAdapter(ABC):
    task: str = "regression"
    needs_feature_names: bool = False
    label: str | None = None

    @staticmethod
    @abstractmethod
    def build_root_node(X_train, y_train, model) -> BaseNode:
        """Convert the model into a linked tree of BaseNode objects and return its root."""

    @staticmethod
    def load_model(model_path: str) -> Any:
        """Optional: load the fitted model from a file in its native format."""
        raise NotImplementedError

    @staticmethod
    def predict(X, model) -> np.ndarray | None:
        """Optional: predict with the original model."""
        return None
```

An adapter for a model that is stored in a file (like partykit) implements `build_root_node` and `load_model`. An
adapter for a model that is fitted inside the application (like PILOT and M5) only needs `build_root_node`.

### Methods

`build_root_node(X_train, y_train, model)` **(required)**

Converts the model into a linked tree of `BaseNode` objects and returns its root. This is usually a
recursive function that starts at the root with all samples and splits the mask at every split node (see
[helpers](#helpers) below).

`load_model(model_path)` **(optional)**

Reads the fitted model from a file on disk and returns it in whatever form is convenient for `build_root_node`: a dict
parsed from JSON, an unpickled Python object, a table, ... Implementing this method is what makes an adapter show up in 
the adapter dropdown of the New Tree card.

`predict(X, model)` **(optional)**

Predicts with the original model. This is useful when the original model is faster, or when its predictions can differ 
slightly from the converted tree. If it is not implemented`VizTree` computes the predictions.

## Writing your own adapter

As an example, suppose that some algorithm exports its trees as JSON in which every node is either a split or a leaf
with a linear model, and children are nested:

```json
{
  "feature_names": ["x1", "x2", "x3"],
  "root": {
    "feature": "x1", "threshold": 0.5,
    "left":  {"intercept": 1.2, "coefficients": {"x2": 0.3}},
    "right": {"intercept": -0.4, "coefficients": {"x2": 1.1, "x3": -0.2}}
  }
}
```

### 1. Create the adapter file

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

### 2. Implement `load_model`

```python
    @staticmethod
    def load_model(model_path: str) -> dict:
        with open(model_path) as f:
            return json.load(f)
```

### 3. Implement `build_root_node`

Start at the root with all training samples, and build the tree recursively. At every node, select the samples that
reach it, create the children first, and then the node itself:

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

Points to keep in mind:

- Map every feature to its **column index** in `X_train` (`pivot_idx`), and give `LinearNodeModel` a coefficient for
  every column, with `0.0` for features the model doesn't use.
- IMoTE sends `x <= pivot_value` to the **left** child. If your algorithm uses `x < threshold` or sends the smaller
  values right, convert the split accordingly, otherwise the samples (and the plots) end up in the wrong node.
- For an additive algorithm, accumulate the models along the path and store the sum in the leaf
  (see [The leaf model is the full model](#the-leaf-model-is-the-full-model)).

### 4. (Optional) implement `predict`

If the original model can still predict, for example because `load_model` returned the fitted model object itself,
implement `predict` to use it:

```python
    @staticmethod
    def predict(X, model) -> np.ndarray:
        return model.predict(X)
```

For an exported file like the JSON above there is no original model to call, so `predict` is left out and `VizTree`
predicts by traversing the converted tree.

### 5. Register the adapter in the application

Import the adapter in `imote/adapters/__init__.py`, so that the `@register_adapter` decorator runs when IMoTE starts:

```python
from .my_tree_adapter import MyTreeAdapter
```

That is all: since `MyTreeAdapter` implements `load_model`, it now appears in the adapter dropdown of the
[New Tree](../user-guide/c-new-tree.md) card, under its registered name or its `label`.

### 6. Load a tree

In the New Tree card, open **Load a tree with an adapter**, select the adapter and the dataset the model was fitted
on, enter the path to the model file and click **Load with Adapter**. If the model can't be loaded or doesn't match
the dataset, the error is shown in the status bar.

A converted tree can also be built directly in Python, without the user interface, which is convenient while
developing and testing an adapter:

```python
from imote import VizTree

model = MyTreeAdapter.load_model("my_tree.json")
viz_tree = VizTree.from_model(MyTreeAdapter, X_train, y_train, model, feature_names=["x1", "x2", "x3"])
assert np.allclose(viz_tree.y_hat, expected_predictions)
```

Comparing `viz_tree.predict(X)` with the predictions of the original implementation is a good check that the splits
and leaf models were converted correctly. The adapter tests in `tests/adapters/` follow this approach.

## Available adapters

| Adapter (name) | Source                                                 | Loads a file | Notes                                                                       |
|----------------|--------------------------------------------------------|--------------|-----------------------------------------------------------------------------|
| `Partykit`     | `lmtree` from the R package partykit, exported to JSON | yes          | Loads trees fitted outside Python, see the [example](partykit_example.md). |
| `Pilot`        | PILOT trees fitted in the application (regression)     | no           | Additive tree using all PILOT node types, models accumulated into leaves.  |
| `M5`           | M5 (`M5Prime`) trees fitted in the application         | no           | `SplitNode`s with linear models in the leaves.                              |

The [partykit example](partykit_example.md) walks through a complete adapter for a model fitted in R, from exporting the
tree to building the `VizTree`. The adapters for PILOT and M5 are explained briefly below. They work on the fitted
model objects directly, so they only implement `build_root_node` and `predict`.

### The PILOT and M5 adapters

#### PILOT

PILOT fits a regression tree greedily: at every node it chooses the best of several node types, fits that model on
the residuals left by its ancestors, and passes the new residuals on to its children. The prediction for an observation
is therefore the sum of all models along its path. The `Pilot` adapter reads the table of nodes from
`model.tree_summary()`, starting at the root, and converts every node according to its PILOT type:

| PILOT node type        | IMoTE node                                        | Models                                                                      |
|------------------------|---------------------------------------------------|-----------------------------------------------------------------------------|
| `lin`                  | `LinearNode`                                      | `SimpleLinearNodeModel` on the node's feature, one child                     |
| `pcon` / `pconc`       | `PconNode` / `PconcNode` (categorical)            | a `ConstantNodeModel` on each side of the split                             |
| `plin`                 | `PlinNode`                                        | a `SimpleLinearNodeModel` on each side of the split                         |
| `blin`                 | `BlinNode`                                        | a `SimpleLinearNodeModel` on each side, continuous at the split point       |
| `con` / `END`          | `LeafNode`                                        | `LinearNodeModel`: the sum of all models on the path, see below             |

While it goes down the tree, the adapter keeps track of two things for every branch:

- **The residuals.** The `y_res` of the root are the `y` values. At every node, the model applied on the branch is
  subtracted, so the `y_res` of a child are exactly the residuals that PILOT fitted that child on.
- **The accumulated model.** The intercepts and coefficients of all models on the path are added up. When a leaf is
  reached, this sum (plus the leaf's own constant) becomes its `LinearNodeModel`, so that the leaf model is the full
  prediction model for its observations (see [The leaf model is the full model](#the-leaf-model-is-the-full-model)).

`predict` uses `model.predict` of the original PILOT model. The `Pilot` adapter is for regression only: binary
classification trees fitted with the classification version of PILOT go through a separate `BcPilot` adapter, which
works on the log-odds scale and is still in development.

#### M5

M5 first grows a regression tree with constant splits, then fits a linear regression in every node, prunes the tree,
and finally smooths the predictions of the leaves with the models of the nodes above them. The fitted `M5Prime` model
(from the `m5py` package) stores its tree in the scikit-learn format (`model.tree_`) and its node models in
`model.node_models`. The `M5` adapter converts:

- every internal node into a plain `SplitNode` on `tree_.feature` and `tree_.threshold`, with no model on either side
  (`NoneNodeModel`), since M5 has no models that are applied in the internal nodes,
- every leaf into a `LeafNode` with the leaf's model: a `LinearNodeModel` with the leaf's coefficients (and `0.0` for
  the features it doesn't use), or a `ConstantNodeModel` when the leaf only has a constant.

Since all models are in the leaves, the `y_res` of every node are the `y` values of the samples that reach it.

`predict` uses `model.predict` of the original M5 model, so `y_hat` includes the smoothing. The leaf models in the
`VizTree` are the unsmoothed models, so predictions computed by traversing the converted tree can differ slightly from
`y_hat`. Because the internal nodes have no models, an M5 tree can't be pruned in IMoTE without refitting it.
