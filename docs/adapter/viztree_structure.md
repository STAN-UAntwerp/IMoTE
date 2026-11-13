# The VizTree object

To write an adapter you need to know what it has to produce. A [`VizTree`](../reference/viz_tree.md) object represents
a tree as a linked structure of node objects, with the root node stored in the `root_node` attribute. Besides the tree
itself it stores:

- the training data `X_train` and `y_train`,
- the predictions of the tree on the training data, `y_hat`,
- optionally the `feature_names` and `target_name`,
- the `task`, `"regression"` or `"classification"`.

The adapter only builds the linked tree of nodes and returns its root. `VizTree` takes care of the rest, including the
split and linear contributions of every training observation used in the (global) prediction plots.

## Nodes

All nodes inherit from the abstract class [`BaseNode`](../reference/nodes/base_node.md). Each node may have one or more
children, which represent the corresponding subtree. Every node, whatever its type, stores:

| Attribute  | Meaning                                                                                                                                                       |
|------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `indices`  | Boolean mask over the rows of `X_train` that is `True` for the training samples that reach this node, i.e. the data used to fit this node and its descendants. |
| `y_res`    | The target values used for training at this node, for the samples selected by `indices`.                                                                      |
| `log_odds` | Classification trees only: log-odds of the samples at this node from the model *before* this node. `None` for regression trees.                              |

The meaning of `y_res` depends on how the algorithm fits the tree:

- In **PILOT**, every node is fitted on the residuals of its ancestors. `y_res` are those residuals: the target minus
  the predictions of all models on the path above the node.
- In **M5** and partykit's `lmtree`, the models are only fitted in the leaves, so `y_res` are simply the `y` values of
  the training samples that reach the node.

### Node types

Each node type stores its own specific information:

| Node type    | Meaning                                                                                                                                                                                    |
|--------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `LeafNode`   | Terminal node with the model that produces the final prediction: a `LinearNodeModel` (multiple linear regression) or a `ConstantNodeModel`.                                                |
| `LinearNode` | Internal node with a single child. It applies a one-dimensional linear model (`SimpleLinearNodeModel`) on one feature before passing all observations to its child.                       |
| `SplitNode`  | Internal node with two children that routes observations based on a split on one feature (`pivot_idx`): observations with `x <= pivot_value` go to the left child. No model is applied. |
| `SplitCNode` | Categorical `SplitNode`: `pivot_value` is the list of categories sent to the left child.                                                                                                   |
| `PconNode`   | Piecewise constant split: a `SplitNode` with a `ConstantNodeModel` on each side.                                                                                                           |
| `PconcNode`  | Categorical `PconNode`.                                                                                                                                                                    |
| `PlinNode`   | Piecewise linear split: a `SplitNode` with a `SimpleLinearNodeModel` on each side.                                                                                                         |
| `BlinNode`   | Broken linear split: a `PlinNode` whose two linear models are continuous at the split point.                                                                                              |

A plain `SplitNode` is all you need for algorithms that only fit models in the leaves, such as M5 and `lmtree`. The
`LinearNode`, `PconNode`, `PlinNode` and `BlinNode` types correspond to the node types of PILOT, which fits a model at
every node.

Each node's model (or models, for a split node) is one of these:

- `LinearNodeModel(coefficients, intercept)`: a full multiple linear regression, with one coefficient per column of
  `X_train` (`0.0` for features it doesn't use). This is what a `LeafNode` holds.
- `ConstantNodeModel(value)`: a constant prediction. Used by a `LeafNode` with no features, and by both sides of a
  `PconNode`/`PconcNode`.
- `SimpleLinearNodeModel(idx, coefficient, intercept)`: a linear model on a single feature. Used by a
  `LinearNode` and by both sides of a `PlinNode`/`BlinNode`.

If an algorithm has a structure that none of these types can express, a new node type can be added without changing
the rest of the application. See the [reference](../reference/reference_overview.md) for the full documentation of all
node types and models.

## The leaf model is the full model

When `VizTree` predicts, it routes each observation down to its leaf and uses **only the model of that leaf**. The
leaf model must therefore be the complete prediction model for the observations in that leaf.

For trees with models only in the leaves (M5, `lmtree`) this is automatic. For additive trees like PILOT, the adapter
has to add up the intercepts and coefficients of all models on the path and store that sum in the `LinearNodeModel` of
the leaf. The models of the internal nodes are still stored, because they are shown in the graph and used in the
prediction plots.

This additive structure is also what allows IMoTE to prune a PILOT tree without refitting it: the root of a pruned
subtree is replaced by a new `LeafNode` whose model is the sum of all models on the path to that node. This is not
possible for M5 trees, where the internal nodes have no models.
