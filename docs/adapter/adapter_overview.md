# Adapter
Different tools exist to fit linear model trees to tabular data. These include the implemented PILOT and M5 algorithms
which can be used to fit new trees in the applications (see [New Tree](../user-guide/c-new-tree.md)).

An **adapter** is the piece of code that formats or translates a fitted model into the internal `VizTree` object used
for visualisation
that the application understands. Adding support for a new type of linear model tree means writing a new adapter.

## VizTree
The internal representation of a linear model tree is a [`VizTree`](../reference/viz_tree.md) object. It provides the
functionality required by the application and can be serialized to a dictionary, allowing trees to be easily saved and
loaded.

A `VizTree` consists of a linked tree of `BaseNode` objects, starting at the root node stored in the `root_node`
attribute. Each node can have children, which contain the information for the corresponding part of the
tree. The tree ends at a leaf node, which has no children.

All nodes are instances of `BaseNode` or one of its subclasses. Different node types represent different structures and
provide different visualisation functionality. See the [reference](../reference/reference_overview.md) for the
full documentation.

Every node stores the row `indices`, `y_res` (residuals or fitted `y` values), and `RSS` (residual sum of squares).

The currently supported node types are:

| Node type                                           | Meaning                                                                                                                                                                                                                                                                                                                                                                |
|-----------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`LeafNode`](../reference/nodes/leaf_node.md)       | Terminal node containing a [`NodeModel`](../reference/nodes/node_model.md), either a [`LinearNodeModel`](../reference/nodes/node_model.md) or [`ConstantNodeModel`](../reference/nodes/node_model.md).                                                                                                                                                                 |
| [`SplitNode`](../reference/nodes/split_node.md)     | Two-child node that routes samples based on a split. Numerical variants include [`PlinNode`](../reference/nodes/split_node.md), [`BlinNode`](../reference/nodes/split_node.md), and [`PconNode`](../reference/nodes/split_node.md), categorical variants include [`SplitCNode`](../reference/nodes/split_node.md) and [`PconcNode`](../reference/nodes/split_node.md). |
| [`LinearNode`](../reference/nodes/internal_node.md) | Single child node that applies a one dimensional linear model before passing the samples to its child.                                                                                                                                                                                                                                                                 |

See the [reference](../reference/reference_overview.md) for the
full documentation. Additional node types can be added if required.

## The adapter interface
An adapter is a class inheriting from [`BaseAdapter`](../reference/adapters/base_adapter.md) and registered with a name
using `@register_adapter("YourFormat")`.

It implements two required methods and one optional method:

- **`load_model`**: Loads the model from its original file format.
- **`build_root_node`**: Converts the loaded model into the `BaseNode` linked tree described above.
- **`predict`** *(optional)*: Predicts directly using the original model instead of the `BaseNode` tree. If this method
  is not implemented, or returns `None`, `VizTree` performs predictions by traversing the `BaseNode` tree.

```python
class BaseAdapter(ABC):
    @staticmethod
    @abstractmethod
    def load_model(model_path) -> Any:
        """Load your model's file format from disk."""

    @staticmethod
    @abstractmethod
    def build_root_node(X_train, y_train, model) -> BaseNode:
        """Convert the loaded model into a BaseNode linked tree."""

    @staticmethod
    def predict(X, model) -> np.ndarray | None:
        """Optional: predict directly via the original model."""
```

Once registered and implemented the adapter shows up as one of the options in the
[new tree](../user-guide/c-new-tree.md) card, from which a tree can be loaded.

## Writing your own adapter
1. Create `imote/adapters/your_format_adapter.py`.
2. Subclass `BaseAdapter` and decorate it with `@register_adapter("YourFormat")`.
   Import it in `imote/adapters/__init__.py` so the adapter is registered when IMoTE starts.
3. Implement `load_model` for your file format.
4. Implement `build_root_node` as described above.
5. (Optional) implement the `predict` function with your model.

See the [partykit adapter](partykit_example.md) for an example of an implemented adapter.

