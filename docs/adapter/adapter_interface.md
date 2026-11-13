# The adapter interface

An adapter is a class that inherits from [`BaseAdapter`](../reference/adapters/base_adapter.md) and is registered
under a name with the `@register_adapter("YourFormat")` decorator. All its methods are static methods (or class
methods), an adapter is never instantiated.

```python
class BaseAdapter(ABC):
    task: str = "regression"            # or "classification"
    needs_feature_names: bool = False   # pass the dataset's feature names to build_root_node and predict
    label: str | None = None            # name in the adapter dropdown, defaults to the registered name

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

## Methods

`build_root_node(X_train, y_train, model)` **(required)**

Converts the model into a linked tree of `BaseNode` objects and returns its root. `X_train` and `y_train` are the data
the model was fitted on, with the columns in the same order as the features of the model. They are needed to compute
the `indices` and `y_res` of every node. This is usually a recursive function that starts at the root with all samples
and divides them over the children at every split.

`load_model(model_path)` **(optional)**

Reads the fitted model from a file on disk and returns it in whatever form is convenient for `build_root_node`: a dict
parsed from JSON, an unpickled Python object, a table, ... Only adapters that implement this method appear in the adapter
dropdown of the [New Tree](../user-guide/c-new-tree.md) card. Adapters for models fitted inside the application, like
PILOT and M5, don't need it.

`predict(X, model)` **(optional)**

Predicts with the original model. This is useful when the original model is faster, or when its predictions can differ
slightly from the converted tree. If it is not implemented, `VizTree`
computes the predictions by traversing the converted tree.
