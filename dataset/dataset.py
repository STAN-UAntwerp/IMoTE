"""Dataset container for regression (for now) data used across the app.

Design notes
------------
- X and y are always stored as numpy arrays internally.
- `cat_ids` are the column indices of X considered categorical. If not
  given explicitly, they are auto-detected: a column counts as
  categorical if it has fewer than `cat_unique_threshold` unique values.
- New data sources are added as classmethods (`from_pmlb`, `from_csv`,
  ...) that all funnel into the same `__init__`, so a future
  `from_openml`, `from_parquet`, etc. is a small addition.
- Classification support later: add a `task_type` field (default
  "regression") and branch on it where needed, without touching the
  constructors' signatures.
"""

from __future__ import annotations

from typing import Optional, List
import numpy as np
import pandas as pd


class Dataset:
    """Container for a dataset: X matrix, y column, and metadata.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    y : array-like of shape (n_samples,)
    feature_names : list of str, optional
        Defaults to x0, x1, ... if not given.
    cat_ids : list of int, optional
        Indices of columns in X that are categorical. If None, they are
        auto-detected using `cat_unique_threshold`.
    name : str, optional
        Human-readable identifier (e.g. the PMLB dataset name or CSV path).
    cat_unique_threshold : int, default 8
        A column is auto-flagged as categorical if it has fewer than
        this many unique values. Only used when `cat_ids` is None.
    """

    def __init__(
        self,
        X,
        y,
        feature_names: Optional[List[str]] = None,
        target_name:Optional[str] = None,
        cat_ids: Optional[np.ndarray] = None,
        name: Optional[str] = None,
        cat_unique_threshold: int = 8,
    ):
        self.X = np.asarray(X)
        self.y = np.asarray(y).ravel()

        if self.X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {self.X.shape}")
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError(
                f"X and y must have the same number of rows, got "
                f"{self.X.shape[0]} and {self.y.shape[0]}"
            )

        if feature_names is None:
            feature_names = [f"X{i}" for i in range(self.X.shape[1])]
        elif len(feature_names) != self.X.shape[1]:
            raise ValueError("feature_names length must match number of X columns")
        self.feature_names = list(feature_names)

        if target_name is None:
            target_name = "y"
        self.target_name = target_name

        if cat_ids is None:
            cat_ids = np.array(
                [i for i in range(self.X.shape[1]) if len(np.unique(self.X[:, i])) < cat_unique_threshold] or [-1]
            )
            print(f"Auto detected categorical columns in dataset {name}, as {cat_ids}")
        self.cat_ids = cat_ids

        self.name = name

    @property
    def n_samples(self) -> int:
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    def __repr__(self) -> str:
        return (
            f"Dataset(name={self.name!r}, n_samples={self.n_samples}, "
            f"n_features={self.n_features}, cat_ids={self.cat_ids})"
        )

    # ------------------------------------------------------------------
    # Constructors for different data sources
    # ------------------------------------------------------------------

    @classmethod
    def from_pmlb(
        cls,
        dataset_name: str,
        cat_ids: Optional[np.ndarray] = None,
        cat_unique_threshold: int = 8,
    ) -> "Dataset":
        #Load a dataset from PMLB.
        from pmlb import fetch_data
        from benchmark_info import PMLB_DATASETS_CAT_IDS

        df = fetch_data(dataset_name)
        y = df["target"].to_numpy()
        X_df = df.drop(columns=["target"])
        X = X_df.to_numpy()
        feature_names = list(X_df.columns)

        if cat_ids is None:
            cat_ids = PMLB_DATASETS_CAT_IDS.get(dataset_name)

        return cls(
            X=X,
            y=y,
            feature_names=feature_names,
            target_name="Target",
            cat_ids=cat_ids,
            name=dataset_name,
            cat_unique_threshold=cat_unique_threshold,
        )

    @classmethod
    def from_csv(
            cls,
            path: str,
            target_col,
            cat_ids: Optional[np.ndarray] = None,
            name: Optional[str] = None,
            cat_unique_threshold: int = 8,
            **read_csv_kwargs,
    ) -> "Dataset":
        # Load a dataset from a CSV file.
        df = pd.read_csv(path, **read_csv_kwargs)

        if isinstance(target_col, str):
            try:
                target_col = int(target_col)
            except ValueError:
                pass

        if isinstance(target_col, str):
            if target_col not in df.columns:
                raise ValueError(f"target_col {target_col!r} not found in columns {list(df.columns)}")
            target_name = target_col
            target_col_name = target_col
        elif isinstance(target_col, int):
            if not (-df.shape[1] <= target_col < df.shape[1]):
                raise ValueError(f"target_col index {target_col} out of range for {df.shape[1]} columns")
            target_name = None
            target_col_name = df.columns[target_col]
        else:
            raise TypeError(f"target_col must be str or int, got {type(target_col)}")

        y = df[target_col_name].to_numpy()
        X_df = df.drop(columns=[target_col_name])
        X = X_df.to_numpy()
        if not np.issubdtype(X.dtype, np.number) or not np.issubdtype(y.dtype, np.number):
            raise TypeError("X and y must contain numeric values")

        X = X.astype(float)
        y = y.astype(float)

        feature_names = list(X_df.columns)

        return cls(
            X=X,
            y=y,
            feature_names=feature_names,
            cat_ids=cat_ids,
            name=name or path,
            target_name=target_name,
            cat_unique_threshold=cat_unique_threshold,
        )
