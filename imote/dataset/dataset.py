from __future__ import annotations

import logging
from typing import Optional, List
import numpy as np

logger = logging.getLogger(__name__)


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
    task : {"regression", "classification"}, default "regression"
    classes : list, optional
        For classification: the original class labels, y holds their codes 0, 1, ...
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
        task: str = "regression",
        classes: Optional[list] = None,
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
            logger.info("Auto-detected categorical columns in dataset %s: %s", name, cat_ids)
        self.cat_ids = cat_ids

        self.name = name
        self.task = task
        self.classes = list(classes) if classes is not None else None

    @property
    def n_samples(self) -> int:
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    def __repr__(self) -> str:
        return (
            f"Dataset(name={self.name!r}, n_samples={self.n_samples}, "
            f"n_features={self.n_features}, cat_ids={self.cat_ids}, task={self.task!r})"
        )
