from __future__ import annotations

import logging
import os
import tempfile
import zlib
from pathlib import Path
from typing import Optional, List
import numpy as np
import pandas as pd

from imote.benchmark_info import PMLB_DATASETS_CAT_IDS
from imote.config import DIR_PMLB_CACHE

logger = logging.getLogger(__name__)


def fetch_pmlb(dataset_name: str, cache_dir: Path = DIR_PMLB_CACHE) -> pd.DataFrame:
    """Returns a PMLB dataset, downloading it only if it isn't saved in cache_dir yet.

    Raises:
        ConnectionError: If the dataset isn't saved yet and can't be downloaded.
    """
    path = Path(cache_dir) / f"{dataset_name}.tsv.gz"
    if path.exists():
        try:
            return pd.read_csv(path, sep="\t", float_precision="round_trip")
        except (OSError, EOFError, ValueError, zlib.error):
            logger.warning("Saved dataset %s can't be read, downloading it again", path)
            path.unlink(missing_ok=True)

    from pmlb import fetch_data
    try:
        df = fetch_data(dataset_name)
    except OSError as e:
        raise ConnectionError(f"Could not download PMLB dataset {dataset_name}: ({e})") from e

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, compression="gzip")
    logger.info(f"Saved PMLB dataset {dataset_name} to {path}")
    return df


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
            logger.info("Auto-detected categorical columns in dataset %s: %s", name, cat_ids)
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
        cache_dir: Path = DIR_PMLB_CACHE,
    ) -> "Dataset":
        # Load a dataset from PMLB, downloading it only if it isn't saved in cache_dir yet.

        df = fetch_pmlb(dataset_name, cache_dir)
        y = df["target"].to_numpy()
        X_df = df.drop(columns=["target"])
        X = X_df.to_numpy()
        feature_names = [str(column) for column in X_df.columns]

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
            target_col_name = target_col
        elif isinstance(target_col, int):
            if not (-df.shape[1] <= target_col < df.shape[1]):
                raise ValueError(f"target_col index {target_col} out of range for {df.shape[1]} columns")
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

        feature_names = [str(column) for column in X_df.columns]

        return cls(
            X=X,
            y=y,
            feature_names=feature_names,
            cat_ids=cat_ids,
            name=name or path,
            target_name=str(target_col_name),
            cat_unique_threshold=cat_unique_threshold,
        )
