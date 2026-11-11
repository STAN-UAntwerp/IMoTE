"""The metadata file (``dataset.toml``) next to every dataset, and turning the raw data into X and y with it.

Example::

    name = "Wine"
    source = "uci"
    source_id = 109
    file = "data.tsv"
    response = "class"  # automatically detected (UCI metadata)
    task = "classification"  # automatically detected (UCI metadata)
    classes = [1, 2, 3]  # automatically detected
    ignore = []

    [read]
    sep = "\\t"
    header = true

    [categorical]
    color = ["blue", "red"]  # automatically detected

Everything a user writes in the file is kept. Missing values are filled in by ``complete_meta`` and marked with
the comment "automatically detected", so it's clear which values were chosen by IMoTE.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import tomlkit
from tomlkit.items import Item
from tomlkit.toml_document import TOMLDocument

logger = logging.getLogger(__name__)

META_FILE_NAME = "dataset.toml"
AUTO = "automatically detected"
TASKS = ("regression", "classification")

# A numeric column with fewer unique values than this is detected as categorical
CAT_UNIQUE_THRESHOLD = 9
# A numeric response with integer values and at most this many unique values is guessed as classification
MAX_CLASSES_GUESS = 5

FILE_HEADER = [
    "IMoTE dataset description. You can edit this file by hand, IMoTE keeps what you write here.",
    f'Values that IMoTE filled in itself are marked "{AUTO}", change them to override them.',
    'task is "regression" or "classification". For classification the response values are coded 0, 1, ... in',
    'the order of "classes" (for two classes, the second one is the positive class).',
    '"ignore" lists the columns that are not used as features.',
]
CATEGORICAL_HEADER = [
    "Categorical feature columns: column = [categories]. Text columns are coded 0, 1, ... in this order,",
    "numeric columns keep their values. Rows with a value that isn't listed are left out.",
    "An empty list [] is filled in automatically. Remove a column to treat it as numeric.",
    "Remove the whole [categorical] table to detect the categorical columns again.",
]


# ----------------------------------------------------------------------
# Reading and writing
# ----------------------------------------------------------------------

def new_meta(name: str, source: str, file: str, source_id=None, read_options: dict | None = None) -> TOMLDocument:
    doc = tomlkit.document()
    for line in FILE_HEADER:
        doc.add(tomlkit.comment(line))
    doc.add(tomlkit.nl())
    doc["name"] = name
    doc["source"] = source
    if source_id is not None:
        doc["source_id"] = source_id
    doc["file"] = file
    read = tomlkit.table()
    for key, value in (read_options or {}).items():
        read[key] = value
    doc["read"] = read
    return doc


def read_meta(path: Path) -> TOMLDocument:
    return tomlkit.parse(Path(path).read_text(encoding="utf-8"))


def write_meta(path: Path, doc: TOMLDocument) -> None:
    Path(path).write_text(tomlkit.dumps(doc), encoding="utf-8")


def to_python(value):
    """Convert numpy/pandas scalars to plain Python values that TOML can store."""
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return int(value) if value.is_integer() and abs(value) < 2 ** 53 else value
    return str(value)


def set_value(container, key: str, value, comment: str | None = None) -> None:
    """Set container[key] and replace its comment (no comment means the value was chosen by the user)."""
    item = tomlkit.item(value)
    if isinstance(value, list) and len(value) > 8:
        item.multiline(True)
    container[key] = item
    stored = container[key]
    if isinstance(stored, Item):
        stored.trivia.comment = f"# {comment}" if comment else ""
        stored.trivia.comment_ws = "  " if comment else ""


def _auto(source: str | None) -> str:
    return f"{AUTO} ({source})" if source else AUTO


# ----------------------------------------------------------------------
# Detection
# ----------------------------------------------------------------------

def is_text(series: pd.Series) -> bool:
    return not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series)


def categories_of(series: pd.Series) -> list:
    values = series.dropna().unique()
    if is_text(series):
        if pd.api.types.is_bool_dtype(series):
            return sorted(bool(v) for v in values)
        return sorted({str(v) for v in values})
    return [to_python(v) for v in sorted(values)]


def guess_task(series: pd.Series) -> str:
    if is_text(series):
        return "classification"
    values = series.dropna().unique()
    if len(values) <= MAX_CLASSES_GUESS and np.all(np.mod(values.astype(float), 1) == 0):
        return "classification"
    return "regression"


def task_kind(task: str | None, classes: list | None) -> str | None:
    """"regression", "binary" or "multiclass" (the tasks in config.METHODS), None for an unknown task.

    A classification task whose classes aren't known yet is taken as binary.
    """
    if task == "regression":
        return "regression"
    if task == "classification":
        return "multiclass" if classes and len(classes) > 2 else "binary"
    return None


@dataclass
class Hints:
    """Values a dataset source already knows, used instead of guessing when the metadata is completed.

    ``source`` is added to the "automatically detected" comment, e.g. "UCI metadata".
    ``categorical_source`` is used instead for the categorical columns when they come from somewhere else.
    """
    response: str | None = None
    task: str | None = None
    categorical: list[str] | None = None
    ignore: list[str] = field(default_factory=list)
    source: str | None = None
    categorical_source: str | None = None


# ----------------------------------------------------------------------
# Completing the metadata
# ----------------------------------------------------------------------

def new_categorical_table():
    """An empty [categorical] table with its explanation comments."""
    table = tomlkit.table()
    for line in CATEGORICAL_HEADER:
        table.add(tomlkit.comment(line))
    return table


def complete_meta(doc: TOMLDocument, df: pd.DataFrame, hints: Hints | None = None) -> bool:
    """Fill in what's missing in the metadata, keeps everything that's already there. Returns whether it changed."""
    hints = hints or Hints()
    before = tomlkit.dumps(doc)
    columns = list(df.columns)

    if not doc.get("response"):
        if hints.response in columns:
            set_value(doc, "response", hints.response, _auto(hints.source))
        else:
            set_value(doc, "response", "target" if "target" in columns else columns[-1], AUTO)
    response = str(doc["response"])

    if "ignore" not in doc:
        ignore = [col for col in hints.ignore if col in columns and col != response]
        set_value(doc, "ignore", ignore, _auto(hints.source) if ignore else None)

    if not doc.get("task") and response in columns:
        if hints.task in TASKS:
            set_value(doc, "task", hints.task, _auto(hints.source))
        else:
            set_value(doc, "task", guess_task(df[response]), AUTO)

    if doc.get("task") == "classification" and not doc.get("classes") and response in columns:
        set_value(doc, "classes", categories_of(df[response]), AUTO)

    excluded = {response, *[str(c) for c in doc.get("ignore", [])]}
    features = [col for col in columns if col not in excluded]
    if "categorical" not in doc:
        table = new_categorical_table()
        if hints.categorical is not None:
            detected = [col for col in hints.categorical if col in features]
            comment = _auto(hints.categorical_source or hints.source)
        else:
            detected = [col for col in features if is_text(df[col]) or df[col].nunique() < CAT_UNIQUE_THRESHOLD]
            comment = AUTO
        for col in detected:
            set_value(table, col, categories_of(df[col]), comment)
        doc["categorical"] = table

    table = doc["categorical"]
    # text columns can only be used as categorical features
    for col in features:
        if is_text(df[col]) and col not in table:
            set_value(table, col, categories_of(df[col]), f"{AUTO} (text column)")
    for col in list(table.keys()):
        if col in columns and not table[col]:
            set_value(table, col, categories_of(df[col]), AUTO)

    return tomlkit.dumps(doc) != before


# ----------------------------------------------------------------------
# Turning the data into X and y
# ----------------------------------------------------------------------

@dataclass
class Prepared:
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    target_name: str
    cat_ids: np.ndarray
    task: str
    classes: list | None
    n_rows_file: int
    n_dropped: int
    encoded_columns: list[str]

    def summary(self) -> str:
        parts = [f"{self.X.shape[0]} rows, {self.X.shape[1]} features"]
        if self.task == "classification":
            kind = "binary" if len(self.classes) == 2 else "multiclass"
            parts.append(f"{len(self.classes)} classes ({kind})")
        n_cat = 0 if np.array_equal(self.cat_ids, [-1]) else len(self.cat_ids)
        parts.append(f"{n_cat} categorical")
        if self.n_dropped:
            parts.append(f"{self.n_dropped} of {self.n_rows_file} rows left out (missing or unlisted values)")
        if self.encoded_columns:
            parts.append(f"text columns coded as numbers: {', '.join(self.encoded_columns)}")
        return "; ".join(parts) + "."


def _code(series: pd.Series, categories: list, column: str, numeric_codes: bool) -> pd.Series:
    """Map a column to numbers with its list of categories, values that aren't listed become NaN.

    numeric_codes: Keep the values of a numeric column instead of coding them 0, 1, ...
    """
    if not categories:
        raise ValueError(f"Column {column!r} has no categories.")
    if not is_text(series) and all(isinstance(c, (int, float)) and not isinstance(c, bool) for c in categories):
        values = series.astype(float)
        cats = [float(c) for c in categories]
        if numeric_codes:
            return values.where(values.isin(cats))
        return values.map({c: i for i, c in enumerate(cats)}).astype(float)
    lookup = {str(c): i for i, c in enumerate(categories)}
    return series.map(lambda v: lookup.get(str(v), np.nan) if pd.notna(v) else np.nan).astype(float)


def prepare(df: pd.DataFrame, meta) -> Prepared:
    """Turn the raw DataFrame into X and y using the (completed) metadata.

    Raises:
        ValueError: The metadata doesn't match the data.
    """
    response = meta.get("response")
    if response not in df.columns:
        raise ValueError(f"Response column {response!r} isn't in the data, the columns are {list(df.columns)}.")
    task = meta.get("task")
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, got {task!r}.")

    excluded = {response, *[str(c) for c in meta.get("ignore", [])]}
    features = [col for col in df.columns if col not in excluded]
    if not features:
        raise ValueError("No feature columns left, check the ignore list.")
    categorical = {str(k): list(v) for k, v in (meta.get("categorical") or {}).items()}

    columns, cat_ids, encoded = {}, [], []
    for i, col in enumerate(features):
        series = df[col]
        if col in categorical:
            cat_ids.append(i)
            columns[col] = _code(series, categorical[col], col, numeric_codes=True)
            if is_text(series):
                encoded.append(col)
        else:
            if is_text(series):
                raise ValueError(f"Column {col!r} contains text, add it to [categorical] or to ignore.")
            columns[col] = series.astype(float)
    X_df = pd.DataFrame(columns, index=df.index)

    classes = None
    if task == "classification":
        classes = list(meta.get("classes") or [])
        if len(classes) < 2:
            raise ValueError(f"A classification response needs at least 2 classes, got {classes}.")
        y = _code(df[response], classes, response, numeric_codes=False)
    else:
        if is_text(df[response]):
            raise ValueError(f"Response {response!r} isn't numeric, use classification for it.")
        y = df[response].astype(float)

    keep = X_df.notna().all(axis=1) & y.notna()
    if not keep.any():
        raise ValueError("No rows left after leaving out rows with missing or unlisted values.")

    return Prepared(
        X=X_df[keep].to_numpy(dtype=float),
        y=y[keep].to_numpy(dtype=float),
        feature_names=features,
        target_name=str(response),
        cat_ids=np.array(cat_ids) if cat_ids else np.array([-1]),
        task=task,
        classes=classes,
        n_rows_file=len(df),
        n_dropped=int((~keep).sum()),
        encoded_columns=encoded,
    )
