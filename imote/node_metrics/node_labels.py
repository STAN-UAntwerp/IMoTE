"""Helpers for the text and colors of the nodes in the tree graph: short feature names and metric labels."""
from __future__ import annotations

import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_hex

from imote.config import NEAR_ZERO, NO_METRIC_COLOR
from imote.node_metrics.node_metric import BaseNodeMetric

WORD_SEPARATOR = r"[_\-.\s]"
VOWELS = set("aeiouAEIOU")
MIN_SYLLABLE_LENGTH = 4
LOG_SCALE_DECADES = 6
TARGET_LOG_SCALE_DECADES = 3


def shorten_name(name: str, max_length: int) -> str:
    """Shortens a feature name to at most max_length characters, keeping it as readable as possible.

    Tries in order:
        1. the leading words that fit (words are split on _, -, . or spaces): cars_per_hour -> cars
        2. the first word cut just before a vowel, keeping at least 4 letters: temperature -> temp
        3. the first word without its vowels
        4. the start of the first word followed by ".."
    """
    if len(name) <= max_length:
        return name
    leading_words = [name[:m.start()] for m in re.finditer(WORD_SEPARATOR, name) if 0 < m.start() <= max_length]
    if leading_words:
        return leading_words[-1].rstrip("_-. ")
    first_word = re.split(WORD_SEPARATOR, name)[0] or name
    for end in range(MIN_SYLLABLE_LENGTH, min(max_length, len(first_word) - 1) + 1):
        if first_word[end] in VOWELS and first_word[end - 1] not in VOWELS:
            return first_word[:end]
    without_vowels = first_word[0] + "".join(c for c in first_word[1:] if c not in VOWELS)
    if len(without_vowels) <= max_length:
        return without_vowels
    return first_word[:max_length - 2] + ".."


def display_names(feature_names: list[str] | None, n_features: int, max_length: int | None = None) -> list[str]:
    """Returns the name to show for every feature: X0, X1, ... without feature names, else the (shortened) names.

    Features whose shortened names are the same get their index added (wind2, wind4), so no two features look alike.
    """
    if feature_names is None:
        return [f"X{i}" for i in range(n_features)]
    if max_length is None:
        return list(feature_names)
    short_names = [shorten_name(name, max_length) for name in feature_names]
    return [name if short_names.count(name) == 1 else name.rstrip(".")[:max_length - len(str(i))] + str(i)
            for i, name in enumerate(short_names)]


def _is_number(value) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_))


def metric_text(value, compact: bool = False) -> str:
    """Formats a metric value for a node label, compact for minimal labels."""
    if not _is_number(value) or not np.isfinite(value):
        return str(value)
    if value == 0:
        return "0"
    if abs(value) < NEAR_ZERO:
        return "≈0"
    if not compact:
        return str(int(value)) if float(value).is_integer() else f"{value:.4g}"
    for limit, suffix in ((1e-9, "n"), (1e-6, "μ"), (1e-3, "m")):
        if abs(value) <= limit:
            return f"{value / limit:.3g}{suffix}"
    for limit, suffix in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
        if abs(value) >= limit:
            return f"{value / limit:.3g}{suffix}"
    return f"{value:.3g}"



def metric_colors(values: list, metric: type[BaseNodeMetric]) -> list[str]:
    """Colors the nodes by their metric values"""

    unique_values = np.unique([float(v) for v in values if _is_number(v) and np.isfinite(v)])
    if metric.higher_is_better is None:
        cmap, start, stop = plt.colormaps["YlGnBu"], 0.05, 0.55
    else:
        cmap, start, stop = plt.colormaps["RdYlGn"], 0.1, 0.9

    colors = []
    for value in values:
        if not _is_number(value) or not np.isfinite(value):
            colors.append(NO_METRIC_COLOR)
            continue
        if metric.value_range is not None:
            low, high = metric.value_range
            if value < low or value > high:
                colors.append(NO_METRIC_COLOR)
                continue
            else:
                scale = 0.5 if high == low else (value - low) / (high - low)
        else:
            scale = 0.5 if len(unique_values) < 2 else np.searchsorted(unique_values, float(value)) / (len(unique_values) - 1)

        if metric.higher_is_better is False:
            scale = 1 - scale
        colors.append(to_hex(cmap(start + scale * (stop - start))))
    return colors
