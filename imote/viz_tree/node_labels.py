"""Helpers for the text and colors of the nodes in the tree graph: short feature names and metric labels."""
from __future__ import annotations

import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_hex

from imote.config import NO_METRIC_COLOR, NEAR_ZERO

WORD_SEPARATOR = r"[_\-.\s]"
VOWELS = set("aeiouAEIOU")
MIN_SYLLABLE_LENGTH = 4


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
    if value != 0 and abs(value) < NEAR_ZERO:
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


def metric_colors(values: list, higher_is_better: bool | None, power: float = 1.0) -> list[str]:
    """Colors every value on a gradient from the lowest to the highest value, better values are green.

    Like the data flow sizes, the position on the gradient is scaled from 0 to 1 and raised to power, so with a power
    below 1 the many small values (e.g. the RSS of the nodes below the root) are spread over more colors. Metrics
    without a better side get a neutral gradient, values that aren't numbers get NO_METRIC_COLOR.
    """
    finite = [float(v) for v in values if _is_number(v) and np.isfinite(v)]
    low, high = (min(finite), max(finite)) if finite else (0.0, 1.0)
    if higher_is_better is None:
        cmap, start, stop = plt.colormaps["YlGnBu"], 0.05, 0.55
    else:
        cmap, start, stop = plt.colormaps["RdYlGn"], 0.1, 0.9

    colors = []
    for value in values:
        if not _is_number(value) or not np.isfinite(value):
            colors.append(NO_METRIC_COLOR)
            continue
        x = ((float(value) - low) / max(high - low, 1e-12)) ** power
        if higher_is_better is False:
            x = 1 - x
        colors.append(to_hex(cmap(start + x * (stop - start))))
    return colors
