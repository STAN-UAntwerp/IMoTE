import pytest

from imote.config import NEAR_ZERO, NO_METRIC_COLOR
from imote.node_metrics.node_metric import BaseNodeMetric
from imote.node_metrics.node_labels import display_names, metric_colors, metric_text, shorten_name

NAMES = ["temperature_at_2m", "cars-per-hour", "wind speed", "electroencephalogram", "age", "x"]


class LowerIsBetter(BaseNodeMetric):
    higher_is_better = False


class HigherIsBetter(BaseNodeMetric):
    higher_is_better = True


class InUnitRange(BaseNodeMetric):
    higher_is_better = True
    value_range = (0.0, 1.0)


class NoBetterSide(BaseNodeMetric):
    pass


# --- Shortening names ---
@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("max_length", [4, 6, 12])
def test_short_name_fits(name, max_length):
    short = shorten_name(name, max_length)
    assert 0 < len(short) <= max_length
    if len(name) <= max_length:
        assert short == name


def test_short_name_keeps_whole_leading_words():
    for name in ["cars_per_hour", "cars-per-hour", "wind speed"]:
        short = shorten_name(name, 6)
        assert name.startswith(short) and name[len(short)] in "_- "


def test_short_name_cuts_before_a_vowel():
    for name in ["temperature", "humidity", "density"]:
        short = shorten_name(name, 6)
        assert name.startswith(short) and len(short) >= 4 and name[len(short)] in "aeiou"


def test_short_name_is_cut_as_last_resort():
    short = shorten_name("bcdfghjklmnp", 6)
    assert short.endswith("..") and "bcdfghjklmnp".startswith(short[:-2])


def test_display_names_without_feature_names():
    assert display_names(None, 3, max_length=6) == ["X0", "X1", "X2"]


def test_display_names_are_unique():
    names = ["temperature_at_2m", "temperature_diff_25m", "wind"]
    short_names = display_names(names, 3, max_length=6)
    assert len(set(short_names)) == 3 and short_names[2] == "wind"
    assert all(len(name) <= 6 for name in short_names)


# --- Metric text ---
@pytest.mark.parametrize("value", [0, 7, 0.0369998, 12.345, -12345.6, 1_880_000.5, 3e12, 1e-9])
def test_compact_metric_text_is_short(value):
    assert len(metric_text(value, compact=True)) <= 6


def test_zero_and_tiny_values():
    tiny = NEAR_ZERO / 10
    assert metric_text(0) == metric_text(0, compact=True) == "0"
    assert metric_text(tiny) == metric_text(-tiny, compact=True) != "0"


def test_metric_text_keeps_non_numbers():
    assert metric_text("-") == metric_text("-", compact=True) == "-"


# --- Metric colors ---
def test_better_values_get_the_same_colors_both_ways():
    lower = metric_colors([1.0, 2.0, 4.0], LowerIsBetter)
    higher = metric_colors([4.0, 2.0, 1.0], HigherIsBetter)
    assert lower == higher and len(set(lower)) == 3


def test_ratio_scale_is_not_dominated_by_one_large_value():
    colors = metric_colors([1_000_000.0, 1.0, 2.0, 4.0], LowerIsBetter)
    assert len(set(colors[1:])) == 3


def test_values_outside_the_range_are_grey():
    assert metric_colors([-0.5, 1.5, 0.5], InUnitRange)[:2] == [NO_METRIC_COLOR, NO_METRIC_COLOR]


def test_fixed_range_does_not_depend_on_the_other_values():
    assert metric_colors([0.5, 0.9], InUnitRange)[0] == metric_colors([0.5, 0.1], InUnitRange)[0]


def test_no_better_side_uses_another_gradient():
    assert not set(metric_colors([1.0, 2.0], NoBetterSide)) & set(metric_colors([1.0, 2.0], LowerIsBetter))


def test_grey_for_non_numbers():
    assert metric_colors(["-", float("nan"), 1.0], LowerIsBetter)[:2] == [NO_METRIC_COLOR, NO_METRIC_COLOR]
