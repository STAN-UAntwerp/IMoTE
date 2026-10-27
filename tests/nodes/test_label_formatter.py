from imote.nodes.label_formatter import LabelFormatter

NAMES = ["first", "second", "third"]


def test_feature_uses_names_when_given():
    assert LabelFormatter(NAMES).feature(1) == "second"
    assert LabelFormatter().feature(1) == "X1"


def test_number_is_short():
    for value in [1 / 3, 123456.789, -0.000123456]:
        assert len(LabelFormatter.number(value)) <= len(f"{value:.3g}")
        assert float(LabelFormatter.number(value)) == float(f"{value:.3g}")


def test_expression_leaves_out_zero_terms():
    expression = LabelFormatter(NAMES).expression([(0, 1.5), (1, 0.0), (2, 2.0)], 0.0)
    assert "first" in expression and "third" in expression and "second" not in expression


def test_expression_has_no_plus_minus():
    expression = LabelFormatter(NAMES).expression([(0, -1.5), (2, -2.0)], -3.0)
    assert "+ -" not in expression and expression.count("-") == 3


def test_expression_of_only_an_intercept():
    assert LabelFormatter().expression([], 2.5) == LabelFormatter.number(2.5)
    assert LabelFormatter().expression([(0, 0.0)], 0.0) == LabelFormatter.number(0.0)
