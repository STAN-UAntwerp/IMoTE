import numpy as np
import pytest

from nodes.node_model import (
    ConstantNodeModel,
    LinearNodeModel,
    NoneNodeModel,
    SimpleLinearNodeModel,
    node_model_from_dict,
)

X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

MODELS = [
    NoneNodeModel(),
    ConstantNodeModel(3.0),
    SimpleLinearNodeModel(1, 2.0, 0.5),
    LinearNodeModel(np.array([0.0, -2.0]), 0.25),
]


@pytest.mark.parametrize("model, x, expected", [
    (NoneNodeModel(), X, [0, 0, 0]),
    (NoneNodeModel(), X[0], [0]),
    (ConstantNodeModel(2.5), X, [2.5, 2.5, 2.5]),
    (ConstantNodeModel(2.5), X[0], [2.5]),
    (SimpleLinearNodeModel(0, 2.0, 1.0), np.array([0.0, 1.0, 2.0]), [1.0, 3.0, 5.0]),
    (LinearNodeModel(np.array([1.0, -1.0]), 0.5), X, [-0.5, -0.5, -0.5]),
    (LinearNodeModel(np.array([2.0, 1.0]), 1.0), X[0], [5.0]),
])
def test_predict(model, x, expected):
    np.testing.assert_array_equal(model.predict(x), expected)


@pytest.mark.parametrize("model", MODELS)
def test_label_is_string(model):
    assert isinstance(model.get_label(), str)


@pytest.mark.parametrize("model", MODELS)
def test_round_trip(model):
    restored = node_model_from_dict(model.to_dict())
    assert type(restored) is type(model)
    np.testing.assert_array_equal(restored.predict(X), model.predict(X))


def test_add_simple_linear_model():
    model = LinearNodeModel(np.zeros(2), 1.0)
    model.add_model(SimpleLinearNodeModel(1, 3.0, 0.5))
    np.testing.assert_array_equal(model.coefficients, [0.0, 3.0])
    assert model.intercept == 1.5


def test_add_constant_model():
    model = LinearNodeModel(np.array([1.0, 2.0]), 1.0)
    model.add_model(ConstantNodeModel(2.0))
    np.testing.assert_array_equal(model.coefficients, [1.0, 2.0])
    assert model.intercept == 3.0


def test_add_linear_model_length_mismatch():
    model = LinearNodeModel(np.zeros(2), 0.0)
    with pytest.raises(ValueError):
        model.add_model(LinearNodeModel(np.zeros(3), 0.0))
