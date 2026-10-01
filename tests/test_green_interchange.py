import sympy as sp

from odeanalysis import LinearDifferentialOperator
from odeanalysis.interchange import green_operator_data


def test_green_operator_data_verifies_constant_coefficient_characteristic_polynomial():
    x = sp.symbols("x", positive=True)
    y = sp.Function("y")
    operator = LinearDifferentialOperator(x, y, (2, -3, 1))

    data = green_operator_data(operator, point=sp.oo)

    lam = data.characteristic_parameter
    assert data.order == 2
    assert data.coefficients == (2, -3, 1)
    assert sp.expand(data.characteristic_polynomial - (lam**2 - 3 * lam + 2)) == 0
    assert data.constant_coefficients is True
    assert data.leading_coefficient_nonzero is True
    assert data.verify() is True


def test_green_operator_data_preserves_variable_coefficients_as_uncertified_input():
    x = sp.symbols("x", positive=True)
    y = sp.Function("y")
    operator = LinearDifferentialOperator(x, y, (1, x, 1))

    data = green_operator_data(operator, point=sp.oo)

    assert data.constant_coefficients is False
    assert data.verify() is True
    assert x in data.characteristic_polynomial.free_symbols


def test_green_operator_data_does_not_claim_unassumed_symbolic_leading_coefficient_is_nonzero():
    x, a = sp.symbols("x a")
    y = sp.Function("y")
    operator = LinearDifferentialOperator(x, y, (1, 0, a))

    data = green_operator_data(operator)

    assert data.constant_coefficients is True
    assert data.leading_coefficient_nonzero is None
    assert data.verify() is True
