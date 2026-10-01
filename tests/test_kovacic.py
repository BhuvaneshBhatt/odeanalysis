"""Certificate-oriented Kovacic analysis tests."""

import sympy as sp

from odeanalysis.kovacic import (
    KovacicOutcome,
    _case3,
    _normal_form_potential,
    _pole_data,
    kovacic_analysis,
)
from odeanalysis.operator import LinearDifferentialOperator

x = sp.symbols("x")
y = sp.Function("y")


def test_kovacic_case_one_certifies_exponential_solutions():
    equation = sp.diff(y(x), x, 2) - y(x)
    result = kovacic_analysis(equation, y, x)
    assert result.outcome is KovacicOutcome.CASE_1
    assert result.case == 1
    assert result.is_liouvillian is True
    assert result.verify()
    assert result.normal_form_potential == 1


def test_kovacic_case_one_certifies_euler_power_solutions():
    equation = x**2 * sp.diff(y(x), x, 2) - 2 * y(x)
    result = kovacic_analysis(equation, y, x)
    assert result.outcome is KovacicOutcome.CASE_1
    assert result.is_liouvillian is True
    assert result.verify()


def test_kovacic_case_two_replays_quadratic_algebraic_certificate():
    equation = 2 * x**2 * sp.diff(y(x), x, 2) - x * sp.diff(y(x), x) + (1 + x) * y(x)
    result = kovacic_analysis(equation, y, x)
    assert result.outcome is KovacicOutcome.CASE_2
    assert result.case == 2
    assert result.is_liouvillian is True
    assert result.case2_certificates
    assert result.verify()


def test_case_three_recurrence_replays_published_n4_example():
    equation = (
        (1 - x) * x**2 * sp.diff(y(x), x, 2)
        + (5 * x - 4) * x * sp.diff(y(x), x)
        + (6 - 9 * x) * y(x)
    )
    operator = LinearDifferentialOperator.from_ode(equation, y, x)
    potential = _normal_form_potential(operator)
    certificates = _case3(potential, x, *_pole_data(potential, x))
    assert len(certificates) == 1
    certificate = certificates[0]
    assert certificate.n == 4
    assert certificate.degree == 0
    assert certificate.verify(potential, x)
    omega = sp.Symbol("omega")
    expected = -((2 * omega * x**2 - 2 * omega * x - x + 2) ** 4) / 16
    assert sp.simplify(certificate.algebraic_log_derivative - expected) == 0


def test_airy_is_certified_non_liouvillian_after_all_three_cases_fail():
    equation = sp.diff(y(x), x, 2) - x * y(x)
    result = kovacic_analysis(equation, y, x)
    assert result.outcome is KovacicOutcome.NO_LIOUVILLIAN_SOLUTION
    assert result.is_liouvillian is False
    assert result.verify()
    assert "Cases 1, 2, and 3" in result.reason
