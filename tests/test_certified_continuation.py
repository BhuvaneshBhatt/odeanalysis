import sympy as sp

from odeanalysis import FirstOrderSystem, certified_system_continuation


def test_certified_constant_system_continuation_contract():
    x = sp.symbols("x")
    system = FirstOrderSystem(x, sp.ImmutableMatrix.diag(1, -1))
    result = certified_system_continuation(system, 0, 1)
    # The optional validated backend may be absent in minimal source environments.
    if result.complete:
        assert result.enclosure is not None
        assert result.enclosure.certified
        assert result.enclosure.method == "arb-matrix-exponential"
    else:
        assert "python-flint" in result.limitation


def test_certified_continuation_refuses_variable_coefficients():
    x = sp.symbols("x")
    system = FirstOrderSystem(x, sp.ImmutableMatrix([[x]]))
    result = certified_system_continuation(system, 0, 1)
    assert not result.complete
    assert "variable-coefficient" in result.limitation
