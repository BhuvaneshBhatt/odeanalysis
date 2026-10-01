"""Canonical scalar linear differential-operator representation."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp
from sympy.core.function import AppliedUndef
from sympy.solvers.deutils import ode_order

from ._assumptions import zero_status


def as_ode_expression(ode: sp.Expr | sp.Equality) -> sp.Expr:
    """Return ``lhs-rhs`` for an ODE, expanded but otherwise unchanged."""

    if isinstance(ode, sp.Equality):
        return sp.expand(ode.lhs - ode.rhs)
    return sp.expand(sp.sympify(ode))


def function_class(function: sp.FunctionClass | sp.Expr) -> sp.FunctionClass:
    """Normalize ``y`` or ``y(x)`` to its undefined SymPy function class."""

    if isinstance(function, sp.FunctionClass):
        return function
    if isinstance(function, AppliedUndef):
        return function.func
    if getattr(function, "is_Function", False) and hasattr(function, "func"):
        return function.func
    raise TypeError("function must be an undefined SymPy function such as y or y(x)")


@dataclass(frozen=True)
class LinearDifferentialOperator:
    """A scalar linear ODE represented by coefficient functions.

    ``coefficients[j]`` multiplies the ``j``-th derivative.  The representation
    retains an optional inhomogeneous remainder, but structural singularity and
    Frobenius analysis concern the homogeneous operator only.
    """

    variable: sp.Symbol
    function: sp.FunctionClass
    coefficients: tuple[sp.Expr, ...]
    inhomogeneous: sp.Expr = sp.S.Zero

    @property
    def order(self) -> int:
        return len(self.coefficients) - 1

    @property
    def leading_coefficient(self) -> sp.Expr:
        return self.coefficients[-1]

    @property
    def homogeneous_expression(self) -> sp.Expr:
        yx = self.function(self.variable)
        return sp.expand(
            sum(self.coefficients[j] * sp.diff(yx, self.variable, j) for j in range(self.order + 1))
        )

    @property
    def expression(self) -> sp.Expr:
        return sp.expand(self.homogeneous_expression + self.inhomogeneous)

    @property
    def is_homogeneous(self) -> bool:
        return self.inhomogeneous == 0

    def normalized(self) -> LinearDifferentialOperator:
        """Return the monic operator obtained by division by its leading coefficient."""

        lead = self.leading_coefficient
        coeffs = (
            *tuple(sp.cancel(sp.together(c / lead)) for c in self.coefficients[:-1]),
            sp.S.One,
        )
        rhs = sp.cancel(sp.together(self.inhomogeneous / lead))
        return LinearDifferentialOperator(self.variable, self.function, coeffs, rhs)

    def coefficient(self, derivative_order: int) -> sp.Expr:
        return self.coefficients[derivative_order]

    def to_sympy_holonomic_operator(self):
        """Convert to SymPy's holonomic differential-operator type when polynomial.

        The conversion is an adapter, not the package's
        canonical representation: SymPy's holonomic algebra is excellent for
        polynomial-coefficient operator arithmetic, while ``odeanalysis`` also
        needs rational/meromorphic coefficients and local singularity metadata.
        """

        if not all(sp.sympify(c).is_polynomial(self.variable) for c in self.coefficients):
            raise ValueError("SymPy holonomic conversion requires polynomial coefficients")
        from sympy.holonomic.holonomic import DifferentialOperators

        ring, dx = DifferentialOperators(sp.EX.old_poly_ring(self.variable), "Dx")
        _ = ring
        result = 0
        for j, coefficient in enumerate(self.coefficients):
            result += coefficient * dx**j
        return result

    def reciprocal_transform(
        self,
        new_function: sp.FunctionClass | sp.Expr,
        new_variable: sp.Symbol,
    ) -> LinearDifferentialOperator:
        """Transform the equation under ``x = 1/t`` and re-extract its operator."""

        g = function_class(new_function)
        yx = self.function(self.variable)
        ut = g(new_variable)
        replacements: dict[sp.Expr, sp.Expr] = {yx: ut}
        current = ut
        for k in range(1, self.order + 1):
            current = sp.expand(-(new_variable**2) * sp.diff(current, new_variable))
            replacements[sp.diff(yx, self.variable, k)] = current
        transformed = self.expression.xreplace(replacements).subs(self.variable, 1 / new_variable)
        transformed = sp.factor(sp.together(transformed))
        return LinearDifferentialOperator.from_ode(transformed, g, new_variable)

    @classmethod
    def from_ode(
        cls,
        ode: sp.Expr | sp.Equality,
        function: sp.FunctionClass | sp.Expr,
        variable: sp.Symbol,
    ) -> LinearDifferentialOperator:
        """Extract a canonical scalar linear operator from a SymPy ODE.

        SymPy's ``ode_order`` is used as the authoritative order detector.  We
        then use a polynomial representation in ``y, y', ..., y^(n)`` to verify
        linearity and extract coefficients exactly.
        """

        equation = as_ode_expression(ode)
        f = function_class(function)
        yx = f(variable)
        order = int(ode_order(equation, yx))
        if order < 1:
            raise ValueError(
                "equation must contain at least one derivative of the dependent function"
            )

        gens = [yx] + [sp.diff(yx, variable, k) for k in range(1, order + 1)]
        try:
            poly = sp.Poly(equation, *gens)
        except sp.PolynomialError as exc:
            raise ValueError(
                "equation must be polynomial and linear in y and its derivatives"
            ) from exc
        if poly.total_degree() > 1:
            raise ValueError("ODE analysis requires a scalar linear ODE")

        coeffs = tuple(sp.cancel(equation.coeff(gens[k])) for k in range(order + 1))
        if any(c.has(*gens) for c in coeffs):
            raise ValueError("ODE coefficients must not depend on y or its derivatives")
        homogeneous = sp.expand(sum(coeffs[k] * gens[k] for k in range(order + 1)))
        remainder = sp.cancel(sp.together(equation - homogeneous))
        if zero_status(coeffs[-1]) is True:
            raise ValueError("could not determine a nonzero leading derivative coefficient")
        return cls(variable=variable, function=f, coefficients=coeffs, inhomogeneous=remainder)


def _coerce_linear_operator(
    ode: sp.Expr | sp.Equality | LinearDifferentialOperator,
    function: sp.FunctionClass | sp.Expr | None,
    variable: sp.Symbol | None,
) -> LinearDifferentialOperator:
    """Return an operator, constructing it from an ODE expression when needed."""

    if isinstance(ode, LinearDifferentialOperator):
        return ode
    if function is None or variable is None:
        raise TypeError("function and variable are required when ode is not an operator")
    return LinearDifferentialOperator.from_ode(ode, function, variable)
