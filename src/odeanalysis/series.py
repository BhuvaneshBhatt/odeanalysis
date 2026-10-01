"""Small sparse Laurent-series representation for formal local calculations.

Powers are integral in a chosen uniformizing parameter ``t``.  A Puiseux
series in a local coordinate ``h`` is therefore represented after choosing
``h = t**r``.  The class provides only the operations needed by
formal ODE recurrences: addition, multiplication, truncation, and the ramified
local derivative ``D_h``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import sympy as sp


@dataclass(frozen=True)
class SparseLaurentSeries:
    """Finite sparse Laurent series in an integral-power uniformizer."""

    variable: sp.Symbol
    terms: tuple[tuple[int, sp.Expr], ...]

    def __post_init__(self) -> None:
        combined: dict[int, sp.Expr] = {}
        for power, coefficient in self.terms:
            if not isinstance(power, int):
                raise TypeError("Laurent powers must be integers")
            coefficient = sp.sympify(coefficient)
            if coefficient == 0:
                continue
            combined[power] = sp.expand(combined.get(power, sp.S.Zero) + coefficient)
        cleaned = tuple(
            (power, coefficient)
            for power, coefficient in sorted(combined.items())
            if coefficient != 0
        )
        object.__setattr__(self, "terms", cleaned)

    @classmethod
    def zero(cls, variable: sp.Symbol) -> SparseLaurentSeries:
        return cls(variable, ())

    @classmethod
    def one(cls, variable: sp.Symbol) -> SparseLaurentSeries:
        return cls(variable, ((0, sp.S.One),))

    @classmethod
    def from_mapping(
        cls,
        variable: sp.Symbol,
        coefficients: Mapping[int, sp.Expr],
    ) -> SparseLaurentSeries:
        return cls(variable, tuple(coefficients.items()))

    @classmethod
    def from_expr(
        cls,
        expression: sp.Expr,
        variable: sp.Symbol,
    ) -> SparseLaurentSeries:
        """Create a finite Laurent series from an expanded Laurent polynomial."""

        expression = sp.expand(sp.sympify(expression))
        if expression == 0:
            return cls.zero(variable)

        terms: list[tuple[int, sp.Expr]] = []
        for term in sp.Add.make_args(expression):
            coefficient, power = term.as_coeff_exponent(variable)
            power = sp.sympify(power)
            if not power.is_Integer:
                raise ValueError(
                    "expression is not a finite Laurent polynomial with integral powers"
                )
            # as_coeff_exponent leaves non-monomial dependence in the coefficient.
            if coefficient.has(variable):
                raise ValueError(
                    "expression is not a finite Laurent polynomial in the requested variable"
                )
            terms.append((int(power), coefficient))
        return cls(variable, tuple(terms))

    @property
    def is_zero(self) -> bool:
        return not self.terms

    @property
    def min_power(self) -> int | None:
        return None if not self.terms else self.terms[0][0]

    @property
    def max_power(self) -> int | None:
        return None if not self.terms else self.terms[-1][0]

    def as_dict(self) -> dict[int, sp.Expr]:
        return dict(self.terms)

    def to_expr(self) -> sp.Expr:
        return sp.Add(
            *(coefficient * self.variable**power for power, coefficient in self.terms)
        )

    def truncate(
        self,
        *,
        min_power: int | None = None,
        max_power: int | None = None,
    ) -> SparseLaurentSeries:
        if min_power is not None and max_power is not None and min_power > max_power:
            return self.zero(self.variable)
        return SparseLaurentSeries(
            self.variable,
            tuple(
                (power, coefficient)
                for power, coefficient in self.terms
                if (min_power is None or power >= min_power)
                and (max_power is None or power <= max_power)
            ),
        )

    def scale(self, scalar: sp.Expr) -> SparseLaurentSeries:
        scalar = sp.sympify(scalar)
        if scalar == 0:
            return self.zero(self.variable)
        return SparseLaurentSeries(
            self.variable,
            tuple((power, scalar * coefficient) for power, coefficient in self.terms),
        )

    def add(
        self,
        other: SparseLaurentSeries,
        *,
        min_power: int | None = None,
        max_power: int | None = None,
    ) -> SparseLaurentSeries:
        self._check_variable(other)
        coefficients = self.as_dict()
        for power, coefficient in other.terms:
            coefficients[power] = coefficients.get(power, sp.S.Zero) + coefficient
        return SparseLaurentSeries.from_mapping(self.variable, coefficients).truncate(
            min_power=min_power,
            max_power=max_power,
        )

    def multiply(
        self,
        other: SparseLaurentSeries,
        *,
        min_power: int | None = None,
        max_power: int | None = None,
    ) -> SparseLaurentSeries:
        self._check_variable(other)
        coefficients: dict[int, sp.Expr] = {}
        for left_power, left_coefficient in self.terms:
            for right_power, right_coefficient in other.terms:
                power = left_power + right_power
                if min_power is not None and power < min_power:
                    continue
                if max_power is not None and power > max_power:
                    continue
                coefficients[power] = (
                    coefficients.get(power, sp.S.Zero)
                    + left_coefficient * right_coefficient
                )
        return SparseLaurentSeries.from_mapping(self.variable, coefficients)

    def derivative(
        self,
        *,
        ramification_index: int = 1,
        min_power: int | None = None,
        max_power: int | None = None,
    ) -> SparseLaurentSeries:
        r"""Differentiate with respect to ``h`` when ``h=t**r``.

        ``D_h(c*t**p) = (p/r) c*t**(p-r)``.
        """

        if ramification_index < 1:
            raise ValueError("ramification_index must be positive")
        result = SparseLaurentSeries(
            self.variable,
            tuple(
                (
                    power - ramification_index,
                    sp.Rational(power, ramification_index) * coefficient,
                )
                for power, coefficient in self.terms
                if power != 0
            ),
        )
        return result.truncate(min_power=min_power, max_power=max_power)

    def _check_variable(self, other: SparseLaurentSeries) -> None:
        if self.variable != other.variable:
            raise ValueError("Laurent series use different uniformizing variables")
