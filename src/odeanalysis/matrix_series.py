"""Sparse matrix Laurent series for formal system calculations.

The scalar formal layer uses :class:`~odeanalysis.series.SparseLaurentSeries`.
This module provides the matrix analogue needed for formal gauge
transformations and block reduction.  Powers are integral in a uniformizing
parameter ``t``; Puiseux series in a local coordinate ``h`` are represented on
``h = t**r``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import sympy as sp
from sympy.matrices.exceptions import NonInvertibleMatrixError


@dataclass(frozen=True)
class MatrixLaurentSeries:
    """Finite sparse Laurent series with matrix coefficients.

    ``terms`` contains pairs ``(power, coefficient_matrix)``.  All coefficient
    matrices have the declared shape.  Formal inversion is supported for
    square series whose lowest-power coefficient is invertible; because an
    inverse is generally infinite, :meth:`inverse` requires a highest output
    power.
    """

    variable: sp.Symbol
    rows: int
    cols: int
    terms: tuple[tuple[int, sp.ImmutableMatrix], ...]

    def __post_init__(self) -> None:
        if self.rows < 0 or self.cols < 0:
            raise ValueError("matrix dimensions must be nonnegative")
        combined: dict[int, sp.Matrix] = {}
        for power, coefficient in self.terms:
            if not isinstance(power, int):
                raise TypeError("Laurent powers must be integers")
            matrix = sp.Matrix(coefficient)
            if matrix.shape != (self.rows, self.cols):
                raise ValueError(
                    "all Laurent coefficient matrices must have the declared shape"
                )
            if matrix.is_zero_matrix:
                continue
            if power in combined:
                combined[power] = combined[power] + matrix
            else:
                combined[power] = matrix
        cleaned: list[tuple[int, sp.ImmutableMatrix]] = []
        for power, coefficient in sorted(combined.items()):
            coefficient = coefficient.applyfunc(sp.expand)
            if not coefficient.is_zero_matrix:
                cleaned.append((power, sp.ImmutableMatrix(coefficient)))
        object.__setattr__(self, "terms", tuple(cleaned))

    @classmethod
    def zero(
        cls,
        variable: sp.Symbol,
        rows: int,
        cols: int | None = None,
    ) -> MatrixLaurentSeries:
        if cols is None:
            cols = rows
        return cls(variable, rows, cols, ())

    @classmethod
    def identity(cls, variable: sp.Symbol, size: int) -> MatrixLaurentSeries:
        return cls(variable, size, size, ((0, sp.ImmutableMatrix(sp.eye(size))),))

    @classmethod
    def from_mapping(
        cls,
        variable: sp.Symbol,
        coefficients: Mapping[int, sp.MatrixBase],
        *,
        shape: tuple[int, int] | None = None,
    ) -> MatrixLaurentSeries:
        items = tuple(coefficients.items())
        if shape is None:
            if not items:
                raise ValueError("shape is required for an empty matrix Laurent series")
            first = sp.Matrix(items[0][1])
            shape = first.shape
        return cls(
            variable,
            int(shape[0]),
            int(shape[1]),
            tuple((power, sp.ImmutableMatrix(matrix)) for power, matrix in items),
        )

    @classmethod
    def from_matrix(
        cls,
        matrix: sp.MatrixBase,
        variable: sp.Symbol,
    ) -> MatrixLaurentSeries:
        """Create a finite Laurent series from a matrix of Laurent polynomials."""

        matrix = sp.Matrix(matrix)
        by_power: dict[int, sp.MutableDenseMatrix] = {}
        for i in range(matrix.rows):
            for j in range(matrix.cols):
                expression = sp.expand(matrix[i, j])
                if expression == 0:
                    continue
                for term in sp.Add.make_args(expression):
                    coefficient, power = term.as_coeff_exponent(variable)
                    power = sp.sympify(power)
                    if not power.is_Integer or coefficient.has(variable):
                        raise ValueError(
                            "matrix entries must be finite Laurent polynomials with integral powers"
                        )
                    integer_power = int(power)
                    target = by_power.setdefault(
                        integer_power,
                        sp.zeros(matrix.rows, matrix.cols),
                    )
                    target[i, j] += coefficient
        return cls.from_mapping(
            variable,
            by_power,
            shape=(matrix.rows, matrix.cols),
        )

    @property
    def shape(self) -> tuple[int, int]:
        return self.rows, self.cols

    @property
    def is_zero(self) -> bool:
        return not self.terms

    @property
    def min_power(self) -> int | None:
        return None if not self.terms else self.terms[0][0]

    @property
    def max_power(self) -> int | None:
        return None if not self.terms else self.terms[-1][0]

    def as_dict(self) -> dict[int, sp.ImmutableMatrix]:
        return dict(self.terms)

    def coefficient(self, power: int) -> sp.ImmutableMatrix:
        if not isinstance(power, int):
            raise TypeError("power must be an integer")
        return self.as_dict().get(
            power,
            sp.ImmutableMatrix(sp.zeros(self.rows, self.cols)),
        )

    def to_matrix(self) -> sp.ImmutableMatrix:
        result = sp.zeros(self.rows, self.cols)
        for power, coefficient in self.terms:
            result += sp.Matrix(coefficient) * self.variable**power
        return sp.ImmutableMatrix(result.applyfunc(sp.expand))

    def truncate(
        self,
        *,
        min_power: int | None = None,
        max_power: int | None = None,
    ) -> MatrixLaurentSeries:
        if min_power is not None and max_power is not None and min_power > max_power:
            return self.zero(self.variable, self.rows, self.cols)
        return MatrixLaurentSeries(
            self.variable,
            self.rows,
            self.cols,
            tuple(
                (power, coefficient)
                for power, coefficient in self.terms
                if (min_power is None or power >= min_power)
                and (max_power is None or power <= max_power)
            ),
        )

    def scale(self, scalar: sp.Expr) -> MatrixLaurentSeries:
        scalar = sp.sympify(scalar)
        if scalar == 0:
            return self.zero(self.variable, self.rows, self.cols)
        return MatrixLaurentSeries(
            self.variable,
            self.rows,
            self.cols,
            tuple(
                (power, sp.ImmutableMatrix(sp.Matrix(coefficient) * scalar))
                for power, coefficient in self.terms
            ),
        )

    def shift(self, power: int) -> MatrixLaurentSeries:
        """Multiply the series by ``variable**power`` without expanding."""

        if not isinstance(power, int):
            raise TypeError("power must be an integer")
        return MatrixLaurentSeries(
            self.variable,
            self.rows,
            self.cols,
            tuple((p + power, coefficient) for p, coefficient in self.terms),
        )

    def add(
        self,
        other: MatrixLaurentSeries,
        *,
        min_power: int | None = None,
        max_power: int | None = None,
    ) -> MatrixLaurentSeries:
        self._check_same_variable(other)
        if self.shape != other.shape:
            raise ValueError(
                "matrix Laurent series have incompatible shapes for addition"
            )
        coefficients: dict[int, sp.Matrix] = {
            power: sp.Matrix(coefficient) for power, coefficient in self.terms
        }
        for power, coefficient in other.terms:
            coefficients[power] = coefficients.get(
                power, sp.zeros(self.rows, self.cols)
            ) + sp.Matrix(coefficient)
        return MatrixLaurentSeries.from_mapping(
            self.variable,
            coefficients,
            shape=self.shape,
        ).truncate(min_power=min_power, max_power=max_power)

    def multiply(
        self,
        other: MatrixLaurentSeries,
        *,
        min_power: int | None = None,
        max_power: int | None = None,
    ) -> MatrixLaurentSeries:
        self._check_same_variable(other)
        if self.cols != other.rows:
            raise ValueError(
                "matrix Laurent series have incompatible shapes for multiplication"
            )
        coefficients: dict[int, sp.Matrix] = {}
        for left_power, left_coefficient in self.terms:
            for right_power, right_coefficient in other.terms:
                power = left_power + right_power
                if min_power is not None and power < min_power:
                    continue
                if max_power is not None and power > max_power:
                    continue
                product = sp.Matrix(left_coefficient) * sp.Matrix(right_coefficient)
                coefficients[power] = (
                    coefficients.get(power, sp.zeros(self.rows, other.cols)) + product
                )
        return MatrixLaurentSeries.from_mapping(
            self.variable,
            coefficients,
            shape=(self.rows, other.cols),
        )

    def derivative(
        self,
        *,
        ramification_index: int = 1,
        min_power: int | None = None,
        max_power: int | None = None,
    ) -> MatrixLaurentSeries:
        r"""Differentiate with respect to ``h`` when ``h=t**r``.

        For a matrix coefficient ``A_p``,
        ``D_h(A_p*t**p) = (p/r) A_p*t**(p-r)``.
        """

        if ramification_index < 1:
            raise ValueError("ramification_index must be positive")
        result = MatrixLaurentSeries(
            self.variable,
            self.rows,
            self.cols,
            tuple(
                (
                    power - ramification_index,
                    sp.ImmutableMatrix(
                        sp.Matrix(coefficient) * sp.Rational(power, ramification_index)
                    ),
                )
                for power, coefficient in self.terms
                if power != 0
            ),
        )
        return result.truncate(min_power=min_power, max_power=max_power)

    def inverse(
        self,
        *,
        max_power: int,
        min_power: int | None = None,
    ) -> MatrixLaurentSeries:
        """Return a truncated formal inverse through ``max_power``.

        If ``A(t)=t**p(A0 + A1*t + ...)`` with invertible ``A0``, the inverse
        is generated recursively from ``A(t) B(t)=I``.  The leading inverse
        power is ``-p``.  A :class:`ValueError` is raised when the leading
        coefficient is singular or the series is zero/non-square.
        """

        if self.rows != self.cols:
            raise ValueError("formal inversion requires a square matrix series")
        if self.is_zero:
            raise ValueError("the zero matrix series is not invertible")
        lead_power = self.min_power
        if lead_power is None:
            raise RuntimeError("nonzero Laurent series has no leading power")
        inverse_lead_power = -lead_power
        if max_power < inverse_lead_power:
            return self.zero(self.variable, self.rows, self.cols)

        coefficients = {
            power - lead_power: sp.Matrix(matrix) for power, matrix in self.terms
        }
        a0 = coefficients[0]
        try:
            b0 = a0.inv()
        except (ValueError, ZeroDivisionError, NonInvertibleMatrixError) as exc:
            raise ValueError("leading matrix coefficient is not invertible") from exc
        if sp.simplify(a0.det()) == 0:
            raise ValueError("leading matrix coefficient is not invertible")

        relative_limit = max_power - inverse_lead_power
        inverse_coefficients: dict[int, sp.Matrix] = {0: b0}
        zero = sp.zeros(self.rows, self.cols)
        for n in range(1, relative_limit + 1):
            convolution = zero.copy()
            for k in range(1, n + 1):
                ak = coefficients.get(k)
                if ak is not None:
                    convolution += ak * inverse_coefficients[n - k]
            inverse_coefficients[n] = -b0 * convolution

        result = MatrixLaurentSeries.from_mapping(
            self.variable,
            {
                inverse_lead_power + relative_power: coefficient
                for relative_power, coefficient in inverse_coefficients.items()
            },
            shape=self.shape,
        )
        return result.truncate(min_power=min_power, max_power=max_power)

    def _check_same_variable(self, other: MatrixLaurentSeries) -> None:
        if self.variable != other.variable:
            raise ValueError(
                "matrix Laurent series use different uniformizing variables"
            )
