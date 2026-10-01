# Fuchsian analysis and apparent singularities

## Riemann schemes

`riemann_scheme()` builds the exponent table of a scalar equation that is Fuchsian on the Riemann sphere. Every singular point, including infinity when singular, must be regular singular and must have a completely resolved indicial root multiset. Irregular or unresolved equations are rejected rather than assigned a partial P-symbol.

```python
import sympy as sp
from odeanalysis import riemann_scheme

x = sp.symbols("x")
y = sp.Function("y")
a = sp.Rational(1, 3)
b = sp.Rational(1, 2)
c = sp.Rational(2, 3)
ode = x * (1 - x) * sp.diff(y(x), x, 2) + (c - (a + b + 1) * x) * sp.diff(y(x), x) - a * b * y(x)

scheme = riemann_scheme(ode, y, x)
assert scheme.points == (0, 1, sp.oo)
assert scheme.exponent_columns == (
    (0, sp.Rational(1, 3)),
    (sp.Rational(-1, 6), 0),
    (sp.Rational(1, 3), sp.Rational(1, 2)),
)
```

## Fuchs relation

For an order-`n` scalar Fuchsian equation with `m` singular points on the sphere, the local exponents satisfy

`sum rho_ij = (m - 2) n (n - 1) / 2`.

`fuchs_relation()` returns the actual sum, expected sum, residual, and a three-valued exact decision. Symbolic residuals that cannot be proved zero or nonzero remain undecided.

## Apparent singularities

`apparent_singularity_analysis()` tests whether a coefficient singularity is removable at the level of the complete local solution space. The package uses the holomorphic definition: all local exponents must be nonnegative integers and a complete logarithmic Frobenius basis must contain no logarithms.

```python
from odeanalysis import apparent_singularity_analysis

apparent_ode = sp.diff(y(x), x, 2) - sp.diff(y(x), x) / x
result = apparent_singularity_analysis(apparent_ode, y, x, point=0)
assert result.exponents == (0, 2)
assert result.apparent is True
```

The decision is `None` when exponent integrality, singularity type, or completeness of the logarithmic basis cannot be certified. Negative integral exponents are not called apparent here because they give meromorphic rather than holomorphic local solutions.
