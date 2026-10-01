# Newton polygons, slopes, and ranks

Differential Newton polygons are primary public objects in `odeanalysis`. For a localized scalar operator

`L = sum_j a_j(h) D_h**j`,

`DifferentialNewtonPolygon` uses points `(j, v(a_j)-j)`, where `v` is the exact local valuation. The lower hull retains collinear points because all of them contribute to the edge characteristic polynomial.

```python
import sympy as sp
from odeanalysis import (
    differential_newton_polygon,
    katz_rank,
    newton_slopes,
    poincare_rank,
    slope_filtration,
)

x = sp.symbols("x", positive=True)
y = sp.Function("y")
airy = sp.diff(y(x), x, 2) - x * y(x)

polygon = differential_newton_polygon(airy, y, x, point=sp.oo)
assert polygon.slopes == (sp.Rational(3, 2), sp.Rational(3, 2))
assert katz_rank(airy, y, x, point=sp.oo) == sp.Rational(3, 2)
assert poincare_rank(airy, y, x, point=sp.oo) == 3
```

## Katz rank versus Poincare rank

The **Katz rank** is the largest positive rational Newton slope, also called the Poincare--Katz rank. It is invariant under formal gauge equivalence and may have a nontrivial denominator when ramification is required.

The primary `poincare_rank()` API reports a different quantity: the ordinary Poincare rank of the package's natural Euler-scaled companion-system presentation. It is therefore representation/gauge dependent. For Airy at infinity, the Katz rank is `3/2` while this companion presentation has Poincare rank `3`. The explicit name `euler_system_poincare_rank` remains available on `IrregularSingularityInvariants` for callers that want the representation dependence visible in the attribute name.

## Slope filtration

`slope_filtration()` groups equal Newton slopes and records their horizontal multiplicities. Nonpositive lower-hull slopes belong to the regular slope-zero piece. Positive pieces retain their exact rational slopes and required ramification denominators.

```python
filtration = slope_filtration(polygon)
assert filtration.rank == 2
assert filtration.pieces[0].slope == sp.Rational(3, 2)
assert filtration.pieces[0].multiplicity == 2
assert filtration.irregularity == 3
assert filtration.ramification_index == 2
```

The weighted sum `sum(multiplicity * slope)` over positive pieces is the formal irregularity.
