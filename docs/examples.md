# Worked examples

These examples emphasize stable mathematical invariants rather than full printed representations.

## Euler--Cauchy: a logarithmic regular-singular block

```python
import sympy as sp
from odeanalysis import levelt_structure

x = sp.symbols("x", positive=True)
y = sp.Function("y")
ode = x**2 * sp.diff(y(x), x, 2) + x * sp.diff(y(x), x)
structure = levelt_structure(ode, y, x, point=0, terms=4)

assert structure.complete
assert structure.ramification_index == 1
assert structure.blocks[0].has_logarithms
```

## Bessel: integer resonance

```python
import sympy as sp
from odeanalysis import frobenius_analysis

x = sp.symbols("x")
y = sp.Function("y")
ode = x**2 * sp.diff(y(x), x, 2) + x * sp.diff(y(x), x) + (x**2 - 1) * y(x)
result = frobenius_analysis(ode, y, x, point=0, terms=5)

assert {root for root, _ in result.root_multiplicities} == {-1, 1}
assert any(item.difference == 2 for item in result.resonances)
```

## Airy at infinity: ramified exponential branches

```python
import sympy as sp
from odeanalysis import levelt_structure

x = sp.symbols("x", positive=True)
y = sp.Function("y")
airy = sp.diff(y(x), x, 2) - x * y(x)
structure = levelt_structure(airy, y, x, point=sp.oo, terms=4)

assert structure.ramification_index == 2
assert tuple(block.dimension for block in structure.blocks) == (1, 1)
```


## First-class linear-system analysis

```python
import sympy as sp
from odeanalysis import (
    FirstOrderSystem,
    analyze_system_singularity,
    formal_system_analysis,
    system_parameter_analysis,
    system_stokes_geometry,
)

x = sp.symbols("x")
system = FirstOrderSystem(x, sp.ImmutableMatrix([[1/x**2, 0], [0, -1/x**2]]))
local = analyze_system_singularity(system, 0)
assert local.kind == "irregular"
assert local.poincare_rank == 1

formal = formal_system_analysis(system, max_power=2)
assert formal.complete and formal.verify()
stokes = system_stokes_geometry(formal)
assert stokes.structural_only

a = sp.symbols("a", real=True)
family = FirstOrderSystem(x, sp.ImmutableMatrix([[a/x, 0], [0, -a/x]]))
parameter_data = system_parameter_analysis(family, (a,), max_resonance_order=2)
assert parameter_data.exhaustive
assert parameter_data.rank_loci and parameter_data.resonance_loci
```

The Stokes result is structural: it does not claim generic analytic Stokes multipliers. See [Linear systems](systems.md) for the full contract.

## Distinct irregular eigenvalues: exact spectral split

```python
import sympy as sp
from odeanalysis import MatrixLaurentSeries, formal_block_diagonalize

t = sp.symbols("t")
connection = MatrixLaurentSeries.from_matrix(sp.diag(t**-2, -(t**-2)), t)
result = formal_block_diagonalize(connection, max_power=3)

assert result.block_dimensions == (1, 1)
assert result.verify()
```

## Nilpotent irregular block: Moser reduction

```python
import sympy as sp
from odeanalysis import MatrixLaurentSeries, moser_reduce

t = sp.symbols("t")
connection = MatrixLaurentSeries.from_matrix(sp.Matrix([[0, t**-2], [1, 0]]), t)
result = moser_reduce(connection, max_power=3)

assert result.complete
assert len(result.steps) == 1
assert result.verify()
```

The test suite contains the same examples as executable structural regressions, so the documentation and mathematical contracts evolve together.

## Hypergeometric equation: Riemann scheme

```python
import sympy as sp
from odeanalysis import riemann_scheme

x = sp.symbols("x")
y = sp.Function("y")
a, b, c = sp.Rational(1, 3), sp.Rational(1, 2), sp.Rational(2, 3)
ode = (
    x * (1 - x) * sp.diff(y(x), x, 2)
    + (c - (a + b + 1) * x) * sp.diff(y(x), x)
    - a * b * y(x)
)
scheme = riemann_scheme(ode, y, x)

assert scheme.points == (0, 1, sp.oo)
assert scheme.fuchs_relation.holds is True
```

## Airy: Newton rank and sector dominance

```python
from odeanalysis import classify_solution_dominance, differential_newton_polygon

x = sp.symbols("x", positive=True)
y = sp.Function("y")
airy = sp.diff(y(x), x, 2) - x * y(x)
polygon = differential_newton_polygon(airy, y, x, point=sp.oo)

assert polygon.katz_rank == sp.Rational(3, 2)
assert polygon.poincare_rank == 3
assert polygon.slopes == (sp.Rational(3, 2), sp.Rational(3, 2))

dominance = classify_solution_dominance(airy, y, x, point=sp.oo)
assert dominance.complete
assert len(dominance.sectors) == 6
```

## Turning points: Airy and Weber uniformization

```python
import sympy as sp
from odeanalysis import (
    airy_uniformization,
    analyze_turning_points,
    weber_uniformization,
)

x = sp.symbols("x", positive=True)
y = sp.Function("y")

airy = sp.diff(y(x), x, 2) - x * y(x)
analysis = analyze_turning_points(airy, y, x)
assert analysis.points[0].multiplicity == 1
assert airy_uniformization(airy, y, x, point=0).is_exact

weber = sp.diff(y(x), x, 2) - x**2 * y(x)
analysis = analyze_turning_points(weber, y, x)
assert analysis.points[0].multiplicity == 2
assert weber_uniformization(weber, y, x, point=0).is_exact
```

For a generic simple turning point such as `Q=x*(1+x)`, the Airy coordinate still verifies exactly but the returned residual is nonzero. That residual is part of the result contract.

## System example 1: resonant Fuchsian residue

```python
import sympy as sp
from odeanalysis import FirstOrderSystem, analyze_system_singularity, formal_system_analysis
x = sp.symbols("x")
S = FirstOrderSystem(x, sp.ImmutableMatrix([[0, 1/x], [0, 2/x]]))
local = analyze_system_singularity(S, 0)
assert local.regular_singular and local.exponents == (0, 2)
assert local.resonances
formal = formal_system_analysis(S, 0)
assert formal.certificate.verified
```

This separates residue resonance from the stronger question of whether logarithms actually occur in a chosen Levelt basis.

## System example 2: transparent irregular Stokes blocks

```python
from odeanalysis import system_stokes_geometry
S = FirstOrderSystem(x, sp.ImmutableMatrix.diag(x**-2, -x**-2))
formal = formal_system_analysis(S, 0)
geometry = system_stokes_geometry(formal)
assert formal.complete
assert geometry.structural_only
assert geometry.pairs
```

The pairwise exponential difference determines rays; no analytic Stokes multiplier is inferred.

## System example 3: nondiagonal irregular spectral splitting

```python
G = sp.ImmutableMatrix([[1, 1], [0, 1]])
D = FirstOrderSystem(x, sp.ImmutableMatrix.diag(x**-2, -x**-2))
S = D.gauge_transform(G)
formal = formal_system_analysis(S, 0, adaptive=True)
assert formal.certificate.verified
assert formal.complete
```

A constant gauge hides the diagonal presentation without changing the formal spectral type.

## System example 4: ramified bounded formal reduction

```python
S = FirstOrderSystem(x, sp.ImmutableMatrix([[0, x**-2], [x**-3, 0]]))
formal = formal_system_analysis(
    S, 0, adaptive=True, max_depth=1, max_adaptive_depth=6,
    max_adaptive_cover_index=12,
)
assert formal.certificate.verified
assert formal.ramification_index >= 1
```

If the configured cover/depth budget is insufficient, `complete` remains false and the certificate records the limitation.

## System example 5: scalar/companion/cyclic correspondence

```python
from odeanalysis import LinearDifferentialOperator, scalar_system_correspondence
y = sp.Function("y")
op = LinearDifferentialOperator(x, y, (x, 1+x, 1))
correspondence = scalar_system_correspondence(op)
assert correspondence.verify()
```

Scalar algorithms remain specialized; the correspondence is an explicit testable bridge rather than an internal replacement of scalar kernels.

## System example 6: parameter formal type and turning confluence

```python
from odeanalysis import system_formal_type_stratification, parameterized_turning_analysis
a = sp.symbols("a", real=True)
S = FirstOrderSystem(x, sp.ImmutableMatrix([[a/x, 0], [0, -a/x]]))
types = system_formal_type_stratification(S, (a,), max_resonance_order=2)
assert types.base.exhaustive

op = LinearDifferentialOperator(x, y, (-(x**2-a), 0, 1))
turning = parameterized_turning_analysis(op, parameters=(a,))
assert turning.exhaustive and turning.transition_polynomials
```

The first decomposition tracks spectral/resonance/formal information; the second isolates the turning-point confluence locus before Airy/Weber uniformization is selected.

## Certified numerical transport: deliberately narrow

```python
from odeanalysis import certified_system_continuation
S = FirstOrderSystem(x, sp.ImmutableMatrix.diag(1, -1))
transport = certified_system_continuation(S, 0, 1)
if transport.complete:
    assert transport.enclosure.certified
```

With `python-flint` installed this encloses the constant-system matrix exponential using Arb balls. Variable-coefficient validated integration is rejected until a rigorous stepwise enclosure backend is implemented.
