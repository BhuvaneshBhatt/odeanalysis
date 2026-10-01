# Linear systems

`odeanalysis` treats first-order linear systems as a public analysis target as well as an internal representation for scalar equations.

## Representation

`FirstOrderSystem` represents

```text
Y' = A(x) Y + b(x).
```

It preserves the forcing term, supports exact changes of independent variable and ramified covers, and applies gauges `Y = G Z` with the full connection law `G^-1 A G - G^-1 G'`. Scalar `LinearDifferentialOperator` objects can be converted to companion systems.

## Singularity analysis

`analyze_system_singularity(system, point)` classifies finite points and infinity. For a local connection with pole order `m`, it reports the leading Laurent matrix and raw Poincare rank `max(0, m-1)`. At a regular singular point it also returns the residue, residue eigenvalues as local exponents, and provable nonzero integral eigenvalue differences as resonances.

```python
import sympy as sp
from odeanalysis import FirstOrderSystem, analyze_system_singularity

x = sp.symbols("x")
system = FirstOrderSystem(x, sp.ImmutableMatrix([[0, 0], [0, 2]]) / x)
local = analyze_system_singularity(system, 0)

assert local.kind == "regular_singular"
assert local.exponents == (0, 2)
assert local.resonances[0].difference == -2 or local.resonances[0].difference == 2
```

Infinity is analyzed after the exact reciprocal transformation `x = 1/t`; the connection Jacobian is therefore included automatically.

The reported Poincare rank is the rank of the supplied connection presentation. Formal gauge reduction may lower it.

## Formal system analysis

`formal_system_analysis()` converts the localized connection to `MatrixLaurentSeries` and runs the existing exact bounded Moser/spectral/Levelt--Turrittin pipeline. `FormalSystemAnalysis` keeps the inexpensive singularity result beside the reduction evidence and exposes completion, ramification, limitation, and `verify()`.

```python
from odeanalysis import formal_system_analysis

irregular = FirstOrderSystem(x, sp.ImmutableMatrix([[1 / x**2, 0], [0, -1 / x**2]]))
formal = formal_system_analysis(irregular, max_power=2)
assert formal.complete
assert formal.verify()
```

This API does not replace the lower-level reduction objects; it gives them a stable system-facing entry point.

## Structural Stokes geometry

`system_stokes_geometry()` compares formal exponential blocks through their pairwise differences. The initial contract is deliberately structural: equal-magnitude and phase-alignment rays are returned, but generic analytic Stokes multipliers are not inferred from formal data.

The formal result retains scalar irregular exponential polynomials extracted from its reduced blocks, so Stokes geometry normally consumes them directly:

```python
geometry = system_stokes_geometry(formal)
assert geometry.structural_only
```

This separation mirrors the scalar API: formal ray/support geometry and analytically normalized Stokes constants are different mathematical objects.

## Parameterized systems

`system_parameter_analysis()` delegates the real parameter decomposition to `semialg`. Its transition polynomial families are kept separately:

- `rank_loci`: leading-matrix minor transitions;
- `collision_loci`: residue characteristic discriminants;
- `resonance_loci`: for 2x2 regular-singular systems, bounded integer eigenvalue-difference conditions;
- `block_loci`: spectral block-collision conditions;
- `stokes_loci`: optional exponential-block degeneracy/phase transitions in explicit real coordinates.

```python
a = sp.symbols("a", real=True)
family = FirstOrderSystem(x, sp.ImmutableMatrix([[a / x, 0], [0, -a / x]]))
parameter_data = system_parameter_analysis(family, (a,), max_resonance_order=4)
assert parameter_data.exhaustive
```

The parameter analyzer intentionally does not guess unresolved symbolic eigensystem or complex-phase conditions. The initial resonance contract is exact but bounded, and the Stokes transition contract requires real semialgebraic coordinates.

## Current boundaries

System singularity analysis is defined for finite rational local valuations. Formal reduction inherits the configured truncation and ramification bounds of the Moser/Levelt--Turrittin machinery. Parameterized resonance discovery is initially specialized to exact 2x2 residue discriminants; higher-dimensional integer spectral relations require a stronger algebraic spectral layer. Generic analytic Stokes and connection matrices remain outside the system contract.

## Detailed guides

- [System singularities](system-singularities.md)
- [Formal system analysis](system-formal-analysis.md)
- [System Stokes geometry](system-stokes.md)
- [Parameterized systems](system-parameters.md)

## Advanced interoperability and certified continuation

Use `scalarize_system()` when a homogeneous system has a certifiably cyclic output; `scalar_system_correspondence()` verifies the standard companion round trip. `formal_system_analysis(..., adaptive=True)` provides bounded adaptive Levelt--Turrittin deepening with a `FormalReductionCertificate`.

`certified_system_continuation()` is a separate validated-numerics API. Its initial backend certifies straight-segment transport for constant homogeneous systems using Arb complex-ball matrix exponentials. Variable-coefficient validated integration, numerical connection matrices, and numerical analytic Stokes matrices remain future extensions and are rejected rather than approximated under the word “certified.”
