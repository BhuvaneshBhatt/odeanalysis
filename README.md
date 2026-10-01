# odeanalysis

`odeanalysis` is a SymPy-based package for exact symbolic, formal, structural,
and certified analysis of **scalar linear ODEs** and **first-order linear
differential systems**. It is designed for structural questions—singularities,
Frobenius/Levelt data, Newton and Levelt--Turrittin structure, Stokes geometry,
turning points, parameter transitions, exact continuation in selected families,
and rigorously enclosed numerical transport—not as a generic nonlinear ODE
solver.

The package distinguishes exact, certified-formal, certified-semialgebraic,
certified-numerical, structural, and incomplete results. See the
[certification model](./docs/certification-model.md) before interpreting a
`complete`, `certified`, or `structural_only` field.

## Choose a workflow

| Starting point | First guide | Typical public entry points |
| --- | --- | --- |
| Scalar equation/operator | [Scalar workflow](./docs/scalar-workflow.md) | `analyze_ode_singularities`, `frobenius_analysis`, `differential_newton_polygon` |
| `Y' = A(x)Y+b(x)` | [System workflow](./docs/system-workflow.md) | `analyze_system_singularity`, `formal_system_analysis`, `system_stokes_geometry` |
| Symbolic parameters | [Parameter workflow](./docs/parameter-workflow.md) | `local_parameter_analysis`, `system_formal_type_stratification`, `parameterized_turning_analysis` |
| Rigorous numerical transport | [Certified continuation](./docs/certified-continuation.md) | `certified_system_continuation` |

The [documentation index](./docs/index.md) gives the complete map.

## Capability overview

| Capability | Scalar | Systems | Parameters | Evidence |
| --- | --- | --- | --- | --- |
| Local singularity classification | Yes | Yes | Yes | Exact where valuations decide |
| Frobenius / residue-Levelt structure | Yes | Yes | Yes | Exact / bounded formal |
| Irregular formal analysis | Yes | Yes | Yes | Exact formal within stated bounds |
| Ramification and exponential parts | Yes | Yes | Yes | Verified formal reduction |
| Structural Stokes geometry | Yes | Yes | Yes | Structural, not generic analytic constants |
| Turning points / uniform WKB | Yes | Via scalarization/model-specific paths | Yes | Exact local structure + formal models |
| Scalar ↔ system interoperability | Companion form | Cyclic scalarization | Bounded by cyclicity proof | Exact transformation replay |
| Canonical recognition | Selected families | Via scalarization where applicable | Partial | Exact pullback verification |
| Exact continuation | Selected canonical families | Selected inherited cases | Partial | Exact formulas |
| Certified numerical continuation | — | Constant homogeneous systems | — | Arb complex-ball enclosure |
| Generic analytic Stokes matrices | No | No | No | Requires stronger global analytic certification |
| Nonlinear ODEs | No | No | No | Outside scope |

The theorem-level boundary is in [capabilities and limitations](./docs/capabilities.md).

## Installation

```bash
python -m pip install odeanalysis
```

Python 3.11+ is required. Core dependencies are SymPy, `funcprops`, and
`semialg`. Rigorous ball continuation uses:

```bash
python -m pip install "odeanalysis[certified]"
```

## Scalar quick start

```python
import sympy as sp
from odeanalysis import analyze_ode_singularities, frobenius_analysis

x = sp.symbols("x")
y = sp.Function("y")
ode = x**2 * sp.diff(y(x), x, 2) + x * sp.diff(y(x), x) + (x**2 - 1) * y(x)

singularities = analyze_ode_singularities(ode, y, x)
frobenius = frobenius_analysis(ode, y, x, point=0, terms=6)
```

Continue with [local analysis](./docs/local-analysis.md),
[Fuchsian analysis](./docs/fuchsian-analysis.md), [Newton invariants](./docs/newton-invariants.md),
[formal asymptotics](./docs/formal-asymptotics.md), [turning points](./docs/turning-points.md),
[canonical equations](./docs/canonical-equations.md), and
[analytic continuation](./docs/analytic-continuation.md).

## System quick start

```python
import sympy as sp
from odeanalysis import (
    FirstOrderSystem,
    analyze_system_singularity,
    formal_system_analysis,
    system_stokes_geometry,
)

x = sp.symbols("x")
system = FirstOrderSystem(x, sp.ImmutableMatrix.diag(x**-2, -(x**-2)))
local = analyze_system_singularity(system, 0)
formal = formal_system_analysis(system, 0, adaptive=True)
stokes = system_stokes_geometry(formal)
assert formal.certificate.verified
```

The system guides are [singularities](./docs/system-singularities.md),
[formal reduction](./docs/system-formal-analysis.md), [Stokes geometry](./docs/system-stokes.md),
and [parameter stratification](./docs/system-parameters.md). The overview remains in
[systems](./docs/systems.md) and the deeper reducer reference is
[Levelt--Turrittin](./docs/levelt-turrittin.md).

## Parameter quick start

```python
from odeanalysis import system_formal_type_stratification

a = sp.symbols("a", real=True)
family = FirstOrderSystem(x, sp.ImmutableMatrix.diag(a / x, -a / x))
strata = system_formal_type_stratification(family, (a,), max_resonance_order=2)
```

A formal-type stratum certifies a **discrete bounded signature**: singularity
kind, leading rank, spectral multiplicity pattern, resonance orders within the
configured bound, ramification, block dimensions, exponential-block data, and
structural Stokes combinatorics. Continuously varying exponent values are not
claimed to be constant merely because a cell is open.

## Certified numerical continuation

```python
from odeanalysis import certified_system_continuation

constant = FirstOrderSystem(x, sp.ImmutableMatrix([[0, 1], [0, 0]]))
transport = certified_system_continuation(constant, 0, 2, precision_bits=192)
```

The current certified backend is deliberately narrow: for constant homogeneous
systems it encloses `exp((b-a)A)` with Arb complex balls. Variable-coefficient
validated integration is refused rather than replaced by tolerance-only
floating-point integration. See [certified continuation](./docs/certified-continuation.md).

## Examples, verification, and boundaries

Start with the [example gallery](./docs/example-gallery.md) and
[worked failures](./docs/worked-failures.md). The older compact
[examples reference](./docs/examples.md) is retained for API-oriented snippets.

For evidence semantics and conventions see [verification](./docs/verification.md),
[mathematical conventions](./docs/conventions.md), [API classification](./docs/api-classification.md), [public API index](./docs/api-reference.md),
[architecture](./docs/architecture.md), and [traceability](./docs/traceability.md).
[Dominance](./docs/dominance.md) documents sector ordering, and the
[user guide](./docs/user-guide.md) remains a broad reference.

## License

GPL-3.0-only. See `LICENSE`.
