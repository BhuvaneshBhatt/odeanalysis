# API classification

`odeanalysis` separates its public surface into three levels so that the package root remains easy to learn while specialized algorithms remain available to advanced users.

## Primary root API

The names in `odeanalysis.__all__` are the stable, documented entry points for ordinary use. They cover representations (`LinearDifferentialOperator`, `FirstOrderSystem`, `MatrixLaurentSeries`), singularity and Frobenius analysis, convergence/local-monodromy evidence, unified `SingularityStructure`, Wronskian/Abel certificates, parameterized scalar and system analysis, finite resonance strata, first-class differential Newton polygons and slope invariants, Fuchsian/Riemann-scheme analysis, apparent-singularity detection, dominant/recessive classification, Liouville normal form, turning-point and uniform-WKB analysis, formal asymptotic solutions, first-class system singularity/formal/Stokes analysis, Levelt--Turrittin reduction, Stokes geometry, formal monodromy, and the `FormalODEData` interchange representation.

A typical import is therefore:

```python
from odeanalysis import (
    analyze_ode_singularities,
    frobenius_analysis,
    singularity_structure,
    wronskian_analysis,
    formal_asymptotic_solutions,
    levelt_structure,
    formal_ode_data,
)
```

## Expert public API

Specialized algorithms and their result objects remain public from the module that defines the mathematical concept. Examples include:

```python
from odeanalysis.newton import DifferentialNewtonEdge
from odeanalysis.irregular import irregular_singularity_invariants
from odeanalysis.levelt import levelt_reduce_regular_singular
from odeanalysis.system import companion_system, formal_block_partition
from odeanalysis.interchange import green_operator_data
from odeanalysis.bell import complete_exponential_bell_polynomial
```

These APIs are supported, but they are not duplicated at the package root. This keeps `dir(odeanalysis)` focused on complete user workflows rather than every implementation component.

## Internal implementation

Modules whose names begin with an underscore are implementation details. Verification objects may be returned by public result objects, but callers should normally invoke the result's `verify()` method rather than construct internal records directly.

The executable classification is stored in `odeanalysis._api_policy`; tests require the root namespace to match that policy exactly.

### Classical recognition and solvability

Primary imports include `CanonicalEquationFamily`, `CanonicalEquationRecognition`, `recognize_canonical_equation`, `transform_to_canonical`, `FirstOrderFactorization`, `factor_differential_operator`, `is_reducible_operator`, `KovacicAnalysis`, `KovacicOutcome`, and `kovacic_analysis`. These are workflow-level APIs; lower-level recognition and Riccati helpers remain implementation details.

### Turning points and uniform WKB

Primary imports include `LiouvilleNormalForm`, `TurningPoint`, `TurningPointAnalysis`, `TurningPointKind`, `WKBExpansion`, `UniformWKBReduction`, `liouville_normal_form`, `analyze_turning_points`, `turning_points`, `classify_turning_point`, `wkb_expansion`, `airy_uniformization`, and `weber_uniformization`. These objects form one second-order Liouville-Green workflow; the branch-sensitive phase coordinate and retained residual are represented directly rather than hidden behind internal helpers.

### Analytic continuation

Primary imports include `CanonicalBasis`, `ConnectionMatrix`, `StokesMatrix`, `LocalMonodromy`, `hypergeometric_connection_matrix`, `kummer_connection_matrices`, `connection_matrix`, `stokes_matrices`, and `local_monodromy`. Connection matrices use `F_a = F_b C_{b<-a}`; coefficient columns transport as `v_b = C_{b<-a} v_a`; actual analytic monodromy is separate from `formal_monodromy()`. Exact analytic constants are exposed only for supported canonical normalizations and are not inferred from purely formal local data.

### First-class system and validated-numerics APIs

Stable public system APIs include `analyze_system_singularity`, `formal_system_analysis`, `system_stokes_geometry`, `system_parameter_analysis`, `system_formal_type_stratification`, `scalarize_system`, `scalar_system_correspondence`, and `parameterized_turning_analysis`. `certified_system_continuation` is public with a deliberately narrow validated contract. Low-level Moser, spectral-splitting, matrix-Laurent, and block-reduction helpers remain expert APIs.

## Result evidence classes

Public result objects should make their evidence level visible. Exact transformation
objects expose `verify()` where practical; bounded formal objects expose completion and
limitations; parameter results distinguish semialgebraic exhaustiveness from sampled
formal data; certified numerical results expose an enclosure and method. See
[certification-model.md](certification-model.md).
