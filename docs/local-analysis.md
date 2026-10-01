# Parameterized local analysis

`odeanalysis` uses `funcprops` to normalize symbolic assumptions and `semialg` 1.3 for certified real parameter-space geometry. ODE code discovers the transition polynomials that matter to local analysis; feasibility, decomposition, implication, equivalence, and coverage certification are delegated to `semialg`.

```python
from odeanalysis import local_parameter_analysis

result = local_parameter_analysis(ode, y, x, point=0, assumptions=True)
```

The result partitions parameter space only across ODE-derived transition loci. `semialg.parametric_cad()` supplies certified parameter cells; cells with identical ODE behavior are coalesced before local/Frobenius analysis. Each `LocalAnalysisStratum` contains the resulting semialgebraic condition, singularity classification, and (for regular singular points) Frobenius data. `select()` uses semialgebraic implication, so callers do not depend on a particular CAD cell syntax.

## Frobenius semantics

`frobenius_analysis(..., assumptions=...)` derives the arbitrary-order recurrence. Integer-separated or repeated indicial roots indicate resonance. `logarithm_may_be_required` records that structural possibility, while `logarithm_required` is stronger: it is true only when the pure-power recurrence has a certified obstruction. Thus resonance alone is never treated as proof of a logarithmic term.

## Infinity

`local_parameter_analysis(..., point=oo)` performs the exact reciprocal transformation and analyzes the transformed problem at zero. The returned singularity remains labelled by `oo`, and its transformed equation is retained.

## Support boundary

Automatic local strata cover the requested real semialgebraic parameter domain when the backend reports complete certified coverage. Transition loci include coefficient-valuation changes and repeated roots of quadratic indicial polynomials. A finite requested prefix of integer-difference resonance loci uses the same semialgebraic feasibility machinery; unbounded Diophantine resonance families and transcendental parameter partitions remain outside this contract.

## Unified local evidence

`frobenius_convergence()` attaches the nearest coefficient singularity, the
coefficient-analyticity radius, and the corresponding open disk in the local
coordinate to a completed Frobenius analysis. Symbolic locations are retained
as exact distances (for example `Abs(a)` or `Min(Abs(a), Abs(b))`) and an
incomplete singularity list is never promoted to a certified infinite radius.
This coefficient radius is distinct from the stronger maximal
solution-continuation radius, which can be larger across apparent singularities
and is left uncertified unless independent continuation evidence is available.  `frobenius_local_monodromy()`
derives diagonal monodromy directly for pure-power bases and reuses the
completed logarithmic Frobenius basis when a certified logarithmic obstruction requires it.

`wronskian_analysis()` exposes Abel's identity and can certify linear
independence of an explicit basis from its Wronskian.  The convenience
`abel_wronskian()` returns the general Abel factor.

`singularity_structure()` is the aggregate workflow. It collects finite and
infinite singularity classifications and, where available, Frobenius data,
convergence geometry, apparent-singularity evidence, and local monodromy in a
single immutable `SingularityStructure`. Regular-singular infinity is analyzed
in the shared reciprocal local coordinate, so it carries the same evidence as
a finite regular singular point. Apparent-singularity decisions reuse completed
Frobenius and certified monodromy evidence rather than rebuilding an independent
local decision path.

For quadratic indicial families, `local_parameter_analysis()` also records
a finite prefix of integer exponent-separation conditions in
`resonance_strata`.  The complement remains explicit rather than pretending a
finite list exhausts all possible integer resonances.

## Structural transition loci

`newton_loci()` reports parameter degeneracies that can change the normalized effective support of the local differential Newton polygon. This contract is invariant under algebraic/scalar normalization that preserves effective differential support; it does **not** claim invariance under arbitrary dependent-variable gauge transformations such as `y = exp(g(x)) u`. `singularity_loci()` detects degree-loss and discriminant loci for finite singularities; `singularity_projective_strata()` interprets nested degree losses with `effective_degree` and cumulative `infinity_multiplicity`. `leading_rank_analysis()` delegates leading Laurent-matrix rank stratification to `semialg.matrix_rank_stratification()`. For downstream asymptotic geometry, `turning_loci()` detects turning-point collisions and `stokes_formal_loci()` detects degeneracies of pairwise leading exponential differences. These functions identify ODE-specific algebraic causes; they do not implement a second region/decomposition framework.


Certified Stokes-ray transition geometry requires a real semialgebraic parameter space. Complex parameters must be represented explicitly by real and imaginary coordinate symbols; unconstrained complex SymPy symbols are rejected rather than silently treated as real. The package convention calls `Re(Delta Q)=0` equal-magnitude/Stokes rays and `Im(Delta Q)=0` phase-alignment/anti-Stokes rays.
