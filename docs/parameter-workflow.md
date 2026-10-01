# Parameter workflow

Parameter analysis separates transition geometry from representative analysis.

`semialg` supplies certified real semialgebraic cells. `odeanalysis` supplies the
ODE/system transition polynomials and evaluates local/formal invariants on certified
representatives.

For systems, `SystemFormalTypeSignature` is deliberately discrete. It records
singularity kind, leading rank, spectral multiplicity pattern, resonance orders up to
the configured bound, ramification, block dimensions, exponential parts, and
structural Stokes ray counts. It does **not** claim continuously varying eigenvalues or
exponents are numerically constant on an open cell.

For turning points, parameter analysis certifies multiplicity/family transitions. This
is not the same as a full uniform asymptotic theory for a moving pair of coalescing
turning points.

For irregular system families, representative formal reductions are not automatically cell-wide certificates: the current transition vocabulary does not claim completeness for every possible ramification or formal-block transition.
