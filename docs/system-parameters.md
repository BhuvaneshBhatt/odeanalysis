# Parameterized linear systems

`system_parameter_analysis()` discovers exact algebraic transition loci for leading rank, residue eigenvalue collisions, bounded integer resonance, formal block collision, and supplied exponential/Stokes degeneracy conditions, then delegates real parameter decomposition to `semialg`.

`system_formal_type_stratification()` adds an exact representative formal signature to every CAD parameter cell. The signature records singularity kind, leading rank, spectral multiplicity pattern, resonance orders through the configured bound, ramification, block dimensions, exponential parts, and structural Stokes-ray counts. Continuously varying residue eigenvalues/exponents remain representative data and are not claimed constant on an open cell. A stratum is marked certified only when the parameter decomposition is exhaustive, the representative formal reduction independently verifies and completes, and the current transition vocabulary is sufficient for that family; this initial cell-wide certification is restricted to ordinary/regular-singular 2x2 bounded signatures.

The certification is intentionally bounded by the discovered transition vocabulary and formal-reduction budgets. Unsupported symbolic spectral conditions remain incomplete rather than being guessed.

For scalar turning points, `parameterized_turning_analysis()` decomposes parameter space by turning-point collision and projective degree-loss loci, then exactly specializes each cell to record finite turning multiplicities, kinds, and the locally valid `airy`/`weber` uniform family where the multiplicity contract determines one. This is the parameter geometry used to decide where Airy versus degenerate Weber local uniformization is valid.


### Certification boundary

The initial transition vocabulary is sufficient to certify the bounded discrete signature for ordinary/regular-singular 2x2 families. Irregular parameter cells are annotated with verified representative reductions but are not promoted to cell-wide formal-type certificates merely because the underlying CAD decomposition is complete.
