# Worked failures and correct refusals

## Formal reduction did not complete

Inspect `FormalReductionCertificate.limitation`, attempted depths/covers, and
`verified`. Increasing a budget may reveal more structure, but an incomplete result is
not a proof that no further splitting or ramification exists.

## Scalarization returned no operator

The selected output row was not certifiably cyclic, the system was inhomogeneous, or
cyclicity depended on unresolved parameters. Choose another output or prove cyclicity;
do not infer that no scalar equation exists.

## Stokes geometry has no analytic matrix

`system_stokes_geometry()` is structural. Formal exponential differences determine
rays and support geometry, not generic analytically normalized Stokes multipliers.

## A parameter cell has a representative exponent value

That value is a sample. Open cells may contain continuously varying exponents. The
formal-type signature records discrete spectral multiplicities and bounded resonance
patterns instead.

## Certified continuation refused `A(x)`

The current Arb backend certifies constant homogeneous transport only. It refuses
variable coefficients rather than replacing enclosure proofs with an ODE-solver
tolerance.
