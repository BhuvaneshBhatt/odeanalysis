# Verification

Search and verification are separate concerns. A reduction algorithm may search among spectral splittings, shears, or covers; the returned object then retains concrete evidence that can be checked without repeating that search.

## What is retained

Formal gauge verification stores the source Laurent connection, the gauge, the transformed connection, and the truncation bound. Spectral verification stores the coefficient matrix, eigenvalues, generalized-eigenspace dimensions, and projectors. Ramification records retain both sides of the pullback. Composite Levelt--Turrittin results retain the sequence of formal stages and ramification steps.

## What `verify()` checks

Verification recomputes the claimed transformation or algebraic identities. Spectral verification checks projector idempotence, invariance, generalized-eigenspace annihilation, pairwise orthogonality, dimensions, and that the projectors sum to the identity. Gauge verification recomputes the formal gauge transform. Composite verification checks that the retained stages chain correctly and that each subordinate verification succeeds.

Tests corrupt individual projectors, dimensions, eigenvalues, source connections, target connections, ramification indices, and stage metadata. These mutations must be rejected.

## Formal versus analytic certification

A successful `verify()` certifies the recorded formal algebra. It is not a proof of analytic summability or of a global connection problem. In particular, the package does not infer numerical Stokes constants from formal-local data.

## System certificates

`FormalReductionCertificate` records the exact reduction search budget and the independent `LeveltTurrittinReduction.verify()` result. `CyclicScalarization.verify()` reconstructs the row-jet relation and scalar coefficients. Formal-type parameter strata are marked certified only when the `semialg` decomposition is exhaustive and the specialized formal reduction verifies and completes. `CertifiedMatrixEnclosure` denotes Arb complex-ball output; failure to obtain a validated enclosure is represented as an incomplete result, not a floating-point substitute.

## Certification vocabulary

The normative vocabulary is defined in [certification-model.md](certification-model.md).
In particular, a semialgebraic cell certificate and a formal-reduction certificate are
separate evidence layers. `SystemFormalTypeSignature` contains discrete bounded data;
it does not turn continuously varying exponent values into constants.

Certified numerical results must be enclosures. The constant-system continuation tests
check exact-reference containment, reversal, and path concatenation with Arb balls.
