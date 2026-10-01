# Example gallery

Each example starts from a mathematical question. Executable system examples live in
`examples/gallery/` and are run by the documentation contract tests.

## 1. Resonant Fuchsian system

**Question:** what does the residue say about exponents and resonance?

Use `analyze_system_singularity()` on `diag(0,2)/x`. The residue spectrum is `(0,2)`
and the nonzero integer difference is recorded as resonance.
Executable: `examples/gallery/system_fuchsian.py`.

## 2. Diagonal irregular system and Stokes rays

**Question:** which exponential blocks and structural rays occur?

For `diag(x^-2,-x^-2)`, verified formal reduction produces the two exponential parts;
`system_stokes_geometry()` then computes pairwise structural rays.
Executable: `examples/gallery/system_irregular_stokes.py`.

## 3. Ramified system bookkeeping

**Question:** how does an exact cover affect the system and formal certificate?

Apply `FirstOrderSystem.ramify(t,2)` and verify the accumulated cover index before
formal reduction. Executable: `examples/gallery/system_ramified.py`.

## 4. Scalar → companion → scalar

**Question:** does the cyclic scalarizer recover the original operator?

`scalar_system_correspondence()` constructs and verifies the exact round trip.
Executable: `examples/gallery/scalar_companion_roundtrip.py`.

## 5. Parameterized residue family

**Question:** where do rank, collision, and bounded resonance patterns change?

Analyze `diag(a/x,-a/x)` with `system_formal_type_stratification()`. The cell signature
is discrete; sampled exponent values are not asserted constant.
Executable: `examples/gallery/system_parameters.py`.

## 6. Certified constant-system transport

**Question:** can a numerical fundamental matrix be enclosed rigorously?

For a constant nilpotent matrix, Arb encloses the exact matrix exponential.
Executable: `examples/gallery/certified_transport.py`.

## 7. Airy turning point

**Question:** how does a simple zero of the Liouville potential select the Airy model?

Use `liouville_normal_form()`, turning-point analysis, and `airy_uniformization()`.
The transformed residual remains available for verification.

## 8. Double turning point

**Question:** when is the local Weber model appropriate?

A double zero is distinguished from a moving coalescing pair. The local Weber
uniformization does not by itself certify a global confluence asymptotic regime.

## 9. Euler--Cauchy recognition

**Question:** can a scale-invariant equation be recognized and its local monodromy
written exactly?

`recognize_canonical_equation()` verifies the Euler transformation; `local_monodromy()`
uses the exact power exponents.

## 10. Hypergeometric continuation

**Question:** when are connection matrices exact rather than numerical?

For supported nonresonant Gauss normalizations, use the canonical continuation API and
its fixed basis convention. See `analytic-continuation.md`.

## 11. Incomplete formal reduction

**Question:** what should a caller do when a bounded reducer stops?

Inspect `FormalReductionCertificate`; do not interpret `complete=False` as absence of
further ramification or splitting. See `worked-failures.md`.

## 12. Correct refusal of uncertified continuation

**Question:** what happens for `Y'=A(x)Y` with variable `A`?

The certified backend returns an explicit limitation rather than a tolerance-based
matrix. Executable: `examples/gallery/failure_variable_certification.py`.

The noncyclic scalarization refusal is also executable in
`examples/gallery/failure_scalarization.py`.
