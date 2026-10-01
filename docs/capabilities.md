# Capabilities and limitations

| Problem | Support | Status of evidence |
| --- | --- | --- |
| Ordinary / regular / irregular point classification | Yes | Exact symbolic classification when local orders are decidable |
| Frobenius indicial data and recurrences | Yes | Exact structural data |
| Resonance and logarithmic Frobenius bases | Yes | Exact formal construction in supported cases |
| Differential Newton polygon and slope filtration | Yes | Exact for finite rational local valuations |
| Katz (Poincare--Katz) and companion Poincare ranks | Yes | Katz is invariant; companion Poincare rank is presentation dependent |
| Riemann schemes and Fuchs relation | Yes | Exact when all singularities and indicial roots resolve |
| Apparent-singularity detection | Yes | Three-valued; requires a complete logarithmic Frobenius basis |
| Dominant/recessive formal branches | Yes | Exact on Stokes sectors when angular/sign ordering resolves |
| Leading and completed scalar exponential parts | Yes | Formal algebra with exact branch data when characteristic roots resolve |
| Formal amplitude series | Yes | Truncated formal construction |
| Liouville normal form for second-order equations | Yes | Exact gauge and effective potential |
| Finite turning-point detection and multiplicity | Yes | Exact when polynomial numerator roots resolve |
| Ordinary Riccati/WKB expansion | Yes | Formal recurrence to requested order |
| Simple turning-point Airy uniformization | Yes | Exact coordinate identity with retained residual |
| Isolated double-turning-point Weber uniformization | Yes | Degenerate Weber model with retained residual |
| Parameterized turning multiplicity/family strata | Yes | `semialg`-certified transition cells; local Airy/degenerate-Weber family labels |
| Uniform coalescing-pair Weber asymptotics | No | Requires a nonzero moving Weber parameter and uniform error control |
| First-order system singularity classification | Yes | Exact for finite rational local valuations |
| System residue exponents and resonance | Yes | Exact when residue spectrum resolves |
| Public formal system analysis | Bounded | Wraps verifiable Moser/Levelt--Turrittin reduction |
| System structural Stokes geometry | Yes | Formal rays from supplied/certified exponential blocks; no generic analytic constants |
| Parameterized system rank/collision geometry | Yes | `semialg`-certified real strata |
| Parameterized 2x2 system resonance | Bounded | Exact integer-difference loci through configured order |
| Parameterized system Stokes transitions | Structural | Exact algebraic loci with explicit real coordinates |
| Exact generalized-eigenspace splitting | Yes | Independently verifiable |
| Moser integer shearing | Bounded | Independently verifiable retained gauge |
| Ramified Levelt--Turrittin reduction | Bounded | Composite independently verifiable formal reduction |
| Regular-singular Levelt normalization | Yes | Exact through configured truncation |
| Formal monodromy | Yes | Formal-local |
| Stokes ray/sector geometry | Yes | Formal geometry from exponential differences |
| Stokes support patterns | Yes | Structural support only |
| Exact canonical Stokes matrices | Partial | Airy and Kummer in fixed canonical normalizations; Gauss is Fuchsian and has none |
| Exact canonical connection matrices | Partial | Nonresonant Gauss 0/1/infinity bases and lateral Kummer infinity bases |
| Actual local monodromy | Partial | Gauss regular singularities and Airy infinity in supported canonical normalizations |
| Generic numerical Stokes constants | No | Requires analytic/global normalization beyond the supported canonical families |
| Generic global connection matrices | No | Requires analytic continuation/global data beyond canonical formulas |
| General nonlinear ODE transseries | No | Belongs in downstream asymptotic analysis |

## Unsupported and unresolved cases

The package leaves a result unresolved when a coefficient has no supported finite rational local valuation, an exact characteristic polynomial cannot be resolved, a repeated irregular obstruction survives the configured Moser/ramification bounds, or a symbolic integer shift cannot be represented by the integral-power Laurent model.

`complete` and `limitation` fields are part of the public result contracts. A partial formal result must not be interpreted as a proof that the omitted structure is absent.

## Classical forms and differential algebra

| Capability | Status | Certification |
| --- | --- | --- |
| Affine Airy recognition | Supported | Exact pullback verification |
| Affine Bessel / modified-Bessel recognition | Supported | Exact pullback verification |
| Affine Gauss / confluent hypergeometric recognition | Supported | Exact pullback verification |
| Rational first-order factorization of second-order scalar operators | Supported | Exact noncommutative product verification |
| Kovacic Case 1 | Supported | Exact rational Riccati certificate |
| Kovacic Case 2 | Supported | Auxiliary-polynomial and quadratic Riccati replay |
| Kovacic Case 3 | Supported | Finite-group recurrence and algebraic-log-derivative polynomial replay |
| Negative Liouvillian decision | Supported | Cases 1--3 exhausted exactly |
| Möbius/gauge recognition of three-regular-singularity equations | Supported | Exact projective pullback and logarithmic-gauge verification |

## Turning-point boundary

Turning-point discovery covers finite zeros of the Liouville normal-form potential. Ordinary WKB is formal away from zeros and poles of that potential. Airy uniformization supports simple zeros and Weber uniformization supports isolated double zeros. `parameterized_turning_analysis()` certifies real parameter cells on which finite turning multiplicities and local family labels are constant. A full uniform asymptotic treatment of a moving coalescing pair with nonzero Weber parameter, higher multiplicities, and turning points at infinity remains outside this contract.

## First-class system extensions

System analysis now distinguishes inexpensive connection singularity analysis from bounded/adaptive formal reduction. Adaptive reduction carries an explicit completeness certificate; parameterized formal-type strata record rank, spectral multiplicity/bounded-resonance data, ramification, formal block dimensions/exponential parts, and structural Stokes signatures. Cell-wide certification is initially restricted to ordinary/regular-singular 2x2 bounded signatures; irregular cells retain verified representative reductions without overclaiming transition completeness. Sampled exponent values are not asserted constant across open cells. Cyclic-vector scalarization is available when cyclicity is certified. Certified numerical continuation is presently limited to constant homogeneous systems through Arb complex-ball matrix exponentials; variable-coefficient validated integration and generic analytic Stokes matrices are not claimed.
