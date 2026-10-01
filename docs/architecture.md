# Architecture

## Layer 1: scalar differential operators

`LinearDifferentialOperator` is the canonical representation. It separates coefficient extraction and coordinate transformations from downstream local algorithms. SymPy's `ode_order()` is reused for order detection; polynomial operators can be exported to `sympy.holonomic` without making that representation canonical.

## Layer 2: local singularity and Frobenius structure

`singularities.py` classifies ordinary, regular-singular, irregular-singular, and unknown points from normalized coefficient pole orders. `frobenius.py` implements arbitrary-order regular-singular indicial and recurrence analysis.

## Layer 3: differential Newton polygon

`newton.py` localizes finite points by `x=x0+h` and infinity by `x=1/h`. For every nonzero term `a_j(h) D_h^j` it computes the finite rational local valuation `v(a_j)` and stores the differential-Newton point

`(j, v(a_j)-j)`.

The lower convex hull is represented explicitly by `DifferentialNewtonPoint`, `DifferentialNewtonEdge`, and `DifferentialNewtonPolygon`. Collinear coefficient points remain attached to an edge so they all contribute to its characteristic polynomial. Scalar multiplication of the complete operator only translates heights and therefore leaves slopes invariant.

Positive edge slopes are the irregularity slopes. `SlopeFiltration` groups equal slopes with horizontal multiplicity; nonpositive edges contribute to the regular slope-zero piece. The maximum is the rational Katz (Poincare--Katz) rank, the horizontal-length-weighted sum is the formal irregularity, and the LCM of positive-slope denominators is the Newton ramification index. The separately reported Poincare rank belongs to the natural Euler-scaled companion presentation and is not a formal-gauge invariant.

## Layer 4: irregular formal structure

`irregular.py` converts each irregular edge into an edge characteristic polynomial. For slope `rho > 0` and nonzero root `c`, the leading logarithmic derivative and integrated exponent are

`c*h**(-(rho+1))`

and

`Q_0 = -c*h**(-rho)/rho`.

`FormalExponentialPart` records this leading Newton data. `WKBAnsatz` retains the exact leading exponential conjugation for callers that need that intermediate representation.

## Layer 5: Bell combinatorics and sparse local series

`bell.py` provides the generic complete exponential Bell polynomial. Rather than obtaining it by summing all partial Bell polynomials independently, it enumerates integer-partition multiplicities `c_i` with `sum(i*c_i)=n` and constructs each monomial once with coefficient `n! / product(c_i! (i!)**c_i)`. `complete_exponential_bell_via_sympy()` remains an independent reference against SymPy's `bell(n, k, ...)` API.

`series.py` provides `SparseLaurentSeries`, a finite sparse representation in a uniformizer `t`. With `h=t**r`, its derivative implements `D_h(c*t**p)=(p/r)c*t**(p-r)`. Addition and multiplication can truncate to a requested power window after every operation. This is the computational representation used by the formal Riccati Bell recurrence whenever its logarithmic derivative is a finite ramified Laurent polynomial.

## Layer 6: Riccati/Bell formal refinement

`formal.py` uses the identity

`D**j(y)/y = B_j(w, w', ..., w**(j-1))`, `w = y'/y`,

with the differential Bell recurrence `B_0=1`, `B_{j+1}=D(B_j)+w B_j`. This converts the linear equation into a nonlinear Riccati equation without expanding exponentials.

For a Newton branch the module passes to the uniformizer `h=t**r`, where `r` is the edge ramification denominator. Starting with the edge root, it recursively solves the leading coefficient equation for every ramified power through `h**(-1)`. Terms strictly below `h**(-1)` integrate to the finite exponential polynomial; the `h**(-1)` coefficient becomes the algebraic power `alpha` in `h**alpha`.

`CompleteFormalExponentialPart` stores both local and original-coordinate forms. Degenerate branches are translated into secondary Newton--Puiseux problems; rational correction exponents enlarge the uniformizing cover exactly. `FormalRefinementError` is reserved for unsupported coefficient valuations, non-polynomial translated differential structure, or characteristic roots that cannot be resolved exactly.

After the exponential and power factors are removed, binomial/Bell conjugation gives the exact amplitude operator coefficients without symbolic exponential cancellation. The Bell factors for this conjugation also use the sparse Laurent recurrence when possible, with the symbolic recurrence retained as a fallback. `formal_amplitude_series()` recursively generates the normalized amplitude in the same uniformizer, and `formal_asymptotic_solutions()` combines all three factors.

## Turning-point and Liouville-Green layer

`turning.py` treats second-order equations globally in the original independent variable rather than as singular germs. It first applies the exact Liouville gauge `y=g u`, `g'/g=-p/2`, producing `u''=Q u`. Finite zeros of `Q` are turning points; their numerator-root multiplicities determine the simple/double/higher classification.

`wkb_expansion()` uses the Riccati equation `epsilon*S' + S**2 = Q` and stores both formal branches coefficient by coefficient. At a simple turning point, the Airy coordinate satisfies `zeta*zeta'**2=Q`; at an isolated double point the Weber coordinate satisfies `zeta**2*zeta'**2=Q`. The dependent-variable factor `(zeta')**(-1/2)` removes the first derivative introduced by the coordinate change. The remaining Schwarzian-type residual is kept explicitly, so a uniform leading model is never represented as an exact canonical transformation unless the residual vanishes.

## Layer 7: Stokes geometry and sector dominance

`stokes.py` consumes `CompleteFormalExponentialPart` objects rather than the original Newton leading parts. For each pair it forms the completed difference `Delta Q_ij = Q_i-Q_j`; if leading terms cancel, the first surviving lower exponential term therefore determines the pairwise geometry.

Ramified branches are lifted to a common cover `h=t**R`, with `R` the LCM of their ramification indices. If the most singular surviving term is `c*t**(-m)`, equal-magnitude directions satisfy `Re(c exp(-I*m*theta))=0` and phase-alignment directions satisfy `Im(c exp(-I*m*theta))=0`. Each lifted ray records its cover angle, local `h`-angle, sheet, and original-variable angle; at infinity the latter is negated because `h=1/x`.

The package calls equal-magnitude rays `stokes_rays` and phase-alignment rays `anti_stokes_rays`, but exposes the invariant names as the primary API because the historical terminology is not universal.

The union of lifted equal-magnitude rays partitions the common cover into open `StokesSector` objects. At a representative interior angle, the first nonzero term of every completed pairwise difference determines exponential dominance. `dominance_levels` groups branches with identical completed exponential polynomials and orders the groups from exponentially largest to smallest. Symbolic ray formulas are retained even when unresolved parameters make global sector ordering impossible.

## Poincare-rank naming

The invariant API distinguishes `katz_rank` from `euler_system_poincare_rank`. The latter is the pole excess in the natural Euler-scaled companion presentation and can change under formal gauge transformations. It is useful diagnostic data but is not presented as the invariant rank of the differential module.

## System and differential-module reduction

The scalar `LinearDifferentialOperator` remains the entry point for scalar analysis, while `FirstOrderSystem` and `MatrixLaurentSeries` provide the invariant system representation needed for repeated irregular blocks. Completed scalar exponential data chooses a common cover and guides the companion-system shearing. `formal_block_diagonalize()` then skips common scalar irregular terms, splits exact generalized eigenspaces when a nonscalar coefficient has distinct eigenvalues, and removes off-block terms through exact Sylvester equations.

A nonscalar irregular coefficient with a single eigenvalue is handled by `moser_reduce()`. Its nilpotent part is moved to Jordan form and normalized integer shears are accepted only when the exact Moser progress measure improves. The first nonscalar irregular coefficient is classified once per shear candidate; the same exact eigenspectrum determines both Moser progress and whether ordinary spectral splitting has become available. This avoids duplicate eigenvalue computations without retaining a process-wide symbolic cache. If integral shearing remains insufficient, `levelt_turrittin_reduce()` introduces bounded covers `t=u**r`, pulls the already-reduced connection back exactly, and resumes Moser/spectral reduction. Every `RamificationStep` stores its source and transformed Laurent connection and can be verified independently. Each formal stage also retains explicit spectral-projector and total-gauge verification records, and the composite Levelt-Turrittin verifier checks the entire stage chain.

The implementation separates these tightly coupled concerns into private modules: `_spectral.py` owns exact irregular spectral classification and projector evidence, `_formal_gauge.py` owns truncated gauge transformations and gauge evidence, `_moser.py` owns integer-shear search, and `_block_common.py` owns shared block-series operations. `block_decomposition.py` coordinates recursive reduction and scalar exponential-block reconstruction. The split is organizational only; it does not add symbolic preprocessing to the reduction path.

Once a block is Fuchsian, `levelt_reduce_regular_singular()` solves the homological equations

`L_n(H) = R H - H R - n H`

through the requested truncation order. Nonresonant image terms are removed, integer exponent classes are normalized by diagonal meromorphic gauges, and absorbed resonant terms enter the residue as the nilpotent logarithmic part. The exposed scalar block therefore has the formal structure

`H(t) exp(Q(t^-1)) t**Lambda exp(N log(t))`.

## Stokes geometry and connection support

`stokes.py` derives equal-magnitude and phase-alignment rays from completed pairwise exponential differences on the common ramified cover. Exact validation checks ray projection, full-turn sector partition, positive widths, and dominance partitions. `stokes_connection_patterns()` adds the formal support envelope for a connection factor at each equal-magnitude boundary: only branch pairs active on that ray may have off-diagonal entries and the diagonal is normalized to one.

The formal local problem does not determine numerical Stokes constants or a preferred orientation of every connection factor. Those require sectorial normalization, analytic continuation, or independent global data. The package therefore validates supplied connection structure without fabricating analytic constants.

Downstream Green-operator interchange uses three-valued nondegeneracy evidence for the leading coefficient. Structural nonzero appearance is not treated as a proof: unresolved symbolic parameters remain unresolved until assumptions establish nonvanishing.

## Support boundaries

The bounded ramified reducer is not a complete algorithm for every differential module. It can stop when exact eigenstructure is unavailable, when a repeated irregular obstruction persists beyond the configured cover depth, or when symbolic integer shifts cannot be represented by the integral-power Laurent model. Sectorial existence and summability theorems, analytically normalized Stokes constants, and global connection matrices remain distinct from the formal-local structure implemented here. Generic nonlinear transseries arithmetic remains in the optional `asymptotic` integration rather than in `odeanalysis` core.

## Scalar/system interoperability and validated numerics

Scalar and system implementations remain parallel rather than replacing scalar kernels with companion-system calls. `interoperability.py` is the explicit bridge: companion construction is unconditional, while reverse scalarization requires a certified cyclic output. `system_analysis.py` owns public system singularity/formal/Stokes/parameter results and delegates reduction to the formal matrix machinery and real parameter geometry to `semialg`. `certified_continuation.py` is isolated from heuristic numerical paths so an ordinary solver tolerance cannot accidentally satisfy a certification contract.

## Evidence layers and traceability

Public workflows separate discovery from replayable evidence. Scalar/system
interoperability carries cyclic matrices; formal systems carry reduction certificates;
parameter geometry is delegated to `semialg`; certified numerical continuation returns
ball enclosures. The maintenance mapping from advertised capabilities to executable
corpora is in [traceability.md](traceability.md).
