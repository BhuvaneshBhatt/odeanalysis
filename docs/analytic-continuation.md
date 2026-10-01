# Analytic continuation and monodromy

`odeanalysis` distinguishes formal local data from analytically normalized continuation data.  The analytic-continuation layer supplies exact data for canonical families whose normalization is fixed explicitly.

## Matrix convention

For canonical fundamental matrices, every `ConnectionMatrix(source=a, target=b)` is

\[
F_a=F_b C_{b\leftarrow a}.
\]

Thus a coefficient column transports in the same named direction,
\(v_b=C_{b\leftarrow a}v_a\). Reversal gives
\(C_{a\leftarrow b}=C_{b\leftarrow a}^{-1}\), while composition is
\(C_{c\leftarrow a}=C_{c\leftarrow b}C_{b\leftarrow a}\). This convention makes reversal, coefficient transport, and composition unambiguous.  Primitive Gauss hypergeometric matrices are stored in factored Gamma-function form, and derived directions are obtained by explicit 2-by-2 inversion and matrix composition rather than heuristic global simplification.

## Gauss hypergeometric connection matrices

`hypergeometric_connection_matrix(a, b, c, source, target)` gives exact nonresonant connection matrices among the standard canonical bases at 0, 1, and infinity.  Resonant parameter strata, where standard bases coalesce or logarithmic limiting formulas are required, are outside the current exact formula contract.

`connection_matrix()` dispatches the same formulas by canonical family name.

## Kummer lateral connections and Stokes data

`kummer_connection_matrices(a, c)` gives the two exact lateral connection matrices at infinity in the package's fixed sectorial normalization.  `stokes_matrices("kummer", a=a, c=c)` returns the corresponding exact unitriangular Stokes factors without expanding Gamma reflection identities.

## Airy Stokes matrices

In the exact-WKB normalization used by the turning-point layer, Airy has three alternating unitriangular Stokes factors with multiplier `I`.  Their ordered product with the square-root formal branch permutation yields the actual one-turn monodromy at infinity.

## Actual versus formal monodromy

`formal_monodromy()` remains a formal-local construction.  `local_monodromy()` represents actual analytic continuation in a supported normalized basis. Positive local circuits are counterclockwise in the local coordinate. At infinity the package uses `t=1/x`, so a positive local circuit is clockwise in the global `x`-plane; this orientation is part of the public convention.  For the Gauss equation at 0, 1, and infinity it returns the regular-singular Frobenius monodromy.  For Airy at infinity it composes the formal branch transformation with all Stokes jumps.

The package does not infer generic Stokes constants or global connection matrices from formal local data alone.  Bessel and modified-Bessel exact Stokes normalizations, resonant hypergeometric limiting formulas, and generic numerical connection problems remain explicit limitations.

## Performance contract

Analytic-continuation certificates avoid `sympy.simplify()` on whole Gamma-function matrices.  Verification keeps coefficients factored and replays exact 2-by-2 algebra structurally.  This is both more deterministic and avoids contaminating a long-lived SymPy process with expensive simplification state.

## Branch, basepoint, and global-loop conventions

Canonical connection formulas include the branch choices of their named bases; branch-sensitive phases are preserved. A monodromy matrix transported from a target basis `b` back to a source basis `a` is `C_{b<-a}^{-1} M_b C_{b<-a}`. Global sphere relations require all local matrices to be transported to one basepoint with paths whose loop ordering is stated explicitly. The package does not treat an unordered product of local-basis matrices as a global invariant.

## Validated numerical continuation

`certified_system_continuation()` is distinct from heuristic numerical continuation. The initial validated backend handles constant homogeneous first-order systems and encloses `exp((b-a)A)` with Arb complex balls through optional `python-flint`. Variable-coefficient validated integration is explicitly reported as unsupported; ordinary solver tolerances are not presented as certification.
