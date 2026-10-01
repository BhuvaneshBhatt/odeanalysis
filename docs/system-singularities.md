# System singularities

`analyze_system_singularity()` is the inexpensive first layer for `Y'=A(x)Y+b(x)`. It localizes exactly at a finite point or at infinity (`t=1/x`), computes the connection pole order and leading matrix, and reports ordinary, regular-singular, or irregular type. At a regular singular point the leading matrix is the residue, its eigenvalues are the local exponents, and nonzero integral eigenvalue differences are reported as resonances.

This layer intentionally does **not** run Levelt--Turrittin reduction. The raw Poincare rank is the pole order minus one; formal gauges may lower it. Constant gauges preserve the residue spectrum. Meromorphic gauges can shift exponents by integers, so exponent classes rather than raw residue eigenvalues are the invariant object.

At infinity the package first performs the exact reciprocal coordinate change, including the derivative factor. Thus residue signs and monodromy orientation follow the local coordinate `t=1/x`, not an ad-hoc global convention.
