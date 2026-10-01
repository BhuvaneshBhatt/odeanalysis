# Mathematical conventions

These conventions are public mathematical contracts and are regression-tested.

- **Scalar operator:** coefficients are ordered from `y` through the highest derivative.
- **System:** `Y' = A(x)Y+b(x)`.
- **Gauge:** for `Y=GZ`, `Z'=(G^{-1}AG-G^{-1}G')Z+G^{-1}b`.
- **Variable change:** for `x=phi(t)`, the connection is multiplied by `phi'(t)`.
- **Infinity:** local coordinate `t=1/x`; this introduces the exact reciprocal Jacobian and sign.
- **Residue exponents:** regular-singular system exponents are eigenvalues of the local residue in the chosen lattice.
- **Resonance:** a nonzero integral residue-eigenvalue difference; repeated equal exponents alone are not recorded as resonance.
- **Raw Poincare rank:** connection pole order minus one and presentation-dependent.
- **Ramification:** `x=t^r`; ramification indices multiply under successive covers.
- **Formal exponential block:** obtained from verified scalar irregular coefficients of a completed formal block.
- **Stokes/equal-magnitude ray:** for leading difference `c t^{-k}`, `Re(c t^{-k})=0`.
- **Phase-alignment ray:** `Im(c t^{-k})=0` in the corresponding structural convention.
- **Connection matrices:** basis conventions are those stated in the analytic-continuation guide; transformations are not silently transposed.
- **Certified numerical transport:** fundamental transport maps data at the start point to data at the end point.

A standard derivative companion system is exact but need not preserve the scalar
operator's raw local pole presentation. Comparisons must use gauge-invariant or
explicitly transformed quantities.
