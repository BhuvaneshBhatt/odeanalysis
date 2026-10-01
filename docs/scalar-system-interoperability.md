# Scalar/system interoperability

`companion_system()` converts a scalar operator to its exact derivative-jet system.
`scalarize_system()` constructs row jets `r_(k+1)=r_k'+r_k A`; when the first `n` rows
form a certifiably invertible cyclic matrix, the next row gives a monic scalar equation.
`scalar_system_correspondence()` verifies the standard round trip.

Cyclicity failure is explicit. Parameter-dependent cyclic determinants are not assumed
nonzero without proof.

Do not compare raw companion pole order with scalar regular-singular classification as
if it were invariant. Constant similarity transformations preserve the scalar equation
when the output row is transformed consistently; meromorphic transformations require
the corresponding lattice/formal interpretation.
