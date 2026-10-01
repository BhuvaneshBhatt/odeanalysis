# Certification model

The package uses evidence words narrowly.

## Exact

A symbolic identity or transformation is replayed exactly with SymPy algebra.

## Certified formal

A formal construction is mathematically verified through a stated truncation,
ramification, recursion, or search bound. `complete=False` means the algorithm did not
prove completion; it does not prove missing structure is absent.

## Certified semialgebraic

A real parameter condition or cell decomposition is certified by `semialg`. A sampled
formal result attached to a cell is a separate layer of evidence. A discrete signature
may be constant even when symbolic eigenvalue values vary continuously.

## Certified numerical

A numerical result is represented by a rigorous enclosure. The initial continuation
backend uses Arb complex-ball arithmetic for constant homogeneous systems. Solver
tolerances alone are never labeled certified.

## Structural

A structural result describes support, rays, sectors, possible couplings, or other
formal geometry without claiming analytically normalized constants. Structural Stokes
geometry therefore does not imply an analytic Stokes matrix.

## Incomplete / unresolved

An explicit limitation or unresolved result is part of the public contract. Callers
must not convert it into a negative theorem.
