# Levelt--Turrittin workflow

The formal reduction pipeline separates discovery of irregular exponential scales from regular-singular normalization. For a scalar equation the conceptual path is:

```text
scalar ODE
  -> LinearDifferentialOperator
  -> companion system
  -> local MatrixLaurentSeries
  -> remove common scalar irregular terms
  -> split distinct generalized eigenspaces
  -> Moser shear unresolved single-eigenvalue blocks
  -> ramify when integral shearing is insufficient
  -> regular-singular Levelt reduction
  -> exponential blocks, exponents, logarithms, monodromy
```

## Spectral splitting

At the first nonscalar irregular Laurent coefficient, distinct exact eigenvalues determine generalized eigenspaces. `formal_block_diagonalize()` records exact projectors and solves off-block Sylvester equations through the requested truncation order. Each retained spectral split can be verified independently.

## Moser reduction

A nonscalar irregular coefficient with a single eigenvalue cannot yet be split spectrally. `moser_reduce()` moves its nilpotent part to Jordan form and searches bounded integer shears. A candidate is accepted only when the exact Moser progress measure improves. Spectral data is computed once for each candidate classification and is not retained in a process-wide cache.

## Ramification

When an integral shear cannot resolve the obstruction, `levelt_turrittin_reduce()` may introduce a bounded cover. For example:

```python
import sympy as sp
from odeanalysis import MatrixLaurentSeries, levelt_turrittin_reduce

t = sp.symbols("t")
connection = MatrixLaurentSeries.from_matrix(sp.Matrix([[0, t**-2], [t**-1, 0]]), t)
result = levelt_turrittin_reduce(connection, max_power=3, max_depth=1)

assert result.complete
assert result.ramification_index == 2
assert result.final_diagonalization.block_dimensions == (1, 1)
assert result.verify()
```

The ramification record stores its source and transformed connection, so verification checks the actual pullback rather than trusting the search history.

## Regular-singular normalization

Once a block is Fuchsian, the regular-singular reducer removes nonresonant positive-degree terms by solving exact homological equations. Integer exponent shifts normalize exponent classes, while resonant terms that cannot be removed contribute to the nilpotent logarithmic part. The resulting structural block has the formal shape `H(t) exp(Q(t^-1)) t**Lambda exp(N log(t))`.

## What this does not prove

The reduction certifies algebraic/formal transformations through the retained truncation. It does not by itself prove sectorial summability, determine numerical Stokes constants, or construct global connection matrices.
