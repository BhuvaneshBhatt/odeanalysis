# Dominant and recessive formal solutions

`classify_solution_dominance()` compares completed formal exponential polynomials on the same ramified cover used by Stokes geometry. On each open equal-magnitude sector it returns dominance levels ordered from exponentially largest to exponentially smallest.

```python
import sympy as sp
from odeanalysis import classify_solution_dominance

x = sp.symbols("x", positive=True)
y = sp.Function("y")
airy = sp.diff(y(x), x, 2) - x * y(x)

analysis = classify_solution_dominance(airy, y, x, point=sp.oo)
assert analysis.complete
assert analysis.common_ramification == 2
for sector in analysis.sectors:
    dominant = sector.dominant_branches
    recessive = sector.recessive_branches
```

Branch indices refer to completed exponential branches. The comparison is at the exponential scale: branches with identical completed exponential polynomials remain tied even when algebraic or logarithmic prefactors differ. If symbolic parameters prevent exact ordering of Stokes boundaries or asymptotic signs, `complete` is false and no sector order is guessed.
