# Formal asymptotics

This guide describes the scalar formal-asymptotic pipeline behind
completed exponential factors, Riccati/Newton--Puiseux refinement,
amplitudes, and formal WKB solutions.

## From Newton edges to exponential factors

At an irregular edge of positive slope `rho`, a nonzero root `c` of the
edge characteristic polynomial gives the leading logarithmic derivative

``` text
y'/y ~ c h^(-(rho + 1))
```

and hence the leading exponential term

``` text
Q_0(h) = -c h^(-rho) / rho.
```

`formal_exponential_parts()` retains this leading Newton data.
`complete_formal_exponential_parts()` continues the calculation until
the finite exponential polynomial and algebraic power prefactor have
been determined.

## Riccati and Bell-polynomial refinement

Writing `w = y'/y`, the differential Bell polynomials satisfy

``` text
B_0 = 1
B_(j+1) = D(B_j) + w B_j.
```

For a scalar operator `L = sum(a_j D^j)`, this converts the equation
into

``` text
L[y] / y = sum(a_j B_j) = 0.
```

The generic helper `complete_exponential_bell_polynomial(n, xs)`
constructs complete exponential Bell polynomials directly from
integer-partition multiplicities.
`complete_exponential_bell_via_sympy()` supplies an independent
reference construction from SymPy's partial Bell polynomials.

The production ODE path uses the differential recurrence because it
generates the required sequence incrementally. During ramified
refinement and exponential/power conjugation, the recurrence operates on
sparse Laurent series rather than materializing large generic Bell
expressions.

## Newton--Puiseux refinement

Refinement is performed in a Newton uniformizer `h = t^r`. The Riccati
differential polynomial is recursively cancelled through
logarithmic-derivative order `h^-1`. Lower powers integrate into the
finite exponential polynomial; the `h^-1` coefficient determines the
algebraic power prefactor.

If a multiple characteristic root is not resolved by the first Newton
edge, the partial logarithmic derivative is translated out and a
secondary Newton--Puiseux problem is formed in perturbation variables.
Differential monomials contribute affine valuations in the unknown
correction exponent, producing secondary characteristic polynomials.

Nonzero secondary roots add new logarithmic-derivative terms. Zero roots
retain unresolved multiplicity and continue to later edges. When a
correction exponent is rational in the current uniformizer, the cover is
refined automatically so the new power becomes integral.

Each completed branch records its multiplicity, final ramification
index, and `RiccatiRefinementStep` history. `FormalRefinementError` is
reserved for genuinely unresolved valuations or characteristic roots and
unsupported non-polynomial differential structure.

## Sparse Laurent recurrences

`SparseLaurentSeries` represents the ramified local series used by the
formal pipeline. Bell recurrences can be truncated by valuation window
after every operation:

``` python
differential_bell_laurent_series(
    ...,
    min_power=min_power,
    max_power=max_power,
)
```

This keeps the formal recurrence sparse and avoids constructing terms
that cannot affect the requested output.

## Formal amplitudes and WKB solutions

After the exponential polynomial and algebraic power have been
completed, `formal_amplitude_series()` conjugates by those factors and
solves for a normalized formal amplitude in the ramified coordinate.

``` python
import sympy as sp
from odeanalysis import complete_formal_exponential_parts, formal_asymptotic_solutions
from odeanalysis.formal import formal_amplitude_series

x = sp.symbols("x", positive=True)
y = sp.Function("y")
airy = sp.diff(y(x), x, 2) - x * y(x)

completed = complete_formal_exponential_parts(airy, y, x, point=sp.oo)
amplitudes = formal_amplitude_series(airy, y, x, point=sp.oo, terms=7)
solutions = formal_asymptotic_solutions(airy, y, x, point=sp.oo, terms=7)
```

For Airy, the two completed exponential polynomials are `-2*x**(3/2)/3`
and `2*x**(3/2)/3`, with algebraic prefactor `x**(-1/4)`.

`formal_asymptotic_solutions()` combines the completed exponential,
algebraic power, and normalized amplitude. The lower-level
`wkb_ansatze()` API remains available when the exact operator conjugated
only by the leading Newton exponential is the desired object.

## Stokes geometry

Completed exponential polynomials feed directly into
`stokes_geometry()`. For each pair of branches, the completed difference

``` text
Delta Q_ij = Q_i - Q_j
```

is formed before its leading surviving term is selected. This is
important when the highest terms of two individual exponentials cancel.

The convention-independent ray terminology is:

-   `equal_magnitude_rays`: `Re(Delta Q_ij) = 0`;
-   `phase_alignment_rays`: `Im(Delta Q_ij) = 0`.

`stokes_rays` and `anti_stokes_rays` are convenience aliases, but the
primary names avoid the historical convention reversal found in parts of
the literature.

Ramified branches are compared on a common uniformizer cover. Sector
dominance is computed there so branch labels remain well-defined through
ramification. The resulting rays are asymptotic tangent directions:
lower terms are retained as formal data, while the most singular
surviving term determines the tangent ray and asymptotic sector
ordering.

See also [Dominant and recessive formal solutions](dominance.md) and the
[Levelt--Turrittin workflow](levelt-turrittin.md).
