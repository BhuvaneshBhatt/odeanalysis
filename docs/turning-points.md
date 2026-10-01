# Turning points and uniform WKB

For a homogeneous second-order equation

`y'' + p(x) y' + q(x) y = 0`,

`odeanalysis` first removes the first-derivative term by the exact Liouville transformation

`y = g u`, `g'/g = -p/2`.

The normal-form equation is

`u'' = Q(x) u`,

with

`Q = p'/2 + p**2/4 - q`.

Turning points are finite zeros of this `Q`, not zeros of the original zeroth-order coefficient before the Liouville gauge. `liouville_normal_form()` exposes the reduction explicitly and `analyze_turning_points()` resolves finite zeros with multiplicity when the numerator of `Q` is polynomial and its roots are exactly available.

## Turning-point classes

`TurningPointKind` distinguishes:

- `SIMPLE`: multiplicity one;
- `DOUBLE`: multiplicity two;
- `HIGHER`: multiplicity three or greater.

Each `TurningPoint` retains the exact leading local coefficient

`Q(x0+h) = c*h**m + ...`

and can replay the multiplicity/leading-coefficient check with `verify()`.

```python
import sympy as sp
from odeanalysis import analyze_turning_points

x = sp.symbols("x")
y = sp.Function("y")
ode = sp.diff(y(x), x, 2) - x * (x - 1) ** 2 * y(x)
analysis = analyze_turning_points(ode, y, x)

assert [(p.point, p.multiplicity) for p in analysis.points] == [(0, 1), (1, 2)]
assert analysis.verify()
```

## Ordinary WKB expansion

`wkb_expansion()` interprets the normal-form equation as

`epsilon**2 u'' = Q u`

and solves the Riccati equation

`epsilon*S' + S**2 = Q`

with

`S = S_0 + epsilon*S_1 + epsilon**2*S_2 + ...`.

The two branches start with `S_0 = +/-sqrt(Q)`. For `n >= 1`,

`S_n = -(S_(n-1)' + sum(S_j*S_(n-j), j=1..n-1))/(2*S_0)`.

The resulting normal-form solution is represented formally as

`u = exp(Integral(S, x)/epsilon)`.

The original dependent variable includes the exact Liouville gauge. WKB expansions are asymptotic away from zeros and poles of `Q`; the turning-point uniformizers below are intended for neighborhoods where ordinary WKB loses uniformity.

## Airy uniformization

At a simple turning point `x0`, define the local phase coordinate `zeta` by

`zeta*(d zeta/dx)**2 = Q(x)`.

Equivalently, on a chosen local square-root branch,

`(2/3)*zeta**(3/2) = Integral(sqrt(Q(s)), (s, x0, x))`.

With

`u = (zeta')**(-1/2) W(zeta)`,

the transformed equation is

`epsilon**2 W'' = (zeta + epsilon**2*psi(zeta)) W`,

where the exact residual is

`psi = zeta'''/(2*zeta'**3) - 3*zeta''**2/(4*zeta'**4)`.

`airy_uniformization()` returns the coordinate, amplitude, residual, and original-variable amplitude. When `psi` vanishes, the equation is exactly Airy after the Liouville-Green transformation. Otherwise the Airy term is the uniform leading model and the residual is retained rather than discarded.

## Weber uniformization

At an isolated double turning point, `weber_uniformization()` uses

`zeta**2*(zeta')**2 = Q(x)`

so that

`epsilon**2 W'' = (zeta**2 + epsilon**2*psi(zeta)) W`.

This is the degenerate Weber/parabolic-cylinder model associated with a double zero. A pair of distinct turning points coalescing with a parameter requires a nonzero Weber parameter and is outside this layer; that belongs to parameter-dependent confluence analysis.

## Exactness and branches

The phase integrals and fractional powers define local analytic branches. The package does not impose a global branch cut. Verification checks the local differential identities used by the Liouville-Green transformation and the exact residual formula.

The current turning-point finder is finite-point only. Turning points at infinity, unresolved symbolic root sets, and higher canonical catastrophes are left unresolved.

## Parameter-dependent turning structure

`parameterized_turning_analysis()` uses the exact turning collision/degree-loss loci and `semialg` to partition real parameter space. Each cell carries an exact specialized list of finite turning multiplicities and kinds. This makes confluence boundaries explicit before selecting Airy or degenerate Weber uniformization. A general parameter-dependent coalescing-pair Weber reduction with nonzero canonical Weber parameter remains outside the current contract.
