# Canonical equation recognition

`recognize_canonical_equation()` recognizes exact transformations to five classical second-order families:

- Airy: `u'' - z*u = 0`;
- Bessel: `z**2*u'' + z*u' + (z**2 - nu**2)*u = 0`;
- modified Bessel: `z**2*u'' + z*u' - (z**2 + nu**2)*u = 0`;
- Gauss hypergeometric: `z*(1-z)*u'' + (c-(a+b+1)*z)*u' - a*b*u = 0`;
- confluent hypergeometric/Kummer: `z*u'' + (c-z)*u' - a*u = 0`.

Airy, Bessel, modified Bessel, Gauss hypergeometric, and Kummer recognizers retain their exact affine path. In addition, every resolved second-order equation with exactly three regular singular points is tested as a Riemann P equation. The three singularities are mapped projectively to `0`, `1`, and infinity by

```text
z = (a*x + b)/(c*x + d),    a*d - b*c != 0,
```

and a gauge

```text
y(x) = g(x) * u(z(x)),
g(x) = z(x)**alpha * (1-z(x))**beta
```

is obtained from the local exponent data. The hypergeometric parameters are then fixed by the shifted exponent columns. Candidate transformations are accepted only after exact coefficient replay.

A successful `CanonicalEquationRecognition` records `variable_transform`, `mobius_coefficients`, `dependent_gauge`, `gauge_log_derivative`, and the canonical parameters. For affine maps, the convenience properties `scale` and `shift` expose the affine coefficients; for genuinely projective maps they are `None`.

The verifier uses the normalized pullback identities

```text
p_x = z' P(z) - 2 h - z''/z'
q_x = z'^2 Q(z) - p_x h - h' - h^2
```

where `h = g'/g`. Storing `h` separately avoids relying on branch-sensitive differentiation of fractional powers in `g`.

```python
import sympy as sp
from odeanalysis import transform_to_canonical

x = sp.symbols("x")
y = sp.Function("y")
t = x - 3
ode = t**2 * sp.diff(y(x), x, 2) + t * sp.diff(y(x), x) + (4 * t**2 - 9) * y(x)

result = transform_to_canonical(ode, y, x)
assert result.family.value == "bessel"
assert result.parameter_map["nu"] == 3
assert result.verify()
```

The projective hypergeometric recognizer requires the three singularities and both local exponents at each singularity to be resolved exactly. Equations with unresolved symbolic singular points or unresolved indicial roots return no recognition rather than guessing.

## Euler--Cauchy family

Exact Euler--Cauchy equations `x^2 y'' + alpha*x*y' + beta*y = 0` are recognized as a canonical family. The origin power exponents are the roots of `rho(rho-1)+alpha*rho+beta=0`; `local_monodromy("euler", 0, alpha=..., beta=...)` returns the corresponding exact diagonal power-basis monodromy.
