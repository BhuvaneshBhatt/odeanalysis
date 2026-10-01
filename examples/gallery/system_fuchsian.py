import sympy as sp

from odeanalysis import FirstOrderSystem, analyze_system_singularity

x = sp.symbols("x")
r = analyze_system_singularity(
    FirstOrderSystem(x, sp.ImmutableMatrix([[0, 0], [0, 2]]) / x)
)
assert r.kind == "regular_singular" and r.exponents == (0, 2) and len(r.resonances) == 1
