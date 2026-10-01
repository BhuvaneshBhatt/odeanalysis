import sympy as sp

from odeanalysis import FirstOrderSystem, system_formal_type_stratification

x = sp.symbols("x")
a = sp.symbols("a", real=True)
s = FirstOrderSystem(x, sp.ImmutableMatrix.diag(a / x, -a / x))
r = system_formal_type_stratification(s, (a,), max_resonance_order=2)
assert r.exhaustive and r.strata
