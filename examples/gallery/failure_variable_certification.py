import sympy as sp

from odeanalysis import FirstOrderSystem, certified_system_continuation

x = sp.symbols("x")
r = certified_system_continuation(FirstOrderSystem(x, sp.ImmutableMatrix([[x]])), 0, 1)
assert not r.complete and "variable-coefficient" in r.limitation
