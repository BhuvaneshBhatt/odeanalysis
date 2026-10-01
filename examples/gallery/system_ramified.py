import sympy as sp

from odeanalysis import FirstOrderSystem, formal_system_analysis

x = sp.symbols("x")
# Exact cover bookkeeping is independently visible even when the reducer needs no extra cover.
s = FirstOrderSystem(x, sp.ImmutableMatrix.diag(x**-2, -(x**-2)))
t = sp.symbols("t")
r = s.ramify(t, 2)
f = formal_system_analysis(r, adaptive=True)
assert r.ramification_index == 2 and f.verify()
