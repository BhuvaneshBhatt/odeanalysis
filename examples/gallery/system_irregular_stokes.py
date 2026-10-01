import sympy as sp

from odeanalysis import FirstOrderSystem, formal_system_analysis, system_stokes_geometry

x = sp.symbols("x")
s = FirstOrderSystem(x, sp.ImmutableMatrix.diag(x**-2, -(x**-2)))
f = formal_system_analysis(s, adaptive=True)
g = system_stokes_geometry(f)
assert f.complete and f.verify() and len(g.pairs) == 1 and g.structural_only
