import sympy as sp

from odeanalysis import FirstOrderSystem, scalarize_system

x = sp.symbols("x")
r = scalarize_system(
    FirstOrderSystem(x, sp.ImmutableMatrix.diag(1, 2)),
    output_row=sp.ImmutableMatrix([[1, 0]]),
)
assert not r.complete and r.limitation
