import sympy as sp

from odeanalysis import FirstOrderSystem, certified_system_continuation

x = sp.symbols("x")
r = certified_system_continuation(
    FirstOrderSystem(x, sp.ImmutableMatrix([[0, 1], [0, 0]])), 0, 2, precision_bits=192
)
assert r.complete and r.enclosure is not None and r.enclosure.certified
