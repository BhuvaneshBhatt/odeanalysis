import sympy as sp

from odeanalysis import LinearDifferentialOperator, scalar_system_correspondence

x = sp.symbols("x")
y = sp.Function("y")
op = LinearDifferentialOperator(x, y, (x + 1, 2 * x, 1))
c = scalar_system_correspondence(op)
assert c.verify()
