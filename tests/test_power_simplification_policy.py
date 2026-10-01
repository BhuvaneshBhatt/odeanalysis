from pathlib import Path

import sympy as sp

from odeanalysis._power_simplify import mixed_powsimp


def test_forced_power_simplification_is_confined_to_formal_helper():
    root = Path(__file__).resolve().parents[1] / "src" / "odeanalysis"
    offenders = []
    for path in root.glob("*.py"):
        if path.name == "_power_simplify.py":
            continue
        text = path.read_text()
        if "powsimp" in text and "force=True" in text:
            offenders.append(path.name)
        if "expand_power_base" in text and "force=True" in text:
            offenders.append(path.name)
    assert offenders == []


def test_mixed_power_simplification_keeps_analytic_coefficient_branches():
    a, b, t = sp.symbols("a b t")
    coefficient = sp.sqrt(a) * sp.sqrt(b)
    result = mixed_powsimp(coefficient, t ** sp.Rational(1, 2) * t ** sp.Rational(1, 2))
    assert result.has(sp.sqrt(a))
    assert result.has(sp.sqrt(b))
    assert not result.has(sp.sqrt(a * b))
