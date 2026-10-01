"""Runtime dependency direction is odeanalysis -> no asymptotic import."""

from __future__ import annotations

import ast
from pathlib import Path


def test_production_odeanalysis_does_not_import_asymptotic():
    root = Path(__file__).parents[1] / "src" / "odeanalysis"
    violations: list[str] = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if any(
                    alias.name == "asymptotic" or alias.name.startswith("asymptotic.")
                    for alias in node.names
                ):
                    violations.append(f"{path}:{node.lineno}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module == "asymptotic" or module.startswith("asymptotic."):
                    violations.append(f"{path}:{node.lineno}")
            elif (
                isinstance(node, ast.Call)
                and node.args
                and isinstance(node.args[0], ast.Constant)
            ):
                target = node.args[0].value
                if not isinstance(target, str) or not (
                    target == "asymptotic" or target.startswith("asymptotic.")
                ):
                    continue
                func = node.func
                dynamic = (isinstance(func, ast.Name) and func.id == "__import__") or (
                    isinstance(func, ast.Attribute) and func.attr == "import_module"
                )
                if dynamic:
                    violations.append(f"{path}:{node.lineno}")
    assert not violations, violations
