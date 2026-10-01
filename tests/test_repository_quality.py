"""Static contracts for maintainable production and documentation source."""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "odeanalysis"


def _source_files():
    return sorted(SRC.rglob("*.py"))


def test_production_avoids_broad_exception_handlers_and_asserts():
    hits = []
    for path in _source_files():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):
                hits.append(f"{path.name}:{node.lineno}:assert")
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    hits.append(f"{path.name}:{node.lineno}:bare-except")
                elif isinstance(node.type, ast.Name) and node.type.id in {
                    "Exception",
                    "BaseException",
                }:
                    hits.append(f"{path.name}:{node.lineno}:{node.type.id}")
    assert hits == []


def test_production_definitions_are_not_overridden_in_one_scope():
    hits = []

    def inspect_scope(path, body, scope):
        seen = {}
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name in seen:
                    hits.append(
                        f"{path.name}:{scope}:{node.name}:{seen[node.name]},{node.lineno}"
                    )
                seen[node.name] = node.lineno
        for node in body:
            if isinstance(node, ast.ClassDef):
                inspect_scope(path, node.body, f"{scope}.{node.name}")

    for path in _source_files():
        tree = ast.parse(path.read_text())
        inspect_scope(path, tree.body, "module")
    assert hits == []


def test_function_local_names_remain_readable():
    hits = []
    for path in _source_files():
        tree = ast.parse(path.read_text())
        for function in (
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ):
            names = {arg.arg for arg in function.args.args + function.args.kwonlyargs}
            names.update(
                node.id
                for node in ast.walk(function)
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
            )
            for name in names:
                if len(name) > 24:
                    hits.append(f"{path.name}:{function.name}:{name}")
    assert hits == []


def test_repository_has_no_trailing_whitespace():
    hits = []
    paths = [ROOT / "README.md", *ROOT.glob("docs/*.md")]
    paths += list((ROOT / "src").rglob("*.py"))
    paths += list((ROOT / "tests").rglob("*.py"))
    for path in paths:
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            if line.rstrip() != line:
                hits.append(f"{path.relative_to(ROOT)}:{number}")
    assert hits == []


def test_current_documentation_is_not_organized_by_release_history():
    pattern = re.compile(
        r"\b(?:version\s+0\.\d+|0\.\d+\s+(?:architecture|layer|reduction)|phase\s+\d+|milestone)\b",
        re.IGNORECASE,
    )
    hits = []
    for path in [ROOT / "README.md", *ROOT.glob("docs/*.md")]:
        if pattern.search(path.read_text()):
            hits.append(str(path.relative_to(ROOT)))
    assert hits == []


def test_package_docstring_and_public_api_are_declared_coherently():
    init_path = SRC / "__init__.py"
    tree = ast.parse(init_path.read_text())
    docstring = ast.get_docstring(tree) or ""
    assert docstring.startswith(
        "Primary public API for symbolic structural ODE analysis."
    )
    assert "api-classification.md" in docstring
    all_assignments = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "__all__"
                for target in node.targets
            )
        )
        or (
            isinstance(node, ast.AugAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "__all__"
        )
    ]
    assert len(all_assignments) == 1


def test_repository_does_not_reference_external_proprietary_system_names():
    forbidden = ("mathe" + "matica", "wol" + "fram")
    hits = []
    paths = [ROOT / "README.md", *ROOT.glob("docs/*.md")]
    paths += list((ROOT / "src").rglob("*.py"))
    paths += list((ROOT / "tests").rglob("*.py"))
    for path in paths:
        text = path.read_text().lower()
        if any(re.search(r"\b" + re.escape(name) + r"\b", text) for name in forbidden):
            hits.append(str(path.relative_to(ROOT)))
    assert hits == []


def test_pytest_disables_unrelated_ddtrace_autoload():
    """Keep ordinary one-process test runs independent of global tracing plugins."""

    pyproject = (ROOT / "pyproject.toml").read_text()
    match = re.search(r'^addopts\s*=\s*"([^"]*)"', pyproject, re.MULTILINE)
    assert match is not None
    options = match.group(1).split()
    assert options[options.index("-p") + 1] == "no:ddtrace"


def test_preloaded_ddtrace_is_neutralized_when_present():
    """A globally discovered tracer must not keep pytest alive at shutdown."""

    import sys

    module = sys.modules.get("ddtrace")
    tracer = getattr(module, "tracer", None)
    if tracer is not None:
        assert tracer.enabled is False


def test_active_tree_uses_timeless_capability_language():
    banned = (
        "future " + "work",
        "for a " + "later",
        "previous private " + "inconsistency",
        "keep their previous " + "meaning",
        "currently " + "requires",
    )
    hits = []
    paths = [ROOT / "README.md", *ROOT.glob("docs/*.md")]
    paths += list((ROOT / "src" / "odeanalysis").rglob("*.py"))
    for path in paths:
        text = path.read_text().lower()
        if any(phrase in text for phrase in banned):
            hits.append(str(path.relative_to(ROOT)))
    assert hits == []
