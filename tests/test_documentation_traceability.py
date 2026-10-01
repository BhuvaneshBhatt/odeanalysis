"""Documentation-to-implementation traceability contracts."""

from __future__ import annotations

import ast
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def _public_names():
    tree = ast.parse((ROOT / "src/odeanalysis/__init__.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
        ):
            return tuple(ast.literal_eval(node.value))
    raise AssertionError("root __all__ not found")


def test_every_root_public_name_is_in_api_reference():
    text = (DOCS / "api-reference.md").read_text()
    missing = [
        name for name in _public_names() if name != "__version__" and f"`{name}`" not in text
    ]
    assert missing == []


def test_new_certification_surfaces_have_reference_and_failure_documentation():
    combined = "\n".join(path.read_text() for path in DOCS.glob("*.md"))
    for name in (
        "FormalReductionCertificate",
        "SystemFormalTypeSignature",
        "CyclicScalarization",
        "CertifiedContinuationResult",
    ):
        assert name in combined
    failures = (DOCS / "worked-failures.md").read_text()
    for phrase in ("did not complete", "no operator", "no analytic matrix", "refused"):
        assert phrase.lower() in failures.lower()


def test_traceability_matrix_names_existing_tests_and_docs():
    text = (DOCS / "traceability.md").read_text()
    for test_name in (
        "test_system_certification_corpus.py",
        "test_scalar_system_differential_corpus.py",
        "test_formal_reduction_adversarial.py",
        "test_system_parameter_formal_types.py",
        "test_certified_continuation_corpus.py",
    ):
        assert test_name in text
        assert (ROOT / "tests" / test_name).is_file()
    for doc in (
        "system-singularities.md",
        "system-formal-analysis.md",
        "system-stokes.md",
        "system-parameters.md",
        "certified-continuation.md",
    ):
        assert doc in text
        assert (DOCS / doc).is_file()


def test_readme_capability_rows_route_to_detailed_docs():
    readme = (ROOT / "README.md").read_text()
    for doc in (
        "capabilities.md",
        "certification-model.md",
        "conventions.md",
        "traceability.md",
    ):
        assert f"./docs/{doc}" in readme


def test_gallery_scripts_execute():
    for path in sorted((ROOT / "examples/gallery").glob("*.py")):
        runpy.run_path(str(path), run_name="__main__")
