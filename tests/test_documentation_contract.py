"""Documentation structure and executable API examples stay aligned."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_GUIDES = {
    "analytic-continuation.md",
    "api-classification.md",
    "capabilities.md",
    "dominance.md",
    "examples.md",
    "fuchsian-analysis.md",
    "levelt-turrittin.md",
    "newton-invariants.md",
    "systems.md",
    "turning-points.md",
    "user-guide.md",
    "verification.md",
}


def test_workflow_guides_exist_and_are_linked_from_readme():
    readme = (ROOT / "README.md").read_text()
    for name in REQUIRED_GUIDES:
        assert (ROOT / "docs" / name).is_file()
        assert f"/docs/{name}" in readme


def test_readme_does_not_teach_expert_names_as_root_imports():
    readme = (ROOT / "README.md").read_text()
    forbidden = (
        "from odeanalysis import irregular_singularity_invariants",
        "from odeanalysis import companion_system",
        "from odeanalysis import LinearDifferentialOperator, green_operator_data",
    )
    assert all(item not in readme for item in forbidden)


def test_capability_page_states_analytic_boundaries():
    text = (ROOT / "docs" / "capabilities.md").read_text()
    assert "Generic numerical Stokes constants | No" in text
    assert "Generic global connection matrices | No" in text
    assert "Ramified Levelt--Turrittin reduction | Bounded" in text


def test_first_class_system_guides_and_advanced_topics_are_documented():
    root = Path(__file__).resolve().parents[1]
    required = {
        "system-singularities.md": ("analyze_system_singularity", "resonance"),
        "system-formal-analysis.md": ("FormalReductionCertificate", "scalarize_system"),
        "system-stokes.md": ("structural", "analytic Stokes"),
        "system-parameters.md": (
            "system_formal_type_stratification",
            "parameterized_turning_analysis",
        ),
    }
    for filename, phrases in required.items():
        text = (root / "docs" / filename).read_text()
        assert all(phrase in text for phrase in phrases)
    examples = (root / "docs" / "examples.md").read_text()
    for number in range(1, 7):
        assert f"System example {number}:" in examples
    readme = (root / "README.md").read_text()
    assert "scalar linear ODEs" in readme
    assert "first-order linear" in readme
    assert "Certified numerical continuation" in readme
