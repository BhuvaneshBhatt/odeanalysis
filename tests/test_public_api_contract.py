"""Contracts for the focused root-level API."""

from __future__ import annotations

import inspect

import odeanalysis
from odeanalysis._api_policy import EXPERT_EXPORTS, PRIMARY_EXPORTS

EXPECTED_ROOT = {
    "CertifiedContinuationResult",
    "CertifiedMatrixEnclosure",
    "CyclicScalarization",
    "FormalReductionCertificate",
    "ParameterizedSystemFormalTypes",
    "ParameterizedTurningAnalysis",
    "ParameterizedTurningStratum",
    "ScalarSystemCorrespondence",
    "SystemFormalTypeSignature",
    "SystemFormalTypeStratum",
    "certified_system_continuation",
    "parameterized_turning_analysis",
    "scalar_system_correspondence",
    "scalarize_system",
    "system_formal_type_stratification",
    "FormalSystemAnalysis",
    "ParameterizedSystemAnalysis",
    "SystemParameterStratum",
    "SystemResonance",
    "SystemSingularityAnalysis",
    "SystemStokesGeometry",
    "SystemStokesPair",
    "analyze_system_singularity",
    "formal_system_analysis",
    "system_parameter_analysis",
    "system_stokes_geometry",
    "CanonicalBasis",
    "ConnectionMatrix",
    "LocalAnalysisStratum",
    "LocalMonodromy",
    "StokesMatrix",
    "connection_matrix",
    "hypergeometric_connection_matrix",
    "kummer_connection_matrices",
    "local_monodromy",
    "stokes_matrices",
    "LiouvilleNormalForm",
    "TurningPoint",
    "TurningPointAnalysis",
    "TurningPointKind",
    "UniformWKBReduction",
    "WKBExpansion",
    "airy_uniformization",
    "analyze_turning_points",
    "classify_turning_point",
    "liouville_normal_form",
    "turning_points",
    "turning_loci",
    "weber_uniformization",
    "wkb_expansion",
    "ApparentSingularityAnalysis",
    "CanonicalEquationFamily",
    "CanonicalEquationRecognition",
    "DifferentialNewtonPolygon",
    "FirstOrderFactorization",
    "FirstOrderSystem",
    "FormalODEData",
    "FrobeniusAnalysis",
    "FrobeniusConvergence",
    "FrobeniusLocalMonodromy",
    "FuchsRelation",
    "KovacicAnalysis",
    "KovacicOutcome",
    "LeveltStructure",
    "LeveltTurrittinReduction",
    "LeadingMatrixRankAnalysis",
    "LinearDifferentialOperator",
    "MatrixLaurentSeries",
    "ODESingularityAnalysis",
    "ODESingularityKind",
    "ParameterizedLocalAnalysis",
    "ResonanceStratum",
    "RiemannScheme",
    "SingularityPointStructure",
    "SingularityStructure",
    "SlopeFiltration",
    "SolutionDominanceAnalysis",
    "StokesGeometry",
    "WronskianAnalysis",
    "__version__",
    "abel_wronskian",
    "analyze_ode_singularities",
    "apparent_singularity_analysis",
    "classify_ode_point",
    "classify_solution_dominance",
    "complete_formal_exponential_parts",
    "factor_differential_operator",
    "differential_newton_polygon",
    "formal_asymptotic_solutions",
    "formal_block_diagonalize",
    "formal_exponential_parts",
    "formal_monodromy",
    "formal_ode_data",
    "frobenius_analysis",
    "frobenius_convergence",
    "frobenius_local_monodromy",
    "fuchs_relation",
    "is_apparent_singularity",
    "is_reducible_operator",
    "katz_rank",
    "kovacic_analysis",
    "levelt_structure",
    "leading_rank_analysis",
    "ProjectivePolynomialStratum",
    "levelt_turrittin_reduce",
    "logarithmic_frobenius_basis",
    "moser_reduce",
    "newton_slopes",
    "newton_loci",
    "ode_singular_points",
    "local_parameter_analysis",
    "poincare_rank",
    "recognize_canonical_equation",
    "riemann_scheme",
    "singularity_structure",
    "singularity_loci",
    "singularity_projective_strata",
    "slope_filtration",
    "stokes_geometry",
    "stokes_formal_loci",
    "stokes_ray_loci",
    "transform_to_canonical",
    "turning_point_projective_strata",
    "wkb_ansatze",
    "wronskian_analysis",
}


def test_root_api_matches_policy_exactly():
    assert set(odeanalysis.__all__) == EXPECTED_ROOT
    assert set(PRIMARY_EXPORTS) | {"__version__"} == EXPECTED_ROOT


def test_primary_api_is_documented_and_directly_importable():
    for name in EXPECTED_ROOT - {"__version__"}:
        obj = getattr(odeanalysis, name)
        assert len((inspect.getdoc(obj) or "").strip()) >= 30, name
    assert odeanalysis.__version__ == "0.25.2"


def test_expert_names_are_public_only_from_defining_submodules():
    for name, module_name in EXPERT_EXPORTS.items():
        assert not hasattr(odeanalysis, name), name
        module = __import__(f"odeanalysis{module_name}", fromlist=[name])
        assert hasattr(module, name), (module_name, name)


def test_dir_exposes_root_contract_without_expert_noise():
    names = set(dir(odeanalysis))
    assert EXPECTED_ROOT <= names
    assert not (set(EXPERT_EXPORTS) & names)
