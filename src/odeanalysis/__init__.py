"""Primary public API for symbolic structural ODE analysis.

Expert APIs remain available from their defining submodules; see
``docs/api-classification.md`` for the stability policy.
"""

__version__ = "0.25.2"

from ._moser import moser_reduce
from .analytic_continuation import (
    CanonicalBasis,
    ConnectionMatrix,
    LocalMonodromy,
    StokesMatrix,
    connection_matrix,
    hypergeometric_connection_matrix,
    kummer_connection_matrices,
    local_monodromy,
    stokes_matrices,
)
from .block_decomposition import (
    LeveltTurrittinReduction,
    formal_block_diagonalize,
    levelt_turrittin_reduce,
)
from .canonical import (
    CanonicalEquationFamily,
    CanonicalEquationRecognition,
    recognize_canonical_equation,
    transform_to_canonical,
)
from .certified_continuation import (
    CertifiedContinuationResult,
    CertifiedMatrixEnclosure,
    certified_system_continuation,
)
from .dominance import (
    SolutionDominanceAnalysis,
    classify_solution_dominance,
)
from .factorization import (
    FirstOrderFactorization,
    factor_differential_operator,
    is_reducible_operator,
)
from .formal import complete_formal_exponential_parts, formal_asymptotic_solutions
from .formal_basis import formal_monodromy, logarithmic_frobenius_basis
from .frobenius import FrobeniusAnalysis, frobenius_analysis
from .fuchsian import (
    ApparentSingularityAnalysis,
    FuchsRelation,
    RiemannScheme,
    apparent_singularity_analysis,
    fuchs_relation,
    is_apparent_singularity,
    riemann_scheme,
)
from .interchange import FormalODEData, formal_ode_data
from .interoperability import (
    CyclicScalarization,
    ScalarSystemCorrespondence,
    scalar_system_correspondence,
    scalarize_system,
)
from .irregular import formal_exponential_parts, wkb_ansatze
from .kovacic import KovacicAnalysis, KovacicOutcome, kovacic_analysis
from .levelt import LeveltStructure, levelt_structure
from .local_analysis import (
    LocalAnalysisStratum,
    ParameterizedLocalAnalysis,
    ResonanceStratum,
    local_parameter_analysis,
)
from .local_structure import (
    FrobeniusConvergence,
    FrobeniusLocalMonodromy,
    SingularityPointStructure,
    SingularityStructure,
    frobenius_convergence,
    frobenius_local_monodromy,
    singularity_structure,
)
from .matrix_series import MatrixLaurentSeries
from .newton import (
    DifferentialNewtonPolygon,
    SlopeFiltration,
    differential_newton_polygon,
    katz_rank,
    newton_slopes,
    poincare_rank,
    slope_filtration,
)
from .operator import LinearDifferentialOperator
from .parameter_wkb import (
    ParameterizedTurningAnalysis,
    ParameterizedTurningStratum,
    parameterized_turning_analysis,
)
from .singularities import (
    ODESingularityAnalysis,
    ODESingularityKind,
    analyze_ode_singularities,
    classify_ode_point,
    ode_singular_points,
)
from .stokes import StokesGeometry, stokes_geometry
from .system import FirstOrderSystem
from .system_analysis import (
    FormalReductionCertificate,
    FormalSystemAnalysis,
    ParameterizedSystemAnalysis,
    ParameterizedSystemFormalTypes,
    SystemFormalTypeSignature,
    SystemFormalTypeStratum,
    SystemParameterStratum,
    SystemResonance,
    SystemSingularityAnalysis,
    SystemStokesGeometry,
    SystemStokesPair,
    analyze_system_singularity,
    formal_system_analysis,
    system_formal_type_stratification,
    system_parameter_analysis,
    system_stokes_geometry,
)
from .transition_loci import (
    LeadingMatrixRankAnalysis,
    ProjectivePolynomialStratum,
    leading_rank_analysis,
    newton_loci,
    singularity_loci,
    singularity_projective_strata,
    stokes_formal_loci,
    stokes_ray_loci,
    turning_loci,
    turning_point_projective_strata,
)
from .turning import (
    LiouvilleNormalForm,
    TurningPoint,
    TurningPointAnalysis,
    TurningPointKind,
    UniformWKBReduction,
    WKBExpansion,
    airy_uniformization,
    analyze_turning_points,
    classify_turning_point,
    liouville_normal_form,
    turning_points,
    weber_uniformization,
    wkb_expansion,
)
from .wronskian import WronskianAnalysis, abel_wronskian, wronskian_analysis

__all__ = (
    "ApparentSingularityAnalysis",
    "CanonicalBasis",
    "CanonicalEquationFamily",
    "CanonicalEquationRecognition",
    "CertifiedContinuationResult",
    "CertifiedMatrixEnclosure",
    "ConnectionMatrix",
    "CyclicScalarization",
    "DifferentialNewtonPolygon",
    "FirstOrderFactorization",
    "FirstOrderSystem",
    "FormalODEData",
    "FormalReductionCertificate",
    "FormalSystemAnalysis",
    "FrobeniusAnalysis",
    "FrobeniusConvergence",
    "FrobeniusLocalMonodromy",
    "FuchsRelation",
    "KovacicAnalysis",
    "KovacicOutcome",
    "LeadingMatrixRankAnalysis",
    "LeveltStructure",
    "LeveltTurrittinReduction",
    "LinearDifferentialOperator",
    "LiouvilleNormalForm",
    "LocalAnalysisStratum",
    "LocalMonodromy",
    "MatrixLaurentSeries",
    "ODESingularityAnalysis",
    "ODESingularityKind",
    "ParameterizedLocalAnalysis",
    "ParameterizedSystemAnalysis",
    "ParameterizedSystemFormalTypes",
    "ParameterizedTurningAnalysis",
    "ParameterizedTurningStratum",
    "ProjectivePolynomialStratum",
    "ResonanceStratum",
    "RiemannScheme",
    "ScalarSystemCorrespondence",
    "SingularityPointStructure",
    "SingularityStructure",
    "SlopeFiltration",
    "SolutionDominanceAnalysis",
    "StokesGeometry",
    "StokesMatrix",
    "SystemFormalTypeSignature",
    "SystemFormalTypeStratum",
    "SystemParameterStratum",
    "SystemResonance",
    "SystemSingularityAnalysis",
    "SystemStokesGeometry",
    "SystemStokesPair",
    "TurningPoint",
    "TurningPointAnalysis",
    "TurningPointKind",
    "UniformWKBReduction",
    "WKBExpansion",
    "WronskianAnalysis",
    "__version__",
    "abel_wronskian",
    "airy_uniformization",
    "analyze_ode_singularities",
    "analyze_system_singularity",
    "analyze_turning_points",
    "apparent_singularity_analysis",
    "certified_system_continuation",
    "classify_ode_point",
    "classify_solution_dominance",
    "classify_turning_point",
    "complete_formal_exponential_parts",
    "connection_matrix",
    "differential_newton_polygon",
    "factor_differential_operator",
    "formal_asymptotic_solutions",
    "formal_block_diagonalize",
    "formal_exponential_parts",
    "formal_monodromy",
    "formal_ode_data",
    "formal_system_analysis",
    "frobenius_analysis",
    "frobenius_convergence",
    "frobenius_local_monodromy",
    "fuchs_relation",
    "hypergeometric_connection_matrix",
    "is_apparent_singularity",
    "is_reducible_operator",
    "katz_rank",
    "kovacic_analysis",
    "kummer_connection_matrices",
    "leading_rank_analysis",
    "levelt_structure",
    "levelt_turrittin_reduce",
    "liouville_normal_form",
    "local_monodromy",
    "local_parameter_analysis",
    "logarithmic_frobenius_basis",
    "moser_reduce",
    "newton_loci",
    "newton_slopes",
    "ode_singular_points",
    "parameterized_turning_analysis",
    "poincare_rank",
    "recognize_canonical_equation",
    "riemann_scheme",
    "scalar_system_correspondence",
    "scalarize_system",
    "singularity_loci",
    "singularity_projective_strata",
    "singularity_structure",
    "slope_filtration",
    "stokes_formal_loci",
    "stokes_geometry",
    "stokes_matrices",
    "stokes_ray_loci",
    "system_formal_type_stratification",
    "system_parameter_analysis",
    "system_stokes_geometry",
    "transform_to_canonical",
    "turning_loci",
    "turning_point_projective_strata",
    "turning_points",
    "weber_uniformization",
    "wkb_ansatze",
    "wkb_expansion",
    "wronskian_analysis",
)


def __dir__() -> list[str]:
    """Return the documented root API and standard module metadata."""

    metadata = {
        "__doc__",
        "__file__",
        "__loader__",
        "__name__",
        "__package__",
        "__path__",
        "__spec__",
    }
    return sorted(metadata | set(__all__))
