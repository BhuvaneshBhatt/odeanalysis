"""Adversarial contracts for bounded/adaptive formal system reduction."""

import sympy as sp

from odeanalysis import FirstOrderSystem, formal_system_analysis

x = sp.symbols("x")


def test_adaptive_search_preserves_verified_information_and_records_budget():
    system = FirstOrderSystem(x, sp.ImmutableMatrix.diag(x**-2, -(x**-2)))
    bounded = formal_system_analysis(system, max_depth=1, adaptive=False)
    adaptive = formal_system_analysis(system, max_depth=1, adaptive=True, max_adaptive_depth=4)
    assert bounded.certificate.verified
    assert adaptive.certificate.verified
    assert adaptive.certificate.attempted_depths[0] == 1
    assert adaptive.certificate.final_depth >= bounded.certificate.final_depth
    assert adaptive.singularity.kind == bounded.singularity.kind
    assert adaptive.singularity.poincare_rank == bounded.singularity.poincare_rank


def test_complete_claim_satisfies_independent_reduced_block_criterion():
    system = FirstOrderSystem(x, sp.ImmutableMatrix.diag(2 * x**-3 + x**-1, -(x**-3) + 3 * x**-1))
    result = formal_system_analysis(system, adaptive=True, max_adaptive_depth=6)
    assert result.certificate.verified
    assert result.complete
    # Every extracted irregular coefficient in a final block must be scalar;
    # otherwise exponential_parts would overstate the completed normal form.
    for block in result.reduction.final_diagonalization.blocks:
        for power, coefficient in block.terms:
            if power < -1:
                m = sp.Matrix(coefficient)
                scalar = sp.simplify(sp.trace(m) / m.rows)
                assert m == scalar * sp.eye(m.rows)


def test_constant_gauge_covariance_of_formal_exponential_parts():
    system = FirstOrderSystem(x, sp.ImmutableMatrix.diag(x**-2, -2 * x**-2))
    gauge = sp.ImmutableMatrix([[1, 1], [1, -1]])
    left = formal_system_analysis(system, adaptive=True)
    right = formal_system_analysis(system.gauge_transform(gauge), adaptive=True)
    assert left.verify() and right.verify()

    def canonical(parts):
        normalized = []
        t = sp.Symbol("t")
        for q in parts:
            symbols = tuple(q.free_symbols)
            normalized.append(sp.expand(q.subs(symbols[0], t)) if symbols else q)
        return sorted(normalized, key=sp.default_sort_key)

    assert canonical(left.exponential_parts) == canonical(right.exponential_parts)
    assert left.ramification_index == right.ramification_index


def test_reciprocal_coordinate_preserves_exact_connection_classification():
    system = FirstOrderSystem(x, sp.ImmutableMatrix.diag(x, -x))
    infinity = formal_system_analysis(system, point=sp.oo, adaptive=True)
    t = sp.symbols("t", positive=True)
    local = system.change_variable(t, 1 / t)
    zero = formal_system_analysis(local, point=0, adaptive=True)
    assert infinity.singularity.kind == zero.singularity.kind
    assert infinity.singularity.poincare_rank == zero.singularity.poincare_rank
    assert infinity.ramification_index == zero.ramification_index


def test_ramification_metadata_composes_exactly():
    t, u = sp.symbols("t u")
    system = FirstOrderSystem(x, sp.ImmutableMatrix.diag(x**-2, -(x**-2)))
    twice = system.ramify(t, 2).ramify(u, 3)
    assert twice.ramification_index == 6
    assert twice.matrix == system.change_variable(u, u**6, ramification_multiplier=6).matrix


def test_formal_diagonalization_verifier_rejects_corrupt_partition_metadata():
    from dataclasses import replace

    from odeanalysis import MatrixLaurentSeries
    from odeanalysis.block_decomposition import formal_block_diagonalize

    connection = MatrixLaurentSeries.from_mapping(
        x, {-2: sp.diag(1, -1), -1: sp.eye(2)}, shape=(2, 2)
    )
    reduction = formal_block_diagonalize(connection, max_power=-1)
    assert reduction.complete and reduction.verify()
    corrupt = replace(reduction, block_dimensions=(1,), block_slices=((0, 1),))
    assert not corrupt.verify()
