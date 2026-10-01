"""Independent verification must reject corrupted retained evidence."""

from __future__ import annotations

from dataclasses import replace

import sympy as sp

from odeanalysis import MatrixLaurentSeries, formal_block_diagonalize, moser_reduce


def test_spectral_verification_rejects_corrupted_projector_and_dimension():
    t = sp.symbols("t")
    connection = MatrixLaurentSeries.from_matrix(sp.diag(t**-2, -(t**-2)), t)
    reduction = formal_block_diagonalize(connection, max_power=3)
    split = reduction.spectral_splits[0]
    verification = split.verification

    assert verification.verify()
    bad_projectors = (sp.ImmutableMatrix(sp.eye(2)), *verification.projectors[1:])
    assert not replace(verification, projectors=bad_projectors).verify()
    assert not replace(verification, dimensions=(2, 1)).verify()


def test_gauge_verification_rejects_corrupted_target_and_source():
    t = sp.symbols("t")
    source = MatrixLaurentSeries.from_matrix(sp.Matrix([[0, t**-2], [1, 0]]), t)
    reduction = moser_reduce(source, max_power=3)
    verification = reduction.gauge_verification

    assert verification is not None
    assert verification.verify()
    wrong = MatrixLaurentSeries.from_matrix(sp.zeros(2), t)
    assert not replace(verification, transformed=wrong).verify()
    assert not replace(verification, source=wrong).verify()


def test_composite_block_verifier_checks_spectral_evidence():
    t = sp.symbols("t")
    connection = MatrixLaurentSeries.from_matrix(sp.diag(t**-2, -(t**-2)), t)
    reduction = formal_block_diagonalize(connection, max_power=3)

    assert reduction.verify()
    split = reduction.spectral_splits[0]
    damaged = replace(split.verification, eigenvalues=(sp.Integer(3), sp.Integer(4)))
    damaged_split = replace(split, verification=damaged)
    assert not replace(reduction, spectral_splits=(damaged_split,)).verify()
