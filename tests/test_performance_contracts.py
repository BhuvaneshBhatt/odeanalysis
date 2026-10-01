"""Structural performance regressions without brittle wall-clock thresholds."""

from __future__ import annotations

import sympy as sp

import odeanalysis._moser as moser_module
from odeanalysis import MatrixLaurentSeries


def test_candidate_heavy_moser_search_bounds_spectral_classifications(monkeypatch):
    t = sp.symbols("t")
    connection = MatrixLaurentSeries.from_matrix(
        sp.Matrix(
            [
                [0, t**-2, 0],
                [0, 0, t**-2],
                [1, 0, 0],
            ]
        ),
        t,
    )
    original = moser_module.first_irregular_spectral_data
    calls = 0

    def counted(series):
        nonlocal calls
        calls += 1
        return original(series)

    monkeypatch.setattr(moser_module, "first_irregular_spectral_data", counted)
    moser_module.moser_reduce(connection, max_power=3, max_steps=2, max_shear=2)

    # Three dimensions and max_shear=2 give only a small bounded candidate set.
    # This guards against accidentally computing the same spectrum in nested
    # helper passes or introducing unbounded repeated classification.
    assert calls <= 20
