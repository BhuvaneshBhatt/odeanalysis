"""Preferred, legacy and absent optional proof-oracle configurations."""

from __future__ import annotations

import runpy
import sys
from types import SimpleNamespace
from unittest.mock import patch

import sympy as sp
from exprtools import zerotest

import odeanalysis._assumptions as boundary


def test_exprtools_is_preferred():
    assert boundary._exprtools_zerotest is zerotest


def test_optional_legacy_fallback_and_no_backend():
    legacy = SimpleNamespace(zerotest=lambda *a, **k: None)
    with patch.dict(sys.modules, {"exprtools": None, "exprtest": legacy}):
        loaded = runpy.run_path(boundary.__file__)
        assert loaded["_exprtools_zerotest"] is legacy.zerotest
    with patch.dict(sys.modules, {"exprtools": None, "exprtest": None}):
        loaded = runpy.run_path(boundary.__file__)
        assert loaded["_exprtools_zerotest"] is None
        assert loaded["zero_status"](sp.sin(sp.Symbol("x"))) is None


def test_unknown_forwarding_and_symbolic_parameter_guard():
    x = sp.Symbol("x")
    with (
        patch.object(boundary, "entails", return_value=None),
        patch.object(boundary, "_exprtools_zerotest", return_value=None) as oracle,
    ):
        assert boundary.zero_status(sp.sin(x)) is None
        oracle.assert_called_once_with(sp.sin(x), assumptions=sp.true, confidence="certified")
        oracle.reset_mock()
        assert boundary.zero_status(x) is None
        oracle.assert_not_called()
