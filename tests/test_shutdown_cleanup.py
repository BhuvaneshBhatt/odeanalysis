from __future__ import annotations

import sys
from types import SimpleNamespace

import conftest


def test_symbolic_cache_cleanup_uses_loaded_cache_without_import(monkeypatch):
    calls = []
    cache_module = SimpleNamespace(clear_cache=lambda: calls.append("cache"))

    monkeypatch.setitem(sys.modules, "sympy.core.cache", cache_module)
    monkeypatch.setattr(conftest.gc, "collect", lambda: calls.append("gc"))

    conftest._release_symbolic_caches()

    assert calls == ["cache", "gc"]


def test_symbolic_cache_cleanup_does_not_import_missing_cache(monkeypatch):
    calls = []

    monkeypatch.delitem(sys.modules, "sympy.core.cache", raising=False)
    monkeypatch.setattr(conftest.gc, "collect", lambda: calls.append("gc"))

    conftest._release_symbolic_caches()

    assert calls == ["gc"]
