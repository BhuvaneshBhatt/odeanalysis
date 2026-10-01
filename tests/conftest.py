"""Test-environment guards for deterministic pytest shutdown."""

from __future__ import annotations

import gc
import sys


def _disable_preloaded_ddtrace() -> None:
    """Disable a globally auto-loaded tracer without importing it ourselves."""

    module = sys.modules.get("ddtrace")
    tracer = getattr(module, "tracer", None)
    if tracer is not None:
        tracer.enabled = False


def _release_symbolic_caches() -> None:
    """Release process-global symbolic state before interpreter finalization."""

    cache_module = sys.modules.get("sympy.core.cache")
    clear_cache = getattr(cache_module, "clear_cache", None)
    if clear_cache is not None:
        clear_cache()
    gc.collect()


_disable_preloaded_ddtrace()


def pytest_sessionfinish(session, exitstatus) -> None:
    """Make symbolic cleanup part of visible pytest teardown."""

    _release_symbolic_caches()
