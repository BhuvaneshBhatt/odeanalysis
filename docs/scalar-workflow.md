# Scalar workflow

For a scalar linear operator, proceed from local structure to increasingly global data.

1. `analyze_ode_singularities()` classifies finite points and infinity.
2. `frobenius_analysis()` constructs regular-singular local data.
3. `differential_newton_polygon()` and completed formal exponential parts describe irregular structure.
4. `stokes_geometry()` and dominance analysis describe formal sector geometry.
5. `liouville_normal_form()`, turning analysis, and WKB address second-order turning points.
6. Canonical recognition and exact continuation are used only for supported verified families.

The standard companion system is an exact representation of the same scalar equation,
but its **raw connection pole order is presentation-dependent**. Do not equate raw
companion Poincare rank with invariant scalar regular-singular classification without
the appropriate meromorphic gauge/shearing analysis.
