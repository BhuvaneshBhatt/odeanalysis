# User guide

This guide is organized by the mathematical question being asked rather than by source module.

## Find and classify singularities

Use `analyze_ode_singularities()` for the complete finite/infinity view, `ode_singular_points()` when only locations are needed, and `classify_ode_point()` for a single germ. All three accept either an ODE expression or a `LinearDifferentialOperator`.

## Inspect Newton slopes and irregularity ranks

Use `differential_newton_polygon()` for the first-class lower polygon. `newton_slopes()` expands its slope multiset with multiplicity, `slope_filtration()` groups equal slopes, `katz_rank()` returns the invariant rational Katz rank, and `poincare_rank()` returns the ordinary Poincare rank of the natural Euler-scaled companion presentation; unlike the Katz rank it is gauge dependent. See [Newton polygons, slopes, and ranks](newton-invariants.md).

## Analyze a regular singular point

Use `frobenius_analysis()` for the indicial polynomial, multiplicities, resonances, recurrences, and logarithmic obstructions. When an explicit logarithmic basis is needed, use `logarithmic_frobenius_basis()` and then `formal_monodromy()`.

## Build a Fuchsian Riemann scheme

Use `riemann_scheme()` for equations regular singular at every singular point on the sphere and `fuchs_relation()` for the exact exponent-sum identity. `apparent_singularity_analysis()` distinguishes holomorphic apparent singularities from logarithmic, branched, meromorphic, or unresolved cases. See [Fuchsian analysis](fuchsian-analysis.md).

## Construct formal asymptotic solutions

`formal_exponential_parts()` gives the leading Newton data. `complete_formal_exponential_parts()` performs Riccati/Newton refinement, and `formal_asymptotic_solutions()` combines the exponential, algebraic, and amplitude factors. `wkb_ansatze()` is the lower-level entry point when the leading WKB conjugation itself is the desired object.

## Analyze turning points and build uniform WKB approximations

For a second-order equation, `liouville_normal_form()` exposes the effective normal-form potential. Use `analyze_turning_points()` or `classify_turning_point()` to locate and classify finite zeros, `wkb_expansion()` for ordinary Riccati/WKB branches away from them, `airy_uniformization()` for simple turning points, and `weber_uniformization()` for isolated double turning points. Uniform reductions retain the exact transformed residual. See [Turning points and uniform WKB](turning-points.md).

## Analyze an irregular singularity

For scalar problems, start with `formal_asymptotic_solutions()` or `levelt_structure()`. For a system represented by `MatrixLaurentSeries`, use `formal_block_diagonalize()` for bounded exact block splitting or `levelt_turrittin_reduce()` when ramification may be necessary.

## Understand Levelt--Turrittin structure

`levelt_structure()` packages the scalar formal decomposition into exponential blocks, exponent classes, logarithmic nilpotent parts, and monodromy data. See [Levelt--Turrittin workflow](levelt-turrittin.md) for the reduction stages and a system example.

## Classify dominant and recessive branches

`classify_solution_dominance()` converts completed Stokes-sector comparisons into explicit dominant and recessive branch sets on the common ramified cover. See [Dominant and recessive formal solutions](dominance.md).

## Compute Stokes geometry and formal monodromy

`stokes_geometry()` derives equal-magnitude and phase-alignment rays from completed exponential differences. `formal_monodromy()` operates on a logarithmic formal basis. These are formal-local objects; numerical Stokes constants and global connection matrices require additional analytic information.

## Work with first-order systems

`FirstOrderSystem` is the exact system representation. Use `analyze_system_singularity()` for finite/infinity classification, residues, exponents and resonances; `formal_system_analysis()` for the bounded verified Moser/Levelt--Turrittin pipeline; `system_stokes_geometry()` for structural block-level Stokes rays; and `system_parameter_analysis()` for `semialg`-certified transition strata. The expert helper `odeanalysis.system.companion_system()` converts a scalar operator to a companion system. See [Linear systems](systems.md).

## Export formal data

`formal_ode_data()` produces a stable, machine-readable `FormalODEData` object containing block structure, ramification, basis vectors, monodromy, optional Stokes data, and provenance. This is the preferred boundary for downstream packages.

## Verification

Formal block, Moser, spectral, ramification, and composite reduction results retain enough evidence to verify their recorded transformations without rerunning discovery. See [Verification](verification.md).

## Capabilities and limitations

See [Capabilities and limitations](capabilities.md) for the supported theorem classes and support boundaries.

## Choosing scalar or system analysis

Use scalar APIs directly for scalar equations; they remain the specialized path for Frobenius, Newton polygons, turning points, WKB and canonical recognition. Use `FirstOrderSystem` for genuine systems or when formal block structure is the object of interest. `companion_system()` and `scalarize_system()` provide explicit conversion boundaries rather than silently routing one representation through the other.
