# Structural Stokes geometry for systems

`system_stokes_geometry()` derives structural Stokes data from Levelt--Turrittin exponential blocks. For every pair it forms the exponential difference, extracts its leading negative-power monomial, and computes equal-magnitude and phase-alignment rays on the chosen cover.

This is deliberately **structural** Stokes geometry. It does not claim generic analytic Stokes multipliers or matrices. Exact analytic Stokes matrices remain available only for canonical families with independently implemented formulas.

The geometry is invariant under permutation of formal blocks and addition of a common exponential factor. Cover refinements change angular coordinates in the expected way. Parameter-dependent phase certification requires explicit real parameter coordinates so that `semialg` receives real polynomial conditions.
