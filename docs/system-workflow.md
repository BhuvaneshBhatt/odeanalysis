# System workflow

For `Y' = A(x)Y+b(x)`:

1. `analyze_system_singularity()` gives inexpensive raw connection data.
2. `formal_system_analysis()` performs bounded or adaptive verified formal reduction.
3. Inspect `FormalReductionCertificate` before using a completed normal form.
4. `system_stokes_geometry()` derives structural rays from verified exponential blocks.
5. `scalarize_system()` may recover a scalar equation when a cyclic output is certified.
6. Parameter families use `system_parameter_analysis()` or `system_formal_type_stratification()`.

Raw pole order is representation-dependent under meromorphic gauges. Constant gauges
preserve the residue spectrum and formal exponential data; general meromorphic gauges
preserve the differential-module formal class, not literal residue representatives.
