# Formal analysis of linear systems

`formal_system_analysis()` wraps the exact Moser/spectral/Levelt--Turrittin machinery in one result. The fixed-budget defaults retain predictable historical behavior. Set `adaptive=True` to deepen the reduction search up to explicit depth and cover budgets.

The result includes a `FormalReductionCertificate`: whether the stored transformations verify, whether the reduction completed, the attempted depths and cover indices, the final ramification index, and any limitation. A partial verified reduction is not relabeled complete.

The reducer supports exact ramified pullbacks, Newton/Moser shearing, generalized-eigenspace spectral splitting, formal Sylvester elimination, regular-singular Levelt reduction, and scalar exponential extraction from completed blocks. Symbolic blocks are retained when exact spectral separation is available; unresolved repeated/nilpotent irregular blocks remain explicit limitations.

`scalarize_system()` supplies the reverse interoperability path when a homogeneous system has a certified cyclic output row. It constructs row jets `r_(k+1)=r_k'+r_k A`, verifies the cyclic matrix, and returns the corresponding monic scalar operator. `scalar_system_correspondence()` certifies the standard scalar -> companion -> scalar round trip.
