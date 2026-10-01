# Documentation / implementation traceability

This matrix is a maintenance contract, not a line-coverage report.

| Capability | Reference/adversarial test | Negative/metamorphic test | Documentation/example |
| --- | --- | --- | --- |
| System singularities | `test_system_certification_corpus.py` | infinity/resonance cases | `system-singularities.md`, gallery 1 |
| Scalar/system interoperability | `test_scalar_system_differential_corpus.py` | noncyclic output | `scalar-system-interoperability.md`, gallery 4 |
| Formal reduction | `test_formal_reduction_adversarial.py` | bounded/adaptive + gauge/cover | `system-formal-analysis.md`, gallery 2/3/11 |
| Parameter formal types | `test_system_parameter_formal_types.py` | reparameterization + multi-point cells | `system-parameters.md`, gallery 5 |
| Structural system Stokes | `test_system_analysis_public.py` | common exponential shift | `system-stokes.md`, gallery 2 |
| Certified continuation | `test_certified_continuation_corpus.py` | reversal/concatenation/refusal | `certified-continuation.md`, gallery 6/12 |
| Scalar local/Frobenius | existing scalar reference suites | robustness/metamorphic suites | `local-analysis.md`, `fuchsian-analysis.md` |
| Turning/WKB | `test_turning_wkb.py` | parameter transition tests | `turning-points.md`, gallery 7/8 |
| Canonical recognition | canonical recognition corpus | projective/adversarial cases | `canonical-equations.md`, gallery 9/10 |

A public capability should not be upgraded in prose without corresponding executable
evidence. A documented limitation should have a refusal/negative test where practical.
