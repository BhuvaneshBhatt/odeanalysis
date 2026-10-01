# Certified numerical continuation

`certified_system_continuation()` is separate from heuristic numerical continuation.

For a constant homogeneous system `Y'=AY`, transport from `a` to `b` is exactly
`exp((b-a)A)`. The certified backend evaluates this matrix exponential with Arb complex
balls and returns serialized enclosures. Exact input conversion is itself part of the
certificate: supported SymPy numbers are constructed inside Arb from exact rational
arithmetic and rigorous named constants (currently `pi`), never by rounding to a decimal
point first. Unsupported numeric expressions are refused rather than certified as rounded
surrogates. Tests enforce reference containment, reversal, concatenation, exact-input,
and precision/refusal contracts.

The backend rejects inhomogeneous systems, symbolic endpoints, unsupported exact/numeric
constants, and variable-coefficient systems. Variable-coefficient certification requires validated integration with path
clearance, subdivision, truncation/remainder bounds, and enclosure-growth control.
Until those contracts exist, generic numerical connection matrices and analytic Stokes
matrices are not advertised as certified.
