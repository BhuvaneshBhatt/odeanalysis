# Documentation map

`odeanalysis` has three mathematical workflows and one rigorous numerical workflow.

## I have a scalar ODE

Start with the [scalar workflow](scalar-workflow.md). Then use
[local analysis](local-analysis.md), [Fuchsian analysis](fuchsian-analysis.md),
[Newton invariants](newton-invariants.md), [formal asymptotics](formal-asymptotics.md),
[turning points](turning-points.md), [canonical equations](canonical-equations.md),
and [analytic continuation](analytic-continuation.md).

## I have a first-order system

Start with the [system workflow](system-workflow.md), then read
[system singularities](system-singularities.md), [formal system analysis](system-formal-analysis.md),
[system Stokes geometry](system-stokes.md), [system parameters](system-parameters.md),
and [scalar/system interoperability](scalar-system-interoperability.md).

## My coefficients contain parameters

Start with the [parameter workflow](parameter-workflow.md). Parameter cells are
semialgebraic objects; formal-type annotations use a bounded discrete signature.
Continuously varying exponent values are sample data, not constant-cell invariants.

## I need rigorous numerical transport

Read [certified continuation](certified-continuation.md). The current backend
certifies constant homogeneous systems with Arb balls and explicitly rejects
variable-coefficient validated integration.

## Evidence and reference

Read the [certification model](certification-model.md) and
[mathematical conventions](conventions.md) before relying on a result in a proof.
The [capabilities matrix](capabilities.md), [verification guide](verification.md),
[API classification](api-classification.md), [architecture](architecture.md), and
[traceability matrix](traceability.md) state the package boundaries.

The [example gallery](example-gallery.md) emphasizes mathematical questions;
[worked failures](worked-failures.md) explains correct refusal and incomplete results.
