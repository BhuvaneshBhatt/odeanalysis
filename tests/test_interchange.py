import sympy as sp

from odeanalysis import (
    FormalODEData,
    formal_ode_data,
)


def test_interchange_repeated_regular_root_has_logarithmic_block():
    x = sp.symbols("x", positive=True)
    y = sp.Function("y")
    ode = x**2 * sp.diff(y(x), x, 2) + x * sp.diff(y(x), x)

    data = formal_ode_data(ode, y, x, point=0)

    assert isinstance(data, FormalODEData)
    assert data.complete
    assert data.dimension == 2
    assert data.operator_order == 2
    assert data.stokes is None
    assert len(data.blocks) == 1
    block = data.blocks[0]
    assert block.exponential_polynomial == 0
    assert block.has_logarithms
    assert block.amplitudes == (1, sp.log(x)) or len(block.amplitudes) == 2
    assert sp.simplify(block.nilpotent_exponent[0, 1] - 1) == 0
    assert data.cover_monodromy == sp.ImmutableMatrix([[1, 2 * sp.pi * sp.I], [0, 1]])


def test_interchange_mixed_irregular_repeated_block_is_isolated():
    x = sp.symbols("x")
    y = sp.Function("y")
    a0 = (2 * x**2 - 3 * x + 2) / (x**6 * (x - 2))
    a1 = (2 * x**3 - 6 * x**2 - 5 * x + 2) / (x**4 * (x - 2))
    a2 = (4 * x**2 - 9 * x - 2) / (x**2 * (x - 2))
    ode = sp.diff(y(x), x, 3) + a2 * sp.diff(y(x), x, 2) + a1 * sp.diff(y(x), x) + a0 * y(x)

    data = formal_ode_data(ode, y, x, point=0, terms=5, include_stokes=False)
    assert data.complete
    assert sorted(block.dimension for block in data.blocks) == [1, 2]
    repeated = next(block for block in data.blocks if block.dimension == 2)
    assert sp.simplify(repeated.exponential_polynomial - 1 / x) == 0
    assert repeated.has_logarithms
    assert sp.simplify(repeated.nilpotent_exponent[0, 1] - 1) == 0


def test_interchange_airyai_stokes_data_uses_blocks_not_internal_parts():
    x = sp.symbols("x", positive=True)
    y = sp.Function("y")
    ode = sp.diff(y(x), x, 2) - x * y(x)

    data = formal_ode_data(ode, y, x, point=sp.oo, terms=4)

    assert data.complete
    assert data.ramification_index == 2
    assert len(data.blocks) == 2
    assert data.stokes is not None
    assert data.stokes.common_ramification == 2
    assert data.stokes.pairs
    assert data.stokes.sector_geometry_complete
    projected_orders = {pair.exponential_order for pair in data.stokes.pairs}
    assert projected_orders == {sp.Rational(3, 2)}


def test_interchange_can_skip_stokes_recomputation():
    x = sp.symbols("x", positive=True)
    y = sp.Function("y")
    ode = sp.diff(y(x), x, 2) - x * y(x)

    data = formal_ode_data(ode, y, x, point=sp.oo, terms=3, include_stokes=False)
    assert data.complete
    assert data.stokes is None


def test_interchange_stokes_metadata_retains_original_plane_angles_and_sheets():
    x = sp.symbols("x", positive=True)
    y = sp.Function("y")
    data = formal_ode_data(sp.diff(y(x), x, 2) - x * y(x), y, x, point=sp.oo, terms=4)
    assert data.stokes is not None
    pair = data.stokes.pairs[0]
    assert {sp.simplify(a / sp.pi) for a in pair.equal_magnitude_original_angles} == {
        sp.Rational(1, 3),
        sp.S.One,
        sp.Rational(5, 3),
    }
    assert pair.equal_magnitude_sheets
    assert all(sector.original_representative_angle is not None for sector in data.stokes.sectors)
    assert all(sector.local_representative_angle is not None for sector in data.stokes.sectors)
    assert all(sector.sheet is not None for sector in data.stokes.sectors)


def test_interchange_retains_reduction_and_stokes_provenance():
    x = sp.symbols("x", positive=True)
    y = sp.Function("y")
    data = formal_ode_data(sp.diff(y(x), x, 2) - x * y(x), y, x, point=sp.oo, terms=4)
    assert data.provenance.reduction_path == (
        "scalar-operator",
        "newton-puiseux",
        "formal-block-reduction",
        "levelt-normalization",
    )
    assert data.provenance.source_block_count == len(data.blocks)
    assert data.provenance.stokes_requested
    assert data.provenance.stokes_computed == (data.stokes is not None)
    if data.stokes is not None:
        assert all(pair.source_branch_pairs for pair in data.stokes.pairs)
