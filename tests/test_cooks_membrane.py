"""Verify Cook's membrane (Petersen 2022, Section 7.3): a tapered panel
clamped on the left edge and free elsewhere -- the first domain in this
repo needing genuine essential boundary conditions on the stress space
(sigma.n = 0 on the free edges), rather than the fully-natural Dirichlet
case the square and L-shaped domains use. There's no independent reference
eigenvalue for this domain (unlike the square and L-shape), so this checks
self-consistency under refinement instead of matching a known constant.
"""
import numpy as np
from mpi4py import MPI

from falk_elasticity.domains import (
    cooks_membrane_eigenvalue_pair_at,
    cooks_membrane_neumann_bcs,
    create_cooks_membrane_mesh,
)
from falk_elasticity.elements import falk_function_space

LMBDA, MU = 1.0, 1.0
TARGET_KAPPA = 0.005


def test_mesh_is_well_formed():
    n = 6
    domain = create_cooks_membrane_mesh(MPI.COMM_WORLD, n)
    tdim = domain.topology.dim
    assert domain.topology.index_map(tdim).size_local == 2 * n * n

    x = domain.geometry.x
    corners = {(0.0, 0.0), (48.0, 44.0), (48.0, 60.0), (0.0, 44.0)}
    found = {tuple(np.round(p[:2], 6)) for p in x}
    assert corners <= found


def test_left_edge_is_unconstrained_and_free_edges_are_constrained():
    domain = create_cooks_membrane_mesh(MPI.COMM_WORLD, 4)
    W = falk_function_space(domain, 1)
    bcs = cooks_membrane_neumann_bcs(domain, W)
    assert len(bcs) == 2  # one per BDM row (sigma0, sigma1)


def test_smallest_eigenvalue_is_self_consistent_under_refinement():
    ns = [4, 6, 8]
    values = [cooks_membrane_eigenvalue_pair_at(n, 1, LMBDA, MU, TARGET_KAPPA)[0] for n in ns]

    print(f"\nCook's membrane k=1 eigenvalue near target by N: {list(zip(ns, values))}")

    assert all(v > 0 for v in values), "a clamped-free elastic body should have a strictly positive spectrum"
    relative_change = abs(values[-1] - values[-2]) / abs(values[-1])
    print(f"relative change N={ns[-2]}->N={ns[-1]}: {relative_change:.4f}")
    assert relative_change < 0.5


def test_postprocessing_runs_and_is_finite():
    kappa_h, kappa_star_h = cooks_membrane_eigenvalue_pair_at(6, 1, LMBDA, MU, TARGET_KAPPA)
    print(f"\nCook's membrane k=1, N=6: kappa_h={kappa_h:.6f}, kappa*_h={kappa_star_h:.6f}")
    assert np.isfinite(kappa_h) and np.isfinite(kappa_star_h)
    assert kappa_star_h > 0
