"""Verify the L-shaped domain mesh and the Falk element eigenvalue solve on
it (Petersen 2022, Section 7.2): a non-convex domain with a re-entrant
corner at the origin, where the eigenfunction is singular and uniform
refinement cannot recover the full O(h^{2k}) rate -- exactly the point of
adaptivity (see test_adaptive.py).
"""
import numpy as np
from mpi4py import MPI

from falk_elasticity.convergence import loglog_rate
from falk_elasticity.domains import create_l_shaped_mesh
from falk_elasticity.eigenvalue import assemble_eigenproblem, solve_eigenproblem

LMBDA, MU = 1.0, 1.0
REFERENCE_KAPPA = 13.591920213185


def smallest_eigenvalue(n: int, k: int, nev: int = 4) -> float:
    domain = create_l_shaped_mesh(MPI.COMM_WORLD, n)
    _, A, B = assemble_eigenproblem(domain, k, LMBDA, MU)
    pairs = solve_eigenproblem(A, B, REFERENCE_KAPPA, nev=nev)
    kappa, _ = min(pairs, key=lambda pair: abs(pair[0] - REFERENCE_KAPPA))
    return kappa


def test_l_shaped_mesh_is_well_formed():
    n = 4
    domain = create_l_shaped_mesh(MPI.COMM_WORLD, n)
    tdim = domain.topology.dim
    num_cells = domain.topology.index_map(tdim).size_local
    # 3 remaining unit squares, each an n x n grid of cells, 2 triangles/cell
    assert num_cells == 3 * 2 * n * n

    x = domain.geometry.x
    assert np.isclose(x[:, 0].min(), -1.0) and np.isclose(x[:, 0].max(), 1.0)
    assert np.isclose(x[:, 1].min(), -1.0) and np.isclose(x[:, 1].max(), 1.0)

    # no vertex should lie in the interior of the removed quadrant
    interior_removed = np.logical_and(x[:, 0] > 1e-9, x[:, 1] < -1e-9)
    assert not interior_removed.any()


def test_smallest_eigenvalue_converges_under_uniform_refinement():
    ns = [4, 8, 12]
    values = [smallest_eigenvalue(n, k=1) for n in ns]
    errors = [abs(v - REFERENCE_KAPPA) for v in values]
    h = np.array([1.0 / n for n in ns])

    print(f"\nL-shaped k=1 smallest eigenvalue by N: {list(zip(ns, values, errors))}")

    assert errors[-1] < 1.0
    rate = loglog_rate(h, np.array(errors))
    print(f"L-shaped k=1 eigenvalue rate under uniform refinement: {rate:.2f} "
          f"(reduced from the smooth-domain rate of 2 by the corner singularity)")
    assert rate > 0.5
