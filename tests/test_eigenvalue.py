"""Verify the Falk element eigenvalue solve against the thesis's reference
eigenvalue on the unit square with homogeneous Dirichlet data (Petersen
2022, Section 7.1): the third eigenvalue in the spectrum,
kappa = 51.294997977322 -- the first two form a multiple eigenvalue, which
is why we look for the closest converged value rather than the third one
SLEPc happens to return.
"""
import numpy as np
import pytest
from mpi4py import MPI
from dolfinx import mesh

from falk_elasticity.convergence import loglog_rate
from falk_elasticity.eigenvalue import assemble_eigenproblem, solve_eigenproblem

LMBDA, MU = 1.0, 1.0
REFERENCE_KAPPA = 51.294997977322


def closest_eigenvalue(n: int, k: int, nev: int = 8) -> float:
    domain = mesh.create_unit_square(MPI.COMM_WORLD, n, n, mesh.CellType.triangle)
    _, A, B = assemble_eigenproblem(domain, k, LMBDA, MU)
    pairs = solve_eigenproblem(A, B, REFERENCE_KAPPA, nev=nev)
    assert pairs, "no eigenvalues converged"
    kappa, _ = min(pairs, key=lambda pair: abs(pair[0] - REFERENCE_KAPPA))
    return kappa


def test_lowest_order_eigenvalue_converges():
    """Petersen (2022), Fig. 8: |kappa - kappa_h| = O(h^2) for k=1."""
    ns = [8, 16, 24]
    values = [closest_eigenvalue(n, k=1) for n in ns]
    errors = [abs(v - REFERENCE_KAPPA) for v in values]
    h = [1.0 / n for n in ns]

    print(f"\nk=1 third eigenvalue by N: {list(zip(ns, values, errors))}")

    assert errors[-1] < 0.5, f"finest-mesh eigenvalue {values[-1]} too far from {REFERENCE_KAPPA}"
    rate = loglog_rate(np.array(h), np.array(errors))
    print(f"k=1 eigenvalue convergence rate: {rate:.2f} (theory: 2)")
    assert rate > 1.3


@pytest.mark.parametrize("k", [2, 3])
def test_higher_order_eigenvalue_is_close(k):
    """A single-mesh smoke test for k=2,3 -- O(h^4)/O(h^6) convergence
    would need meshes fine enough that the log-log rate is noise-dominated
    at this problem size, so this only checks the value itself is close.
    """
    value = closest_eigenvalue(12, k)
    error = abs(value - REFERENCE_KAPPA)
    print(f"\nk={k}, N=12 third eigenvalue: {value} (error {error:.4f})")
    assert error < 0.1
