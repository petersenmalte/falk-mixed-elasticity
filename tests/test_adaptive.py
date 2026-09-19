"""Verify the Solve -> Estimate -> Mark -> Refine loop on the L-shaped
domain (Petersen 2022, Section 7.2): the estimator eta should decrease with
each adaptive step, tracking the true postprocessed-eigenvalue error down
despite the corner singularity that stalls uniform refinement
(test_l_shaped.py).
"""
from mpi4py import MPI

from falk_elasticity.adaptive import doerfler_mark, refine_marked
from falk_elasticity.domains import create_l_shaped_mesh
from falk_elasticity.estimator import solve_estimate

LMBDA, MU = 1.0, 1.0
REFERENCE_KAPPA = 13.591920213185


def test_estimator_decreases_under_adaptive_refinement():
    domain = create_l_shaped_mesh(MPI.COMM_WORLD, 4)
    etas = []
    errors = []

    for step in range(4):
        kappa_h, kappa_star_h, eta, eta2 = solve_estimate(domain, 1, LMBDA, MU, REFERENCE_KAPPA)
        etas.append(eta)
        errors.append(abs(kappa_star_h - REFERENCE_KAPPA))
        print(
            f"\nadaptive step {step}: kappa_h={kappa_h:.6f}, kappa*_h={kappa_star_h:.6f}, "
            f"error={errors[-1]:.6f}, eta={eta:.6f}"
        )

        marked = doerfler_mark(eta2.x.array, theta=0.5)
        domain = refine_marked(domain, marked)

    print(f"\netas: {etas}")
    print(f"errors: {errors}")

    assert etas[-1] < etas[0]
    assert errors[-1] < errors[0]
