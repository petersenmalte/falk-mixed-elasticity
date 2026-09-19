"""Shared convergence-study helpers used by both the test suite
(tests/test_convergence.py) and scripts/convergence_study.py."""
import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import fem, mesh

from .elements import as_skew, as_stress
from .manufactured import trigonometric_solution
from .problem import solve_source_problem

QUADRATURE_DEGREE = 16


def l2_error(domain: mesh.Mesh, expr_exact: ufl.core.expr.Expr, expr_h: ufl.core.expr.Expr) -> float:
    dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": QUADRATURE_DEGREE})
    diff = expr_exact - expr_h
    local = fem.assemble_scalar(fem.form(ufl.inner(diff, diff) * dx))
    total = domain.comm.allreduce(local, op=MPI.SUM)
    return float(np.sqrt(total))


def errors_at(n: int, k: int, lmbda: float = 1.0, mu: float = 1.0):
    """Solve the manufactured Falk source problem on an n x n unit-square
    mesh and return (h, ||sigma-sigma_h||_0, ||gamma-gamma_h||_0, ||u-u_h||_0).
    """
    domain = mesh.create_unit_square(MPI.COMM_WORLD, n, n, mesh.CellType.triangle)
    exact = trigonometric_solution(domain, lmbda, mu)
    wh = solve_source_problem(domain, k, lmbda, mu, exact.f, QUADRATURE_DEGREE)

    sigma0_h, sigma1_h, u_h, q_h = ufl.split(wh)
    sigma_h = as_stress(sigma0_h, sigma1_h)
    gamma_h = as_skew(q_h)

    e_sigma = l2_error(domain, exact.sigma, sigma_h)
    e_gamma = l2_error(domain, exact.gamma, gamma_h)
    e_u = l2_error(domain, exact.u, u_h)
    return 1.0 / n, e_sigma, e_gamma, e_u


def loglog_rate(h, errors) -> float:
    """Least-squares log-log slope across the given refinement levels."""
    slope, _ = np.polyfit(np.log(h), np.log(errors), 1)
    return float(slope)
