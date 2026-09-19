"""The a posteriori error estimator for the postprocessed eigenvalue,
Petersen (2022) Section 6 (the display between eq. 67 and Theorem 6.4):

    eta^2 = ||C^{-1} sigma_h + gamma_h - grad_Th u*_h||_0^2
          + sum_T h_T^2 ||kappa*_h u*_h + div(sigma_h)||_{0,T}^2
          + sum_E h_E^{-1} ||[u*_h]||_{0,E}^2
          + ||skw(sigma_h)||_0^2

computed as a DG0 cellwise indicator eta_T^2 (so it can drive Doerfler
marking for adaptive refinement), plus the single global scalar.
"""
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, mesh
from dolfinx.fem.petsc import assemble_vector

from .eigenvalue import assemble_eigenproblem, solve_eigenproblem
from .elements import as_skew, as_stress
from .materials import C_inv
from .postprocessing import postprocess, postprocessed_eigenvalue


def eigenvalue_estimator(
    domain: mesh.Mesh,
    lmbda: float,
    mu: float,
    sigma_h,
    gamma_h,
    u_star_h: fem.Function,
    kappa_star_h: float,
    quadrature_degree: int = 16,
) -> fem.Function:
    """Return a DG0 Function holding the cellwise indicator eta_T^2."""
    dg0 = fem.functionspace(domain, ("DG", 0))
    v0 = ufl.TestFunction(dg0)

    dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": quadrature_degree})
    dS = ufl.Measure("dS", domain=domain, metadata={"quadrature_degree": quadrature_degree})
    ds = ufl.Measure("ds", domain=domain, metadata={"quadrature_degree": quadrature_degree})

    h = ufl.CellDiameter(domain)
    h_e = ufl.FacetArea(domain)

    residual_stress = C_inv(sigma_h, lmbda, mu) + gamma_h - ufl.grad(u_star_h)
    residual_interior = kappa_star_h * u_star_h + ufl.div(sigma_h)
    skw_sigma = ufl.skew(sigma_h)
    jump_u = ufl.jump(u_star_h)

    indicator = (
        ufl.inner(residual_stress, residual_stress) * v0 * dx
        + h**2 * ufl.inner(residual_interior, residual_interior) * v0 * dx
        + ufl.inner(skw_sigma, skw_sigma) * v0 * dx
        + (1.0 / ufl.avg(h_e)) * ufl.inner(jump_u, jump_u) * ufl.avg(v0) * dS
        + (1.0 / h_e) * ufl.inner(u_star_h, u_star_h) * v0 * ds
    )

    eta2 = fem.Function(dg0)
    assemble_vector(eta2.x.petsc_vec, fem.form(indicator))
    eta2.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    return eta2


def global_estimator(eta2: fem.Function) -> float:
    total = float(eta2.x.array.sum())
    total = eta2.function_space.mesh.comm.allreduce(total, op=MPI.SUM)
    return float(np.sqrt(max(total, 0.0)))


def solve_estimate(domain: mesh.Mesh, k: int, lmbda: float, mu: float, reference_kappa: float, nev: int = 8):
    """Solve the eigenproblem, postprocess the mode closest to
    reference_kappa, and compute the a posteriori estimator for it.
    Returns (kappa_h, kappa_star_h, eta, eta2_dg0).
    """
    W, A, B = assemble_eigenproblem(domain, k, lmbda, mu)
    pairs = solve_eigenproblem(A, B, reference_kappa, nev=nev)
    kappa_h, x = min(pairs, key=lambda pair: abs(pair[0] - reference_kappa))

    eigenmode = fem.Function(W)
    eigenmode.x.array[:] = x
    sigma_h = as_stress(eigenmode.sub(0), eigenmode.sub(1))
    gamma_h = as_skew(eigenmode.sub(3))
    u_h = eigenmode.sub(2)

    u_star_h = postprocess(domain, k, lmbda, mu, sigma_h, gamma_h, u_h)
    kappa_star_h = postprocessed_eigenvalue(domain, sigma_h, u_star_h)

    eta2 = eigenvalue_estimator(domain, lmbda, mu, sigma_h, gamma_h, u_star_h, kappa_star_h)
    eta = global_estimator(eta2)
    return kappa_h, kappa_star_h, eta, eta2
