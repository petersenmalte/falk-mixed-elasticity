"""Local (element-wise) postprocessing, Petersen (2022) eq. (52)/(53), and
the postprocessed eigenvalue (Definition 5.5, eq. 59). Improves the O(h^{2k})
eigenvalue rate to O(h^{2k+2}) (Theorem 5.7) using only the already-computed
(sigma_h, u_h, gamma_h) -- no new global system beyond what's already
block-diagonal by construction: both the postprocessed displacement space
and its Lagrange multiplier are broken (discontinuous) with no facet terms,
so a "global" solve here is exactly independent per element, just expressed
as one assembly instead of a per-triangle loop.
"""
import basix
import basix.ufl
import ufl
from mpi4py import MPI
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem

from .eigenvalue import assemble_eigenproblem, solve_eigenproblem
from .elements import as_skew, as_stress
from .materials import C_inv


def postprocess(
    domain: mesh.Mesh,
    k: int,
    lmbda: float,
    mu: float,
    sigma_h,
    gamma_h,
    u_h,
    quadrature_degree: int = 16,
) -> fem.Function:
    """Solve (52)/(53) for u*_h in the broken P_k(T;R^2) space:

        (grad u*_h, grad v)_T + (v, g)_T = (C^{-1} sigma_h + gamma_h, grad v)_T
        (u*_h, w)_T                      = (u_h, w)_T

    for all (v, w) in P_k(T;R^2) x P_{k-1}(T;R^2), on every element T.
    """
    cell = basix.CellType.triangle
    u_star_el = basix.ufl.element("DG", cell, k, shape=(2,))
    g_el = basix.ufl.element("DG", cell, k - 1, shape=(2,))
    Wstar = fem.functionspace(domain, basix.ufl.mixed_element([u_star_el, g_el]))

    u_star, g = ufl.TrialFunctions(Wstar)
    v_star, w = ufl.TestFunctions(Wstar)

    dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": quadrature_degree})
    rhs_stress = C_inv(sigma_h, lmbda, mu) + gamma_h

    a_star = (
        ufl.inner(ufl.grad(u_star), ufl.grad(v_star)) * dx
        + ufl.inner(v_star, g) * dx
        + ufl.inner(u_star, w) * dx
    )
    L_star = ufl.inner(rhs_stress, ufl.grad(v_star)) * dx + ufl.inner(u_h, w) * dx

    problem = LinearProblem(
        a_star,
        L_star,
        bcs=[],
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
        petsc_options_prefix="falk_postprocessing_",
    )
    wstar = problem.solve()

    reason = problem.solver.getConvergedReason()
    if reason <= 0:
        raise RuntimeError(f"postprocessing KSP failed to converge (reason={reason})")

    return wstar.sub(0).collapse()


def postprocessed_eigenvalue(
    domain: mesh.Mesh, sigma_h, u_star_h: fem.Function, quadrature_degree: int = 16
) -> float:
    """Definition 5.5: kappa*_h = -(div(sigma_h), u*_h) / (u*_h, u*_h)."""
    dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": quadrature_degree})
    numerator = fem.assemble_scalar(fem.form(-ufl.inner(ufl.div(sigma_h), u_star_h) * dx))
    denominator = fem.assemble_scalar(fem.form(ufl.inner(u_star_h, u_star_h) * dx))
    numerator = domain.comm.allreduce(numerator, op=MPI.SUM)
    denominator = domain.comm.allreduce(denominator, op=MPI.SUM)
    return numerator / denominator


def eigenvalue_pair_at(n: int, k: int, lmbda: float, mu: float, reference_kappa: float, nev: int = 8):
    """Solve the eigenproblem on an n x n unit-square mesh, postprocess the
    mode closest to reference_kappa, and return (h, kappa_h, kappa_star_h).
    """
    domain = mesh.create_unit_square(MPI.COMM_WORLD, n, n, mesh.CellType.triangle)
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
    return 1.0 / n, kappa_h, kappa_star_h
