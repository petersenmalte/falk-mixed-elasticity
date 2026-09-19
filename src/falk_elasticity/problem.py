"""Assembly and solve of the Falk mixed source problem, Petersen (2022)
eq. (11), specialised to the pure-Dirichlet case (u = 0 on the whole
boundary). There, Sigma_g = Sigma_0 is the full, unconstrained
BDM_k(Omega; R^{2x2}) space and the Dirichlet data enters naturally -- no
essential boundary condition needs to be assembled at all.
"""
import ufl
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem

from .elements import as_skew, as_stress, falk_function_space
from .materials import C_inv


def solve_source_problem(
    domain: mesh.Mesh,
    k: int,
    lmbda: float,
    mu: float,
    f: ufl.core.expr.Expr,
    quadrature_degree: int = 16,
) -> fem.Function:
    """Solve for (sigma_h, u_h, gamma_h) in the order-k Falk space."""
    W = falk_function_space(domain, k)
    sigma0, sigma1, u, q = ufl.TrialFunctions(W)
    tau0, tau1, v, p = ufl.TestFunctions(W)

    sigma = as_stress(sigma0, sigma1)
    tau = as_stress(tau0, tau1)
    gamma = as_skew(q)
    eta = as_skew(p)

    dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": quadrature_degree})

    a = (
        ufl.inner(C_inv(sigma, lmbda, mu), tau) * dx
        + ufl.inner(ufl.div(tau), u) * dx
        + ufl.inner(gamma, tau) * dx
        + ufl.inner(ufl.div(sigma), v) * dx
        + ufl.inner(sigma, eta) * dx
    )
    L = -ufl.inner(f, v) * dx

    problem = LinearProblem(
        a,
        L,
        bcs=[],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    return problem.solve()
