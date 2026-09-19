"""A manufactured smooth solution on the unit square with homogeneous
Dirichlet data, used to verify the a priori rates of Petersen (2022), eq. (1):
    ||sigma - sigma_h||_0 + ||gamma - gamma_h||_0 = O(h^{k+1}),
    ||u - u_h||_0                                 = O(h^k).
"""
from dataclasses import dataclass

import ufl
from dolfinx import mesh

from .materials import C


@dataclass
class ManufacturedSolution:
    u: ufl.core.expr.Expr
    sigma: ufl.core.expr.Expr
    gamma: ufl.core.expr.Expr
    f: ufl.core.expr.Expr


def trigonometric_solution(domain: mesh.Mesh, lmbda: float, mu: float) -> ManufacturedSolution:
    """u = (sin(pi x) sin(pi y), sin(pi x) sin(pi y)), zero on d((0,1)^2).

    sigma, gamma, and f follow from the strong form (Petersen 2022, eq. 8) by
    symbolic differentiation in UFL, so the triple is exact by construction:
    C^{-1} sigma - grad(u) + gamma = 0 holds identically.
    """
    x = ufl.SpatialCoordinate(domain)
    s = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    u = ufl.as_vector([s, s])

    grad_u = ufl.grad(u)
    eps = ufl.sym(grad_u)
    gamma = ufl.skew(grad_u)
    sigma = C(eps, lmbda, mu)
    f = -ufl.div(sigma)

    return ManufacturedSolution(u=u, sigma=sigma, gamma=gamma, f=f)
