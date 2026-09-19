"""Isotropic material law C and its inverse, Petersen (2022) eq. (3)-(4)."""
import ufl


def C(eps: ufl.core.expr.Expr, lmbda: float, mu: float) -> ufl.core.expr.Expr:
    """Hooke's law: C eps = lambda tr(eps) I + 2 mu eps."""
    dim = 2
    return lmbda * ufl.tr(eps) * ufl.Identity(dim) + 2.0 * mu * eps


def C_inv(tau: ufl.core.expr.Expr, lmbda: float, mu: float) -> ufl.core.expr.Expr:
    """Inverse material law. Stays bounded as lambda -> infinity (eq. 4),
    which is what keeps the method locking-free in the incompressible limit.
    """
    dim = 2
    return (1.0 / (2.0 * mu)) * (
        tau - (lmbda / (2.0 * (lmbda + mu))) * ufl.tr(tau) * ufl.Identity(dim)
    )
