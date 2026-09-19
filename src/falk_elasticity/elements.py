"""The Falk element triple (Petersen 2022, Fig. 2/3): BDM_k stress, a broken
P_{k-1} displacement, and a continuous-Lagrange-degree-k scalar multiplier
whose embedding S2(q) = [[0, q], [-q, 0]] represents the skew-symmetric part
of the gradient, enforcing sigma = sigma^T only weakly (in the mean)."""
import basix
import basix.ufl
import ufl
from dolfinx import fem, mesh


def falk_function_space(domain: mesh.Mesh, k: int) -> fem.FunctionSpace:
    """Mixed space Sigma_h x U_h x X_h for the order-k Falk element, k >= 1.

    Sub 0, 1: the two rows of the stress tensor, each in BDM_k.
    Sub 2:    the displacement u_h, in (discontinuous P_{k-1})^2.
    Sub 3:    the scalar q_h, in continuous P_k, with gamma_h = S2(q_h).
    """
    if k < 1:
        raise ValueError("the Falk element is only defined for k >= 1")
    cell = basix.CellType.triangle
    bdm = basix.ufl.element("BDM", cell, k)
    u_el = basix.ufl.element("DG", cell, k - 1, shape=(2,))
    q_el = basix.ufl.element("Lagrange", cell, k)
    mixed_el = basix.ufl.mixed_element([bdm, bdm, u_el, q_el])
    return fem.functionspace(domain, mixed_el)


def as_stress(row0: ufl.core.expr.Expr, row1: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
    """Assemble the 2x2 stress tensor from its two BDM_k row fields."""
    return ufl.as_matrix([[row0[0], row0[1]], [row1[0], row1[1]]])


def as_skew(q: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
    """S2(q) = [[0, q], [-q, 0]], Petersen (2022) Section 2."""
    return ufl.as_matrix([[0, q], [-q, 0]])
