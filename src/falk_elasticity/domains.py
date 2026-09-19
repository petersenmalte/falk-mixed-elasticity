"""Domain geometries beyond the unit square, Petersen (2022) Section 7.

dolfinx.mesh has no built-in non-convex domain, so this builds a structured
triangulation directly from explicit vertex/cell arrays.
"""
import basix
import basix.ufl
import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import fem
from dolfinx import mesh as dmesh

from .eigenvalue import assemble_eigenproblem, solve_eigenproblem
from .elements import as_skew, as_stress, falk_function_space
from .postprocessing import postprocess, postprocessed_eigenvalue


def create_l_shaped_mesh(comm, n: int) -> dmesh.Mesh:
    """Omega = (-1,1)x(0,1) U (-1,0)x(-1,0) (Petersen 2022, Section 7.2): the
    square (-1,1)^2 with its bottom-right quadrant removed, giving a
    re-entrant corner at the origin. Built as a uniform h=1/n grid over the
    bounding box with the excluded quadrant's cells dropped.
    """
    h = 1.0 / n
    ncells = 2 * n

    vertex_index: dict[tuple[int, int], int] = {}
    points: list[tuple[float, float]] = []
    cells: list[list[int]] = []

    def vertex(i: int, j: int) -> int:
        key = (i, j)
        if key not in vertex_index:
            vertex_index[key] = len(points)
            points.append((-1.0 + i * h, -1.0 + j * h))
        return vertex_index[key]

    for i in range(ncells):
        for j in range(ncells):
            cx, cy = -1.0 + (i + 0.5) * h, -1.0 + (j + 0.5) * h
            if cx > 0.0 and cy < 0.0:
                continue  # bottom-right quadrant: not part of the domain
            v00, v10 = vertex(i, j), vertex(i + 1, j)
            v01, v11 = vertex(i, j + 1), vertex(i + 1, j + 1)
            cells.append([v00, v10, v11])
            cells.append([v00, v11, v01])

    x = np.array(points, dtype=np.float64)
    c = np.array(cells, dtype=np.int64)
    coord_element = basix.ufl.element("Lagrange", basix.CellType.triangle, 1, shape=(2,))
    return dmesh.create_mesh(comm, c, ufl.Mesh(coord_element), x)


def create_cooks_membrane_mesh(comm, n: int) -> dmesh.Mesh:
    """Cook's membrane (Petersen 2022, Section 7.3): a tapered panel with
    the classic benchmark corners (0,0), (48,44), (48,60), (0,44), clamped
    on the left edge x=0 and free elsewhere. Built as an n x n structured
    grid on the reference square, mapped to the physical quadrilateral by
    bilinear (transfinite) interpolation of its four corners.
    """
    p00, p10 = np.array([0.0, 0.0]), np.array([48.0, 44.0])
    p01, p11 = np.array([0.0, 44.0]), np.array([48.0, 60.0])

    def physical(xi: float, eta: float) -> tuple[float, float]:
        p = (
            (1 - xi) * (1 - eta) * p00
            + xi * (1 - eta) * p10
            + (1 - xi) * eta * p01
            + xi * eta * p11
        )
        return float(p[0]), float(p[1])

    h = 1.0 / n
    points = [physical(i * h, j * h) for j in range(n + 1) for i in range(n + 1)]

    def idx(i: int, j: int) -> int:
        return j * (n + 1) + i

    cells: list[list[int]] = []
    for i in range(n):
        for j in range(n):
            v00, v10 = idx(i, j), idx(i + 1, j)
            v01, v11 = idx(i, j + 1), idx(i + 1, j + 1)
            cells.append([v00, v10, v11])
            cells.append([v00, v11, v01])

    x = np.array(points, dtype=np.float64)
    c = np.array(cells, dtype=np.int64)
    coord_element = basix.ufl.element("Lagrange", basix.CellType.triangle, 1, shape=(2,))
    return dmesh.create_mesh(comm, c, ufl.Mesh(coord_element), x)


def cooks_membrane_neumann_bcs(domain: dmesh.Mesh, W: fem.FunctionSpace) -> list:
    """sigma.n = 0 strongly on every boundary edge except the clamped left
    edge x=0: constrain both BDM row sub-spaces' normal-component dofs
    there to zero, encoding a free (traction-zero) boundary elsewhere.
    """
    fdim = domain.topology.dim - 1
    free_facets = dmesh.locate_entities_boundary(domain, fdim, lambda x: ~np.isclose(x[0], 0.0))
    bcs = []
    for sub in (0, 1):
        w_sub, _ = W.sub(sub).collapse()
        dofs = fem.locate_dofs_topological((W.sub(sub), w_sub), fdim, free_facets)
        zero = fem.Function(w_sub)
        bcs.append(fem.dirichletbc(zero, dofs, W.sub(sub)))
    return bcs


def cooks_membrane_eigenvalue_pair_at(n: int, k: int, lmbda: float, mu: float, target_kappa: float, nev: int = 6):
    """Solve the eigenproblem on Cook's membrane at resolution n, postprocess
    the mode closest to target_kappa, and return (kappa_h, kappa_star_h).
    """
    domain = create_cooks_membrane_mesh(MPI.COMM_WORLD, n)
    W = falk_function_space(domain, k)
    bcs = cooks_membrane_neumann_bcs(domain, W)
    _, A, B = assemble_eigenproblem(domain, k, lmbda, mu, bcs=bcs, W=W)
    pairs = solve_eigenproblem(A, B, target_kappa, nev=nev)
    kappa_h, x = min(pairs, key=lambda pair: abs(pair[0] - target_kappa))

    eigenmode = fem.Function(W)
    eigenmode.x.array[:] = x
    sigma_h = as_stress(eigenmode.sub(0), eigenmode.sub(1))
    gamma_h = as_skew(eigenmode.sub(3))
    u_h = eigenmode.sub(2)

    u_star_h = postprocess(domain, k, lmbda, mu, sigma_h, gamma_h, u_h)
    kappa_star_h = postprocessed_eigenvalue(domain, sigma_h, u_star_h)
    return kappa_h, kappa_star_h
